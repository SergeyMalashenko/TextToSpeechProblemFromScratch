from preprocess import get_dataset, collate_fn_transformer
from torch.utils.data import DataLoader

from network import *
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm

import torch as t
import torch.nn as nn


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * (warmup_step ** 0.5) * min(step_num * (warmup_step ** -1.5), step_num ** -0.5)
    for pg in optimizer.param_groups:
        pg["lr"] = lr


@t.no_grad()
def _stop_metrics(stop_logits, stop_target, valid_mask, mel_lengths, thr=0.5):
    """
    stop_logits: (B, T) or (B, T, 1)
    stop_target: (B, T) float {0,1}
    valid_mask:  (B, T) bool   True for real frames (non-padding)
    mel_lengths: (B,) long
    """
    if stop_logits.dim() == 3:
        stop_logits = stop_logits.squeeze(-1)

    probs = t.sigmoid(stop_logits)
    pred = (probs > thr)

    # Evaluate metrics on all positions where stop_target is defined (we include padding too,
    # because we set stop_target=1 at last frame and after it).
    # If you want to evaluate only on valid frames + last frame, you can change eval_mask.
    eval_mask = t.ones_like(stop_target, dtype=t.bool)

    y = stop_target.bool()
    yhat = pred

    tp = ((yhat & y) & eval_mask).sum().item()
    fp = ((yhat & ~y) & eval_mask).sum().item()
    fn = ((~yhat & y) & eval_mask).sum().item()
    tn = ((~yhat & ~y) & eval_mask).sum().item()

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))

    # p_last_mean and p_prev_mean at last real indices
    B, T = probs.shape
    last_idx = (mel_lengths - 1).clamp(min=0)          # (B,)
    prev_idx = (mel_lengths - 2).clamp(min=0)          # (B,)

    b_idx = t.arange(B, device=probs.device)
    p_last = probs[b_idx, last_idx]
    p_prev = probs[b_idx, prev_idx]

    p_last_mean = float(p_last.mean().item())
    p_prev_mean = float(p_prev.mean().item())

    # early_stop_rate: any predicted stop at real frames before last frame
    # For each sample, check positions [0, last_idx-1] (valid frames only) for any stop==1
    early = []
    for b in range(B):
        Li = int(mel_lengths[b].item())
        if Li <= 1:
            early.append(0.0)
            continue
        early_pred = yhat[b, :Li-1].any().item()
        early.append(1.0 if early_pred else 0.0)
    early_stop_rate = float(sum(early) / max(1, len(early)))

    return acc, prec, rec, early_stop_rate, p_prev_mean, p_last_mean


def main():
    dataset = get_dataset()
    global_step = 0

    m = nn.DataParallel(Model().cuda())
    
    m.train()
    
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    # Stop loss (use logits + pos_weight). Keep your original idea of pos_weight=5.
    pos_weight = t.tensor([5.0], device="cuda")
    stop_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    writer = SummaryWriter()
    
    dataloader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=collate_fn_transformer,
        drop_last=True,
        num_workers=32,
        pin_memory=True,
    )

    for epoch in range(hp.epochs):
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description(f"Processing at epoch {epoch}")
            global_step += 1

            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            character, mel, pos_text, pos_mel, _, _ = data
            
            character = character.cuda(non_blocking=True)
            mel       = mel      .cuda(non_blocking=True)
            pos_text  = pos_text .cuda(non_blocking=True)
            pos_mel   = pos_mel  .cuda(non_blocking=True)
            
            go = t.zeros((mel.size(0), 1, mel.size(2)), device=mel.device, dtype=mel.dtype)
            mel_input = t.cat([go, mel[:, :-1, :]], dim=1)  # (B, T, 80)
            # ===== build lengths & masks from pos_mel =====
            # pos_mel is 1..T, padded with 0
            valid_mask  = pos_mel.ne(0)                   # (B, T) bool
            mel_lengths = valid_mask.long().sum(dim=1)   # (B,) long
            
            B, T = pos_mel.shape
            last_idx = (mel_lengths - 1).clamp(min=0)
            b_idx = t.arange(B, device=mel.device)
            
            # stable stop target: only last frame is 1
            stop_target = t.zeros((B, T), device=mel.device)
            stop_target[b_idx, last_idx] = 1.0
            
            # ===== forward =====
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m(
                character, mel_input, pos_text, pos_mel
            )

            # stop_preds expected shape (B, T, 1) -> squeeze to (B, T)
            if stop_preds.dim() == 3:
                stop_logits = stop_preds.squeeze(-1)
            else:
                stop_logits = stop_preds

            # ===== losses =====
            mel_loss      = nn.L1Loss()(mel_pred    , mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)
            
            # stop loss on all time positions (incl padding), using stop_target defined above
            stop_loss = stop_criterion(stop_logits, stop_target)

            # You can tune this weight; start small-ish so it doesn't dominate.
            stop_weight = 0.0
            loss = mel_loss + post_mel_loss + stop_weight * stop_loss

            # ===== metrics =====
            with t.no_grad():
                acc, prec, rec, early_stop_rate, p_prev_mean, p_last_mean = _stop_metrics(
                    stop_logits, stop_target, valid_mask, mel_lengths, thr=0.5
                )

            # ===== logging =====
            writer.add_scalars(
                "training_loss",
                {
                    "mel_loss": float(mel_loss.item()),
                    "post_mel_loss": float(post_mel_loss.item()),
                    "stop_loss": float(stop_loss.item()),
                    "total": float(loss.item()),
                },
                global_step,
            )

            writer.add_scalars(
                "stop_stats",
                {
                    "accuracy"       : float(acc),
                    "precision"      : float(prec),
                    "recall"         : float(rec),
                    "early_stop_rate": float(early_stop_rate),
                    "p_prev_mean"    : float(p_prev_mean),
                    "p_last_mean"    : float(p_last_mean),
                },
                global_step,
            )

            # ===== attention logging (safe & meaningful) =====
            # attn tensors are (h*B, Tq, Tk). We'll log 1 sample (b=0), head=0.
            if global_step % hp.image_step == 1:
                b = 0
                h = 0
                H = m.module.decoder.selfattn_layers[0].h  # should be 4 in your code

                def log_attn_stack(tag_prefix, attn_list):
                    # attn_list: list of layers, each (h*B, Tq, Tk)
                    for layer_i, attn in enumerate(attn_list):
                        if attn.dim() != 3:
                            continue
                        hb, Tq, Tk = attn.shape
                        if hb < H * (b + 1):
                            continue
                        idx = h * B + b  # head-major packing: first all batch for head0, then head1, ...
                        if idx >= hb:
                            continue

                        A = attn[idx]  # (Tq, Tk), already softmaxed
                        # normalize to 0..1 for image
                        A_img = A.unsqueeze(0)  # (1, Tq, Tk)
                        writer.add_image(f"{tag_prefix}/layer{layer_i}", A_img, global_step)

                log_attn_stack("attn/encdec", attn_probs)
                log_attn_stack("attn/enc", attns_enc)
                log_attn_stack("attn/dec", attns_dec)

            # ===== backward =====
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()

            if global_step % hp.save_step == 0:
                os.makedirs(hp.checkpoint_path, exist_ok=True)
                t.save(
                    {"model": m.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step},
                    os.path.join(hp.checkpoint_path, f"checkpoint_transformer_{global_step}.pth.tar"),
                )


if __name__ == "__main__":
    main()
