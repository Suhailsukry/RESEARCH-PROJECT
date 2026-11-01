"""Train the BiLSTM-CTC model on the (mock or real) synthetic-sentence dataset.

    python -m signbridge_brain.train --config configs/default.yaml --seed 1

Only the TRAIN and VAL splits are read here. Test / real splits are touched only by
evaluate.py --final (quality gate: test set never used for tuning).
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import load_config
from .dataset import SequenceDataset, collate_ctc
from .decoding.ctc import collapse
from .features.landmarks import feature_dim
from .manifest import assert_signer_independent, load_manifest, select
from .metrics import corpus_scores
from .models import build_model
from .utils import append_jsonl, pick_device, run_info, set_seed
from .vocab import load_vocab_bundle


@torch.no_grad()
def validate(model, loader, criterion, blank: int, device) -> dict:
    model.eval()
    loss_sum, n_seq, frames, blank_frames = 0.0, 0, 0, 0
    refs, hyps = [], []
    for x, xl, y, yl in loader:
        x, y = x.to(device), y.to(device)
        logp, out_lens = model(x, xl)
        loss = criterion(logp.transpose(0, 1), y, out_lens.cpu(), yl)
        loss_sum += float(loss) * len(xl)
        n_seq += len(xl)
        pred = logp.argmax(dim=-1).cpu().numpy()
        for b in range(len(xl)):
            p = pred[b, : int(out_lens[b])]
            frames += len(p)
            blank_frames += int((p == blank).sum())
            hyps.append(collapse(p, blank))
            refs.append(y[b, : int(yl[b])].cpu().tolist())
    scores = corpus_scores(refs, hyps)
    scores.update(val_loss=loss_sum / max(n_seq, 1), blank_ratio=blank_frames / max(frames, 1))
    return scores


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train BiLSTM-CTC")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-name", default="bilstm_ctc")
    ap.add_argument("--resume", action="store_true", help="continue from <run_dir>/last.pt")
    ap.add_argument("--abort-on-collapse", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    set_seed(cfg.seed)
    device = pick_device(cfg.train.device)

    bundle = load_vocab_bundle(cfg.data.vocab_file)
    vocab = bundle.vocab
    root = Path(cfg.data.root)
    items = load_manifest(root)
    assert_signer_independent(items)
    train_items = select(items, split=cfg.data.train_split, source="synthetic")
    val_items = select(items, split=cfg.data.val_split, source="synthetic")
    if not train_items or not val_items:
        raise SystemExit("Empty train/val split. Run: python scripts/make_mock_data.py --out data/mock")

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        SequenceDataset(root, train_items, vocab, cfg.features, cfg.augment, train=True),
        batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_ctc,
        num_workers=cfg.train.num_workers, generator=g)
    val_loader = DataLoader(
        SequenceDataset(root, val_items, vocab, cfg.features),
        batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_ctc,
        num_workers=cfg.train.num_workers)

    model_kwargs = dict(
        input_dim=feature_dim(cfg.features), num_classes=vocab.size, proj_dim=cfg.model.proj_dim,
        hidden_size=cfg.model.hidden_size, num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout, input_dropout=cfg.model.input_dropout)
    model = build_model(cfg.model.type, model_kwargs).to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.train.epochs, 1))

    run_dir = Path(cfg.train.runs_dir) / f"{args.run_name}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.jsonl"
    start_epoch, best_wer = 1, float("inf")

    if args.resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"])
        ts = state["extra"]["train_state"]
        optim.load_state_dict(ts["optim"])
        sched.load_state_dict(ts["sched"])
        start_epoch, best_wer = ts["epoch"] + 1, ts["best_wer"]
        print(f"Resumed from epoch {ts['epoch']} (best val WER {best_wer:.3f})")
    else:
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
        import json
        with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_info(cfg.seed, cfg.to_dict()), f, indent=1)

    print(f"device={device} train={len(train_items)} val={len(val_items)} "
          f"input_dim={model_kwargs['input_dim']} classes={vocab.size} params="
          f"{sum(p.numel() for p in model.parameters()):,}")

    def ckpt(path, epoch):
        save_checkpoint(
            path, model_type=cfg.model.type, model=model, model_kwargs=model_kwargs,
            feature_cfg=cfg.features, vocab=vocab, sinhala=bundle.sinhala, seed=cfg.seed,
            extra={"train_state": {"optim": optim.state_dict(), "sched": sched.state_dict(),
                                   "epoch": epoch, "best_wer": float(best_wer)}})

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for x, xl, y, yl in train_loader:
            x, y = x.to(device), y.to(device)
            logp, out_lens = model(x, xl)
            loss = criterion(logp.transpose(0, 1), y, out_lens.cpu(), yl)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optim.step()
            loss_sum += float(loss) * len(xl)
            n += len(xl)
        sched.step()

        v = validate(model, val_loader, criterion, vocab.blank_id, device)
        record = {"epoch": epoch, "train_loss": loss_sum / max(n, 1), "val_loss": v["val_loss"],
                  "val_wer": v["wer"], "val_sent_acc": v["sentence_acc"], "blank_ratio": v["blank_ratio"],
                  "empty_hyp_ratio": v["empty_hyp_ratio"], "lr": optim.param_groups[0]["lr"],
                  "sec": round(time.time() - t0, 1)}

        collapsed = False
        if epoch % cfg.train.blank_check_every == 0:  # README note 7: check every 5 epochs
            collapsed = v["empty_hyp_ratio"] >= 0.9 or v["blank_ratio"] >= 0.98
            record["collapse_flag"] = bool(collapsed)
            if collapsed:
                print(f"  !! CTC BLANK COLLAPSE suspected at epoch {epoch} (blank_ratio={v['blank_ratio']:.3f}, "
                      f"empty_hyps={v['empty_hyp_ratio']:.2f}). If it persists past ~epoch 15: lower LR, "
                      f"shorten sequences, reduce face features, or switch to the fallback classifier.")
        append_jsonl(log_path, record)
        print(f"epoch {epoch:3d} | train {record['train_loss']:.3f} | val {v['val_loss']:.3f} | "
              f"WER {v['wer']:.3f} | sent_acc {v['sentence_acc']:.3f} | blank {v['blank_ratio']:.2f} | "
              f"{record['sec']}s")

        if v["wer"] < best_wer:
            best_wer = v["wer"]
            ckpt(run_dir / "best.pt", epoch)
        ckpt(run_dir / "last.pt", epoch)  # every epoch (Colab timeouts)
        if collapsed and args.abort_on_collapse:
            raise SystemExit("Aborted: blank collapse")

    print(f"Done. Best val WER {best_wer:.3f}. Checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
