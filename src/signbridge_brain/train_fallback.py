"""Train the Appendix-B fallback: isolated-window classifier (class 0 = background)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import load_config
from .dataset import ClipDataset
from .features.landmarks import feature_dim
from .manifest import assert_signer_independent, load_clip_manifest, select
from .models import build_model
from .utils import append_jsonl, pick_device, set_seed
from .vocab import load_vocab_bundle


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train sliding-window fallback classifier")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default="fallback")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed(cfg.seed)
    device = pick_device(cfg.train.device)
    fb = cfg.fallback

    bundle = load_vocab_bundle(cfg.data.vocab_file)
    vocab = bundle.vocab
    root = Path(cfg.data.root)
    clips = load_clip_manifest(root)
    assert_signer_independent(clips)
    tr, va = select(clips, split="train"), select(clips, split="val")
    if not tr or not va:
        raise SystemExit("No isolated clips found. Run scripts/make_mock_data.py first.")

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    tr_loader = DataLoader(ClipDataset(root, tr, vocab, cfg.features, fb.window), batch_size=fb.batch_size,
                           shuffle=True, generator=g)
    va_loader = DataLoader(ClipDataset(root, va, vocab, cfg.features, fb.window), batch_size=fb.batch_size)

    kwargs = dict(input_dim=feature_dim(cfg.features), num_classes=vocab.size, hidden_size=fb.hidden_size,
                  num_layers=fb.num_layers, dropout=fb.dropout)
    model = build_model("fallback_window", kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=fb.lr, weight_decay=cfg.train.weight_decay)
    ce = nn.CrossEntropyLoss()

    run_dir = Path(cfg.train.runs_dir) / f"{args.run_name}_seed{cfg.seed}"
    best = -1.0
    for epoch in range(1, fb.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            loss = ce(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            tot, n = tot + float(loss) * len(y), n + len(y)
        model.eval()
        correct, m = 0, 0
        with torch.no_grad():
            for x, y in va_loader:
                correct += int((model(x.to(device)).argmax(-1).cpu() == y).sum())
                m += len(y)
        acc = correct / max(m, 1)
        append_jsonl(run_dir / "metrics.jsonl", {"epoch": epoch, "train_loss": tot / max(n, 1), "val_clip_acc": acc})
        print(f"epoch {epoch:3d} | train {tot / max(n, 1):.3f} | val clip acc {acc:.3f}")
        extra = {"decode": {"window": fb.window, "stride": fb.stride, "threshold": fb.threshold},
                 "epoch": epoch, "val_clip_acc": acc}
        save_checkpoint(run_dir / "last.pt", model_type="fallback_window", model=model, model_kwargs=kwargs,
                        feature_cfg=cfg.features, vocab=vocab, sinhala=bundle.sinhala, seed=cfg.seed, extra=extra)
        if acc > best:
            best = acc
            save_checkpoint(run_dir / "best.pt", model_type="fallback_window", model=model, model_kwargs=kwargs,
                            feature_cfg=cfg.features, vocab=vocab, sinhala=bundle.sinhala, seed=cfg.seed, extra=extra)
    print(f"Done. Best val clip acc {best:.3f}. Checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
