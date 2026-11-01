"""Generate mock landmark data.   python scripts/make_mock_data.py --out data/mock"""
import argparse
import json

from signbridge_brain.data.mock import build_mock_dataset
from signbridge_brain.vocab import load_vocab_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mock")
    ap.add_argument("--vocab", default="configs/mock_vocab.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-signers", type=int, default=4)
    ap.add_argument("--sents-per-signer", type=int, default=30)
    ap.add_argument("--real-sents-per-signer", type=int, default=15)
    ap.add_argument("--isolated-per-gloss", type=int, default=4)
    a = ap.parse_args()
    b = load_vocab_bundle(a.vocab)
    stats = build_mock_dataset(
        a.out, b.vocab.glosses, b.sentences, seed=a.seed, n_train_signers=a.train_signers,
        sents_per_signer=a.sents_per_signer, real_sents_per_signer=a.real_sents_per_signer,
        isolated_per_gloss_per_signer=a.isolated_per_gloss,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
