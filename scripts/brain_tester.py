import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from signbridge_brain import infer
from signbridge_brain.manifest import load_manifest, select
from signbridge_brain.metrics import edit_counts

st.set_page_config(page_title="SignBridge Brain Tester (Component B)", layout="wide")

st.title("SignBridge Brain Tester - Component B")
st.caption("MediaPipe Features -> BiLSTM-CTC -> Glosses -> Sinhala | WER/Accuracy Demo")

# Sidebar
st.sidebar.header("Configuration")
runs = Path("runs")
ckpt_options = []
for name in ["pp1_demo_seed1/best.pt", "fallback_seed1/best.pt", "fallback_seed5/best.pt", "fallback_seed9/best.pt"]:
    p = runs / name
    if p.exists():
        ckpt_options.append(str(p))
if not ckpt_options:
    st.sidebar.error("No checkpoints found")
    st.stop()

ckpt_path = st.sidebar.selectbox("Select Checkpoint", ckpt_options)
st.sidebar.success(f"Using: {Path(ckpt_path).parent.name}")

try:
    ck = torch.load(ckpt_path, map_location="cpu")
    with st.sidebar.expander("Checkpoint Info (Self-Contained)"):
        st.write("model_type:", ck.get("model_type"))
        st.write("seed:", ck.get("seed"))
        st.write("feature_cfg:", ck.get("feature_cfg"))
        st.write("model_kwargs:", ck.get("model_kwargs"))
except Exception as e:
    st.sidebar.warning(str(e))

# Tabs at the top
tab1, tab2, tab3 = st.tabs(["Brain Prediction (on samples)", "WER Calculator (Manual Test)", "Evaluation Results"])

with tab1:
    st.subheader("Predict on Mock Landmark Sequences")
    root = Path("data/mock")
    items = load_manifest(root)
    val_items = select(items, split="val", source="synthetic")
    options = []
    for it in val_items:
        options.append((it.path + " | True: " + " ".join(it.glosses), it))
    if not options:
        st.warning("No samples found")
        st.stop()
    sample = st.selectbox("Select a landmark sequence (mock)", options, format_func=lambda x: x[0])[1]
    raw = np.load(root / sample.path)
    st.info("Input: " + sample.path + " | Shape: " + str(raw.shape) + " | True Glosses: " + " ".join(sample.glosses))
    if st.button("Run Brain Prediction", type="primary"):
        with st.spinner("Running inference..."):
            pred = infer.load_predictor(ckpt_path).predict(raw)
        c1, c2, c3 = st.columns(3)
        c1.metric("True Glosses", " ".join(sample.glosses))
        c2.metric("Predicted Glosses", " ".join(pred.glosses) if pred.glosses else "(empty)")
        c3.metric("Predicted Sinhala", pred.sinhala if pred.sinhala else "(empty)")
        st.metric("Confidence", f"{pred.confidence:.4f}")
        ref = sample.glosses
        hyp = pred.glosses
        if ref:
            s, d, i_cnt = edit_counts(ref, hyp)
            n = len(ref)
            wer = (s + d + i_cnt) / n if n else 0.0
            sent_acc = 1.0 if hyp == ref else 0.0
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Substitutions (S)", s)
            m2.metric("Deletions (D)", d)
            m3.metric("Insertions (I)", i_cnt)
            m4.metric("WER on this sample", f"{wer*100:.1f}%")
            st.success(f"Sentence Accuracy: {sent_acc*100:.0f}%")

with tab2:
    st.subheader("Manual WER Calculator")
    st.caption("Type reference and hypothesis glosses to calculate WER live")
    col1, col2 = st.columns(2)
    with col1:
        ref_str = st.text_input("Reference Glosses (space separated)", "I HAVE CHEST PAIN")
    with col2:
        hyp_str = st.text_input("Hypothesis Glosses (space separated)", "I PAIN")
    ref = ref_str.strip().split()
    hyp = hyp_str.strip().split()
    if ref:
        s, d, i_cnt = edit_counts(ref, hyp)
        n = len(ref)
        wer = (s + d + i_cnt) / n if n else 0.0
        sent_acc = 1.0 if hyp == ref else 0.0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Substitutions (S)", s)
        c2.metric("Deletions (D)", d)
        c3.metric("Insertions (I)", i_cnt)
        c4.metric("WER", f"{wer*100:.1f}%")
        c5.metric("Sentence Accuracy", f"{sent_acc*100:.0f}%")
        st.code(f"WER = (S + D + I) / N_ref = ({s} + {d} + {i_cnt}) / {n} = {wer*100:.1f}%")
        if hyp == ref:
            st.success("Perfect match! WER = 0% | Sentence Accuracy = 100%")
        else:
            st.warning("Mismatch - see edit counts above")

with tab3:
    st.subheader("Evaluation Results")
    runs_path = Path("runs")
    for name, label in [("pp1_demo_seed1", "BiLSTM-CTC (pp1_demo_seed1)"), ("fallback_seed1", "Fallback (seed 1)"), ("fallback_seed5", "Fallback (seed 5)"), ("fallback_seed9", "Fallback (seed 9)")]:
        p = runs_path / name / "eval_val_greedy.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
                s_res = d["by_source"]["synthetic"]
                st.write("**" + label + "** — WER: **" + f"{s_res['wer']*100:.1f}%" + "** | Sentence Acc: **" + f"{s_res['sentence_accuracy']*100:.1f}%" + "** | Seqs: " + str(s_res['n_seq']))
            except Exception as e:
                st.write(f"{label}: {e}")
    st.info("Targets: WER < 30% (synthetic), < 50% (real) | Sentence Acc > 60%")
    st.caption("Results on MOCK data to validate pipeline.")
