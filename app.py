"""
app.py  -  Streamlit UI wrapper for AI-Powered Fragment Reassembly
"""
import random
import time
import streamlit as st
from reassembler import FragmentReassembler

st.set_page_config(page_title="Fragment Reassembly", page_icon="🧩", layout="wide")
st.title("🧩 AI-Powered Fragment Reassembly")
st.caption("Paste shuffled text fragments and let the AI reconstruct their original order.")

with st.sidebar:
    st.header("⚙️ Settings")
    use_fallback = st.toggle("Use TF-IDF fallback (no model download)", value=False)
    beam_width   = st.slider("Beam search width", 1, 10, 3)
    seed         = st.number_input("Random seed", value=42, step=1)
    evaluate     = st.toggle("Show evaluation metrics", value=True)

DEMO = [
    "The study began by identifying the primary sources of pollution in the watershed.",
    "Water samples were collected from twelve different locations along the river.",
    "Each sample was preprocessed to remove outliers and normalise chemical readings.",
    "Machine learning classifiers, including Random Forest and SVM, were then applied.",
    "Feature engineering reduced dimensionality while retaining 97% of the variance.",
    "The team concluded that ensemble methods outperform any single classifier.",
    "These findings suggest AI-driven monitoring can serve as an early-warning system.",
    "The result was a significant improvement in water quality across all tested sites.",
]

st.subheader("Input Fragments")
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Load demo text"):
        st.session_state["fragment_text"] = "\n".join(DEMO)

default_text = st.session_state.get("fragment_text", "\n".join(DEMO))
raw_input = st.text_area(
    "Enter one fragment per line:",
    value=default_text,
    height=220,
    key="fragment_text",
)

fragments = [ln.strip() for ln in raw_input.splitlines() if ln.strip()]
st.info(f"**{len(fragments)} fragments** detected.")

if st.button("🔀 Reassemble", type="primary", disabled=len(fragments) < 2):
    random.seed(int(seed))
    shuffled = fragments[:]
    random.shuffle(shuffled)

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("Shuffled order (input)")
        for i, f in enumerate(shuffled, 1):
            st.markdown(f"`{i}.` {f}")

    with st.spinner("Running semantic reassembly..."):
        reassembler = FragmentReassembler(use_fallback=use_fallback, beam_width=int(beam_width))
        t0 = time.perf_counter()
        ordered, indices, confidence = reassembler.reassemble(shuffled)
        elapsed = time.perf_counter() - t0

    with col_out:
        st.subheader("Reassembled order (output)")
        for i, f in enumerate(ordered, 1):
            st.markdown(f"`{i}.` {f}")

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence Score", f"{confidence:.4f}")
    m2.metric("Time Elapsed", f"{elapsed*1000:.1f} ms")
    m3.metric("Fragments", len(ordered))

    if evaluate:
        st.subheader("📊 Evaluation Metrics")
        original_indices = {frag: idx for idx, frag in enumerate(fragments)}
        predicted = [original_indices[shuffled[i]] for i in indices]
        metrics = reassembler.evaluate(fragments, predicted)
        e1, e2, e3 = st.columns(3)
        e1.metric("Perfect Match", "✅ Yes" if metrics["perfect_match"] else "❌ No")
        e2.metric("Position Accuracy", f"{metrics['accuracy']*100:.1f}%")
        e3.metric("Kendall Tau", f"{metrics['kendall_tau']:.4f}")
