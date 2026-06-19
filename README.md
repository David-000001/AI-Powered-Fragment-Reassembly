<div align="center">

# 🧩 AI-Powered Fragment Reassembly

### *Reconstruct scrambled text with zero labels — powered by semantic AI*

[![Live Demo](https://img.shields.io/badge/Live_Demo-Netlify-00C7B7?style=for-the-badge&logo=netlify)](https://ai-fragment-reassembly.netlify.app)
[![GitHub](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/David-000001/AI-Powered-Fragment-Reassembly)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)](https://python.org)

</div>

---

## What Is This?

Given a set of **shuffled text fragments**, this engine reconstructs their **most likely original order** using semantic similarity — no labels, no position markers, pure AI reasoning over meaning.

---

## Live Web Demo

The web app runs **entirely in the browser**. No install, no API key.

Deploy your own in 30 seconds on Netlify.

---

## How It Works

| Step | Component | Detail |
|------|-----------|--------|
| 1 | Embedding | TF-IDF vectors (browser) or sentence-transformers (Python) |
| 2 | Similarity | Pairwise cosine similarity matrix N x N |
| 3 | Ordering | Greedy nearest-neighbour OR Beam Search |
| 4 | Anchor | Fragment with lowest mean similarity = likely opener |
| 5 | Evaluation | Position accuracy + Kendall Tau rank correlation |

---

## Project Structure

```
AI-Powered-Fragment-Reassembly/
├── index.html       Web app entry point (Netlify)
├── style.css        Dark-mode design system
├── app.js           Client-side AI engine (TF-IDF + beam search)
├── netlify.toml     Netlify config + security headers
├── main.py          Python CLI entry point
├── reassembler.py   Core ordering logic
├── embeddings.py    Sentence embedding wrapper
├── requirements.txt Python dependencies
└── README.md
```

---

## Deploy to Netlify

1. Fork this repository
2. Go to app.netlify.com, click Add new site > Import from Git
3. Select your fork — Netlify auto-detects netlify.toml
4. Click Deploy site

No build step required.

---

## Python CLI

```bash
git clone https://github.com/David-000001/AI-Powered-Fragment-Reassembly.git
cd AI-Powered-Fragment-Reassembly
pip install -r requirements.txt

# Run built-in demo
python main.py

# Custom fragments (one per line)
python main.py --input my_fragments.txt

# Evaluate accuracy
python main.py --input my_fragments.txt --evaluate

# TF-IDF fallback (offline)
python main.py --fallback

# Adjust beam width
python main.py --beam 5
```

---

## Tech Stack

| Layer | Web App | Python CLI |
|-------|---------|------------|
| Language | JavaScript ES2022 | Python 3.9+ |
| Embeddings | TF-IDF client-side | sentence-transformers |
| Similarity | Cosine vanilla JS | scikit-learn |
| Search | Beam Search | Greedy + Beam Search |
| Metrics | Kendall Tau, Accuracy | Kendall Tau, Accuracy |
| Deploy | Netlify static | Local or any Python env |

---

## Use Cases

- Document reconstruction after data corruption
- Forensic text analysis of fragmented records
- NLP research on semantic coherence
- Educational AI demos
- Content pipeline quality assurance

---

## Author

**Avijit Chandra Dey**
B.Sc. Computer Science (Artificial Intelligence)
University of Malaya

[![GitHub](https://img.shields.io/badge/GitHub-David--000001-181717?logo=github)](https://github.com/David-000001)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/avijitchandradey)

---

<div align="center">
  <sub>Built with semantic AI and cosine similarity</sub>
</div># AI-Powered Fragment Reassembly

An AI-driven Python system that takes a set of shuffled text fragments and reconstructs their most likely original order using semantic sentence embeddings and intelligent search algorithms.

---

## Overview

When documents, paragraphs, or records are fragmented and their order is lost, this system uses **sentence-level semantic similarity** to intelligently reassemble them — no labels or positional markers required.

**How it works:**
1. Each fragment is encoded into a dense vector using a pre-trained sentence-transformer model (`all-MiniLM-L6-v2`)
2. A cosine similarity matrix is built between all fragment pairs
3. A **greedy nearest-neighbour chain** (or optional **beam search**) finds the ordering that maximises consecutive semantic coherence
4. Evaluation metrics (accuracy, Kendall Tau) compare the predicted order against the ground truth

---

## Features

- Semantic reassembly using `sentence-transformers`
- Beam search for improved accuracy on smaller inputs
- TF-IDF fallback for offline / resource-constrained environments
- Kendall Tau and position-accuracy evaluation metrics
- Coloured CLI output via `colorama`
- Supports custom fragment files (one fragment per line)

---

## Project Structure

```
AI-Powered-Fragment-Reassembly/
├── main.py           # Entry point & CLI
├── reassembler.py    # Core ordering logic (greedy + beam search)
├── embeddings.py     # Sentence embedding wrapper with TF-IDF fallback
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/David-000001/AI-Powered-Fragment-Reassembly.git
cd AI-Powered-Fragment-Reassembly

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~80 MB). Use `--fallback` to skip this.

---

## Usage

### Run the built-in demo
```bash
python main.py
```

### Reassemble fragments from a file (one fragment per line)
```bash
python main.py --input my_fragments.txt
```

### Evaluate against ground-truth (file must be in correct order)
```bash
python main.py --input my_fragments.txt --evaluate
```

### Use TF-IDF fallback (no internet required)
```bash
python main.py --fallback
```

### Adjust beam search width
```bash
python main.py --beam 5
```

---

## Example Output

```
────────────────────────────────────────────────────────────
  INPUT — Shuffled Fragments
────────────────────────────────────────────────────────────
  [ 1] Machine learning classifiers, including Random Forest and SVM, were then applied.
  [ 2] The study began by identifying the primary sources of pollution in the watershed.
  ...

  Reassembling 8 fragments ...

────────────────────────────────────────────────────────────
  OUTPUT — Reassembled Order
────────────────────────────────────────────────────────────
  [ 1] The study began by identifying the primary sources of pollution in the watershed.
  [ 2] Water samples were collected from twelve different locations along the river.
  ...

  Confidence Score : 0.7812
  Time Elapsed     : 342.5 ms

────────────────────────────────────────────────────────────
  EVALUATION METRICS
────────────────────────────────────────────────────────────
  Result           : ✔  PERFECT MATCH
  Position Accuracy: 100.0%
  Kendall Tau      : 1.0000  (1.0 = perfect)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Similarity | `scikit-learn` (cosine similarity) |
| Fallback | TF-IDF vectorisation |
| Ordering | Greedy chain + Beam search |
| CLI | `argparse`, `colorama` |

---

## Author

**Avijit Chandra Dey**  
B.Sc. Computer Science (AI) — University of Malaya  
[GitHub](https://github.com/David-000001) · [LinkedIn](https://linkedin.com/in/avijit-chandra-dey-340b61294)
