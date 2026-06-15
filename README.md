# AI-Powered Fragment Reassembly

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
