"""
main.py
-------
Entry point for the AI-Powered Fragment Reassembly system.

Usage
-----
  # Demo mode (built-in example)
  python main.py

  # Custom fragments from a text file (one fragment per line)
  python main.py --input fragments.txt

  # Evaluate against ground-truth (fragments already in correct order)
  python main.py --input fragments.txt --evaluate

  # Use TF-IDF fallback (no internet / GPU required)
  python main.py --fallback
"""

from __future__ import annotations

import argparse
import random
import sys
import time

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

from reassembler import FragmentReassembler

# ---------------------------------------------------------------------------
# Demo text (fragmented paragraph)
# ---------------------------------------------------------------------------
DEMO_TEXT = [
    "The result was a significant improvement in water quality across all tested sites.",
    "Water samples were collected from twelve different locations along the river.",
    "Machine learning classifiers, including Random Forest and SVM, were then applied to the processed data.",
    "The study began by identifying the primary sources of pollution in the watershed.",
    "Feature engineering reduced dimensionality while retaining 97% of the variance.",
    "Each sample was preprocessed to remove outliers and normalise chemical readings.",
    "These findings suggest that AI-driven monitoring can serve as an early-warning system for pollution events.",
    "The team concluded that a combination of ensemble methods outperforms any single classifier.",
]


def _color(text: str, color: str) -> str:
    if HAS_COLOR:
        return f"{color}{text}{Style.RESET_ALL}"
    return text


def _print_header(title: str) -> None:
    bar = "─" * 60
    print()
    print(_color(bar, Fore.CYAN if HAS_COLOR else ""))
    print(_color(f"  {title}", Fore.CYAN if HAS_COLOR else ""))
    print(_color(bar, Fore.CYAN if HAS_COLOR else ""))


def load_fragments(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def run(
    fragments: list[str],
    evaluate: bool = False,
    use_fallback: bool = False,
    beam_width: int = 3,
) -> None:
    original_order = fragments[:]
    shuffled = fragments[:]
    random.shuffle(shuffled)

    _print_header("INPUT — Shuffled Fragments")
    for i, frag in enumerate(shuffled, 1):
        print(f"  [{i:>2}] {frag}")

    print()
    print("  Reassembling", _color(f"{len(shuffled)} fragments", Fore.YELLOW if HAS_COLOR else ""), "...")

    reassembler = FragmentReassembler(use_fallback=use_fallback, beam_width=beam_width)

    t0 = time.perf_counter()
    ordered, indices, confidence = reassembler.reassemble(shuffled)
    elapsed = time.perf_counter() - t0

    _print_header("OUTPUT — Reassembled Order")
    for i, frag in enumerate(ordered, 1):
        print(f"  [{i:>2}] {frag}")

    print()
    conf_color = Fore.GREEN if confidence >= 0.6 else Fore.YELLOW if HAS_COLOR else ""
    print(f"  Confidence Score : {_color(f'{confidence:.4f}', conf_color)}")
    print(f"  Time Elapsed     : {elapsed*1000:.1f} ms")

    if evaluate:
        # Map shuffled indices back to original positions
        original_indices = {frag: idx for idx, frag in enumerate(original_order)}
        gt_order = [original_indices[frag] for frag in shuffled]
        predicted_as_original = [original_indices[shuffled[i]] for i in indices]

        metrics = reassembler.evaluate(original_order, predicted_as_original)

        _print_header("EVALUATION METRICS")
        pm_str = _color("✔  PERFECT MATCH", Fore.GREEN if HAS_COLOR else "") if metrics["perfect_match"] \
            else _color("✘  Partial match", Fore.RED if HAS_COLOR else "")
        print(f"  Result           : {pm_str}")
        print(f"  Position Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"  Kendall Tau      : {metrics['kendall_tau']:.4f}  (1.0 = perfect)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Powered Text Fragment Reassembly"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a text file with one fragment per line.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate against ground-truth order (assumes input is already ordered).")
    parser.add_argument("--fallback", action="store_true",
                        help="Use TF-IDF fallback instead of sentence-transformers.")
    parser.add_argument("--beam", type=int, default=3,
                        help="Beam width for search (default: 3, use 1 for greedy).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible shuffling.")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.input:
        try:
            fragments = load_fragments(args.input)
        except FileNotFoundError:
            print(f"Error: file '{args.input}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        print(_color("No --input provided. Running built-in demo...", Fore.MAGENTA if HAS_COLOR else ""))
        fragments = DEMO_TEXT[:]

    run(
        fragments=fragments,
        evaluate=args.evaluate or args.input is None,
        use_fallback=args.fallback,
        beam_width=args.beam,
    )


if __name__ == "__main__":
    main()
