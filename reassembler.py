"""
reassembler.py
--------------
Core logic for AI-powered text fragment reassembly.

Strategy
--------
1. Embed all fragments with a sentence-transformer model.
2. Build a cosine-similarity matrix between consecutive pairs.
3. Solve the ordering problem as a greedy nearest-neighbour chain,
   starting from the fragment with the lowest average similarity to
   all others (likely an opening sentence).
4. Optionally re-rank with beam search for better accuracy on longer inputs.
"""

from __future__ import annotations

import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

from embeddings import EmbeddingModel


class FragmentReassembler:
    """
    Reassembles a list of text fragments into their most likely original order.

    Parameters
    ----------
    model_name   : HuggingFace sentence-transformer model to use.
    use_fallback : Force TF-IDF fallback (useful for testing without GPU/internet).
    beam_width   : Width for beam search (1 = greedy).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_fallback: bool = False,
        beam_width: int = 3,
    ):
        self.encoder = EmbeddingModel(model_name=model_name, use_fallback=use_fallback)
        self.beam_width = beam_width

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reassemble(self, fragments: list[str]) -> tuple[list[str], list[int], float]:
        """
        Reassemble fragments into the most likely order.

        Returns
        -------
        ordered_fragments : list[str]   – fragments in predicted order
        order_indices     : list[int]   – original index positions
        confidence        : float       – mean chain similarity score (0–1)
        """
        if len(fragments) == 0:
            return [], [], 0.0
        if len(fragments) == 1:
            return fragments[:], [0], 1.0

        embeddings = self.encoder.encode(fragments)
        sim_matrix = cosine_similarity(embeddings)

        if self.beam_width > 1 and len(fragments) <= 12:
            order = self._beam_search(sim_matrix, self.beam_width)
        else:
            order = self._greedy_chain(sim_matrix)

        ordered = [fragments[i] for i in order]
        confidence = self._chain_score(sim_matrix, order)
        return ordered, order, float(confidence)

    def evaluate(self, original: list[str], predicted_order: list[int]) -> dict:
        """
        Compute evaluation metrics given the ground-truth order (0,1,2,...).

        Returns a dict with:
          - perfect_match  : bool  – entire sequence correct
          - accuracy       : float – fraction of fragments in correct position
          - kendall_tau    : float – rank correlation (-1 to 1)
        """
        n = len(original)
        correct = sum(1 for i, idx in enumerate(predicted_order) if idx == i)
        accuracy = correct / n if n > 0 else 0.0
        tau = self._kendall_tau(list(range(n)), predicted_order)
        return {
            "perfect_match": predicted_order == list(range(n)),
            "accuracy": round(accuracy, 4),
            "kendall_tau": round(tau, 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _greedy_chain(self, sim: np.ndarray) -> list[int]:
        """Greedy nearest-neighbour chain starting from the best anchor."""
        n = sim.shape[0]
        # Anchor: fragment with lowest mean similarity (often the opening)
        mean_sims = sim.mean(axis=1) - sim.diagonal()  # exclude self-similarity
        start = int(np.argmin(mean_sims))

        visited = [False] * n
        order = [start]
        visited[start] = True

        for _ in range(n - 1):
            current = order[-1]
            scores = sim[current].copy()
            scores[visited] = -np.inf  # mask already-visited
            nxt = int(np.argmax(scores))
            order.append(nxt)
            visited[nxt] = True

        return order

    def _beam_search(self, sim: np.ndarray, beam_width: int) -> list[int]:
        """Beam search for better ordering on small inputs (n ≤ 12)."""
        n = sim.shape[0]
        # beams: list of (score, partial_order, visited_set)
        beams = [(0.0, [i], {i}) for i in range(n)]

        for step in range(n - 1):
            candidates = []
            for score, order, visited in beams:
                current = order[-1]
                for nxt in range(n):
                    if nxt in visited:
                        continue
                    new_score = score + float(sim[current, nxt])
                    candidates.append((new_score, order + [nxt], visited | {nxt}))
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[:beam_width]

        best = max(beams, key=lambda x: x[0])
        return best[1]

    def _chain_score(self, sim: np.ndarray, order: list[int]) -> float:
        """Mean cosine similarity between consecutive fragments in the chain."""
        if len(order) < 2:
            return 1.0
        scores = [sim[order[i], order[i + 1]] for i in range(len(order) - 1)]
        return float(np.mean(scores))

    @staticmethod
    def _kendall_tau(ground_truth: list[int], predicted: list[int]) -> float:
        """Normalised Kendall tau-b rank correlation."""
        n = len(ground_truth)
        if n <= 1:
            return 1.0
        concordant = discordant = 0
        rank_pred = {v: i for i, v in enumerate(predicted)}
        for i, j in itertools.combinations(range(n), 2):
            gt_order = ground_truth[i] < ground_truth[j]
            pd_order = rank_pred[i] < rank_pred[j]
            if gt_order == pd_order:
                concordant += 1
            else:
                discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / total if total > 0 else 0.0
