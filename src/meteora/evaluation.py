from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set


@dataclass(frozen=True)
class SelectionMetrics:
    precision: float
    recall: float
    f1: float
    correct_chunk_found: bool

    def to_dict(self):
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "correct_chunk_found": self.correct_chunk_found,
        }


def precision_recall_f1(
    selected_chunks: Iterable[int], gold_chunks: Iterable[int]
) -> SelectionMetrics:
    selected: Set[int] = set(selected_chunks)
    gold: Set[int] = set(gold_chunks)
    overlap = len(selected.intersection(gold))
    precision = overlap / len(selected) if selected else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SelectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        correct_chunk_found=overlap > 0,
    )
