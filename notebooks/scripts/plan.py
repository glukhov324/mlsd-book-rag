from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .chunking import Chunk


@dataclass(frozen=True)
class ChapterQuota:
    chapter_id: str
    chapter_index: int
    n_chunks: int
    final_total: int
    final_single: int
    final_scattered: int
    candidate_total: int
    candidate_single: int
    candidate_scattered: int


@dataclass(frozen=True)
class BookPlan:
    book_title: str
    n_final: int
    candidate_pool_ratio: float
    single_ratio: float
    chapters: tuple[ChapterQuota, ...]

    @property
    def n_candidate_target(self) -> int:
        return sum(c.candidate_total for c in self.chapters)

    @property
    def n_final_target(self) -> int:
        return sum(c.final_total for c in self.chapters)


def split_single_scattered(total: int, single_ratio: float) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    single = round(total * single_ratio)
    single = min(single, total)
    return single, total - single


def distribute_int(weights: Sequence[int], total: int) -> list[int]:
    if total <= 0:
        return [0] * len(weights)
    if not weights or sum(weights) == 0:
        base, rem = divmod(total, len(weights))
        return [base + (1 if i < rem else 0) for i in range(len(weights))]

    weight_sum = sum(weights)
    raw = [total * w / weight_sum for w in weights]
    floors = [int(x) for x in raw]
    assigned = sum(floors)
    remainders = [(raw[i] - floors[i], i) for i in range(len(weights))]
    remainders.sort(reverse=True)
    for k in range(total - assigned):
        floors[remainders[k % len(remainders)][1]] += 1
    return floors


def plan_book(
    chunks: Sequence[Chunk],
    *,
    n_final: int = 100,
    candidate_pool_ratio: float = 1.5,
    single_ratio: float = 0.7,
    min_chunks_for_quota: int = 1,
) -> BookPlan:
    """Build per-chapter quotas for candidates and final records."""
    if not chunks:
        raise ValueError("no chunks")

    by_chapter: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_chapter.setdefault(chunk.chapter_id, []).append(chunk)

    chapter_ids = sorted(
        by_chapter, key=lambda chapter_id: by_chapter[chapter_id][0].chapter_index
    )
    weights = [len(by_chapter[chapter_id]) for chapter_id in chapter_ids]
    eligible = [i for i, weight in enumerate(weights) if weight >= min_chunks_for_quota]
    if not eligible:
        raise ValueError("no eligible chapters")
    eligible_weights = [weights[i] for i in eligible]

    n_final_single, n_final_scattered = split_single_scattered(
        n_final, single_ratio
    )
    n_cand_single, n_cand_scattered = split_single_scattered(
        round(n_final * candidate_pool_ratio), single_ratio
    )

    final_single = [0] * len(chapter_ids)
    final_scattered = [0] * len(chapter_ids)
    for idx, value in zip(
        eligible, distribute_int(eligible_weights, n_final_single), strict=True
    ):
        final_single[idx] = value
    for idx, value in zip(
        eligible, distribute_int(eligible_weights, n_final_scattered), strict=True
    ):
        final_scattered[idx] = value

    cand_single = [0] * len(chapter_ids)
    cand_scattered = [0] * len(chapter_ids)
    for idx, value in zip(
        eligible, distribute_int(eligible_weights, n_cand_single), strict=True
    ):
        cand_single[idx] = value
    for idx, value in zip(
        eligible, distribute_int(eligible_weights, n_cand_scattered), strict=True
    ):
        cand_scattered[idx] = value

    quotas = []
    for i, chapter_id in enumerate(chapter_ids):
        chapter_chunks = by_chapter[chapter_id]
        quotas.append(
            ChapterQuota(
                chapter_id=chapter_id,
                chapter_index=chapter_chunks[0].chapter_index,
                n_chunks=len(chapter_chunks),
                final_total=final_single[i] + final_scattered[i],
                final_single=final_single[i],
                final_scattered=final_scattered[i],
                candidate_total=cand_single[i] + cand_scattered[i],
                candidate_single=cand_single[i],
                candidate_scattered=cand_scattered[i],
            )
        )
    return BookPlan(
        book_title=chunks[0].book_title,
        n_final=n_final,
        candidate_pool_ratio=candidate_pool_ratio,
        single_ratio=single_ratio,
        chapters=tuple(quotas),
    )
