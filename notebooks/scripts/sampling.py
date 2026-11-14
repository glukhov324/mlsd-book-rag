from __future__ import annotations

import random
from typing import Sequence

from .chunking import Chunk


def chapter_chunks_sorted(chunks: Sequence[Chunk], chapter_id: str) -> list[Chunk]:
    part = [c for c in chunks if c.chapter_id == chapter_id]
    return sorted(part, key=lambda c: c.chunk_index)


def min_chapter_chunks_for_scattered(
    scattered_chunk_counts: Sequence[int],
    *,
    min_index_gap: int = 2,
) -> int:
    counts = [c for c in scattered_chunk_counts if c >= 2]
    if not counts:
        raise ValueError("scattered_chunk_counts must include at least one value >= 2")
    return max((c - 1) * min_index_gap + 1 for c in counts)


def sample_scattered_chunks(
    chunks: list[Chunk],
    rng: random.Random,
    *,
    scattered_chunk_counts: Sequence[int] = (2, 3),
    min_index_gap: int = 2,
    attempts: int = 64,
) -> tuple[Chunk, ...] | None:
    counts = sorted({c for c in scattered_chunk_counts if c >= 2})
    if not counts:
        raise ValueError("scattered_chunk_counts must include at least one value >= 2")

    k = rng.choice(counts)
    min_n = (k - 1) * min_index_gap + 1
    if len(chunks) < min_n:
        return None

    for _ in range(attempts):
        idxs = sorted(rng.sample(range(len(chunks)), k))
        if all(
            idxs[t + 1] - idxs[t] >= min_index_gap for t in range(k - 1)
        ):
            return tuple(chunks[i] for i in idxs)
    return None


def shuffled_chunks(chunks: list[Chunk], rng: random.Random) -> list[Chunk]:
    out = list(chunks)
    rng.shuffle(out)
    return out
