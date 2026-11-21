from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from tqdm.auto import tqdm

from .categories import QUESTION_CATEGORY_VALUES, QuestionCategory
from .chunking import Chunk, split_epub
from .context_mode import ContextMode
from .plan import BookPlan, ChapterQuota, distribute_int, plan_book
from .question_pipeline import (
    QuestionRecord,
    process_chunk,
    process_scattered,
)
from .sampling import (
    chapter_chunks_sorted,
    min_chapter_chunks_for_scattered,
    sample_scattered_chunks,
    shuffled_chunks,
)


@dataclass
class ChapterCollection:
    quota: ChapterQuota
    single_candidates: list[QuestionRecord] = field(default_factory=list)
    scattered_candidates: list[QuestionRecord] = field(default_factory=list)
    single_final: list[QuestionRecord] = field(default_factory=list)
    scattered_final: list[QuestionRecord] = field(default_factory=list)
    attempts_single: int = 0
    attempts_scattered: int = 0


@dataclass
class CollectResult:
    plan: BookPlan
    by_chapter: dict[str, ChapterCollection]

    @property
    def candidates(self) -> list[QuestionRecord]:
        return [
            record
            for chapter in self.by_chapter.values()
            for record in (
                *chapter.single_candidates,
                *chapter.scattered_candidates,
            )
        ]

    @property
    def finals(self) -> list[QuestionRecord]:
        return [
            record
            for chapter in self.by_chapter.values()
            for record in (*chapter.single_final, *chapter.scattered_final)
        ]

    def plan_vs_actual(self) -> list[dict[str, Any]]:
        rows = []
        for chapter in self.by_chapter.values():
            quota = chapter.quota
            rows.append(
                {
                    "chapter_id": quota.chapter_id,
                    "n_chunks": quota.n_chunks,
                    "final_plan": quota.final_total,
                    "final_actual": len(chapter.single_final)
                    + len(chapter.scattered_final),
                    "candidate_plan": quota.candidate_total,
                    "candidate_actual": len(chapter.single_candidates)
                    + len(chapter.scattered_candidates),
                    "final_single_plan": quota.final_single,
                    "final_single_actual": len(chapter.single_final),
                    "final_scattered_plan": quota.final_scattered,
                    "final_scattered_actual": len(chapter.scattered_final),
                    "attempts_single": chapter.attempts_single,
                    "attempts_scattered": chapter.attempts_scattered,
                }
            )
        return rows

    def final_distribution(self) -> list[dict[str, Any]]:
        return final_distribution_rows(self.plan, self.finals)

    def bucket_deficits(self) -> list[dict[str, Any]]:
        targets = final_bucket_targets(self.plan)
        counts = _bucket_counts(self.finals)
        rows = []
        for (category, mode), target in targets.items():
            actual = counts.get((category, mode), 0)
            if actual < target:
                rows.append(
                    {
                        "category": category.value,
                        "context_mode": mode.value,
                        "target": target,
                        "actual": actual,
                        "need": target - actual,
                    }
                )
        rows.sort(key=lambda row: -row["need"])
        return rows


def attempts_budget(
    plan: BookPlan, max_attempts_single: int, max_attempts_scattered: int
) -> int:
    total = 0
    for quota in plan.chapters:
        if quota.candidate_total <= 0:
            continue
        total += min(quota.candidate_single, max_attempts_single)
        if quota.candidate_scattered > 0:
            total += min(quota.candidate_scattered, max_attempts_scattered)
    return total


class CollectionProgress:
    def __init__(
        self,
        plan: BookPlan,
        max_attempts_single: int,
        max_attempts_scattered: int,
    ) -> None:
        self.candidate_target = plan.n_candidate_target
        self.n_accepted = 0
        self._bar = tqdm(
            total=attempts_budget(
                plan, max_attempts_single, max_attempts_scattered
            ),
            desc="сбор",
            unit="попытка",
        )

    def after_attempt(
        self,
        chapter_id: str,
        mode: str,
        chapter_candidates: int,
        chapter_goal: int,
        accepted: bool,
    ) -> None:
        if accepted:
            self.n_accepted += 1
        self._bar.update(1)
        self._bar.set_postfix(
            глава=chapter_id,
            режим=mode,
            глава_канд=f"{chapter_candidates}/{chapter_goal}",
            всего_канд=f"{self.n_accepted}/{self.candidate_target}",
        )

    def close(self) -> None:
        self._bar.close()


@dataclass
class ChunkTryTracker:
    touches: dict[str, int] = field(default_factory=dict)

    def mark(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self.touches[chunk_id] = self.touches.get(chunk_id, 0) + 1

    def score(self, chunk_id: str) -> int:
        return self.touches.get(chunk_id, 0)


def _bucket_counts(
    records: Sequence[QuestionRecord],
) -> dict[tuple[QuestionCategory, ContextMode], int]:
    counts: dict[tuple[QuestionCategory, ContextMode], int] = {}
    for record in records:
        key = (record.category, record.context_mode)
        counts[key] = counts.get(key, 0) + 1
    return counts


def final_distribution_rows(
    plan: BookPlan, finals: list[QuestionRecord]
) -> list[dict[str, Any]]:
    targets = final_bucket_targets(plan)
    return [
        {
            "category": category.value,
            "context_mode": mode.value,
            "target": targets[(category, mode)],
            "actual": sum(
                record.category == category and record.context_mode == mode
                for record in finals
            ),
        }
        for category in QuestionCategory
        for mode in ContextMode
    ]


def _parse_chunk_ids(cell: Any) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [str(value) for value in cell]
    text = str(cell).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(value) for value in parsed]
    except (SyntaxError, ValueError):
        pass
    return [part.strip() for part in text.split(",") if ":" in part]


def records_from_xlsx(
    xlsx_path: str | Path, chunks: Sequence[Chunk]
) -> list[QuestionRecord]:
    """Restore records using stable chunk IDs from an earlier export."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    dataframe = pd.read_excel(xlsx_path)
    records = []
    for _, row in dataframe.iterrows():
        ids = _parse_chunk_ids(row.get("chunk_ids"))
        if not ids:
            continue
        try:
            selected_chunks = tuple(by_id[chunk_id] for chunk_id in ids)
        except KeyError:
            continue
        mode_raw = row.get("context_mode")
        if isinstance(mode_raw, str) and mode_raw in {
            mode.value for mode in ContextMode
        }:
            mode = ContextMode(mode_raw)
        else:
            mode = (
                ContextMode.SINGLE
                if len(selected_chunks) == 1
                else ContextMode.SCATTERED
            )
        try:
            category = QuestionCategory(str(row["category"]))
        except (KeyError, ValueError):
            continue
        first = selected_chunks[0]
        records.append(
            QuestionRecord(
                book_title=first.book_title,
                chapter_index=first.chapter_index,
                chapter_id=first.chapter_id,
                context_mode=mode,
                chunk_indices=tuple(
                    chunk.chunk_index for chunk in selected_chunks
                ),
                global_chunk_indices=tuple(
                    chunk.global_chunk_index for chunk in selected_chunks
                ),
                chunk_ids=tuple(chunk.chunk_id for chunk in selected_chunks),
                chunks=selected_chunks,
                question=str(row.get("question") or "").strip(),
                answer=str(row.get("answer") or "").strip(),
                evidence="",
                category=category,
                validate_reason="",
            )
        )
    return unique_questions(records, len(records))


def unique_questions(
    records: list[QuestionRecord], limit: int
) -> list[QuestionRecord]:
    seen = set()
    unique = []
    for record in records:
        key = record.question.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(record)
        if len(unique) >= limit:
            break
    return unique


def final_bucket_targets(
    plan: BookPlan,
) -> dict[tuple[QuestionCategory, ContextMode], int]:
    category_totals = distribute_int([1] * len(QuestionCategory), plan.n_final)
    n_single = round(plan.n_final * plan.single_ratio)
    single_totals = distribute_int(category_totals, n_single)
    return {
        (category, mode): (
            single_totals[i]
            if mode == ContextMode.SINGLE
            else category_totals[i] - single_totals[i]
        )
        for i, category in enumerate(QuestionCategory)
        for mode in ContextMode
    }


def select_dynamic_finals(
    candidates: list[QuestionRecord],
    plan: BookPlan,
    rng: random.Random,
    *,
    pinned: Sequence[QuestionRecord] = (),
) -> list[QuestionRecord]:
    targets = final_bucket_targets(plan)
    pinned_list = unique_questions(list(pinned), len(pinned))
    pinned_keys = {
        record.question.casefold().strip() for record in pinned_list
    }
    pool = [
        record
        for record in unique_questions(candidates, len(candidates))
        if record.question.casefold().strip() not in pinned_keys
    ]
    rng.shuffle(pool)

    selected = list(pinned_list)
    reserve = []
    counts = {key: 0 for key in targets}
    for record in selected:
        counts[(record.category, record.context_mode)] += 1
    for record in pool:
        key = (record.category, record.context_mode)
        if len(selected) < plan.n_final and counts[key] < targets[key]:
            selected.append(record)
            counts[key] += 1
        else:
            reserve.append(record)

    category_targets = {
        category: sum(targets[(category, mode)] for mode in ContextMode)
        for category in QuestionCategory
    }
    mode_targets = {
        mode: sum(targets[(category, mode)] for category in QuestionCategory)
        for mode in ContextMode
    }

    def deviation(records: list[QuestionRecord]) -> int:
        return sum(
            abs(
                sum(record.category == category for record in records)
                - target
            )
            for category, target in category_targets.items()
        ) + sum(
            abs(
                sum(record.context_mode == mode for record in records) - target
            )
            for mode, target in mode_targets.items()
        )

    while reserve and len(selected) < plan.n_final:
        best = min(
            range(len(reserve)),
            key=lambda index: deviation([*selected, reserve[index]]),
        )
        selected.append(reserve.pop(best))
    return selected


def _topup_deficits(
    plan: BookPlan,
    pool: list[QuestionRecord],
    pinned: Sequence[QuestionRecord] = (),
) -> list[tuple[QuestionCategory, ContextMode, int]]:
    targets = final_bucket_targets(plan)
    pinned_counts = _bucket_counts(pinned)
    pool_counts = _bucket_counts(pool)
    gaps = []
    for key, target in targets.items():
        after_pinned = target - min(target, pinned_counts.get(key, 0))
        need = max(0, after_pinned - pool_counts.get(key, 0))
        if need > 0:
            gaps.append((*key, need))
    gaps.sort(key=lambda item: -item[2])
    return gaps


def _tracker_from_records(
    records: Sequence[QuestionRecord],
) -> ChunkTryTracker:
    tracker = ChunkTryTracker()
    for record in records:
        tracker.mark(record.chunk_ids)
    return tracker


def _skeleton_by_chapter(
    plan: BookPlan,
) -> dict[str, ChapterCollection]:
    return {
        quota.chapter_id: ChapterCollection(quota=quota)
        for quota in plan.chapters
    }


def _assign_finals_to_chapters(
    by_chapter: dict[str, ChapterCollection],
    finals: list[QuestionRecord],
) -> None:
    for chapter in by_chapter.values():
        chapter.single_final.clear()
        chapter.scattered_final.clear()
    for record in finals:
        chapter = by_chapter[record.chapter_id]
        if record.context_mode == ContextMode.SINGLE:
            chapter.single_final.append(record)
        else:
            chapter.scattered_final.append(record)


def _pick_single_chunk(
    chunks: list[Chunk], tracker: ChunkTryTracker, rng: random.Random
) -> Chunk:
    return min(
        chunks, key=lambda chunk: (tracker.score(chunk.chunk_id), rng.random())
    )


def top_up_candidates(
    chunks: list[Chunk],
    plan: BookPlan,
    candidates: list[QuestionRecord],
    rng: random.Random,
    tracker: ChunkTryTracker,
    *,
    max_attempts: int,
    scattered_chunk_counts: tuple[int, ...],
    min_chunks_for_scattered: int,
    pinned: Sequence[QuestionRecord] = (),
    show_progress: bool = True,
) -> list[QuestionRecord]:
    if max_attempts <= 0:
        return candidates
    chapter_ids = {chunk.chapter_id for chunk in chunks}
    by_chapter = {
        chapter_id: chapter_chunks_sorted(chunks, chapter_id)
        for chapter_id in chapter_ids
    }
    pool = list(candidates)
    attempts = 0
    bar = tqdm(
        total=max_attempts,
        desc="добор",
        unit="попытка",
        disable=not show_progress,
    )
    while attempts < max_attempts:
        gaps = _topup_deficits(plan, pool, pinned)
        if not gaps:
            break
        category, mode, _ = gaps[0]
        shuffled_chapter_ids = list(by_chapter)
        rng.shuffle(shuffled_chapter_ids)
        for chapter_id in shuffled_chapter_ids:
            if attempts >= max_attempts:
                break
            chapter_chunks = by_chapter[chapter_id]
            if mode == ContextMode.SINGLE:
                if not chapter_chunks:
                    continue
                chunk = _pick_single_chunk(chapter_chunks, tracker, rng)
                tracker.mark([chunk.chunk_id])
                record = process_chunk(chunk)
            else:
                if len(chapter_chunks) < min_chunks_for_scattered:
                    continue
                group = sample_scattered_chunks(
                    chapter_chunks,
                    rng,
                    scattered_chunk_counts=scattered_chunk_counts,
                )
                if group is None:
                    attempts += 1
                    bar.update(1)
                    continue
                tracker.mark([chunk.chunk_id for chunk in group])
                record = process_scattered(group)
            attempts += 1
            bar.update(1)
            if record is None:
                continue
            pool.append(record)
            if (
                record.category == category
                and record.context_mode == mode
            ):
                break
    bar.close()
    return pool


def collect_chapter(
    chapter_id: str,
    quota: ChapterQuota,
    all_chunks: list[Chunk],
    rng: random.Random,
    *,
    max_attempts_single: int,
    max_attempts_scattered: int,
    min_chunks_for_scattered: int,
    scattered_chunk_counts: tuple[int, ...],
    progress: CollectionProgress | None = None,
) -> ChapterCollection:
    state = ChapterCollection(quota=quota)
    if quota.candidate_total <= 0:
        return state
    chapter_chunks = chapter_chunks_sorted(all_chunks, chapter_id)
    if not chapter_chunks:
        return state

    singles = shuffled_chunks(chapter_chunks, rng)
    single_index = 0
    while (
        len(state.single_candidates) < quota.candidate_single
        and state.attempts_single < max_attempts_single
    ):
        state.attempts_single += 1
        record = process_chunk(singles[single_index % len(singles)])
        single_index += 1
        if record is not None:
            state.single_candidates.append(record)
        if progress is not None:
            progress.after_attempt(
                chapter_id,
                "single",
                len(state.single_candidates),
                quota.candidate_single,
                record is not None,
            )

    if (
        quota.candidate_scattered > 0
        and len(chapter_chunks) >= min_chunks_for_scattered
    ):
        while (
            len(state.scattered_candidates) < quota.candidate_scattered
            and state.attempts_scattered < max_attempts_scattered
        ):
            state.attempts_scattered += 1
            group = sample_scattered_chunks(
                chapter_chunks,
                rng,
                scattered_chunk_counts=scattered_chunk_counts,
            )
            record = None if group is None else process_scattered(group)
            if record is not None:
                state.scattered_candidates.append(record)
            if progress is not None:
                progress.after_attempt(
                    chapter_id,
                    "scattered",
                    len(state.scattered_candidates),
                    quota.candidate_scattered,
                    record is not None,
                )
    return state


def collect_book(
    epub_path: str | Path,
    *,
    n_final: int = 100,
    candidate_pool_ratio: float = 1.5,
    single_ratio: float = 0.7,
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
    max_attempts_single: int = 40,
    max_attempts_scattered: int = 40,
    scattered_chunk_counts: tuple[int, ...] = (2, 3),
    min_chunks_for_scattered: int | None = None,
    balance_top_up: bool = False,
    max_topup_attempts: int = 60,
    continue_from_xlsx: str | Path | None = None,
    keep_existing_finals: bool = True,
    run_chapter_pass: bool | None = None,
    seed: int = 0,
    show_progress: bool = True,
) -> CollectResult:
    """Collect, validate, balance, and select synthetic book Q&A records."""
    counts = tuple(sorted({count for count in scattered_chunk_counts if count >= 2}))
    if not counts:
        raise ValueError(
            "scattered_chunk_counts must include at least one value >= 2"
        )
    if min_chunks_for_scattered is None:
        min_chunks_for_scattered = min_chapter_chunks_for_scattered(counts)

    chunks = split_epub(
        epub_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    plan = plan_book(
        chunks,
        n_final=n_final,
        candidate_pool_ratio=candidate_pool_ratio,
        single_ratio=single_ratio,
    )
    rng = random.Random(seed)

    pinned = []
    if continue_from_xlsx is not None:
        loaded = records_from_xlsx(continue_from_xlsx, chunks)
        if not loaded:
            raise ValueError(
                f"Could not load records from {continue_from_xlsx}; "
                "check chunk IDs and EPUB chunking settings"
            )
        pinned = list(loaded) if keep_existing_finals else []
        by_chapter = _skeleton_by_chapter(plan)
        candidates = unique_questions(list(loaded), len(loaded))
    else:
        by_chapter = {}
        candidates = []

    if run_chapter_pass is None:
        run_chapter_pass = continue_from_xlsx is None
    if run_chapter_pass:
        progress = (
            CollectionProgress(
                plan, max_attempts_single, max_attempts_scattered
            )
            if show_progress
            else None
        )
        try:
            for quota in plan.chapters:
                chapter = collect_chapter(
                    quota.chapter_id,
                    quota,
                    chunks,
                    rng,
                    max_attempts_single=max_attempts_single,
                    max_attempts_scattered=max_attempts_scattered,
                    min_chunks_for_scattered=min_chunks_for_scattered,
                    scattered_chunk_counts=counts,
                    progress=progress,
                )
                by_chapter[quota.chapter_id] = chapter
                candidates.extend(chapter.single_candidates)
                candidates.extend(chapter.scattered_candidates)
        finally:
            if progress is not None:
                progress.close()
        candidates = unique_questions(candidates, len(candidates))

    tracker = _tracker_from_records(candidates)
    existing_keys = {
        record.question.casefold().strip() for record in candidates
    }
    do_top_up = max_topup_attempts > 0 and (
        balance_top_up or continue_from_xlsx is not None
    )
    if do_top_up:
        candidates = top_up_candidates(
            chunks,
            plan,
            candidates,
            rng,
            tracker,
            max_attempts=max_topup_attempts,
            scattered_chunk_counts=counts,
            min_chunks_for_scattered=min_chunks_for_scattered,
            pinned=pinned,
            show_progress=show_progress,
        )
        for record in candidates:
            if record.question.casefold().strip() in existing_keys:
                continue
            chapter = by_chapter.get(record.chapter_id)
            if chapter is None:
                continue
            if record.context_mode == ContextMode.SINGLE:
                chapter.single_candidates.append(record)
            else:
                chapter.scattered_candidates.append(record)

    finals = select_dynamic_finals(candidates, plan, rng, pinned=pinned)
    _assign_finals_to_chapters(by_chapter, finals)
    return CollectResult(plan=plan, by_chapter=by_chapter)


def records_to_rows(
    records: list[QuestionRecord],
) -> list[dict[str, Any]]:
    return [record.as_row() for record in records]


def stratified_sample(
    dataframe: pd.DataFrame,
    n: int,
    strata: Sequence[str],
    *,
    seed: int = 0,
) -> pd.DataFrame:
    if n <= 0 or dataframe.empty:
        return dataframe.iloc[0:0].copy()
    if len(dataframe) <= n:
        return dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)
    missing = [column for column in strata if column not in dataframe.columns]
    if missing:
        raise ValueError(f"columns not in dataframe: {missing}")

    sizes = dataframe.groupby(list(strata), observed=True).size()
    targets = distribute_int(sizes.tolist(), n)
    picked = []
    for keys, target in zip(sizes.index, targets, strict=True):
        if target <= 0:
            continue
        mask = pd.Series(True, index=dataframe.index)
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        for column, value in zip(strata, key_tuple, strict=True):
            mask &= dataframe[column] == value
        subset = dataframe.loc[mask]
        picked.extend(
            subset.sample(
                n=min(target, len(subset)), random_state=seed
            ).index.tolist()
        )
    if len(picked) < n:
        remainder = dataframe.drop(picked)
        picked.extend(
            remainder.sample(
                n=min(n - len(picked), len(remainder)), random_state=seed
            ).index.tolist()
        )
    if len(picked) > n:
        picked = (
            dataframe.loc[picked]
            .sample(n=n, random_state=seed)
            .index.tolist()
        )
    return dataframe.loc[picked].reset_index(drop=True)


def stratified_sample_equal_categories(
    dataframe: pd.DataFrame, n: int, *, seed: int = 0
) -> pd.DataFrame:
    if n <= 0 or dataframe.empty:
        return dataframe.iloc[0:0].copy()
    if "category" not in dataframe.columns:
        raise ValueError("column 'category' required")
    if len(dataframe) <= n:
        return dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)

    targets = dict(
        zip(
            QUESTION_CATEGORY_VALUES,
            distribute_int([1] * len(QUESTION_CATEGORY_VALUES), n),
            strict=True,
        )
    )
    picked = []
    for category, target in targets.items():
        subset = dataframe[dataframe["category"] == category]
        picked.extend(
            subset.sample(
                n=min(target, len(subset)), random_state=seed
            ).index.tolist()
        )
    if len(picked) < n:
        remainder = dataframe[~dataframe.index.isin(picked)]
        picked.extend(
            remainder.sample(
                n=min(n - len(picked), len(remainder)), random_state=seed
            ).index.tolist()
        )
    if len(picked) > n:
        picked = (
            dataframe.loc[picked]
            .sample(n=n, random_state=seed)
            .index.tolist()
        )
    return dataframe.loc[picked].reset_index(drop=True)
