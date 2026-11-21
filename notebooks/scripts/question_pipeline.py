# ruff: noqa: E501

import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from .categories import CATEGORY_GUIDE, QuestionCategory
from .chunking import Chunk
from .context_mode import ContextMode
from .ollama_json import chat_json
from .schemas import GENERATE_SCHEMA, VALIDATE_SCHEMA

GENERATOR_MODEL = "llama3.1:8b"
VALIDATOR_MODEL = "gemma4:12b"

ANSWER_WORDS_TARGET = (20, 30)
ANSWER_WORDS_LO, ANSWER_WORDS_HI = 15, 35

CHARACTER_NAMING_RULE = """В формулировке вопроса не используй «главный герой», «герой романа», «протагонист» без имени — называй персонажа **по имени** или **по роли в сцене** («рассказчик», «мать Пети», «офицер в кабине»). Допустимо «главный герой [Имя]» только если в фрагменте явно ясно, кто это, и имя названо."""

QUESTION_FORM_RULE = """Формулировка вопроса:
• **Один вопрос — один фокус** (один факт, одно действие, одна связь). Нельзя два вопроса в одном: «…, и почему…», «…, и как…», «…, а также что…». Один знак «?».
• Вопрос должен звучать так, как его задал бы внимательный читатель, а не разметчик датасета. Не пиши «в данном фрагменте», «согласно тексту», «в отрывке», «в чанке» или «на основании контекста».
• Вопрос понятен сам по себе: назови человека, предмет, событие или ситуацию вместо неясных «он», «она», «это», «там».
• Разнообразь начало: не зацикливайся на «Почему» и «Где». Подбирай под сцену: **кто / кого / кем**, **что произошло / что сделал**, **где / когда**, **сколько / какой**, **удалось ли / признал ли** (исход да/нет), **из-за чего / о чём** — что естественнее."""

_COMPOUND_Q = re.compile(
    r"\sи\s+(также\s+)?(как|почему|что|где|когда|сколько|какой|чей|каким\s+образом|зачем)\b",
    re.IGNORECASE,
)
_META_Q = re.compile(
    r"\b(?:согласно|по)\s+(?:данному\s+)?(?:тексту|фрагменту|отрывку|чанку)"
    r"|\bв\s+(?:этом|данном|представленном)\s+"
    r"(?:тексте|фрагменте|отрывке|чанке)"
    r"|\bна\s+основании\s+(?:контекста|текста|фрагмента)\b",
    re.IGNORECASE,
)


def question_form_ok(question: str) -> bool:
    q = question.strip()
    if not q:
        return False
    if q.count("?") != 1 or not q.endswith("?"):
        return False
    return _COMPOUND_Q.search(q) is None and _META_Q.search(q) is None


GENERATE_SYSTEM = f"""Твоя роль — внимательный читатель и редактор. Составь естественную пару «вопрос–ответ» по сцене из книги. Вопрос должен быть интересен обычному читателю, а ответ — полностью проверяться по переданному тексту.

{CHARACTER_NAMING_RULE}

Тебе дают один фрагмент. Сначала реши, можно ли задать по нему самостоятельный и содержательный вопрос без домыслов и знаний о других частях книги.

Верни `able_to_create_answer=true`, только если такой вопрос реально составить. Верни false, если сцена обрывочна, ответ неоднозначен, требуется внешний контекст или получается лишь викторина о случайной бытовой детали. При false поля question, answer и evidence оставь пустыми.

{QUESTION_FORM_RULE}

{CATEGORY_GUIDE}

Когда able_to_create_answer=true:
• **category** — одна из трёх подписей выше; вопрос и ответ должны ей соответствовать.
• question — один естественный вопрос с понятным предметом и одним знаком «?».
• answer — прямой, связный ответ на **{ANSWER_WORDS_TARGET[0]}–{ANSWER_WORDS_TARGET[1]} слов. Не начинай с «В тексте говорится» и не упоминай фрагмент, контекст или процесс проверки.
• evidence — короткая дословная цитата, которая подтверждает ответ. Не пересказывай её своими словами.

Если able_to_create_answer=false: category — любая из трёх (не используется).

Перед отправкой проверь: вопрос можно показать человеку без технических пояснений; ответ отвечает именно на него; каждый факт ответа подтверждён цитатой."""

GENERATE_SYSTEM_SCATTERED = f"""Твоя роль — внимательный читатель и редактор. Составь естественную пару «вопрос–ответ», для которой действительно нужны несколько разнесённых частей одной главы.

{CHARACTER_NAMING_RULE}

Даны **несколько отрывков одной главы** (обычно 2–3, они не идут подряд). Ответ должен честно следовать из их объединения без знаний о других частях книги.

Верни `able_to_create_answer=false`, если связи нет, её приходится додумывать или для полного ответа достаточно только одной части.

{QUESTION_FORM_RULE}

{CATEGORY_GUIDE}

При true:
• выбери **category** и сформулируй один естественный question;
• дай прямой answer на **{ANSWER_WORDS_TARGET[0]}–{ANSWER_WORDS_TARGET[1]} слов без слов «фрагмент», «отрывок», «контекст» или «чанк»;
• в evidence приведи короткие дословные опоры из всех частей, необходимых для ответа, разделив цитаты знаком ` | `.

Перед отправкой проверь, что удаление любой использованной части делает ответ неполным. При false category — любая из трёх (не используется), остальные поля пустые."""

VALIDATE_SYSTEM = f"""Ты независимый литературный редактор. Проверь пару «вопрос–ответ» по переданной сцене. Не исправляй пару и не додумывай за автора: только прими или отклони.

Ставь accept=true, если:
1. вопрос звучит естественно и понятен без слов «данный фрагмент», «текст», «отрывок», «контекст» или «чанк»;
2. в вопросе один фокус и ровно один знак «?»;
3. ответ прямо отвечает на вопрос, содержит **{ANSWER_WORDS_LO}–{ANSWER_WORDS_HI} слов и не ссылается на процесс проверки;
4. каждый факт ответа следует из сцены, а evidence — дословная и достаточная опора;
5. вопрос содержателен: это не случайная мелочь и не пустая викторина.

Ставь accept=false, если нарушен хотя бы один пункт; есть домысел; неясно, к кому относятся «он/она»; персонаж назван только «главным героем» без имени или однозначной роли; вопрос составной («… и почему/как/что…»).

{CATEGORY_GUIDE}

Верни category из той же тройки. Она должна совпадать с сутью пары и предложенной генератором категорией; иначе accept=false.

reason — одна конкретная фраза о главной причине отказа; если пара принята, верни пустую строку."""

VALIDATE_SYSTEM_SCATTERED = (
    VALIDATE_SYSTEM
    + "\n\nДополнительное правило: полный ответ должен требовать все заявленные части. Evidence должен содержать дословные опоры из каждой необходимой части. Если одной части достаточно или связь между частями додумана, ставь accept=false."
)


@dataclass(frozen=True)
class QuestionCandidate:
    question: str
    answer: str
    evidence: str
    category: QuestionCategory


@dataclass(frozen=True)
class QuestionRecord:
    book_title: str
    chapter_index: int
    chapter_id: str
    context_mode: ContextMode
    chunk_indices: tuple[int, ...]
    global_chunk_indices: tuple[int, ...]
    chunk_ids: tuple[str, ...]
    chunks: tuple[Chunk, ...]
    question: str
    answer: str
    evidence: str
    category: QuestionCategory
    validate_reason: str

    @property
    def chunk(self) -> Chunk:
        return self.chunks[0]

    def as_row(self) -> dict[str, Any]:
        return {
            "book_title": self.book_title,
            "chapter_index": self.chapter_index,
            "chapter_id": self.chapter_id,
            "context_mode": self.context_mode.value,
            "chunk_indices": list(self.chunk_indices),
            "global_chunk_indices": list(self.global_chunk_indices),
            "chunk_ids": list(self.chunk_ids),
            "chunks_text": "\n\n---\n\n".join(c.text for c in self.chunks),
            "category": self.category.value,
            "question": self.question,
            "answer": self.answer,
            "evidence": self.evidence,
            "selected_text": "",
            "is_spoiler": 0,
            "selection_anchor_chunk_id": "",
            "reader_anchor_chunk_id": "",
            "reading_progress_char": None,
            "sel_start": None,
            "sel_end": None,
        }


def selection_offsets(chunk: Chunk, selected_text: str) -> tuple[int, int]:
    """Границы выделения в координатах главы: те же, что `char_start`/`char_end` чанка."""
    text = selected_text
    if not text:
        raise ValueError("empty selected_text")
    local = chunk.text.find(text)
    if local < 0:
        raise ValueError(f"{chunk.chunk_id}: selected_text is not in the anchor chunk")
    sel_start = chunk.char_start + local
    sel_end = sel_start + len(text)
    if sel_end > chunk.char_end:
        raise ValueError(f"{chunk.chunk_id}: sel_end past chunk.char_end")
    return sel_start, sel_end


def source_from_chunks(
    chunks: Sequence[Chunk], context_mode: ContextMode
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("chunks is empty")
    chapter_id = chunks[0].chapter_id
    if any(c.chapter_id != chapter_id for c in chunks):
        raise ValueError("all chunks must belong to the same chapter")
    return {
        "book_title": chunks[0].book_title,
        "chapter_index": chunks[0].chapter_index,
        "chapter_id": chapter_id,
        "context_mode": context_mode,
        "chunk_indices": tuple(c.chunk_index for c in chunks),
        "global_chunk_indices": tuple(c.global_chunk_index for c in chunks),
        "chunk_ids": tuple(c.chunk_id for c in chunks),
        "chunks": tuple(chunks),
    }


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def answer_length_ok(text: str) -> bool:
    n_words = word_count(text)
    return ANSWER_WORDS_LO <= n_words <= ANSWER_WORDS_HI


def chunk_context(chunk: Chunk) -> str:
    return (
        f"Книга: {chunk.book_title}\n"
        f"Глава: {chunk.chapter_id} (индекс {chunk.chapter_index})\n"
        f"Чанк: {chunk.chunk_id}\n\n"
        f"{chunk.text}"
    )


def chunks_context(chunks: Sequence[Chunk]) -> str:
    if len(chunks) == 1:
        return chunk_context(chunks[0])
    header = (
        f"Книга: {chunks[0].book_title}\n"
        f"Глава: {chunks[0].chapter_id} (индекс {chunks[0].chapter_index})\n"
    )
    parts = [header]
    for n, chunk in enumerate(chunks, start=1):
        parts.append(
            f"--- часть {n} (чанк {chunk.chunk_id}) ---\n{chunk.text}\n"
        )
    return "\n".join(parts)


def generate_question(
    chunks: Sequence[Chunk], context_mode: ContextMode
) -> QuestionCandidate | None:
    if context_mode == ContextMode.SINGLE:
        if len(chunks) != 1:
            raise ValueError("single mode requires exactly one chunk")
        system = GENERATE_SYSTEM
    else:
        if len(chunks) < 2:
            raise ValueError("scattered mode requires multiple chunks")
        system = GENERATE_SYSTEM_SCATTERED
    data = chat_json(
        GENERATOR_MODEL,
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Фрагмент:\n\n{chunks_context(chunks)}",
            },
        ],
        GENERATE_SCHEMA,
        temperature=0.4,
        num_predict=1024,
    )
    if not data or not data.get("able_to_create_answer"):
        return None
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    evidence = (data.get("evidence") or "").strip()
    if (
        not question
        or not answer
        or not answer_length_ok(answer)
        or not question_form_ok(question)
    ):
        return None
    try:
        category = QuestionCategory((data.get("category") or "").strip())
    except ValueError:
        return None
    return QuestionCandidate(question, answer, evidence, category)


def validate_question(
    chunks: Sequence[Chunk],
    candidate: QuestionCandidate,
    context_mode: ContextMode,
) -> QuestionRecord | None:
    if not answer_length_ok(candidate.answer) or not question_form_ok(
        candidate.question
    ):
        return None
    system = (
        VALIDATE_SYSTEM
        if context_mode == ContextMode.SINGLE
        else VALIDATE_SYSTEM_SCATTERED
    )
    data = chat_json(
        VALIDATOR_MODEL,
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Фрагмент:\n\n{chunks_context(chunks)}\n\n"
                    f"Категория (генератор): {candidate.category.value}\n"
                    f"Вопрос: {candidate.question}\n"
                    f"Ответ ({word_count(candidate.answer)} слов): "
                    f"{candidate.answer}\n"
                    f"Цитата: {candidate.evidence}"
                ),
            },
        ],
        VALIDATE_SCHEMA,
        temperature=0.1,
        num_predict=512,
    )
    if not data or not data.get("accept"):
        return None
    try:
        category = QuestionCategory(data["category"])
    except (KeyError, ValueError):
        return None
    meta = source_from_chunks(chunks, context_mode)
    return QuestionRecord(
        question=candidate.question,
        answer=candidate.answer,
        evidence=candidate.evidence,
        category=category,
        validate_reason=(data.get("reason") or "").strip(),
        **meta,
    )


def process_chunks_group(
    chunks: Sequence[Chunk], context_mode: ContextMode
) -> QuestionRecord | None:
    candidate = generate_question(chunks, context_mode)
    if candidate is None:
        return None
    return validate_question(chunks, candidate, context_mode)


def process_chunk(chunk: Chunk) -> QuestionRecord | None:
    return process_chunks_group((chunk,), ContextMode.SINGLE)


def process_chunks(chunks: Iterable[Chunk]) -> Iterator[QuestionRecord]:
    for chunk in chunks:
        record = process_chunk(chunk)
        if record is not None:
            yield record


def process_scattered(chunks: Sequence[Chunk]) -> QuestionRecord | None:
    return process_chunks_group(chunks, ContextMode.SCATTERED)
