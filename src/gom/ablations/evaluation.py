"""
Correct VQA scoring for the ablation studies (VQAv1/VQAv2 protocol).

Why this module exists
----------------------
``gom.vqa.runner.evaluate`` scores an answer with a single case-insensitive
string equality against the *majority* human answer. That is wrong in two ways
for this study:

1. **Reasoning / "thinking" models.** ``LlamaV-o1`` and any ``*-Thinking``
   model (e.g. ``Qwen3-VL-8B-Thinking``) emit reasoning tokens — often inside
   ``<think>...</think>`` — *before* the final answer. ``run_vqa`` only trims
   output when the literal token ``Answer:`` is present, so the reasoning trace
   is scored verbatim and every such answer misses. Their accuracy comes out
   artificially low. This is the concrete correctness risk flagged for the
   prompting set.
2. **The official VQA metric is not exact string match.** The paper states VQA
   is *"evaluated using accuracy-based metrics following the official evaluation
   protocols defined by their respective creators"* (§4). For VQAv1/VQAv2
   (Antol et al. 2015; Goyal et al. 2017) that protocol is:
     * a canonical answer **normalization** (lowercase, strip punctuation,
       expand contractions, map number words to digits, drop the articles
       a/an/the), and
     * a **soft** score ``acc = min(1, #humans_who_gave_that_answer / 3)``,
       averaged over the ten "leave-one-human-out" subsets of the 10 answers.

This module reproduces that protocol and adds robust final-answer extraction so
reasoning-model outputs are scored on their conclusion rather than their trace.
It lives under ``ablations/`` and is called from ``run_experiments.py`` instead
of ``runner.evaluate`` — ``gom.vqa.runner`` is left untouched, and the raw
(un-extracted) model output stays in ``raw_results.json`` for debugging.

Scope note
----------
Extraction reliably recovers a concise answer when the model ends with an
``Answer:``-style marker or a short final line (which every prompt template in
``prompts.py`` requests). It deliberately does **not** try to distil a short
answer out of an arbitrary free-form sentence — that would require an LLM judge
and could silently inflate/deflate scores. Keep prompting the models for terse
answers; this scorer then handles normalization, the soft 10-answer metric, and
reasoning-trace stripping.

References
----------
Official VQA evaluation code (the ``processPunctuation`` / ``processDigitArticle``
routines below are ported from it):
https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 1) Final-answer extraction (handles reasoning / "thinking" model output)
# ---------------------------------------------------------------------------

# ``<think> ... </think>`` (Qwen3 thinking format and similar). DOTALL so the
# block may span many lines; non-greedy so multiple blocks are each removed.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# A dangling opener with no closer (truncated reasoning): drop everything after
# it as well, otherwise the trace would be scored as the answer.
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)

# Markers that separate the reasoning from the final answer. Ordered by
# specificity; we split on the LAST occurrence of the first marker that matches.
_ANSWER_MARKERS = (
    r"final answer\s*[:\-]",
    r"answer\s*[:\-]",
    r"the answer is",
)

# Leading list/markdown bullets and emphasis we strip from a candidate answer.
_LEADING_JUNK_RE = re.compile(r"^\s*(?:[-*>#]+|\d+[.)])\s*")
_MD_EMPHASIS_RE = re.compile(r"[*_`]+")


def _strip_reasoning(text: str) -> str:
    """Remove ``<think>`` reasoning blocks from a raw model response."""
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _THINK_OPEN_RE.sub(" ", text)
    return text


def extract_final_answer(raw: Optional[str]) -> str:
    """
    Recover the concise final answer from a (possibly reasoning-heavy) response.

    Steps, in order:
      1. Strip ``<think>...</think>`` blocks (closed or dangling).
      2. If an answer marker (``Answer:``, ``Final answer:``, ``The answer is``)
         is present, keep only the text after its LAST occurrence.
      3. Otherwise fall back to the last non-empty line (prompt templates ask
         the model to put the answer on the final line).
      4. Strip surrounding quotes, markdown emphasis, and leading list bullets.

    Idempotent: running it on an already-extracted answer is a no-op, so it is
    safe to apply at eval time even though ``run_vqa`` already trims on
    ``Answer:``.
    """
    if not raw:
        return ""

    text = _strip_reasoning(str(raw)).strip()
    if not text:
        return ""

    lowered = text.lower()
    cut = -1
    for marker in _ANSWER_MARKERS:
        for m in re.finditer(marker, lowered):
            cut = max(cut, m.end())
        if cut != -1:
            break
    # Only honour the marker if there is real text after it; a trailing bare
    # "Answer:" (nothing after) should NOT blank out an otherwise fine response.
    if cut != -1 and text[cut:].strip():
        text = text[cut:].strip()
    else:
        # No usable marker: take the last non-empty line. For a plain
        # non-reasoning answer this is simply the whole (single-line) answer,
        # so the extractor is a safe no-op on ordinary model output.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[-1]

    text = _LEADING_JUNK_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    # Trim surrounding quotes and trailing sentence punctuation for readability
    # (scoring re-normalizes anyway, so this only tidies the stored value).
    text = text.strip().strip('"\'').strip()
    text = text.rstrip('.,;:!?').strip().strip('"\'').strip()
    return text


# ---------------------------------------------------------------------------
# 2) Official VQA answer normalization (ported from the VQA eval toolkit)
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id": "I'd", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itll": "it'll", "its": "it's", "lets": "let's", "maam": "ma'am",
    "mightve": "might've", "mustve": "must've", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's", "theyd": "they'd",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "wed": "we'd", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whod": "who'd", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
}

_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

_ARTICLES = {"a", "an", "the"}

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (_COMMA_STRIP.search(text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out, count=0)
    return out


def _process_digit_article(text: str) -> str:
    out_words = []
    for word in text.lower().split():
        word = _NUMBER_MAP.get(word, word)
        if word in _ARTICLES:
            continue
        out_words.append(word)
    # Expand contractions that lost their apostrophe during punctuation stripping.
    for i, word in enumerate(out_words):
        if word in _CONTRACTIONS:
            out_words[i] = _CONTRACTIONS[word]
    return " ".join(out_words)


def normalize_vqa_answer(answer: Optional[str]) -> str:
    """Apply the official VQA answer normalization to a single answer string."""
    if not answer:
        return ""
    text = str(answer).replace("\n", " ").replace("\t", " ").strip()
    text = _process_punctuation(text)
    text = _process_digit_article(text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3) Official soft VQA accuracy for one prediction against 10 human answers
# ---------------------------------------------------------------------------

def vqa_soft_accuracy(prediction: str, human_answers: Sequence[str]) -> float:
    """
    Official VQAv1/VQAv2 per-question accuracy in [0, 1].

    ``acc = mean over the 10 "leave-one-out" subsets of min(1, matches/3)`` where
    ``matches`` counts, among the other 9 human answers, those equal to the
    (normalized) prediction. Reduces to the standard closed form
    ``min(1, total_matches / 3)`` but we compute it exactly as the toolkit does.

    Falls back to a plain normalized exact match when fewer than 10 human
    answers are available (so partial/legacy datasets still score sensibly).
    """
    pred = normalize_vqa_answer(prediction)
    gts = [normalize_vqa_answer(a) for a in human_answers if isinstance(a, str)]
    if not gts:
        return 0.0
    if len(gts) < 10:
        return 1.0 if any(pred == g for g in gts) else 0.0

    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        matching = sum(1 for g in others if g == pred)
        accs.append(min(1.0, matching / 3.0))
    return sum(accs) / len(accs)


# ---------------------------------------------------------------------------
# 4) Drop-in replacement for gom.vqa.runner.evaluate
# ---------------------------------------------------------------------------

def _human_answers(record: Dict[str, Any]) -> List[str]:
    """The 10 human answers stashed in metadata by build_vqa_examples."""
    meta = record.get("metadata") or {}
    answers = meta.get("answers")
    if isinstance(answers, list) and answers:
        return [a for a in answers if isinstance(a, str)]
    # Fallback: the single majority answer, if that is all we have.
    gold = record.get("answer")
    return [gold] if isinstance(gold, str) and gold else []


def evaluate_vqa(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Correct VQA scorer for ablation runs. Drop-in for ``runner.evaluate``.

    For each result it extracts the final answer from ``generated_answer``
    (stripping reasoning traces), then computes:

      * ``vqa_accuracy``  — mean official soft VQA accuracy over the 10 human
        answers (percentage, 0–100). **This is the metric to report.**
      * ``exact_percent`` / ``exact`` — strict normalized equality against the
        majority human answer, kept for continuity with the previous metric and
        as a lower bound.
      * ``avg_time`` — mean processing time.

    Returns an empty dict when no record carries any gold answer.
    """
    gold = [r for r in results if _human_answers(r)]
    if not gold:
        return {}

    soft_scores: List[float] = []
    strict_hits = 0
    for r in gold:
        pred = extract_final_answer(r.get("generated_answer", ""))
        humans = _human_answers(r)
        soft_scores.append(vqa_soft_accuracy(pred, humans))

        # Strict: prediction vs the majority (most frequent) human answer.
        majority = r.get("answer") or (max(set(humans), key=humans.count) if humans else "")
        if normalize_vqa_answer(pred) == normalize_vqa_answer(majority) and normalize_vqa_answer(majority):
            strict_hits += 1

    n = len(gold)
    times = [r.get("processing_time", 0.0) for r in gold]
    return {
        "total": n,
        "vqa_accuracy": 100.0 * sum(soft_scores) / n,
        "exact": strict_hits,
        "exact_percent": 100.0 * strict_hits / n,
        "avg_time": (sum(times) / n) if n else 0.0,
    }
