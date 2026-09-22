"""OpenStax AP textbooks as a secondary, out-of-distribution eval set.

SciQ passages are middle and early high school level. This module turns pinned OpenStax CNXML
(Biology for AP Courses, College Physics for AP Courses 2e, Chemistry 2e) into prompts with the
same schema as data/eval/prompts.jsonl:

1. Passages: body paragraphs of each section module, with exercises, figures, tables, notes,
   equations and end-of-section material removed. Short paragraphs are merged and long ones cut at
   a sentence boundary so lengths fall in the SciQ passage range (150 to 1,200 characters).
2. Screens, in order: 8-gram containment >= 0.5 against any passage the v1 or v2 adapters were
   trained or validated on, then the same against every SciQ support passage, then the tagger.
3. Target LOs: the tagger's own top-1 objective, kept only when it is in the book's subject and its
   score clears tau. The valid split tagged each SciQ question and answer; here a prompt with a book
   reference question (step 4) is tagged the same way from that question, and the rest, which have
   no question, from the passage text. Passages are screened on the passage tag.
4. References: human-written multiple-choice questions from the same section (Biology 2e review
   questions for biology, since the AP edition's embedded exercises don't publish keys; AP test prep
   questions for physics; end-of-chapter exercises for chemistry), keyed by the book's own solution.
   A passage gets the closest one by embedding cosine if it clears REF_MIN_COS.
"""

from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eduai.curriculum.tagger import tag_text
from eduai.data import explain, leakage

MIN_CHARS = 150
MAX_CHARS = explain.MAX_PASSAGE_CHARS
MAX_INLINE_MATH = 3
REF_MIN_COS = 0.6
LETTERS = "ABCD"

# Sections that hold end-of-section or non-prose material.
SKIP_SECTION = re.compile(
    r"learning-objectives|summary|problems-exercises|conceptual-questions|ap-test-prep|exercises|"
    r"key-equations|multiple-choice|visual-exercise|critical-thinking|review|free-response|"
    r"science-practice|chapter-review|reading-discard|contrib-auth|glossary|key-concepts|interactive"
)
SKIP_TAGS = {
    "exercise",
    "example",
    "note",
    "figure",
    "table",
    "list",
    "footnote",
    "glossary",
    "definition",
    "media",
    "equation",
    "rule",
}
# A paragraph containing one of these can't be rendered as plain prose.
REJECT_TAGS = {"figure", "table", "media", "image", "equation", "list", "footnote", "code", "preformat"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _math_text(el: ET.Element) -> str:
    t = _local(el.tag)
    kids = list(el)
    if t in ("mi", "mn", "mo", "mtext", "ms"):
        return (el.text or "").strip()
    if t == "msub" and len(kids) == 2:
        return f"{_math_text(kids[0])}_{_math_text(kids[1])}"
    if t == "msup" and len(kids) == 2:
        return f"{_math_text(kids[0])}^{_math_text(kids[1])}"
    if t == "msubsup" and len(kids) == 3:
        return f"{_math_text(kids[0])}_{_math_text(kids[1])}^{_math_text(kids[2])}"
    if t == "mfrac" and len(kids) == 2:
        return f"({_math_text(kids[0])})/({_math_text(kids[1])})"
    if t == "msqrt":
        return "sqrt(" + "".join(_math_text(k) for k in kids) + ")"
    if t in ("mover", "munder", "munderover") and kids:
        return _math_text(kids[0])
    if t == "annotation" or t == "annotation-xml":
        return ""
    return "".join(_math_text(k) for k in kids)


XREF = "\x00"


class Unrenderable(Exception):
    pass


def inline_text(el: ET.Element, max_math: int = MAX_INLINE_MATH) -> str:
    """Plain text of a paragraph-like element. Raises Unrenderable for figures, lists, equations,
    too much math, or cross-references whose text would be missing."""
    n_math = 0

    def rec(e: ET.Element) -> str:
        nonlocal n_math
        out = [e.text or ""]
        for c in e:
            t = _local(c.tag)
            if t in REJECT_TAGS:
                raise Unrenderable(t)
            if t == "math":
                n_math += 1
                if n_math > max_math:
                    raise Unrenderable("math")
                out.append(_math_text(c))
            elif t == "link" and c.get("target-id") and not (c.text or len(c)):
                out.append(XREF)  # "(Figure 2.3)" style reference; see clean()
            elif t == "newline":
                out.append(" ")
            else:
                out.append(rec(c))
            out.append(c.tail or "")
        return "".join(out)

    return clean(rec(el))


def clean(text: str) -> str:
    """Whitespace and dropped cross-references. A reference inside parentheses is removed with them;
    one that is part of the sentence ("shows a worker") makes the paragraph unrenderable."""
    text = re.sub(r"\s+", " ", text)
    x = re.escape(XREF)
    text = re.sub(rf"\(\s*(?:see\s*)?(?:(?:{x}|and|or|,|;|–|-)\s*)*\)", "", text, flags=re.I)
    if XREF in text:
        raise Unrenderable("xref")
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Module:
    book: str
    module_id: str
    title: str
    chapter: str
    number: str  # "chapter.section" in collection order
    paragraphs: list[tuple[str, str | None]] = field(default_factory=list)  # (section path, text)
    mcqs: list[dict] = field(default_factory=list)


def prose(txt: str) -> bool:
    """A self-contained paragraph: starts with a capital or digit and ends a sentence. This drops
    paragraphs that lead into or continue a removed equation ("... becomes", "where v is ...")."""
    return bool(re.match(r"[A-Z0-9“\"(]", txt)) and bool(re.search(r"[.?!][”\")]?$", txt))


def _body_paragraphs(content: ET.Element) -> list[tuple[str, str | None]]:
    """(section id, text) in reading order. Text is None where something was removed (a figure, an
    equation, an unrenderable paragraph), so chunks never join prose across a gap."""
    out: list[tuple[str, str | None]] = []

    def walk(e: ET.Element, sec: str) -> None:
        for c in e:
            t = _local(c.tag)
            if t == "section":
                if SKIP_SECTION.search(c.get("class", "")):
                    continue
                walk(c, c.get("id", sec))
            elif t == "para":
                try:
                    txt = inline_text(c)
                except Unrenderable:
                    txt = None
                if txt and not prose(txt):
                    txt = None
                if txt != "":
                    out.append((sec, txt))
            elif t in SKIP_TAGS:
                if t == "equation" and out and out[-1][0] == sec:
                    out[-1] = (sec, None)  # the paragraph before a display equation leads into it
                out.append((sec, None))
            else:
                walk(c, sec)

    walk(content, "")
    return out


_OPT = re.compile(r"^\(([a-e])\)\s*(.+)$", re.S)
_KEY_ONLY = re.compile(r"^\(?([a-eA-E])\)?\.?$")


def _key_letter(solution: ET.Element) -> str | None:
    try:
        s = inline_text(solution, max_math=20)
    except Unrenderable:
        return None
    m = _KEY_ONLY.match(s.strip())
    letters = {m.group(1).lower()} if m else set(re.findall(r"\(([a-e])\)", s))
    if len(letters) != 1:
        return None
    x = letters.pop()
    return x.upper() if x in "abcd" else None


ROMAN = ("I", "II", "III", "IV", "V", "VI")


def parse_mcq(ex: ET.Element) -> dict | None:
    """A four-option, single-key multiple-choice exercise with a published solution, else None."""
    problem = next((c for c in ex if _local(c.tag) == "problem"), None)
    solution = next((c for c in ex if _local(c.tag) == "solution"), None)
    if problem is None or solution is None:
        return None
    for d in problem.iter():
        t = _local(d.tag)
        if t in ("figure", "media", "image", "table", "equation") or (t == "link" and d.get("target-id")):
            return None
    stem_parts, options = [], []
    try:
        for c in problem:
            t = _local(c.tag)
            if t == "list" and c.get("number-style") == "lower-alpha":
                options += [inline_text(i, max_math=6) for i in c if _local(i.tag) == "item"]
            elif t == "list" and c.get("number-style") == "upper-roman":
                # statements the options refer to ("I and III"); they belong to the stem
                items = [inline_text(i, max_math=6) for i in c if _local(i.tag) == "item"]
                if len(items) > len(ROMAN):
                    return None
                stem_parts.append(" ".join(f"{r}. {x}" for r, x in zip(ROMAN, items, strict=False)))
            elif t == "list":
                return None  # any other list would be dropped from the stem, so skip the exercise
            elif t == "para":
                lists = [x for x in c if _local(x.tag) == "list"]
                if lists and lists[0].get("number-style") == "lower-alpha":
                    head = clean(c.text or "")
                    if head:
                        stem_parts.append(head)
                    options += [inline_text(i, max_math=6) for i in lists[0] if _local(i.tag) == "item"]
                    continue
                txt = inline_text(c, max_math=6)
                m = _OPT.match(txt)
                if m:
                    options.append(m.group(2).strip())
                elif txt:
                    stem_parts.append(txt)
    except Unrenderable:
        return None
    key = _key_letter(solution)
    stem = " ".join(stem_parts).strip()
    if len(options) != 4 or key is None or not stem or any(not o for o in options):
        return None
    choices = dict(zip(LETTERS, options, strict=True))
    if len({o.lower() for o in options}) < 4:
        return None
    return {"exercise_id": ex.get("id"), "stem": stem, "choices": choices, "key": key}


def parse_module(path: Path, book: str, chapter: str, number: str) -> Module:
    root = ET.parse(path).getroot()
    title = next((clean("".join(c.itertext())) for c in root if _local(c.tag) == "title"), "")
    content = next(c for c in root if _local(c.tag) == "content")
    mod = Module(book, path.parent.name, title, chapter, number)
    mod.paragraphs = _body_paragraphs(content)
    for ex in content.iter():
        if _local(ex.tag) == "exercise":
            q = parse_mcq(ex)
            if q:
                mod.mcqs.append(q)
    return mod


def collection_modules(collection: Path) -> list[tuple[str, str, str]]:
    """(module id, chapter title, "chapter.section") for chapter modules in collection order."""
    root = ET.parse(collection).getroot()
    out: list[tuple[str, str, str]] = []
    chap = 0

    def walk(e: ET.Element) -> None:
        nonlocal chap
        for c in e:
            t = _local(c.tag)
            if t == "subcollection":
                content = next((x for x in c if _local(x.tag) == "content"), None)
                if content is None:
                    continue
                if any(_local(x.tag) == "module" for x in content):
                    chap += 1
                    ctitle = next(("".join(x.itertext()) for x in c if _local(x.tag) == "title"), "")
                    sec = 0
                    for x in content:
                        if _local(x.tag) == "module":
                            out.append((x.get("document"), ctitle, f"{chap}.{sec}"))
                            sec += 1
                else:
                    walk(content)
            elif t == "content":
                walk(c)

    walk(root)
    return out


BACKREF = re.compile(
    r"(This|These|That|Those|Such|Here|Thus|Therefore|Hence|So|Whichever|Similarly|"
    r"The (last|above|preceding|following|previous|same))\b"
)


def chunks(paragraphs: list[tuple[str, str | None]]) -> list[str]:
    """Merge consecutive short paragraphs of one section; cut long ones at a sentence boundary."""
    out: list[str] = []
    buf, buf_sec = "", None
    after_gap = True
    for sec, txt in paragraphs:
        if txt is not None and after_gap and BACKREF.match(txt):
            txt = None  # "This relationship is known as ..." after a removed equation or list
        after_gap = txt is None
        if txt is None or (buf and sec != buf_sec):
            if len(buf) >= MIN_CHARS:
                out.append(buf)
            buf = ""
        if txt is None:
            continue
        buf_sec = sec
        buf = f"{buf} {txt}".strip() if buf else txt
        if len(buf) >= MIN_CHARS:
            out.append(buf)
            buf = ""
    if len(buf) >= MIN_CHARS:
        out.append(buf)
    res = []
    for c in out:
        if len(c) > MAX_CHARS:
            kept: list[str] = []
            for s in explain.sentences(c):
                if sum(len(x) + 1 for x in kept) + len(s) > MAX_CHARS:
                    break
                kept.append(s)
            c = " ".join(kept)
        if MIN_CHARS <= len(c) <= MAX_CHARS:
            res.append(c)
    return res


def fit_length(chunk: str, target: int) -> str:
    """Leading sentences of `chunk` up to about `target` characters, never below MIN_CHARS."""
    out: list[str] = []
    for s in explain.sentences(chunk):
        n = sum(len(x) + 1 for x in out)
        if out and n >= MIN_CHARS and n + len(s) > target:
            break
        out.append(s)
    res = " ".join(out)
    return res if MIN_CHARS <= len(res) <= MAX_CHARS else chunk


def load_book(raw: Path, book: dict) -> list[Module]:
    repo_dir = raw / book["repo"].split("/")[1]
    col = repo_dir / "collections" / f"{book['collection']}.collection.xml"
    mods = []
    for mid, chapter, number in collection_modules(col):
        m = parse_module(repo_dir / "modules" / mid / "index.cnxml", book["key"], chapter, number)
        if number.endswith(".0") or m.title.lower().startswith("introduction"):
            m.paragraphs = []  # chapter openers
        mods.append(m)
    return mods


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def screen_and_select(
    books: list[dict],
    modules: dict[str, list[Module]],
    tagger,
    embedder,
    train_passages: list[str],
    sciq_supports: list[str],
    train_qa: list[tuple[str, str]],
    n_per_subject: int = 50,
    seed: int = 20260918,
    target_lengths: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Pick up to n_per_subject screened passages per passage book, at most one per section per
    round, and attach a same-section reference MCQ where one is close enough. With target_lengths
    (the SciQ eval passage lengths), each chunk is cut to a length drawn from that list."""
    rng = random.Random(seed)
    sft_idx = leakage.ShingleIndex(train_passages)
    sciq_idx = leakage.ShingleIndex(sciq_supports)
    ref_vecs = embedder.encode([q for q, _ in train_qa]) if train_qa else None
    ref_answers = [a for _, a in train_qa]
    by_key = {b["key"]: b for b in books}
    ref_book_for = {"bio": "bio2e", "phys": "phys", "chem": "chem"}
    stats: dict = {}
    selected: list[dict] = []
    for b in books:
        if "passages" not in b["role"]:
            continue
        st = Counter()
        mods = [m for m in modules[b["key"]] if m.paragraphs]
        ref_mods = {_norm_title(m.title): m for m in modules[ref_book_for[b["key"]]]}
        order = mods[:]
        rng.shuffle(order)
        per_mod = {m.module_id: chunks(m.paragraphs) for m in order}
        st["modules"] = len(order)
        st["chunks"] = sum(len(v) for v in per_mod.values())
        for v in per_mod.values():
            rng.shuffle(v)
        kept: list[dict] = []
        used_refs: set[tuple[str, str]] = set()
        for _round in range(3):
            for m in order:
                if len(kept) >= n_per_subject:
                    break
                cands = per_mod[m.module_id]
                while cands and len(kept) < n_per_subject:
                    passage = cands.pop(0)
                    if target_lengths:
                        passage = fit_length(passage, rng.choice(target_lengths))
                    st["screened"] += 1
                    if sft_idx.best(passage, both_ways=True)[0] >= leakage.CONTAINMENT_MAX:
                        st["dropped_sft_containment"] += 1
                        continue
                    if sciq_idx.best(passage, both_ways=True)[0] >= leakage.CONTAINMENT_MAX:
                        st["dropped_sciq_containment"] += 1
                        continue
                    tag = tagger.tag_texts([passage])[0]
                    if tag.subject != b["subject"]:
                        st["dropped_tag_off_subject"] += 1
                        continue
                    if not tag.aligned:
                        st["dropped_tag_below_tau"] += 1
                        continue
                    kept.append(
                        {
                            "book": b["key"],
                            "subject": b["subject"],
                            "module": m,
                            "passage": passage,
                            "lo_id": tag.lo_id,
                            "tag_score": round(float(tag.score), 4),
                            "label_source": "tagger-passage",
                        }
                    )
                    break  # one passage per section per round
            if len(kept) >= n_per_subject:
                break
        # references
        pv = embedder.encode([k["passage"] for k in kept]) if kept else np.zeros((0, 1))
        for k, v in zip(kept, pv, strict=True):
            m = k["module"]
            rm = m if ref_book_for[b["key"]] == b["key"] else ref_mods.get(_norm_title(m.title))
            pool = [q for q in (rm.mcqs if rm else []) if (rm.module_id, q["exercise_id"]) not in used_refs]
            k["reference"] = None
            if not pool:
                st["no_section_mcq"] += 1
                continue
            texts = [f"{q['stem']} {q['choices'][q['key']]}" for q in pool]
            qv = embedder.encode(texts)
            if ref_vecs is not None:
                # the leak screen compares Q+A in the training side's format (sft_qa uses tag_text);
                # the passage cosine below keeps the plain text the committed prompts were picked with
                lv = embedder.encode([tag_text(q["stem"], q["choices"][q["key"]]) for q in pool])
                leaks = leakage.same_answer_qa_leaks(
                    lv, [q["choices"][q["key"]] for q in pool], ref_vecs, ref_answers
                )
                bad = {i for i, _, _ in leaks}
                st["ref_dropped_same_answer_qa"] += len(bad)
            else:
                bad = set()
            cos = qv @ v
            best = max((i for i in range(len(pool)) if i not in bad), key=lambda i: cos[i], default=None)
            if best is None or cos[best] < REF_MIN_COS:
                st["ref_below_min_cos"] += 1
                continue
            q = pool[best]
            used_refs.add((rm.module_id, q["exercise_id"]))
            rb = by_key[ref_book_for[b["key"]]]
            k["reference"] = {
                "question": q["stem"],
                "answer": q["choices"][q["key"]],
                "choices": q["choices"],
                "key": q["key"],
                "provenance": {
                    "book": rb["title"],
                    "repo": rb["repo"],
                    "sha": rb["sha"],
                    "module": rm.module_id,
                    "section": rm.title,
                    "exercise_id": q["exercise_id"],
                    "cos_to_passage": round(float(cos[best]), 4),
                },
            }
            st["with_reference"] += 1
            # As on the valid split, a prompt with a source question is labeled from that question
            # and its answer, when the tagger keeps it in the subject and above tau.
            rt = tagger.tag_texts([tag_text(q["stem"], q["choices"][q["key"]])])[0]
            if rt.subject == b["subject"] and rt.aligned:
                k.update(lo_id=rt.lo_id, tag_score=round(float(rt.score), 4), label_source="tagger-reference")
                st["labeled_from_reference"] += 1
        st["kept"] = len(kept)
        st["mcqs_parsed_in_reference_book"] = sum(len(x.mcqs) for x in modules[ref_book_for[b["key"]]])
        stats[b["subject"]] = dict(st)
        selected += kept
    return selected, stats


def build_rows(
    selected: list[dict],
    books: list[dict],
    tax,
    seed: int = 20260918,
    stimulus_frac: float = 0.4,
    misconception_frac: float = 0.3,
) -> tuple[list[dict], dict]:
    """Prompt rows in the data/eval/prompts.jsonl schema, with the main eval's format quotas per
    subject. Target misconceptions need a human wrong option, so they go to referenced rows only."""
    from eduai.prompts import GenerationRequest

    rng = random.Random(seed)
    by_key = {b["key"]: b for b in books}
    rows, quota = [], {}
    for subj in dict.fromkeys(s["subject"] for s in selected):
        group = [s for s in selected if s["subject"] == subj]
        k = len(group)
        stim = set(rng.sample(range(k), round(stimulus_frac * k)))
        with_ref = [i for i, s in enumerate(group) if s["reference"]]
        want = round(misconception_frac * k)
        mis = set(rng.sample(with_ref, min(want, len(with_ref))))
        diffs = (["easy", "medium", "hard"] * (k // 3 + 1))[:k]
        rng.shuffle(diffs)
        quota[subj] = {
            "n": k,
            "stimulus": len(stim),
            "misconception": len(mis),
            "misconception_target": want,
            "with_reference": len(with_ref),
        }
        for i, s in enumerate(group):
            lo = tax.lo(s["lo_id"])
            ref = s["reference"]
            miscon = None
            if i in mis:
                miscon = rng.choice([v for letter, v in ref["choices"].items() if letter != ref["key"]])
            req = GenerationRequest(
                subject=lo.subject_name,
                unit=lo.unit_name,
                topic=lo.topic_name,
                lo_id=lo.id,
                lo_text=lo.text,
                difficulty=diffs[i],
                format="stimulus" if i in stim else "standard",
                passage=s["passage"],
                target_misconception=miscon,
            )
            m, b = s["module"], by_key[s["book"]]
            rows.append(
                {
                    "id": f"ox-{s['book']}-{i:03d}",
                    "request": req.__dict__,
                    "group": None,
                    "reference": ref,
                    "label_source": s["label_source"],
                    "tag_score": s["tag_score"],
                    "source": {
                        "book": b["title"],
                        "repo": b["repo"],
                        "sha": b["sha"],
                        "module": m.module_id,
                        "section": f"{m.number} {m.title}",
                        "chapter": m.chapter,
                        "license": b["license"],
                    },
                }
            )
    return rows, quota


def load_sources(path: Path) -> list[dict]:
    return json.loads(path.read_text())["books"]
