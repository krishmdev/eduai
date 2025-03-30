from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False)
models_app = typer.Typer(no_args_is_help=True)
curriculum_app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
tagger_app = typer.Typer(no_args_is_help=True)
bank_app = typer.Typer(no_args_is_help=True)
app.add_typer(models_app, name="models", help="Download and verify pinned model artifacts.")
app.add_typer(curriculum_app, name="curriculum", help="Inspect and validate the curriculum taxonomy.")
app.add_typer(data_app, name="data", help="Build the SciQ-derived datasets.")
app.add_typer(tagger_app, name="tagger", help="Curriculum tagger evaluation.")
app.add_typer(bank_app, name="bank", help="Item bank maintenance and generation.")
eval_app = typer.Typer(no_args_is_help=True)
app.add_typer(eval_app, name="eval", help="Base vs few-shot vs fine-tuned comparison.")
console = Console()


@models_app.command("fetch")
def models_fetch(keys: list[str], update_lock: bool = typer.Option(False, "--update-lock")) -> None:
    from eduai.models import download, read_lock, write_lock

    download(keys)
    new = [k for k in keys if update_lock or k not in read_lock()]
    if new:
        write_lock(new)
        console.print(f"recorded hashes for {', '.join(new)} in models.lock")
    console.print(f"fetched {', '.join(keys)}")


@models_app.command("verify")
def models_verify(keys: list[str]) -> None:
    from eduai.models import verify

    problems = verify(keys)
    for p in problems:
        console.print(f"[red]{p}[/red]")
    if problems:
        raise typer.Exit(1)
    console.print(f"verified {', '.join(keys)} against models.lock")


@curriculum_app.command("validate")
def curriculum_validate() -> None:
    from eduai.curriculum.taxonomy import load_subjects, validate

    subjects = load_subjects()
    problems = validate(subjects)
    for s in subjects:
        console.print(f"{s.id:6} {s.name:28} units={len(s.units):2} objectives={len(s.objectives)}")
    for p in problems:
        console.print(f"[red]{p}[/red]")
    if problems:
        raise typer.Exit(1)


@curriculum_app.command("neighbors")
def curriculum_neighbors() -> None:
    from eduai.curriculum.embedder import get_embedder
    from eduai.curriculum.neighbors import build_graph, save_graph, similarity_stats
    from eduai.curriculum.taxonomy import default_taxonomy

    graph = build_graph(default_taxonomy(), get_embedder("bge-small"))
    save_graph(graph)
    console.print(similarity_stats(graph))


@data_app.command("build")
def data_build(
    sciq: Path = typer.Option(..., help="Directory with SciQ train/valid/test.json"),
    out: Path = typer.Option(Path("data")),
    reports: Path = typer.Option(Path("reports")),
    seed: int = 7,
    containment_link: bool = typer.Option(
        False, help="v2 grouping: also link passages with >=50% 8-gram overlap"
    ),
    samples: bool = True,
) -> None:
    import time

    from eduai.curriculum.tagger import build_tagger
    from eduai.curriculum.taxonomy import default_taxonomy
    from eduai.data.build_sft import Builder, save_card
    from eduai.data.samples import write_samples
    from eduai.data.sciq import load_sciq
    from eduai.manifest import write_manifest

    t0 = time.time()
    tax = default_taxonomy()
    tagger = build_tagger(tax, "bge-small", rerank=True)
    if tagger.tau is None:
        raise typer.BadParameter("no tau calibration; run `eduai tagger eval` first")
    items = load_sciq(sciq)
    result = Builder(
        tax, tagger, tagger.embedder, seed=seed, containment=0.5 if containment_link else None
    ).run(items, out)
    save_card(result.card, reports)
    if samples:
        write_samples(out)
    write_manifest(
        reports / "data_card_manifest.json",
        {
            "task": "data-build",
            "seed": seed,
            "tagger": tagger.mode,
            "tau": tagger.tau,
            "embedder_id": tagger.embedder.embedder_id,
            "seconds": round(time.time() - t0, 1),
        },
    )
    console.print((reports / "data_card.md").read_text())


@data_app.command("eval-candidates")
def data_eval_candidates(data: Path = typer.Option(Path("data")), n: int = 230) -> None:
    from eduai.curriculum.embedder import get_embedder
    from eduai.data.eval_prompts import candidates
    from eduai.data.sciq import write_jsonl

    kept, stats = candidates(data, get_embedder("bge-small"), n)
    write_jsonl(data / "eval" / "candidates.jsonl", kept)
    (data / "eval" / "candidates_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    console.print(stats)


@data_app.command("eval-prompts")
def data_eval_prompts(
    data: Path = typer.Option(Path("data")),
    labels: Path = typer.Option(...),
    n: int = 150,
    seed: int = 20260924,
) -> None:
    from eduai.curriculum.taxonomy import default_taxonomy
    from eduai.data.eval_prompts import build_prompts, write
    from eduai.data.sciq import read_jsonl

    cands = read_jsonl(data / "eval" / "candidates.jsonl")
    lab = {r["id"]: r["lo_id"] for r in read_jsonl(labels)}
    rows, stats = build_prompts(cands, lab, default_taxonomy(), n, seed)
    stats["leakage_filter"] = json.loads((data / "eval" / "candidates_stats.json").read_text())
    write(rows, data / "eval" / "prompts.jsonl")
    Path("reports/eval_prompts_card.json").write_text(json.dumps(stats, indent=2) + "\n")
    console.print(stats)


@tagger_app.command("sample-gold")
def tagger_sample_gold(
    sciq: Path = typer.Option(..., help="Directory with SciQ train/valid/test.json"),
    out: Path = typer.Option(Path("data/gold/tagging_gold_candidates.jsonl")),
    n: int = 150,
    seed: int = 20260923,
) -> None:
    import random

    from eduai.data.sciq import load_sciq, write_jsonl

    items = load_sciq(sciq)
    rng = random.Random(seed)
    picked = rng.sample(items, n)
    write_jsonl(out, [{"id": it.id, "question": it.question, "answer": it.correct} for it in picked])
    console.print(f"wrote {n} candidates to {out}")


@tagger_app.command("eval")
def tagger_eval(
    gold: Path = typer.Option(Path("data/gold/tagging_gold.jsonl")),
    out: Path = typer.Option(Path("reports/tagger_eval.md")),
) -> None:
    from eduai.curriculum.tagger import build_tagger
    from eduai.curriculum.taxonomy import default_taxonomy
    from eduai.data.sciq import read_jsonl
    from eduai.evaluation import tagging
    from eduai.manifest import write_manifest

    tax = default_taxonomy()
    taggers = [
        build_tagger(tax, "minilm", rerank=False),
        build_tagger(tax, "bge-small", rerank=False),
        build_tagger(tax, "bge-small", rerank=False, query_prefix=False),
        build_tagger(tax, "bge-small", rerank=True),
        build_tagger(tax, "bge-small", rerank=True, use_gate=False),
    ]
    results = tagging.run(tax, gold, taggers)
    tagging.save_calibration(results)
    manifest = out.with_name("tagger_eval_manifest.json")
    write_manifest(manifest, {"task": "tagger-eval", "gold": str(gold)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tagging.render_markdown(results, read_jsonl(gold), manifest.name))
    (out.with_suffix(".json")).write_text(json.dumps([r.__dict__ for r in results], indent=2) + "\n")
    console.print(out.read_text())


@bank_app.command("add-generated")
def bank_add_generated(
    arm: str = "finetuned", out: Path = typer.Option(Path("data/samples/generated_items.jsonl"))
) -> None:
    """Promote eval outputs that passed every check (and aren't a copy of their source) into the bank."""
    import random
    import re

    from eduai.curriculum.taxonomy import default_taxonomy
    from eduai.data.difficulty import LABEL_B
    from eduai.data.sciq import read_jsonl, write_jsonl
    from eduai.generation.validate import parse

    tax = default_taxonomy()
    rng = random.Random(0)
    prompts = {p["id"]: p for p in read_jsonl(Path("data/eval/prompts.jsonl"))}
    passed = {
        r["id"]
        for r in read_jsonl(Path("reports/eval/per_item.jsonl"))
        if r["arm"] == arm and r.get("all_checks") and not r.get("source_copy")
    }
    rows = []
    for g in read_jsonl(Path(f"reports/eval/gen_{arm}.jsonl")):
        if g["id"] not in passed:
            continue
        item, _ = parse(g["text"])
        # The fine-tuned model puts the key at A far more often than 25%; reshuffle before banking.
        old = list(item["choices"].items())
        rng.shuffle(old)
        new_choices = {"ABCD"[i]: text for i, (_, text) in enumerate(old)}
        new_answer = "ABCD"[[k for k, _ in old].index(item["answer"])]
        item["explanation"] = re.sub(
            r"^[ABCD] is correct\.", f"{new_answer} is correct.", item["explanation"]
        )
        item["choices"], item["answer"] = new_choices, new_answer
        req = prompts[g["id"]]["request"]
        lo = tax.lo(req["lo_id"])
        rows.append(
            {
                "id": f"gen-{g['id']}",
                "stem": item["stem"],
                "stimulus": item.get("stimulus"),
                "choices": item["choices"],
                "answer": item["answer"],
                "explanation": item["explanation"],
                "lo_id": lo.id,
                "subject": lo.subject,
                "unit_id": lo.unit_id,
                "difficulty": req["difficulty"],
                "b": LABEL_B[req["difficulty"]],
                "source": f"generated:{arm}",
                "source_item": g["id"],
                "grounded": True,
                "ungrounded": False,
                "aligned": True,
            }
        )
    write_jsonl(out, rows)
    console.print(f"wrote {len(rows)} generated items to {out}")


@app.command("simulate")
def simulate(students: int = 500, out: Path = typer.Option(Path("reports")), seed: int = 20260923) -> None:
    from eduai.curriculum.taxonomy import default_taxonomy
    from eduai.manifest import write_manifest
    from eduai.sim import experiments, plots, report

    cfg = experiments.SimConfig(students=students, seed=seed)
    res = experiments.run_all(default_taxonomy(), cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sim_results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    plots.sim_figures(res, out / "figures")
    write_manifest(
        out / "sim_manifest.json",
        {"task": "simulate", "students": students, "seed": seed, "seconds": res["seconds"]},
    )
    (out / "sim_report.md").write_text(report.render(res, "sim_manifest.json"))
    console.print(f"wrote {out / 'sim_report.md'} in {res['seconds']} s")


def _novelty_index():
    from eduai.curriculum.embedder import get_embedder
    from eduai.data.sciq import read_jsonl
    from eduai.generation.dedup import NoveltyIndex

    bank = read_jsonl(Path("data/bank/sciq_items.jsonl"))
    groups = {d["id"]: d["tags"]["group"] for d in read_jsonl(Path("data/items_tagged.jsonl"))}
    train = [r["stem"] for r in read_jsonl(Path("data/sft/train_stems.jsonl"))]
    return NoveltyIndex(
        get_embedder("bge-small"),
        [b["stem"] for b in bank],
        [b["id"] for b in bank],
        [groups[b["id"].removeprefix("sciq-")] for b in bank],
        train,
    )


@eval_app.command("generate")
def eval_generate(
    arm: str, prompts: Path = typer.Option(Path("data/eval/prompts.jsonl")), limit: int = None
) -> None:
    from eduai.evaluation.compare import generate_arm, load_prompts

    out = generate_arm(arm, load_prompts(prompts), limit=limit)
    console.print(f"wrote {out}")


@eval_app.command("judge")
def eval_judge(
    judge: str = "llama-1b", prompts: Path = typer.Option(Path("data/eval/prompts.jsonl"))
) -> None:
    from eduai.evaluation.compare import judge_all, load_prompts

    console.print(f"wrote {judge_all(load_prompts(prompts), judge)}")


@eval_app.command("score")
def eval_score(
    prompts: Path = typer.Option(Path("data/eval/prompts.jsonl")), out: Path = typer.Option(Path("reports"))
) -> None:
    from eduai.curriculum.tagger import build_tagger
    from eduai.curriculum.taxonomy import default_taxonomy
    from eduai.evaluation import compare, report
    from eduai.manifest import write_manifest

    res = compare.score(compare.load_prompts(prompts), build_tagger(default_taxonomy()), _novelty_index())
    (out / "eval.json").write_text(json.dumps(res, indent=2) + "\n")
    manifest = out / "eval_manifest.json"
    if not manifest.exists():
        write_manifest(manifest, {"task": "eval-score"})
    card = json.loads((out / "eval_prompts_card.json").read_text())
    (out / "eval_report.md").write_text(report.render(res, manifest.name, card))
    console.print((out / "eval_report.md").read_text())


@app.command("serve")
def serve(port: int = 8001, host: str = "127.0.0.1") -> None:
    from eduai.web.app import serve as run

    run(port=port, host=host)


@app.command("fetch-adapter")
def fetch_adapter(
    source: str = typer.Option(None, help="Local tarball path or URL instead of the release"),
) -> None:
    import importlib.util

    from eduai.config import ROOT

    spec = importlib.util.spec_from_file_location("fetch_adapter", ROOT / "scripts" / "fetch_adapter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    console.print(f"adapter installed at {mod.fetch(source=source)}")


if __name__ == "__main__":
    app()
