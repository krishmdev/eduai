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
