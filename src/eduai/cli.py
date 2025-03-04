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


if __name__ == "__main__":
    app()
