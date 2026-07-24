from __future__ import annotations

import logging
import math
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from eduai.config import ROOT, Settings, get_settings
from eduai.curriculum.taxonomy import default_taxonomy
from eduai.data.samples import DEMO_BANK
from eduai.generation.bank import ItemBank
from eduai.web.service import (
    ASSESSMENT_MAX,
    PRACTICE_LENGTHS,
    AnswerConflict,
    Service,
    SessionFinished,
    SessionNotFound,
)

HERE = Path(__file__).parent
log = logging.getLogger("eduai.web")

GENERATED_ITEMS = ROOT / "data" / "samples" / "generated_items.jsonl"


def coarse_sd(sd: float) -> float:
    """Posterior SD rounded up to 0.1, for display during an assessment."""
    return math.ceil(sd * 10 - 1e-9) / 10


def bank_sources(settings: Settings) -> list[Path]:
    """Full SciQ-derived bank if it was built locally, else the committed demo sample."""
    if settings.bank_path:
        paths = [settings.bank_path]
    else:
        full = settings.data_dir / "bank" / "sciq_items.jsonl"
        paths = [full if full.exists() else DEMO_BANK]
    if GENERATED_ITEMS.exists():
        paths.append(GENERATED_ITEMS)
    return paths


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="EduAI", docs_url="/api/docs", openapi_url="/api/openapi.json")
    tax = default_taxonomy()
    bank = ItemBank(settings.db_path)
    for src in bank_sources(settings):
        bank.load_jsonl(src)

    from eduai.llm.select import select_backend

    backend, skipped = select_backend(settings)
    service = Service(settings.db_path, bank, tax)
    app.state.service = service
    app.state.backend = backend
    app.state.backend_skipped = skipped
    app.state.canary = None
    if settings.egress_canary:
        from eduai.egress import probe

        app.state.canary = probe()
        log.warning("egress canary: %s", app.state.canary)

    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals["backend_name"] = backend.name
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def pct(x):
        return "–" if x is None else f"{x:.0%}"

    templates.env.filters["pct"] = pct

    def render(request: Request, name: str, status: int = 200, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx, status_code=status)

    def public_progress(prog: dict) -> dict:
        """During a running assessment, hide anything that reveals right/wrong per answer."""
        if prog["mode"] == "assessment" and not prog["done"]:
            out = {k: v for k, v in prog.items() if k not in ("correct", "theta", "units")}
            # The size of each SD drop hints at right/wrong, so only coarse steps are shown.
            out["sd"] = coarse_sd(prog["sd"])
            return out
        return prog

    def page_progress(prog: dict) -> dict:
        """For templates: same data, but a running assessment only ever sees coarse SD steps."""
        if prog["mode"] == "assessment" and not prog["done"]:
            return {**prog, "sd": coarse_sd(prog["sd"])}
        return prog

    def get_state(sid: str) -> dict:
        try:
            return service.progress(sid)
        except SessionNotFound as exc:
            raise HTTPException(404, "session not found") from exc

    # -- HTML -----------------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return render(
            request,
            "index.html",
            subjects=service.subjects(),
            practice_lengths=PRACTICE_LENGTHS,
            assessment_max=ASSESSMENT_MAX,
        )

    @app.post("/sessions")
    def create_session(
        request: Request, subject: str = Form(""), mode: str = Form(""), length: str = Form("10")
    ):
        # Every field is optional at the form layer so bad input gets a friendly page, not a raw 422.
        try:
            sid = service.create(subject, mode, length if mode == "practice" else None)
        except ValueError as exc:
            friendly = {
                "mode": "Choose practice or assessment.",
                "subject": "Choose one of the subjects below.",
                "length": "Choose a practice length from the list.",
            }
            key = next((k for k in friendly if k in str(exc)), None)
            return render(
                request,
                "index.html",
                400,
                subjects=service.subjects(),
                error=friendly.get(key, "That didn't work. Check your choices and try again."),
                practice_lengths=PRACTICE_LENGTHS,
                assessment_max=ASSESSMENT_MAX,
            )
        return RedirectResponse(f"/sessions/{sid}", status_code=303)

    @app.get("/sessions/{sid}", response_class=HTMLResponse)
    def session_page(request: Request, sid: str):
        prog = get_state(sid)
        if prog["done"]:
            return RedirectResponse(f"/sessions/{sid}/report", status_code=303)
        q = service.next_question(sid)
        return render(request, "session.html", progress=page_progress(prog), q=q)

    @app.get("/sessions/{sid}/next", response_class=HTMLResponse)
    def next_fragment(request: Request, sid: str):
        prog = get_state(sid)
        q = None if prog["done"] else service.next_question(sid)
        if q is None:
            resp = HTMLResponse("")
            resp.headers["HX-Redirect"] = f"/sessions/{sid}/report"
            return resp
        return render(request, "_question.html", progress=page_progress(prog), q=q)

    @app.post("/sessions/{sid}/answer", response_class=HTMLResponse)
    def answer_fragment(request: Request, sid: str, choice: str = Form("")):
        get_state(sid)
        try:
            res = service.answer(sid, choice)
        except SessionFinished:
            resp = HTMLResponse("")
            resp.headers["HX-Redirect"] = f"/sessions/{sid}/report"
            return resp
        except AnswerConflict:
            resp = HTMLResponse("")
            resp.headers["HX-Redirect"] = f"/sessions/{sid}"
            return resp
        except ValueError as exc:
            return render(request, "_error.html", 400, message=str(exc))
        return render(request, "_feedback.html", progress=page_progress(service.progress(sid)), res=res)

    @app.get("/sessions/{sid}/report", response_class=HTMLResponse)
    def report_page(request: Request, sid: str):
        prog = get_state(sid)
        if prog["mode"] == "assessment" and not prog["done"]:
            return render(request, "report_pending.html", progress=page_progress(prog))
        return render(request, "report.html", r=service.report(sid))

    # -- JSON -----------------------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "backend": backend.name,
            "skipped": skipped,
            "items": bank.count(),
            "egress_canary": app.state.canary,
        }

    @app.get("/api/subjects")
    def api_subjects():
        return service.subjects()

    @app.post("/api/sessions")
    def api_create(body: dict):
        try:
            sid = service.create(body.get("subject", ""), body.get("mode", ""), body.get("length"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"id": sid, **public_progress(service.progress(sid))}

    @app.get("/api/sessions/{sid}")
    def api_progress(sid: str):
        return public_progress(get_state(sid))

    @app.get("/api/sessions/{sid}/next")
    def api_next(sid: str):
        prog = get_state(sid)
        q = None if prog["done"] else service.next_question(sid)
        if q is None:
            return {"done": True, "progress": public_progress(service.progress(sid))}
        item = {
            k: q.item.get(k) for k in ("id", "stem", "stimulus", "choices", "lo_id", "unit_id", "difficulty")
        }
        return {"done": False, "number": q.number, "item": item, "lo_text": q.lo_text, "unit": q.unit_name}

    @app.post("/api/sessions/{sid}/answer")
    def api_answer(sid: str, body: dict):
        get_state(sid)
        try:
            res = service.answer(sid, body.get("choice", ""))
        except SessionFinished as exc:
            raise HTTPException(409, "session is finished") from exc
        except AnswerConflict as exc:
            raise HTTPException(409, "this question was already answered") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        res.pop("item")
        prog = public_progress(service.progress(sid))
        if prog["mode"] == "assessment":
            # No feedback during an assessment; the key is revealed in the report.
            res = {"choice": res["choice"], "done": res["done"]}
        return {**res, "progress": prog}

    @app.get("/api/sessions/{sid}/report")
    def api_report(sid: str):
        prog = get_state(sid)
        if prog["mode"] == "assessment" and not prog["done"]:
            # Results stay hidden until the assessment ends.
            keep = ("id", "mode", "subject", "subject_name", "answered", "max_items", "done", "sd_stop")
            return {**{k: prog[k] for k in keep}, "sd": coarse_sd(prog["sd"]), "results_available": False}
        return {**service.report(sid), "results_available": True}

    @app.post("/api/generate")
    def api_generate(body: dict):
        if not isinstance(body.get("passage"), str) or body.get("lo_id") not in tax.objectives:
            raise HTTPException(400, "lo_id (a known objective id) and passage are required")
        if backend.name == "bank-only":
            return JSONResponse({"error": "bank-only mode: no generator available", "skipped": skipped}, 503)
        from eduai.generation.generator import Generator
        from eduai.generation.validate import validate_item
        from eduai.prompts import GenerationRequest

        lo = tax.lo(body["lo_id"])
        req = GenerationRequest(
            subject=lo.subject_name,
            unit=lo.unit_name,
            topic=lo.topic_name,
            lo_id=lo.id,
            lo_text=lo.text,
            difficulty=body.get("difficulty", "medium"),
            format=body.get("format", "standard"),
            passage=body["passage"],
            target_misconception=body.get("target_misconception"),
        )
        from eduai.evaluation.compare import fixed_shots

        # Base models get the same two fixed examples as the eval's best arm; the adapter runs 0-shot.
        shots = [] if backend.name == "mlx+adapter" else fixed_shots()
        gen = Generator(backend, shots=shots).generate(req)
        res = validate_item(gen.text, req)
        return {
            "ok": res.ok,
            "reason": res.reason,
            "detail": res.detail,
            "item": res.item,
            "checks": res.checks,
            "raw": gen.text,
            "seconds": gen.seconds,
            "note": "structural checks only; key, alignment and novelty checks run in the batch pipeline",
        }

    @app.exception_handler(404)
    def not_found(request: Request, exc):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": getattr(exc, "detail", "not found")}, 404)
        if request.url.path.startswith("/sessions/"):
            message = "That session doesn't exist. Sessions live in this machine's local database."
        else:
            message = "There's no page at this address."
        return render(request, "error.html", 404, title="Not found", message=message)

    return app


def serve(port: int = 8001, host: str = "127.0.0.1") -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
