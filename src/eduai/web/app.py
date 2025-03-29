from __future__ import annotations

import logging
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
        request: Request, subject: str = Form(...), mode: str = Form(...), length: int = Form(10)
    ):
        try:
            sid = service.create(subject, mode, length if mode == "practice" else None)
        except ValueError as exc:
            return render(
                request,
                "index.html",
                400,
                subjects=service.subjects(),
                error=str(exc),
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
        return render(request, "session.html", progress=prog, q=q)

    @app.get("/sessions/{sid}/next", response_class=HTMLResponse)
    def next_fragment(request: Request, sid: str):
        prog = get_state(sid)
        q = None if prog["done"] else service.next_question(sid)
        if q is None:
            resp = HTMLResponse("")
            resp.headers["HX-Redirect"] = f"/sessions/{sid}/report"
            return resp
        return render(request, "_question.html", progress=prog, q=q)

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
        return render(request, "_feedback.html", progress=service.progress(sid), res=res)

    @app.get("/sessions/{sid}/report", response_class=HTMLResponse)
    def report_page(request: Request, sid: str):
        get_state(sid)
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
        return {"id": sid, **service.progress(sid)}

    @app.get("/api/sessions/{sid}")
    def api_progress(sid: str):
        return get_state(sid)

    @app.get("/api/sessions/{sid}/next")
    def api_next(sid: str):
        prog = get_state(sid)
        q = None if prog["done"] else service.next_question(sid)
        if q is None:
            return {"done": True, "progress": service.progress(sid)}
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
        return {**res, "progress": service.progress(sid)}

    @app.get("/api/sessions/{sid}/report")
    def api_report(sid: str):
        get_state(sid)
        return service.report(sid)

    @app.post("/api/generate")
    def api_generate(body: dict):
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
        gen = Generator(backend).generate(req)
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
        return render(
            request,
            "error.html",
            404,
            title="Not found",
            message="That session doesn't exist. It may have been created on another machine.",
        )

    return app


def serve(port: int = 8001, host: str = "127.0.0.1") -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
