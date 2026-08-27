"""The blind audit must not turn missing or malformed verdicts into positive scores."""

import json

import pytest

from scripts.blind_audit import draw, score


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_audit_rejects_bad_verdicts_and_sheet_changes(tmp_path):
    gen = tmp_path / "gen"
    gen.mkdir()
    item = {
        "stem": "What produces ATP?",
        "choices": {"A": "mitochondria", "B": "nucleus", "C": "ribosome", "D": "vacuole"},
        "answer": "A",
        "explanation": "Mitochondria produce ATP.",
        "lo_id": "BIO.2.1.b",
        "difficulty": "easy",
    }
    prompts = tmp_path / "prompts.jsonl"
    _write(
        prompts,
        [
            {
                "id": "p1",
                "request": {
                    "passage": "Mitochondria make ATP.",
                    "lo_id": "BIO.2.1.b",
                    "lo_text": "Identify organelles.",
                    "target_misconception": None,
                },
            }
        ],
    )
    _write(gen / "per_item.jsonl", [{"arm": a, "id": "p1", "usable": True} for a in ("v2", "base")])
    for arm in ("v2", "base"):
        _write(gen / f"gen_{arm}.jsonl", [{"id": "p1", "text": json.dumps(item)}])
    out = tmp_path / "audit"
    draw(gen, prompts, ["v2", "base"], 1, 7, out)
    with pytest.raises(SystemExit, match="already has an audit"):
        draw(gen, prompts, ["v2", "base"], 1, 7, out)

    sheet = [json.loads(x) for x in (out / "sheet.jsonl").read_text().splitlines()]
    sheet[0].update(key_correct="false", lo_fit=True)
    sheet[1].update(key_correct=True, lo_fit=False)
    _write(out / "sheet.jsonl", sheet)
    with pytest.raises(SystemExit, match="boolean"):
        score(out)

    sheet[0]["key_correct"] = False
    _write(out / "sheet.jsonl", sheet + [sheet[0]])
    with pytest.raises(SystemExit, match="once each"):
        score(out)
    _write(out / "sheet.jsonl", sheet[:1])
    with pytest.raises(SystemExit, match="exactly"):
        score(out)

    _write(out / "sheet.jsonl", sheet)
    result = score(out)
    assert sorted(v["n"] for v in result.values()) == [1, 1]
    draw(gen, prompts, ["v2", "base"], 1, 7, out, force=True)
    assert not (out / "score.json").exists()


def _stub_server(replies):
    """OpenAI-style stub: pops the next reply for every request and records the bodies."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                seen.append(body)
                msg = replies.pop(0)
            data = json.dumps(
                {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"completion_tokens": 42}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_llm_judge_against_a_stub_server(tmp_path):
    from scripts.blind_audit import ChatClient, extract_verdict, kappa, llm_judge, llm_score, strip_reasoning

    gen = tmp_path / "gen"
    gen.mkdir()
    item = {
        "stem": "What produces ATP?",
        "choices": {"A": "mitochondria", "B": "nucleus", "C": "ribosome", "D": "vacuole"},
        "answer": "A",
        "explanation": "Mitochondria produce ATP.",
    }
    prompts = tmp_path / "prompts.jsonl"
    req = {"passage": "Mitochondria make ATP.", "lo_id": "BIO.2.1.b", "lo_text": "Identify organelles."}
    _write(prompts, [{"id": f"p{i}", "request": {**req, "target_misconception": None}} for i in range(2)])
    arms = ("finetuned-v2", "base-2shot")
    _write(
        gen / "per_item.jsonl", [{"arm": a, "id": f"p{i}", "usable": True} for a in arms for i in range(2)]
    )
    for arm in arms:
        _write(gen / f"gen_{arm}.jsonl", [{"id": f"p{i}", "text": json.dumps(item)} for i in range(2)])
    _write(
        gen / "judge_llama-1b.jsonl",
        [{"arm": a, "id": f"p{i}", "agrees": True} for a in arms for i in range(2)],
    )
    out = tmp_path / "audit"
    draw(gen, prompts, list(arms), 2, 7, out)

    ok = '{"key_correct": true, "lo_fit": true, "distractors_plausible": false, "notes": "fine"}'
    bad = '{"key_correct": false, "lo_fit": true, "distractors_plausible": true, "notes": "two answers"}'
    replies = [
        {"role": "assistant", "content": '<think>Hmm, {"key_correct": maybe}</think>no json here'},
        {"role": "assistant", "content": ok},  # the retry
        {"role": "assistant", "content": ok, "reasoning_content": "server-side reasoning"},
        {"role": "assistant", "content": "<|channel>thought\nlet me see<channel|>" + bad},
        {"role": "assistant", "content": "thought\nplain text with {braces}\n" + ok},
    ]
    srv, seen = _stub_server(replies)
    try:
        client = ChatClient(f"http://127.0.0.1:{srv.server_port}/v1", "stub-model")
        m = llm_judge(out, "stub", client, thinking=True, concurrency=1, meta={"server": {"commit": "abc"}})
    finally:
        srv.shutdown()
    assert m["n"] == 4 and m["first_try_parse_failure_rate"] == 0.25 and m["unparsed_after_retry"] == 0
    assert m["thinking"] and m["max_thinking_tokens"] == 3000 and m["server"]["commit"] == "abc"
    bodies = json.dumps(seen)
    assert not any(a in bodies for a in arms), "the arm must never reach the judge"
    assert all(
        "response_format" not in b and b["chat_template_kwargs"] == {"enable_thinking": True} for b in seen
    )
    assert seen[1]["messages"][-1]["role"] == "user" and len(seen[1]["messages"]) == 4
    with pytest.raises(SystemExit, match="runs once"):
        llm_judge(out, "stub", client)
    srv, _ = _stub_server([{"role": "assistant", "content": ok}])
    try:
        port = srv.server_port
        with pytest.raises(SystemExit, match="ignored enable_thinking"):
            llm_judge(out, "nothink", ChatClient(f"http://127.0.0.1:{port}/v1", "m"), thinking=True)
    finally:
        srv.shutdown()

    res = llm_score(out, gen)
    j = res["judges"]["stub"]
    assert sum(r["n"] for r in j["by_arm"].values()) == 4
    assert j["agreement_with_llama-1b_key"]["n"] == 4 and j["agreement_with_llama-1b_key"]["agree"] == 0.75

    assert strip_reasoning("<think>a</think>b") == ("b", "<think>a</think>")
    assert extract_verdict('x {"key_correct": "yes", "lo_fit": true, "distractors_plausible": true}') is None
    assert extract_verdict("r " + ok)["_start"] == 2
    assert kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert kappa([True, True], [True, False]) is None
