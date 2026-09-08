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
                msg = dict(replies.pop(0))
            usage = {"completion_tokens": 42, **msg.pop("_usage", {})}
            data = json.dumps(
                {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": usage}
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

    srv, seen = _stub_server([{"role": "assistant", "content": ok, "reasoning_content": "r"}] * 2)
    try:
        port = srv.server_port
        m = llm_judge(out, "sub", ChatClient(f"http://127.0.0.1:{port}/v1", "m"), concurrency=1, per_arm=1)
    finally:
        srv.shutdown()
    key = json.loads((out / "key.json").read_text())
    judged = [json.loads(line)["audit_id"] for line in (out / "llm_sub" / "verdicts.jsonl").open()]
    assert m["n"] == 2 and m["subset_first_per_arm"] == 1
    assert sorted(key[a]["arm"] for a in judged) == sorted(arms)
    first = {}
    for r in (json.loads(line) for line in (out / "sheet.jsonl").open()):
        first.setdefault(key[r["audit_id"]]["arm"], r["audit_id"])
    assert sorted(judged) == sorted(first.values())
    assert not any(a in json.dumps(seen) for a in arms)

    res = llm_score(out, gen)
    assert res["kappa"]["stub vs sub"]["n"] == 2
    j = res["judges"]["stub"]
    assert sum(r["n"] for r in j["by_arm"].values()) == 4
    assert j["agreement_with_llama-1b_key"]["n"] == 4 and j["agreement_with_llama-1b_key"]["agree"] == 0.75

    assert strip_reasoning("<think>a</think>b") == ("b", "<think>a</think>")
    assert extract_verdict('x {"key_correct": "yes", "lo_fit": true, "distractors_plausible": true}') is None
    assert extract_verdict("r " + ok)["_start"] == 2
    assert kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert kappa([True, True], [True, False]) is None


def test_pilot_keeps_lengths_only_and_sets_the_cap(tmp_path):
    from scripts.blind_audit import ChatClient, pilot, pilot_cap, thinking_stats

    assert pilot_cap([100] * 10) == 1024
    assert pilot_cap([1000] * 9 + [3000]) == 1280  # p90 = 1200 -> 1280
    assert pilot_cap([2049] * 10) == 2304
    out = tmp_path / "pilot"
    out.mkdir()
    sheet = [
        {"audit_id": f"a{i}", "passage": "p", "lo_id": "x", "lo_text": "y", "item": {}} for i in range(3)
    ]
    _write(out / "sheet.jsonl", sheet)
    ok = '{"key_correct": true, "lo_fit": true, "distractors_plausible": true, "notes": "n"}'
    replies = [
        {"role": "assistant", "content": ok, "reasoning_content": "r", "_usage": {"thinking_tokens": t}}
        for t in (500, 2000, 6000)
    ]
    srv, seen = _stub_server(replies)
    try:
        rec = pilot(out, "stub", ChatClient(f"http://127.0.0.1:{srv.server_port}/v1", "m"), concurrency=1)
    finally:
        srv.shutdown()
    assert sorted(rec["thinking_tokens"]) == [500, 2000, 6000] and rec["at_safety_cap"] == 1
    assert rec["cap"] == pilot_cap([500, 2000, 6000])
    assert all(b["max_thinking_tokens"] == 6000 for b in seen)
    saved = (out / "pilot_stub.json").read_text()
    assert "key_correct" not in saved and "notes" not in saved
    with pytest.raises(SystemExit, match="runs once"):
        pilot(out, "stub", None)
    st = thinking_stats([{"thinking_tokens": [1024]}, {"thinking_tokens": [10, 1024]}, {}], 1024)
    assert st["items_at_cap"] == 2 and st["n_reported"] == 3


def test_finalize_partial_keeps_finished_verdicts_in_sheet_order(tmp_path):
    from scripts.blind_audit import finalize_partial

    out = tmp_path / "a"
    (out / "llm_j").mkdir(parents=True)
    _write(out / "sheet.jsonl", [{"audit_id": f"a{i}"} for i in range(4)])
    (out / "key.json").write_text(
        json.dumps({f"a{i}": {"arm": "v2" if i % 2 else "base", "id": f"p{i}"} for i in range(4)})
    )
    v = {"first_parse_failed": False, "parsed": True, "reasoning_from": "server", "thinking_tokens": [8]}
    _write(
        out / "llm_j" / "verdicts.partial.jsonl",
        [{**v, "audit_id": a, "done_at": t} for a, t in (("a2", 5.0), ("a0", 9.0))],
    )
    m = finalize_partial(out, "j", "hard stop", {"max_thinking_tokens": 8})
    assert m["partial"] and m["n"] == 2 and m["n_by_arm"] == {"base": 2, "v2": 0}
    assert m["thinking_tokens"]["items_at_cap"] == 2 and m["last_done_at_seconds"] == 9.0
    ids = [json.loads(line)["audit_id"] for line in (out / "llm_j" / "verdicts.jsonl").open()]
    assert ids == ["a0", "a2"] and not (out / "llm_j" / "verdicts.partial.jsonl").exists()
    with pytest.raises(SystemExit, match="nothing to finalize"):
        finalize_partial(out, "j", "x", {})
