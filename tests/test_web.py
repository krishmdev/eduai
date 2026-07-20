import re

import pytest
from fastapi.testclient import TestClient

from eduai.config import Settings
from eduai.data.samples import DEMO_BANK
from eduai.web.app import create_app


@pytest.fixture
def client(tmp_path):
    s = Settings(backend="bank", db_path=tmp_path / "t.db", bank_path=DEMO_BANK, data_dir=tmp_path)
    return TestClient(create_app(s))


def test_health_reports_bank_only(client):
    h = client.get("/api/health").json()
    assert h["backend"] == "bank-only" and h["items"] > 100


def test_practice_session_via_api(client):
    sid = client.post("/api/sessions", json={"subject": "BIO", "mode": "practice", "length": 5}).json()["id"]
    seen = set()
    for _ in range(5):
        nxt = client.get(f"/api/sessions/{sid}/next").json()
        assert not nxt["done"]
        assert nxt["item"]["id"] not in seen
        seen.add(nxt["item"]["id"])
        assert "answer" not in nxt["item"]
        res = client.post(f"/api/sessions/{sid}/answer", json={"choice": "A"}).json()
        assert res["answer"] in "ABCD"
    assert client.get(f"/api/sessions/{sid}/next").json()["done"]
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["summary"]["answered"] == 5
    assert any(u["mastery"] is not None for u in rep["units"])
    assert all(u["mastery"] is None for u in rep["units"] if u["answered"] == 0)
    assert client.post(f"/api/sessions/{sid}/answer", json={"choice": "A"}).status_code == 409


def test_assessment_stops_by_precision_or_cap(client):
    sid = client.post("/api/sessions", json={"subject": "CHEM", "mode": "assessment"}).json()["id"]
    n = 0
    while not client.get(f"/api/sessions/{sid}/next").json()["done"]:
        res = client.post(f"/api/sessions/{sid}/answer", json={"choice": "B"}).json()
        assert set(res) == {"choice", "done", "progress"}  # no key or feedback during an assessment
        n += 1
        assert n <= 30
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["sd"] < 0.5 or n == 30


def test_answer_calibrates_item_in_bank(client):
    app = client.app
    sid = client.post("/api/sessions", json={"subject": "BIO", "mode": "practice", "length": 3}).json()["id"]
    item_id = client.get(f"/api/sessions/{sid}/next").json()["item"]["id"]
    _, n0, _ = app.state.service.bank.item_b(item_id)
    client.post(f"/api/sessions/{sid}/answer", json={"choice": "C"})
    _, n1, _ = app.state.service.bank.item_b(item_id)
    assert n1 == n0 + 1


def test_html_flow(client):
    r = client.get("/")
    assert r.status_code == 200 and "AP Biology" in r.text
    r = client.post(
        "/sessions", data={"subject": "BIO", "mode": "practice", "length": "10"}, follow_redirects=False
    )
    assert r.status_code == 303
    page = client.get(r.headers["location"]).text
    assert 'name="choice"' in page and "Mastery by unit" in page
    sid = r.headers["location"].rsplit("/", 1)[1]
    fb = client.post(f"/sessions/{sid}/answer", data={"choice": "A"}).text
    assert "hx-swap-oob" in fb and re.search(r"Correct\.|Not quite", fb)
    assert client.get(f"/sessions/{sid}/report").status_code == 200


def test_bad_input(client):
    assert client.post("/api/sessions", json={"subject": "XX", "mode": "practice"}).status_code == 400
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.get("/sessions/nope").status_code == 404
    sid = client.post("/api/sessions", json={"subject": "BIO", "mode": "practice"}).json()["id"]
    client.get(f"/api/sessions/{sid}/next")
    assert client.post(f"/api/sessions/{sid}/answer", json={"choice": "Z"}).status_code == 400
    assert client.post("/api/generate", json={"lo_id": "BIO.2.1.a", "passage": "x"}).status_code == 503
    assert client.post("/api/generate", json={"passage": "x"}).status_code == 400
    assert client.post("/api/generate", json={"lo_id": "NOPE", "passage": "x"}).status_code == 400
    for bad in (-5, 0, 61, "abc", 2.7, True):
        r = client.post("/api/sessions", json={"subject": "BIO", "mode": "assessment", "length": bad})
        assert r.status_code == 400, bad


def test_concurrent_submits_record_one_response(client):
    import sqlite3
    import threading

    sid = client.post("/api/sessions", json={"subject": "BIO", "mode": "practice", "length": 10}).json()["id"]
    item = client.get(f"/api/sessions/{sid}/next").json()["item"]["id"]
    bank = client.app.state.service.bank
    _, n0, _ = bank.item_b(item)
    codes = []
    barrier = threading.Barrier(4)

    def go():
        barrier.wait()
        codes.append(client.post(f"/api/sessions/{sid}/answer", json={"choice": "A"}).status_code)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(codes) == [200, 409, 409, 409]
    _, n1, _ = bank.item_b(item)
    con = sqlite3.connect(bank.path)
    rows = con.execute("SELECT COUNT(*) FROM responses WHERE session_id = ?", (sid,)).fetchone()[0]
    assert rows == 1 and n1 - n0 == 1


def test_backend_order_prefers_base_two_shot():
    from eduai.evaluation.compare import fixed_shots
    from eduai.llm.select import ORDER

    assert ORDER[0] == "mlx-base"
    shots = fixed_shots()
    assert sorted(r.format for r, _ in shots) == ["standard", "stimulus"]


def test_assessment_progress_hides_results_until_done(client):
    sid = client.post("/api/sessions", json={"subject": "BIO", "mode": "assessment"}).json()["id"]
    client.get(f"/api/sessions/{sid}/next")
    res = client.post(f"/api/sessions/{sid}/answer", json={"choice": "A"}).json()
    assert not {"correct", "theta", "units"} & set(res["progress"])
    assert not {"correct", "theta", "units"} & set(client.get(f"/api/sessions/{sid}").json())
    page = client.get(f"/sessions/{sid}").text
    assert "Precision" in page and "correct ·" not in page


def test_generic_404_and_in_progress_report(client):
    r = client.get("/definitely-not-a-page")
    assert r.status_code == 404 and "no page at this address" in r.text
    sid = client.post("/api/sessions", json={"subject": "BIO", "mode": "practice", "length": 5}).json()["id"]
    client.get(f"/api/sessions/{sid}/next")
    client.post(f"/api/sessions/{sid}/answer", json={"choice": "A"})
    rep = client.get(f"/sessions/{sid}/report").text
    assert "In progress" in rep and "Continue session" in rep and "-0.00" not in rep
