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
        assert res["explanation"] is None  # no feedback in assessment mode
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
