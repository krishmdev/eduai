import threading

from eduai.generation.bank import ItemBank
from eduai.kt.elo import ItemCalibration


def _row(i, diff="medium"):
    return {
        "id": f"x{i}",
        "stem": f"What is item {i}?",
        "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "answer": "A",
        "lo_id": "BIO.2.1.a",
        "subject": "BIO",
        "unit_id": "BIO.2",
        "difficulty": diff,
    }


def test_label_b_until_min_responses(tmp_path):
    bank = ItemBank(tmp_path / "b.db", ItemCalibration(k0=0.2, min_n=3))
    bank.add([_row(1, "hard")])
    for _ in range(2):
        bank.calibrate("x1", theta_pre=2.0, correct=True)
    item = bank.get("x1")
    assert item["b"] == 1.0 and item["b_hat"] < 1.0
    bank.calibrate("x1", theta_pre=2.0, correct=True)
    item = bank.get("x1")
    assert item["b"] == item["b_hat"] < 1.0 and item["n_responses"] == 3


def test_concurrent_calibration_counts_every_response(tmp_path):
    bank = ItemBank(tmp_path / "b.db")
    bank.add([_row(1)])
    threads = [
        threading.Thread(target=lambda: [bank.calibrate("x1", 0.0, True) for _ in range(25)])
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    b, n, label = bank.item_b("x1")
    assert n == 100 and b < label


def test_only_aligned_items_are_served(tmp_path):
    bank = ItemBank(tmp_path / "b.db")
    r = _row(2)
    r["aligned"] = False
    bank.add([_row(1), r])
    assert [c["id"] for c in bank.candidates("BIO.2.1.a")] == ["x1"]
