import numpy as np
import pytest

from eduai.curriculum.embedder import EmbedderError, HashingEmbedder, VectorIndex, cached_encode


class FlakyEmbedder(HashingEmbedder):
    def __init__(self, dim=64, fail_after=None):
        super().__init__(dim)
        self.embedder_id = f"flaky/test/v2/{dim}/x"
        self.fail_after = fail_after
        self.calls = 0

    def encode(self, texts, *, query=False):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("provider down")
        return super().encode(texts, query=query)


def test_generation_switch_keeps_old_index_when_new_provider_fails(tmp_path):
    idx = VectorIndex(tmp_path / "idx")
    old = HashingEmbedder(64)
    gen1 = idx.build(old, ["a", "b"], ["cell membrane", "plate tectonics"])
    assert idx.current().embedder_id == old.embedder_id

    new = FlakyEmbedder(64, fail_after=0)
    with pytest.raises(RuntimeError):
        idx.build(new, ["a", "b"], ["cell membrane", "plate tectonics"])
    live = idx.current()
    assert live.embedder_id == old.embedder_id
    np.testing.assert_array_equal(live.vectors, gen1.vectors)
    assert not list((tmp_path / "idx").glob(".building-*"))


def test_query_with_different_embedder_is_refused(tmp_path):
    idx = VectorIndex(tmp_path / "idx")
    idx.build(HashingEmbedder(64), ["a"], ["osmosis"])
    with pytest.raises(EmbedderError):
        idx.load_for(FlakyEmbedder(64))
    new = FlakyEmbedder(64)
    idx.build(new, ["a"], ["osmosis"])
    assert idx.load_for(new).embedder_id == new.embedder_id


def test_cache_is_keyed_by_embedder_id(tmp_path):
    a, b = HashingEmbedder(64), FlakyEmbedder(64)
    va = cached_encode(a, ["x y"], cache_dir=tmp_path)
    vb = cached_encode(b, ["x y"], cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 2
    np.testing.assert_array_equal(va, vb)  # same hashing, different identity -> separate files


def test_cache_is_keyed_by_device_and_reuses_legacy_mps_entries(tmp_path):
    import hashlib

    from eduai.curriculum.embedder import text_hash

    cpu, mps = HashingEmbedder(64), HashingEmbedder(64)
    mps.device = "mps"
    cached_encode(cpu, ["x y"], cache_dir=tmp_path)
    cached_encode(mps, ["x y"], cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 2

    # An entry under the old key (no device) is picked up by an MPS embedder, not re-encoded.
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    base = f"{mps.embedder_id}|0|{text_hash(['a b'])}"
    marker = np.full((1, 64), 7.0, dtype=np.float32)
    np.save(legacy_dir / f"{hashlib.sha256(base.encode()).hexdigest()[:24]}.npy", marker)
    np.testing.assert_array_equal(cached_encode(mps, ["a b"], cache_dir=legacy_dir), marker)
    assert len(list(legacy_dir.glob("*.npy"))) == 1
    assert not np.array_equal(cached_encode(cpu, ["a b"], cache_dir=legacy_dir), marker)
