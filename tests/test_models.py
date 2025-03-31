"""Exercises the real pinned bge-small + cross-encoder tagger. Skipped when `make models` hasn't run."""

import pytest

from eduai.models import available

pytestmark = pytest.mark.skipif(not (available("bge-small") and available("rerank")),
                                reason="pinned embedding models not downloaded (run `make models`)")


def test_real_tagger_ranks_obvious_items(taxonomy):
    from eduai.curriculum.tagger import build_tagger, tag_text

    tagger = build_tagger(taxonomy, "bge-small", rerank=True)
    res = tagger.tag_texts([
        tag_text("Which process do plants use to turn light energy into chemical energy?", "photosynthesis"),
        tag_text("What is the pH of a neutral solution at 25 C?", "7"),
    ])
    assert res[0].subject == "BIO" and "BIO.3.2.a" in res[0].top3
    assert res[1].subject == "CHEM" and "CHEM.8.1.b" in res[1].top3
    aligned = tagger.is_aligned([tag_text("What process do plants use to make glucose from sunlight?",
                                          "photosynthesis")], ["BIO.3.2.a"])[0]
    assert aligned["target_in_top3"]
