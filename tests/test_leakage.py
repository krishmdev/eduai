import numpy as np

from eduai.data.leakage import (
    ShingleIndex,
    containment_pairs,
    norm_answer,
    same_answer_qa_leaks,
    shingles,
)


def test_shingles_and_shingle_index():
    text1 = "the mitochondria are the powerhouses of the cell and generate ATP"
    text2 = "the mitochondria are the powerhouses of the cell and store energy"
    s1 = shingles(text1, n=5)
    s2 = shingles(text2, n=5)
    assert len(s1) > 0
    assert len(s1.intersection(s2)) > 0

    idx = ShingleIndex([text1, text2], n=5)
    score, match_idx = idx.best(text1)
    assert score == 1.0
    assert match_idx == 0


def test_containment_pairs():
    text1 = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    text2 = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
    pairs = containment_pairs([text1, text2], threshold=0.5, n=8)
    assert (0, 1) in pairs


def test_norm_answer_strips_leading_articles():
    assert norm_answer("the mitochondria") == "mitochondria"
    assert norm_answer("A Mitochondrion") == "mitochondrion"
    assert norm_answer("an electron") == "electron"
    assert norm_answer("photosynthesis") == "photosynthesis"
    assert norm_answer("THE cell wall") == "cell wall"


def test_same_answer_qa_leaks_detects_article_mismatches():
    # 2 queries, 2 refs
    q_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    ref_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    q_answers = ["the mitochondria", "photosynthesis"]
    ref_answers = ["mitochondria", "a plant"]

    # Match when answers are semantically equal after stripping articles
    leaks = same_answer_qa_leaks(q_vecs, q_answers, ref_vecs, ref_answers, threshold=0.8)
    assert len(leaks) == 1
    assert leaks[0] == (0, 0, 1.0)
