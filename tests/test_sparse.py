"""M1.2 sparse encoder: aligned vectors, sensible weights, deterministic output."""
from retrieval.sparse import Bm25SparseEncoder, SparseVector


def test_encode_returns_aligned_vector():
    encoder = Bm25SparseEncoder().fit(["the quick brown fox", "the lazy dog"])
    vec = encoder.encode("quick fox")
    assert isinstance(vec, SparseVector)
    assert len(vec.indices) == len(vec.values)
    assert all(v > 0 for v in vec.values)


def test_rare_term_outweighs_common_term():
    corpus = ["common word here"] * 5 + ["rareunicorn"]
    encoder = Bm25SparseEncoder().fit(corpus)
    common = encoder.encode("common")
    rare = encoder.encode("rareunicorn")
    assert max(rare.values) > max(common.values)


def test_encoding_is_deterministic():
    encoder = Bm25SparseEncoder().fit(["alpha beta", "beta gamma"])
    assert encoder.encode("alpha beta").values == encoder.encode("alpha beta").values
