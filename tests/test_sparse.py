"""M1.2 sparse encoder: stateless, aligned, saturating term frequency (Qdrant adds idf)."""
from retrieval.sparse import SparseEncoder, SparseVector


def test_encode_returns_aligned_sorted_vector():
    vec = SparseEncoder().encode("quick brown fox")
    assert isinstance(vec, SparseVector)
    assert len(vec.indices) == len(vec.values)
    assert vec.indices == sorted(vec.indices)
    assert all(v > 0 for v in vec.values)


def test_term_frequency_saturates():
    encoder = SparseEncoder(k1=1.5)
    one = encoder.encode("word").values[0]
    two = encoder.encode("word word").values[0]
    assert two > one          # repeating helps
    assert two < 2 * one      # but sublinearly


def test_encoding_is_deterministic():
    encoder = SparseEncoder()
    assert encoder.encode("alpha beta").values == encoder.encode("alpha beta").values
