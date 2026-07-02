"""M1.2 chunking: correct splitting, metadata, and stable checksums."""
from ingest.chunk import chunk_records, split_text


def test_short_text_is_one_chunk():
    assert split_text("a b c", max_words=10) == ["a b c"]


def test_long_text_splits_with_overlap():
    text = " ".join(str(i) for i in range(50))
    pieces = split_text(text, max_words=20, overlap=5)
    assert len(pieces) > 1
    assert pieces[0].split()[-5:] == pieces[1].split()[:5]


def test_chunk_records_metadata_and_ids():
    recs = [{"id": "R1", "text": "hello world", "lang": "en", "rating": 5}]
    chunks = chunk_records(recs, id_field="id", text_field="text",
                           lang_field="lang", meta_fields=["rating"], doc_type="review")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == "R1"
    assert chunk.metadata["lang"] == "en"
    assert chunk.metadata["rating"] == 5
    assert chunk.metadata["doc_type"] == "review"
    assert len(chunk.metadata["checksum"]) == 64


def test_checksum_changes_with_text():
    a = chunk_records([{"id": "1", "text": "aaa"}], id_field="id", text_field="text")[0]
    b = chunk_records([{"id": "1", "text": "bbb"}], id_field="id", text_field="text")[0]
    assert a.metadata["checksum"] != b.metadata["checksum"]
