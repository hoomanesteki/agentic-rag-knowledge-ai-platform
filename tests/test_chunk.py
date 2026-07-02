"""M2.4 chunking: sentence/token-aware splitting, contextual prefix, metadata, checksums."""
import re

from ingest.chunk import chunk_records, split_text

_SENT = re.compile(r"(?<=[.!?])\s+")


def test_short_text_is_one_chunk():
    assert split_text("one short sentence only.") == ["one short sentence only."]


def test_long_text_splits_on_sentences_with_overlap():
    text = " ".join("Sentence number {} sits here.".format(i) for i in range(30))
    pieces = split_text(text, max_tokens=20, overlap_sentences=1)
    assert len(pieces) > 1
    first = set(_SENT.split(pieces[0]))
    second = set(_SENT.split(pieces[1]))
    assert first & second  # a whole sentence overlaps the chunk boundary


def test_chunk_records_metadata_and_ids():
    recs = [{"id": "R1", "text": "hello world.", "lang": "en", "rating": 5}]
    chunk = chunk_records(recs, id_field="id", text_field="text",
                          lang_field="lang", meta_fields=["rating"], doc_type="review")[0]
    assert chunk.id == "R1"
    assert chunk.metadata["lang"] == "en"
    assert chunk.metadata["rating"] == 5
    assert chunk.metadata["doc_type"] == "review"
    assert len(chunk.metadata["checksum"]) == 64


def test_context_prefix_stored_in_metadata_not_text():
    recs = [{"id": "R1", "text": "runs small.", "product_id": "P002"}]
    with_ctx = chunk_records(recs, id_field="id", text_field="text",
                             context_fields=["product_id"])[0]
    assert with_ctx.text == "runs small."                       # display text stays clean
    assert with_ctx.metadata["context"] == "[product_id=P002]"  # prefix is embedded separately
    without = chunk_records(recs, id_field="id", text_field="text")[0]
    assert "context" not in without.metadata


def test_context_prefix_skips_missing_and_none_fields():
    recs = [{"id": "R1", "text": "runs small.", "product_id": None}]
    chunk = chunk_records(recs, id_field="id", text_field="text", context_fields=["product_id"])[0]
    assert "context" not in chunk.metadata  # None value is not turned into "product_id=None"


def test_over_budget_sentence_emits_one_chunk():
    # a single no-punctuation "sentence" far over budget must terminate and emit one chunk
    pieces = split_text("word " * 1000, max_tokens=50)
    assert len(pieces) == 1


def test_multi_chunk_ids_use_suffix():
    text = " ".join("Sentence number {} sits here.".format(i) for i in range(30))
    chunks = chunk_records([{"id": "R1", "text": text}], id_field="id", text_field="text",
                           max_tokens=20)
    assert len(chunks) > 1
    assert [c.id for c in chunks][:2] == ["R1#0", "R1#1"]


def test_overlap_zero_produces_no_shared_sentence():
    text = " ".join("Sentence number {} sits here.".format(i) for i in range(30))
    pieces = split_text(text, max_tokens=20, overlap_sentences=0)
    joined = " ".join(pieces)
    assert joined.count("Sentence number 0 sits here.") == 1  # no duplication without overlap


def test_empty_text_skipped():
    assert chunk_records([{"id": "R1", "text": "   "}], id_field="id", text_field="text") == []


def test_checksum_changes_with_text():
    a = chunk_records([{"id": "1", "text": "aaa."}], id_field="id", text_field="text")[0]
    b = chunk_records([{"id": "1", "text": "bbb."}], id_field="id", text_field="text")[0]
    assert a.metadata["checksum"] != b.metadata["checksum"]
