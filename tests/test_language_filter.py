"""A shopper writing in one language must not get another language's chunk in the top-k (the French
return-policy duplicate leaking behind the English one on the most-asked question). Retrieval drops
mismatched-language chunks, keeps untagged ones, and defers when the language is unknown."""
from adapters.factory import make_embedder, make_store
from pipeline.answer import _query_lang, retrieve
from retrieval.sparse import SparseEncoder


def test_query_lang_detects_en_fr_and_defers_on_other_scripts():
    assert _query_lang("what is your return policy") == "en"
    assert _query_lang("Quelle est votre politique de retour") == "fr"
    assert _query_lang("Ou en est ma commande") == "fr"
    assert _query_lang("送料はいくらですか") is None  # other script -> no filtering


def _seed():
    emb, enc, store = make_embedder("fake"), SparseEncoder(), make_store("memory")
    docs = [
        {"id": "EN", "text": "you can return items within 30 days",
         "payload": {"doc_type": "guide", "lang": "en",
                     "text": "you can return items within 30 days"}},
        {"id": "FR", "text": "vous pouvez retourner sous 30 jours",
         "payload": {"doc_type": "guide", "lang": "fr",
                     "text": "vous pouvez retourner sous 30 jours"}},
        {"id": "NA", "text": "returns policy details here",
         "payload": {"doc_type": "guide", "text": "returns policy details here"}},  # untagged
    ]
    dense = emb.embed([d["text"] for d in docs])
    store.upsert([{**docs[i], "dense": dense[i],
                   "sparse": {"indices": enc.encode(docs[i]["text"]).indices,
                              "values": enc.encode(docs[i]["text"]).values}} for i in range(3)])
    return emb, store


def test_retrieve_drops_other_language_chunks_keeps_untagged():
    emb, store = _seed()
    langs = {(h.get("payload") or {}).get("lang")
             for h in retrieve("return policy", emb, store, top_k=8)}
    assert "fr" not in langs      # the French chunk is dropped for an English query
    assert langs <= {"en", None}  # English and untagged chunks remain


def test_retrieve_defers_when_the_query_language_is_unknown():
    emb, store = _seed()
    # a query in neither en nor fr defers (no filter), so nothing is dropped on language grounds
    assert len(retrieve("送料 policy", emb, store, top_k=8)) == 3
