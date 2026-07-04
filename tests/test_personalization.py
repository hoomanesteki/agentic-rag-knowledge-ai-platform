"""A logged-in shopper's taste profile: it must personalize from their OWN order history and never
leak the order-level PII (order numbers, tracking, shipping city) personalization does not need.

The profile powers "welcome back" recommendations biased to what someone actually buys. These
checks keep it useful (products/colors/size survive) and safe (identifiers do not), and confirm it
stays silent for anonymous shoppers and for accounts with no history.
"""
from adapters.factory import make_embedder, make_store
from pipeline.answer import _personal_profile_note
from retrieval.sparse import SparseEncoder

# Two real-shaped order docs for the demo account, plus the account summary doc.
_ORDERS = [
    {"id": "ODPROFILE", "doc_type": "order", "email": "info@esteki.ca",
     "text": ("Customer account on file: Jordan Avery, email info@esteki.ca (account key). "
              "Member of Aster Circle. Jordan Avery has placed 20 orders in the past 12 months.")},
    {"id": "OD100200", "doc_type": "order", "email": "info@esteki.ca",
     "text": ("Order AS100200 for Jordan Avery (info@esteki.ca): 1x Aster Base Shell "
              "(Storm Blue, size M), $224, placed 2025-07-19, shipped to Toronto via FedEx, "
              "tracking 771092284417. Status: delivered.")},
    {"id": "OD100201", "doc_type": "order", "email": "info@esteki.ca",
     "text": ("Order AS100201 for Jordan Avery (info@esteki.ca): 1x Aster Cloud Short "
              "(Black, size M), $74, placed 2025-08-08, shipped to Toronto via Canada Post. "
              "Status: delivered.")},
]


def _seed(docs):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    dense = embedder.embed([d["text"] for d in docs])
    store.upsert([
        {"id": d["id"], "text": d["text"], "payload": d, "dense": dv,
         "sparse": {"indices": sv.indices, "values": sv.values}}
        for d, dv, sv in zip(docs, dense, [encoder.encode(d["text"]) for d in docs])
    ])
    return embedder, store


def test_profile_captures_taste_signal_from_own_orders():
    embedder, store = _seed(_ORDERS)
    note = _personal_profile_note(("Jordan Avery", "info@esteki.ca"), embedder, store)
    assert note is not None
    low = note.lower()
    # useful preference signal survives
    assert "jordan" in low
    assert "aster base shell" in low or "aster cloud short" in low
    assert "storm blue" in low or "black" in low
    assert "size m" in low
    assert "aster circle" in low


def test_profile_never_leaks_order_level_pii():
    embedder, store = _seed(_ORDERS)
    note = _personal_profile_note(("Jordan Avery", "info@esteki.ca"), embedder, store)
    assert note is not None
    # order numbers, tracking, and shipping locations are PII creep the profile must not carry
    for secret in ("as100200", "as100201", "771092284417", "toronto", "fedex"):
        assert secret not in note.lower(), "profile leaked '{}'".format(secret)


def test_no_profile_without_login():
    embedder, store = _seed(_ORDERS)
    assert _personal_profile_note(None, embedder, store) is None
    assert _personal_profile_note(("", ""), embedder, store) is None


def test_no_profile_for_account_without_history():
    embedder, store = _seed(_ORDERS)
    # a different, order-less account gets no profile rather than someone else's
    assert _personal_profile_note(("Pat Lee", "stranger@example.com"), embedder, store) is None
