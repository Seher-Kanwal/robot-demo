from conftest import make_crop
from unknown_tracker import UnknownTracker

REMOTE_BOX = (100, 100, 300, 180)
BEAR_BOX = (600, 300, 760, 500)


def _step(tracker, embedder, dets, now):
    """dets: list of (oid, view, box)"""
    crops = [make_crop(oid, view, (box[3] - box[1], box[2] - box[0]), seed=int(now * 10)) for oid, view, box in dets]
    return tracker.update([d[2] for d in dets], embedder.embed(crops) if dets else [], crops, now)


def _jitter(box, k):
    return tuple(v + (k % 3) * 4 for v in box)


def test_one_question_per_object_despite_jitter(embedder):
    tr = UnknownTracker()
    asked = []
    for k in range(10):
        _step(tr, embedder, [(1, 0, _jitter(REMOTE_BOX, k))], now=k * 0.2)
        q = tr.pop_new_question()
        if q is not None:
            asked.append(q)
    assert len(asked) == 1
    assert len(tr.items) == 1 and tr.asking()["id"] == asked[0]


def test_needs_several_sightings_before_asking(embedder):
    tr = UnknownTracker(min_hits=3)
    _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=0.0)
    _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=0.2)
    assert tr.asking() is None
    _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=0.4)
    assert tr.asking() is not None


def test_one_at_a_time_fifo(embedder):
    tr = UnknownTracker()
    for k in range(4):
        _step(tr, embedder, [(1, 0, REMOTE_BOX), (2, 0, BEAR_BOX)], now=k * 0.2)
    first = tr.asking()
    assert first is not None and tr.queue_len() == 1
    assert tr.take(first["id"]) is first
    _step(tr, embedder, [(2, 0, BEAR_BOX)], now=1.0)
    second = tr.asking()
    assert second is not None and second["id"] != first["id"]


def test_identity_follows_moved_object_by_appearance(embedder):
    tr = UnknownTracker()
    for k in range(3):
        _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=k * 0.2)
    (res,) = _step(tr, embedder, [(1, 0, (900, 500, 1100, 580))], now=0.8)  # no overlap, same look
    assert len(tr.items) == 1 and res[1] == "asking"


def test_skip_ignores_object_for_the_session(embedder):
    tr = UnknownTracker()
    for k in range(4):
        _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=k * 0.2)
    assert tr.skip(tr.asking()["id"]) is not None
    for k in range(5):
        (res,) = _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=1.0 + k * 0.2)
        assert res == (None, "ignored")
    assert tr.asking() is None and tr.pop_new_question() is None


def test_paused_tracker_does_not_ask(embedder):
    tr = UnknownTracker()
    tr.paused = True
    for k in range(5):
        _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=k * 0.2)
    assert tr.asking() is None and tr.queue_len() == 1
    tr.paused = False
    _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=1.2)
    assert tr.asking() is not None


def test_unseen_items_expire_but_not_the_current_question(embedder):
    tr = UnknownTracker(expire_s=10.0)
    for k in range(4):
        _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=k * 0.2)
    _step(tr, embedder, [(2, 0, BEAR_BOX)], now=1.0)  # seen once, never confirmed
    _step(tr, embedder, [], now=20.0)
    assert [i["state"] for i in tr.items.values()] == ["asking"]


def test_does_not_start_asking_about_something_no_longer_visible(embedder):
    tr = UnknownTracker(ask_visible_s=1.5)
    tr.paused = True
    for k in range(4):
        _step(tr, embedder, [(1, 0, REMOTE_BOX)], now=k * 0.2)
    tr.paused = False
    _step(tr, embedder, [], now=5.0)
    assert tr.asking() is None


def test_purge_similar_drops_duplicate_identities(embedder):
    tr = UnknownTracker()
    _step(tr, embedder, [(1, 0, REMOTE_BOX), (2, 0, BEAR_BOX)], now=0.0)
    tr.purge_similar(embedder.embed([make_crop(1, 0)])[0])
    assert len(tr.items) == 1
