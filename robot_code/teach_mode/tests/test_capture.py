import numpy as np
import pytest

from capture_session import CaptureSession
from conftest import DIM, make_crop, make_frame

SHAPE = (720, 1280, 3)        # center mark at (640, 360), slack 0.04 * 720 = 28.8 px
CENTER = (600, 300, 760, 420)  # covers the mark
CORNER = (50, 50, 200, 200)    # nowhere near it


def _session(embedder, oid=1, **kw):
    return CaptureSession("obj_x", "cup", embedder.embed([make_crop(oid, 0)]), **kw)


def _cands(embedder, dets):
    return [d[2] for d in dets], embedder.embed([make_crop(oid, view) for oid, view, _ in dets])


def _weak():
    """An embedding ~0.45 similar to object 1: enough to keep following, not to acquire."""
    e = np.zeros((1, DIM), np.float32)
    e[0, 1], e[0, 10] = 0.5, 0.866
    return e


def test_clutter_on_the_mark_is_never_picked(embedder):
    cap = _session(embedder)
    # first screenshot: a phone sits on the mark, the cup is off to the side
    boxes, embs = _cands(embedder, [(1, 3, CORNER), (2, 0, CENTER)])
    assert cap.select_target(boxes, embs, SHAPE, now=0.0) == 0
    assert cap.target_box == CORNER and cap.public()["target_level"] == "match"


def test_no_target_when_nothing_looks_like_it(embedder):
    cap = _session(embedder)
    couch = (100, 100, 1200, 700)
    boxes, embs = _cands(embedder, [(2, 0, CENTER), (9, 0, couch)])
    assert cap.select_target(boxes, embs, SHAPE, now=0.0) is None
    assert not cap.target_in_frame and cap.target_score is None and cap.public()["target_level"] is None


def test_weak_lookalike_is_not_acquired(embedder):
    cap = _session(embedder)
    # second screenshot: the monitor scored ~0.4 against the hand and got picked
    assert cap.select_target([CENTER], _weak(), SHAPE, now=0.0) is None


def test_tracked_object_is_followed_at_the_lower_threshold(embedder):
    cap = _session(embedder)
    assert cap.select_target(*_cands(embedder, [(1, 3, CENTER)]), SHAPE, now=0.0) == 0
    turned = (610, 305, 770, 425)  # same place, now showing a side that looks less alike
    assert cap.select_target([turned], _weak(), SHAPE, now=0.1) == 0 and cap.target_box == turned
    # out of sight for longer than follow_s: has to look like it again to be picked up
    assert cap.select_target([turned], _weak(), SHAPE, now=2.0) is None


def test_big_background_box_behind_the_object_does_not_take_over(embedder):
    cap = _session(embedder)
    cap.select_target(*_cands(embedder, [(1, 3, CENTER)]), SHAPE, now=0.0)
    monitor = (200, 0, 900, 450)  # overlaps the object's last box, but IoU ~0.06
    assert cap.select_target([monitor], _weak(), SHAPE, now=0.1) is None


def test_track_starts_where_the_object_was_named(embedder):
    cap = _session(embedder, seed_box=CORNER, now=5.0)
    assert cap.select_target([CORNER], _weak(), SHAPE, now=5.2) == 0
    fresh = _session(embedder)
    assert fresh.select_target([CORNER], _weak(), SHAPE, now=5.2) is None


def test_followed_object_beats_a_lookalike_on_the_mark(embedder):
    cap = _session(embedder)
    cap.select_target(*_cands(embedder, [(1, 3, CORNER)]), SHAPE, now=0.0)
    boxes, embs = _cands(embedder, [(1, 0, CENTER), (1, 3, CORNER)])
    assert cap.select_target(boxes, embs, SHAPE, now=0.1) == 1


def test_mark_picks_between_lookalikes(embedder):
    cap = _session(embedder)
    boxes, embs = _cands(embedder, [(1, 0, CORNER), (1, 3, CENTER)])  # the corner one even looks more alike
    assert cap.select_target(boxes, embs, SHAPE, now=0.0) == 1


def test_slack_around_the_mark(embedder):
    just_off = (660, 300, 800, 420)  # left edge 20 px right of the mark: within the slack, counts as on it
    boxes, embs = _cands(embedder, [(1, 3, CENTER), (1, 0, just_off)])
    assert _session(embedder).select_target(boxes, embs, SHAPE, now=0.0) == 1  # both on it -> more similar
    too_far = (680, 300, 800, 420)   # 40 px: not on it
    boxes, embs = _cands(embedder, [(1, 3, CENTER), (1, 0, too_far)])
    assert _session(embedder).select_target(boxes, embs, SHAPE, now=0.0) == 0


def test_shoot_uses_target_box_records_similarity_and_flags(embedder):
    cap = _session(embedder)
    cap.select_target(*_cands(embedder, [(1, 0, CENTER)]), SHAPE, now=0.0)
    view = cap.shoot(make_frame([(1, 0, CENTER)]), embedder.embed)
    assert view["origin"] == "capture" and "duplicate" in view["badges"]  # same viewpoint as the seed
    assert view["sim"] > 0.99

    cap.select_target(*_cands(embedder, [(1, 4, CENTER)]), SHAPE, now=0.1)
    view2 = cap.shoot(make_frame([(1, 4, CENTER)]), embedder.embed)
    assert view2["badges"] == [] and 0.62 <= view2["sim"] < 0.95
    assert len(cap.views) == 2 and cap.guide_text().startswith("View 3")


def test_press_with_nothing_like_it_in_view_saves_nothing(embedder):
    cap = _session(embedder)
    cap.select_target(*_cands(embedder, [(2, 0, CENTER)]), SHAPE, now=0.0)
    with pytest.raises(ValueError, match="no cup in view"):
        cap.shoot(make_frame([(2, 0, CENTER)]), embedder.embed)
    assert cap.views == []


def test_similarity_is_against_the_views_collected_so_far(embedder):
    cap = _session(embedder)
    cap.select_target(*_cands(embedder, [(1, 4, CENTER)]), SHAPE, now=0.0)
    before = cap.target_score
    cap.shoot(make_frame([(1, 4, CENTER)]), embedder.embed)
    cap.select_target(*_cands(embedder, [(1, 4, CENTER)]), SHAPE, now=0.1)
    assert before < 0.95 and cap.target_score > 0.99  # that viewpoint is now in the reference


def test_levels(embedder):
    cap = _session(embedder)
    assert [cap.level(s) for s in (None, 0.97, 0.7, 0.5, 0.2)] == [None, "dup", "match", "new", "different"]


def test_delete_view(embedder):
    cap = _session(embedder)
    frame = make_frame([(1, 3, CENTER)])
    cap.select_target(*_cands(embedder, [(1, 3, CENTER)]), SHAPE, now=0.0)
    cap.shoot(frame, embedder.embed)
    cap.shoot(frame, embedder.embed)
    assert cap.delete(1) and [v["idx"] for v in cap.views] == [2]
    assert not cap.delete(99)
