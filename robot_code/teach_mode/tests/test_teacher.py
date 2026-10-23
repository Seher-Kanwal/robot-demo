"""End-to-end walk through the 7 steps on synthetic frames (no camera, detector or GPU)."""
import threading
import time

from conftest import make_frame
from multiview_memory import MultiViewMemory
from teacher import STEP_ASK, STEP_MEMORY, STEP_MOVE, STEP_RECOGNIZE, STEP_VIEWS, Teacher
from unknown_tracker import UnknownTracker

OOD, COUCH, PERSON = 20, 18, 0
VOC = {0: "person", 18: "couch", 20: ""}
TABLE = (100, 100, 300, 180)
IN_HAND = (500, 250, 700, 400)
SOFA = (900, 500, 1100, 580)
COUCH_BOX = (700, 380, 1250, 700)
BEAR_BOX = (350, 120, 500, 300)  # right of TABLE, so on a stopped image the TABLE object is #1


def make_teacher(tmp_path, embedder, **kw):
    memory = MultiViewMemory(tmp_path / "mem", embedder)
    return Teacher(embedder, memory, UnknownTracker(), VOC, OOD, person_class_id=PERSON,
                   negatives_dir=tmp_path / "neg", **kw)


def run(t, objects, now):
    """objects: list of (oid, view, box, label[, detector score -- default 0.97])"""
    frame = make_frame([(o[0], o[1], o[2]) for o in objects], seed=int(now * 10))
    scores = [o[4] if len(o) > 4 else 0.97 for o in objects]
    return t.process(frame, [o[2] for o in objects], [o[3] for o in objects], scores, now=now)


def shoot(t, objects, now):
    out = {}
    th = threading.Thread(target=lambda: out.update(view=t.request_shot(timeout=5)))
    th.start()
    for _ in range(500):
        with t.lock:
            if t._shots:
                break
        time.sleep(0.005)
    run(t, objects, now)
    th.join()
    return out["view"]


def events(t, kind):
    return [e for e in t.snapshot(0)["events"] if e["type"] == kind]


def test_full_teach_and_recognize_flow(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)

    # 1-3: an unknown remote on the table (plus a known couch) -> asked about exactly once
    for k in range(6):
        run(t, [(1, 0, TABLE, OOD), (9, 0, COUCH_BOX, COUCH)], now=k * 0.2)
    snap = t.snapshot(0)
    assert len(events(t, "ask")) == 1
    assert snap["current_step"] == STEP_ASK and snap["asking"]["crop"].startswith("data:image/jpeg")
    assert snap["known_labels"] == ["couch"]

    # 3 -> 5: answering stores view 0 and opens the capture window
    res = t.answer(snap["asking"]["item_id"], "Um, it's a TV remote.")
    assert res["name"] == "TV remote"
    assert t.memory.get(res["object_id"])["n_views"] == 1
    assert t.current_step(now=1.3) == STEP_VIEWS

    # 4: user holds it up and presses Capture twice; no new question about the object in hand
    for k, view in enumerate([2, 4]):
        run(t, [(1, view, IN_HAND, OOD)], now=1.4 + k * 0.4)
        v = shoot(t, [(1, view, IN_HAND, OOD)], now=1.6 + k * 0.4)
        assert v["origin"] == "capture"
    cap = t.snapshot(0)["capture"]
    assert cap["target_found"] and len(cap["views"]) == 2 and len(cap["seed_views"]) == 1
    assert len(events(t, "ask")) == 1

    # 5: Done commits the views
    assert t.finish_capture(now=2.5)["n_views"] == 3
    assert t.current_step(now=2.6) == STEP_MEMORY
    assert t.current_step(now=9.0) == STEP_MOVE
    assert events(t, "memorized")[0]["n_views"] == 3

    # 6-7: put it down on the sofa (a viewpoint never captured) -> recognized, announced once
    for k in range(15):
        run(t, [(1, 5, SOFA, OOD)], now=9.0 + k * 0.2)
    rec = events(t, "recognized")
    assert len(rec) == 1 and "TV remote" in rec[0]["text"]
    assert t.snapshot(0)["recognized"][0]["name"] == "TV remote"
    assert t._step_hold[0] == STEP_RECOGNIZE and t._awaiting_move is None

    # a different unknown object is asked about, not mistaken for the remote
    for k in range(6):
        run(t, [(1, 5, SOFA, OOD), (2, 0, TABLE, OOD)], now=12.0 + k * 0.2)
    assert len(events(t, "ask")) == 2 and len(events(t, "recognized")) == 1


def test_reannounces_after_object_was_out_of_sight(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(1, 0, TABLE, OOD)], now=k * 0.2)
    t.answer(t.snapshot(0)["asking"]["item_id"], "remote")
    t.finish_capture(now=1.0)
    for k in range(10):
        run(t, [(1, 1, SOFA, OOD)], now=6.0 + k * 0.2)
    assert len(events(t, "recognized")) == 1
    run(t, [], now=8.5)  # out of sight for 5 s ...
    for k in range(10):
        run(t, [(1, 1, SOFA, OOD)], now=13.5 + k * 0.2)  # ... back at the same spot
    assert len(events(t, "recognized")) == 2


def test_not_announced_while_being_carried(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(1, 0, TABLE, OOD)], now=k * 0.2)
    t.answer(t.snapshot(0)["asking"]["item_id"], "remote")
    t.finish_capture(now=1.0)
    for k in range(12):  # keeps moving across the frame
        box = (100 + 80 * k, 300, 300 + 80 * k, 380)
        run(t, [(1, 1, box, OOD)], now=6.0 + k * 0.2)
    assert events(t, "recognized") == []


def test_skip_never_asks_again_and_saves_negative(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(3, 0, TABLE, OOD)], now=k * 0.2)
    assert t.skip(t.snapshot(0)["asking"]["item_id"])
    for k in range(10):
        run(t, [(3, 0, TABLE, OOD)], now=1.0 + k * 0.2)
    assert len(events(t, "ask")) == 1
    assert len(list((tmp_path / "neg").glob("*.jpg"))) == 1


def test_cancel_keeps_the_named_view(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(1, 0, TABLE, OOD)], now=k * 0.2)
    oid = t.answer(t.snapshot(0)["asking"]["item_id"], "remote")["object_id"]
    shoot(t, [(1, 2, IN_HAND, OOD)], now=1.0)
    assert t.cancel_capture()
    assert t.memory.get(oid)["n_views"] == 1 and t.capture is None


def test_capture_window_draws_only_the_object_on_the_center_mark(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(1, 0, TABLE, OOD)], now=k * 0.2)
    t.answer(t.snapshot(0)["asking"]["item_id"], "remote")
    # the remote (another side) on the mark, an unknown thing on the table, a known couch
    objs = [(1, 2, IN_HAND, OOD), (2, 0, TABLE, OOD), (9, 0, COUCH_BOX, COUCH)]
    robot, _ = run(t, objs, now=1.0)
    frame = make_frame([o[:3] for o in objs], seed=10)
    cap = t.snapshot(0)["capture"]
    assert cap["target_found"] and cap["target_level"] == "match" and 0.62 <= cap["target_score"] < 0.95
    for x1, y1, x2, y2 in (TABLE, COUCH_BOX):  # no boxes on the other objects (inset: COUCH_BOX touches IN_HAND)
        assert (robot[y1 + 8:y2, x1 + 8:x2] == frame[y1 + 8:y2, x1 + 8:x2]).all()
    x1, y1, x2, y2 = IN_HAND
    assert not (robot[y1:y2, x1:x2] == frame[y1:y2, x1:x2]).all()
    v = shoot(t, objs, now=1.2)
    assert v["origin"] == "capture" and v["sim"] == cap["target_score"]


def test_known_class_boxes_only_matched_when_enabled(tmp_path, embedder):
    for enabled, expect in ((False, 0), (True, 1)):
        t = make_teacher(tmp_path / str(enabled), embedder, match_known_boxes=enabled)
        for k in range(4):
            run(t, [(1, 0, TABLE, OOD)], now=k * 0.2)
        t.answer(t.snapshot(0)["asking"]["item_id"], "remote")
        t.finish_capture(now=1.0)
        for k in range(10):  # moved, and now the detector calls it a couch
            run(t, [(1, 1, SOFA, COUCH)], now=6.0 + k * 0.2)
        assert len(events(t, "recognized")) == expect


def test_answer_rejects_empty_name_and_stale_question(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(1, 0, TABLE, OOD)], now=k * 0.2)
    item_id = t.snapshot(0)["asking"]["item_id"]
    try:
        t.answer(item_id, "  it's a  ")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        t.answer(item_id + 999, "remote")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


# ---------------------------------------------------------------- explore mode
def _teach_remote(t):
    for k in range(4):
        run(t, [(1, 0, TABLE, OOD)], now=k * 0.2)
    t.answer(t.snapshot(0)["asking"]["item_id"], "TV remote")
    t.finish_capture(now=1.0)


def test_explore_shows_only_known_and_remembered_objects(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    _teach_remote(t)
    t.set_mode("explore")
    assert events(t, "mode")[-1]["mode"] == "explore"
    asks_before = len(events(t, "ask"))
    scene = [(9, 0, COUCH_BOX, COUCH), (1, 3, SOFA, OOD, 0.97), (2, 0, TABLE, OOD, 0.99)]  # couch painted first
    for k in range(10):  # a couch, the remembered remote on it (confident), an unknown bear (confident)
        robot, _ = run(t, scene, now=10.0 + k * 0.2)
    snap = t.snapshot(0)
    assert [r["name"] for r in snap["recognized"]] == ["TV remote"]
    assert snap["known_labels"] == ["couch"]
    assert snap["asking"] is None and len(events(t, "ask")) == asks_before  # the bear is never asked about
    assert len(events(t, "recognized")) == 1
    assert snap["current_step"] == 7
    # the unknown bear is not drawn at all: its box area is untouched in the robot view
    frame = make_frame([o[:3] for o in scene], seed=int((10.0 + 9 * 0.2) * 10))  # same frame as the last run()
    x1, y1, x2, y2 = TABLE
    assert (robot[y1:y2, x1:x2] == frame[y1:y2, x1:x2]).all()


def test_explore_drops_ood_boxes_below_detector_threshold(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder, explore_ood_conf=0.95)
    _teach_remote(t)
    t.set_mode("explore")
    for k in range(10):
        run(t, [(1, 3, SOFA, OOD, 0.92)], now=10.0 + k * 0.2)  # remembered, but detector not sure enough
    assert t.snapshot(0)["recognized"] == [] and len(events(t, "recognized")) == 0
    for k in range(10):
        run(t, [(1, 3, SOFA, OOD, 0.96)], now=13.0 + k * 0.2)
    assert len(events(t, "recognized")) == 1


def test_explore_greets_remembered_objects_again(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    _teach_remote(t)
    for k in range(10):
        run(t, [(1, 3, SOFA, OOD)], now=6.0 + k * 0.2)
    assert len(events(t, "recognized")) == 1
    t.set_mode("explore")  # starting a new exploration: everything gets announced once more
    for k in range(10):
        run(t, [(1, 3, SOFA, OOD)], now=8.2 + k * 0.2)
    assert len(events(t, "recognized")) == 2


def test_back_to_teach_asks_again_and_mode_rules(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(2, 0, TABLE, OOD)], now=k * 0.2)
    assert t.snapshot(0)["asking"] is not None
    t.set_mode("explore")
    assert t.snapshot(0)["asking"] is None  # pending question dropped
    t.set_mode("teach")
    for k in range(4):
        run(t, [(2, 0, TABLE, OOD)], now=2.0 + k * 0.2)
    assert t.snapshot(0)["asking"] is not None and len(events(t, "ask")) == 2

    t.answer(t.snapshot(0)["asking"]["item_id"], "teddy bear")  # capture window now open
    try:
        t.set_mode("explore")
        raise AssertionError("expected ValueError while capturing")
    except ValueError:
        pass
    t.finish_capture(now=3.0)
    t.set_mode("explore")
    try:
        t.start_capture(t.memory.list_objects()[0]["id"])
        raise AssertionError("expected ValueError: no capture in explore mode")
    except ValueError:
        pass


# ---------------------------------------------------------------- stop on the image (teach)
def test_stops_on_image_and_asks_about_each_object_in_turn(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    scene = [(2, 0, BEAR_BOX, OOD), (1, 0, TABLE, OOD)]  # two unknown objects in one image
    for k in range(4):
        robot, _ = run(t, scene, now=k * 0.2)
    snap = t.snapshot(0)
    assert snap["frozen"] == {"total": 2, "pending": 2, "named": 0, "skipped": 0}
    assert snap["asking"]["index"] == 1 and snap["asking"]["total"] == 2
    assert len(events(t, "ask")) == 1 and "2 unfamiliar objects" in events(t, "ask")[0]["text"]
    assert snap["current_step"] == STEP_ASK

    # the robot view stays on the stopped image while the camera keeps moving
    frozen_view = robot
    moved, _ = run(t, [(5, 0, (400, 400, 600, 500), OOD)], now=1.0)
    assert (moved == frozen_view).all()

    # object 1 (leftmost = TABLE remote): named -> still on the stopped image, asking about object 2
    first = t.answer(snap["asking"]["item_id"], "TV remote", now=1.1)
    snap = t.snapshot(0)
    assert snap["capture"] is None and snap["asking"]["index"] == 2 and len(events(t, "ask")) == 2
    still, _ = run(t, [(5, 0, (400, 400, 600, 500), OOD)], now=1.2)
    assert not (still == frozen_view).all()  # the first box now carries its name ...
    assert t.snapshot(0)["frozen"] == {"total": 2, "pending": 1, "named": 1, "skipped": 0}  # ... but still paused

    # object 2 named -> every object on the image is named -> capture windows, one per object
    t.answer(snap["asking"]["item_id"], "teddy bear", now=1.3)
    cap = t.snapshot(0)["capture"]
    assert cap is not None and cap["object_id"] == first["object_id"]
    live, _ = run(t, [(1, 2, IN_HAND, OOD)], now=1.4)
    assert not (live == frozen_view).all()  # the capture window needs the live picture
    shoot(t, [(1, 2, IN_HAND, OOD)], now=1.5)
    t.finish_capture(now=1.6)
    assert "Move it somewhere else" not in events(t, "memorized")[-1]["text"]  # one more to go
    assert t.snapshot(0)["capture"]["name"] == "teddy bear" and t.snapshot(0)["asking"] is None
    t.finish_capture(now=2.0)
    assert "Move it somewhere else" in events(t, "memorized")[-1]["text"]

    # all named -> live again; both are in memory
    snap = t.snapshot(0)
    assert snap["frozen"] is None and snap["asking"] is None
    assert sorted(o["name"] for o in t.memory.list_objects()) == ["TV remote", "teddy bear"]
    assert t.memory.get(first["object_id"])["n_views"] == 2
    live_again, _ = run(t, [(5, 0, (400, 400, 600, 500), OOD)], now=2.2)
    assert not (live_again == frozen_view).all()


def test_skipped_objects_get_no_capture_window(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(2, 0, BEAR_BOX, OOD), (1, 0, TABLE, OOD)], now=k * 0.2)
    first = t.snapshot(0)["asking"]
    t.answer(first["item_id"], "TV remote", now=1.0)
    assert t.snapshot(0)["capture"] is None
    t.skip(t.snapshot(0)["asking"]["item_id"], now=1.1)
    assert t.snapshot(0)["capture"]["name"] == "TV remote"
    assert t.cancel_capture(now=1.2)  # cancelling doesn't reopen it
    snap = t.snapshot(0)
    assert snap["capture"] is None and snap["frozen"] is None


def test_skip_rest_goes_live_and_never_asks_about_them_again(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    scene = [(2, 0, BEAR_BOX, OOD), (1, 0, TABLE, OOD)]
    for k in range(4):
        run(t, scene, now=k * 0.2)
    assert t.skip_rest(now=1.0) == 2
    assert t.snapshot(0)["frozen"] is None and events(t, "resume")
    for k in range(10):
        run(t, scene, now=3.0 + k * 0.2)
    assert len(events(t, "ask")) == 1 and t.snapshot(0)["frozen"] is None


def test_skip_one_moves_to_the_next(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    for k in range(4):
        run(t, [(2, 0, BEAR_BOX, OOD), (1, 0, TABLE, OOD)], now=k * 0.2)
    first = t.snapshot(0)["asking"]
    assert t.skip(first["item_id"], now=1.0)
    nxt = t.snapshot(0)["asking"]
    assert nxt["index"] == 2 and nxt["item_id"] != first["item_id"]
    assert not t.skip(first["item_id"])  # already skipped


def test_no_recognition_announcements_while_stopped(tmp_path, embedder):
    t = make_teacher(tmp_path, embedder)
    _teach_remote(t)
    for k in range(4):  # a new unknown appears -> stop on that image
        run(t, [(2, 0, BEAR_BOX, OOD)], now=6.0 + k * 0.2)
    assert t.snapshot(0)["frozen"] is not None
    for k in range(10):  # meanwhile the remembered remote sits in view, live
        run(t, [(1, 3, SOFA, OOD)], now=7.0 + k * 0.2)
    assert events(t, "recognized") == []
    t.skip_rest(now=9.0)
    for k in range(10):  # moved since it was taught -> greeted once we're live again
        run(t, [(1, 3, SOFA, OOD)], now=9.2 + k * 0.2)
    assert len(events(t, "recognized")) == 1
