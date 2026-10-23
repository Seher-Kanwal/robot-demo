"""
The /assistant page's brain: "Robot, where are my headphones?" -> "I found your headphones on
the sofa, next to the teddy bear, about 1.6 m ahead, slightly to my left."

robot_web_app.py calls Assistant.observe() once per processed frame (live --realsense loop
or robot_client.py pushes) with the detector + VLM results it already computed; nothing
here runs the detector again. A question is answered by, in order:

  1. live      -- the object is in the latest frame: where it is right now (+ depth position)
  2. memory    -- it was seen earlier: where it was last seen, and how long ago
  3. grounding -- neither, but Qwen3-VL finds it when asked about the full current frame
                  (small objects the VOC-trained detector never proposes)
  4. not found -- keep looking, and say so as soon as it shows up

Every asked-for object stays on a short watch list afterwards, so the page also gets
proactive messages: "There they are! I found your headphones on the table." when it
first appears, and "...they are now on the table" when it moves to a different surface.

Only find/locate requests are understood -- the robot itself is driven by hand.
"""
import base64
import re
import threading
import time
from collections import OrderedDict, deque

import cv2
from PIL import Image

import scene_memory as sm

LIVE_MAX_AGE_S = 3.0        # the latest frame counts as "what I see right now" for this long
SEARCH_TTL_S = 600.0        # an asked-for object stays on the watch list this long
MAX_SEARCHES = 3
EVENT_COOLDOWN_S = 5.0      # per searched object, so a flickering detection can't chatter
MATCH_THRESHOLD = 0.8
MAX_EVENTS = 100
_NUM = r"\s*(-?\d+(?:\.\d+)?)\s*"
BBOX_RE = re.compile(r"\[" + ",".join([_NUM] * 4) + r"\]")

# --- understanding the request -------------------------------------------------

WAKE_RE = re.compile(r"^\s*(?:(?:hey|hi|hello|ok|okay)\s+)?(?:robot|robo|robert)\b[\s,.:;!-]*", re.I)
CANCEL_RE = re.compile(r"\b(?:stop (?:looking|searching)|never ?mind|cancel|forget (?:it|about))\b(?:\s+(?:for|about))?\s*(?P<t>.*)$", re.I)
LOCATE_PATTERNS = [re.compile(p, re.I) for p in (
    # "where are my X", "where's X", "where did I leave X", "where can I find X", "where did my X go"
    r"\bwhere(?:'s|’s)?\s+(?:(?:is|are|was|were|did|do|does|have|has|had|can|could|would|will)\s+)?"
    r"(?:(?:i|you|we)\s+(?:find|put|left|leave|placed|place|keep|kept|see|saw|seen|last\s+(?:see|saw))\s+)?(?P<t>.+)",
    r"\b(?:i\s+(?:can't|can’t|cannot|can\s+not)\s+find|i\s+lost|i'm\s+looking\s+for|i\s+am\s+looking\s+for"
    r"|help\s+me\s+find)\s+(?P<t>.+)",
    r"\b(?:find|locate|look\s+for|search\s+for|show\s+me|point\s+(?:me\s+)?to|spot)\s+(?P<t>.+)",
    r"\b(?:do|can|did)\s+you\s+see\s+(?P<t>.+)",
    r"\bhave\s+you\s+seen\s+(?P<t>.+)",
    r"\b(?:is|are)\s+there\s+(?P<t>.+)",
)]
DETERMINER_RE = re.compile(r"^(?:my|our|the|a|an|any|some|your|his|her|their|this|that|these|those)\s+", re.I)
TRAILING_RE = re.compile(
    r"(?:\s+(?:please|for\s+me|right\s+now|now|again|at|located|kept|gone|go|went|is|are|anywhere|around\s+here))+$",
    re.I)
LOCATION_CLAUSE_RE = re.compile(r"\s+(?:in|on|near|under|at|around|inside|from|somewhere)\s+(?:the|my|this|that|here)\b.*$", re.I)
PLURAL_EXCEPTIONS = ("ss", "us", "is")
QUESTION_WORDS = ("what", "who", "how", "why", "when", "which")
NOT_OBJECTS = {"hello", "hi", "hey", "thanks", "thank you", "yes", "no", "okay", "ok", "stop", "help", "test",
               "testing", "good", "nice", "great", "cool", "sorry"}

SYNONYM_GROUPS = [
    {"sofa", "couch", "settee"},
    {"tv", "television", "monitor", "screen", "tv monitor"},
    {"table", "dining table", "desk", "coffee table", "side table"},
    {"remote", "remote control", "tv remote"},
    {"headphones", "headphone", "headset", "earphones", "earbuds"},
    {"phone", "cellphone", "cell phone", "mobile phone", "smartphone", "mobile"},
    {"cup", "mug"},
    {"bottle", "water bottle"},
    {"plant", "potted plant", "houseplant"},
    {"laptop", "notebook", "notebook computer"},
    {"glasses", "spectacles", "eyeglasses"},
    {"teddy bear", "teddy", "stuffed animal", "plush toy", "plush", "stuffed toy"},
    {"bag", "backpack", "handbag", "rucksack"},
    {"keys", "key", "keychain", "car keys"},
    {"wallet", "purse"},
    {"chair", "armchair"},
]


def _variants(text):
    out = {text}
    if text.endswith("es"):
        out.add(text[:-2])
    if text.endswith("s"):
        out.add(text[:-1])
    else:
        out.add(text + "s")
    return out


_GROUP_OF = {}
for _i, _group in enumerate(SYNONYM_GROUPS):
    for _name in _group:
        for _v in _variants(_name):
            _GROUP_OF.setdefault(_v, set()).add(_i)


def is_plural(target):
    last = (target.split() or [""])[-1]
    return last.endswith("s") and not last.endswith(PLURAL_EXCEPTIONS)


def match_score(target, label):
    """How well a detected/named label matches what the user asked for: 1.0 same word
    (modulo plural), 0.9 same synonym group, 0.8 one's words are a subset of the other's
    *with the same head noun* ("remote" / "tv remote" match; "tv" / "tv remote" don't)."""
    t, l = sm.norm_label(target), sm.norm_label(label)
    if not t or not l:
        return 0.0
    if l in _variants(t) or t in _variants(l):
        return 1.0
    if _GROUP_OF.get(t, set()) & _GROUP_OF.get(l, set()):
        return 0.9
    a, b = t.split(), l.split()
    for small, large in ((a, b), (b, a)):
        large_tokens = set().union(*(_variants(w) for w in large))
        if set(small) <= large_tokens and small[-1] in _variants(large[-1]):
            return 0.8
    return 0.0


class Query:
    def __init__(self, intent, target=None, possessive="the", text=""):
        self.intent = intent          # "locate" | "cancel" | "unsupported"
        self.target = target
        self.possessive = possessive  # "your" when the user said "my"/"our", else "the"
        self.text = text

    @property
    def plural(self):
        return bool(self.target) and is_plural(self.target)

    def obj(self):
        return f"{self.possessive} {self.target}"

    def they(self):
        return "they" if self.plural else "it"

    def them(self):
        return "them" if self.plural else "it"

    def be(self):
        return "are" if self.plural else "is"


def _clean_target(raw):
    """'my headphones please?' -> ('headphones', 'your')."""
    t = re.sub(r"[?!.,;:\"“”]+", " ", raw or "").strip()
    t = LOCATION_CLAUSE_RE.sub("", t)
    possessive = "the"
    while True:
        m = DETERMINER_RE.match(t)
        if not m:
            break
        if m.group(0).strip().lower() in ("my", "our"):
            possessive = "your"
        t = t[m.end():]
    t = TRAILING_RE.sub("", t).strip()
    return " ".join(t.lower().split()), possessive


def parse_query(text):
    raw = (text or "").strip()
    body = WAKE_RE.sub("", raw).strip()
    if not body:
        return Query("unsupported", text=raw)
    m = CANCEL_RE.search(body)
    if m:
        target, possessive = _clean_target(m.group("t"))
        return Query("cancel", target or None, possessive, raw)
    for pattern in LOCATE_PATTERNS:
        m = pattern.search(body)
        if m:
            target, possessive = _clean_target(m.group("t"))
            if target:
                return Query("locate", target, possessive, raw)
    # A bare object name ("headphones?", "my keys") is a request to find it too.
    target, possessive = _clean_target(body)
    if target and len(target.split()) <= 3 and target not in NOT_OBJECTS and not target.startswith(QUESTION_WORDS):
        return Query("locate", target, possessive, raw)
    return Query("unsupported", text=raw)


def _ago(seconds):
    if seconds < 5:
        return "just now"
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return "a minute ago" if minutes == 1 else f"{minutes} minutes ago"
    hours = int(round(minutes / 60))
    return "an hour ago" if hours == 1 else f"{hours} hours ago"


def _where(sighting, with_position=True):
    text = sighting["relation"]["text"]
    if with_position and sighting.get("position"):
        text += ", " + sighting["position"]["text"]
    return text


# --- the assistant ---------------------------------------------------------------

class Assistant:
    def __init__(self, voc_names, ood_class_id, grounding=True):
        self.voc_names = voc_names
        self.ood_class_id = ood_class_id
        self.grounding = grounding
        self.scene = sm.SceneMemory()
        self._lock = threading.RLock()
        self._latest = None  # newest observed frame; see observe()
        self._frame_id = 0
        self._searches = OrderedDict()  # normalized target -> search state
        self._events = deque(maxlen=MAX_EVENTS)
        self._next_id = 1

    # -- per-frame input ---------------------------------------------------------
    def observe(self, im0, boxes, labels, scores, ood_labels_by_row, preview_b64, depth_info=None, source=None):
        """Called by robot_web_app.py for every processed frame. CPU-only (box geometry,
        a median over depth pixels, one small resize) -- no model calls."""
        objects = []
        for row in range(len(boxes)):
            cls = int(labels[row])
            if cls == self.ood_class_id:
                label, src = ood_labels_by_row.get(row, ("unknown", "vlm1"))
            else:
                label, src = self.voc_names[cls], "detector"
            if sm.norm_label(label) in sm.SKIP_LABELS:
                continue
            objects.append({"label": label, "source": src,
                            "box": [float(v) for v in boxes[row].tolist()], "score": float(scores[row])})

        ts = time.time()
        sightings = sm.compute_sightings(objects, im0.shape, depth_info)
        with self._lock:
            self._frame_id += 1
            frame_id = self._frame_id
            self._latest = {
                "frame_id": frame_id, "ts": ts, "im0": im0, "depth": depth_info, "objects": objects,
                "sightings": sightings, "preview": preview_b64, "source": source,
                "resolution": f"{im0.shape[1]}x{im0.shape[0]}", "im_shape": im0.shape[:2],
            }
        self.scene.record(frame_id, ts, im0, sightings)
        with self._lock:
            self._update_searches(sightings, ts)

    # -- watch list / proactive messages -------------------------------------------
    @staticmethod
    def _best_match(sightings, target):
        best, best_score = None, 0.0
        for s in sightings:
            score = match_score(target, s["label"])
            if score > best_score or (score == best_score and best is not None and s["score"] > best["score"]):
                best, best_score = s, score
        return best if best_score >= MATCH_THRESHOLD else None

    def _emit(self, kind, search, text, sighting=None):
        event = {
            "id": self._next_id, "ts": time.time(), "kind": kind, "target": search["query"].target,
            "text": text, "snapshot": self._snapshot_uri(sighting["key"]) if sighting else None,
        }
        self._next_id += 1
        self._events.append(event)
        search["last_event"] = event["ts"]
        print(f"[assistant] {kind}: {text}")

    def _update_searches(self, sightings, ts):
        for key, search in list(self._searches.items()):
            if ts - search["started"] > SEARCH_TTL_S:
                del self._searches[key]
                continue
            q = search["query"]
            match = self._best_match(sightings.values(), q.target)
            search["hits"].append(match is not None)
            if match is None:
                search["pending_support"], search["pending_count"] = None, 0
                continue
            cooled = ts - search["last_event"] >= EVENT_COOLDOWN_S
            support = match["relation"]["support"]
            if search["state"] == "searching":
                if sum(search["hits"]) >= 2 and cooled:
                    there = "There they are!" if q.plural else "There it is!"
                    self._emit("found", search, f"{there} I found {q.obj()} {_where(match)}.", match)
                    search["state"], search["support"] = "tracking", support
            elif support and support != search["support"]:
                if search["support"] is None:
                    search["support"] = support  # just a better view of where it already was
                    continue
                if support == search["pending_support"]:
                    search["pending_count"] += 1
                else:
                    search["pending_support"], search["pending_count"] = support, 1
                if search["pending_count"] >= 2 and cooled:
                    self._emit("moved", search,
                               f"I found {q.obj()} again. {q.they().capitalize()} {q.be()} now {_where(match)}.", match)
                    search["support"], search["pending_support"], search["pending_count"] = support, None, 0
            else:
                search["pending_support"], search["pending_count"] = None, 0

    def _watch(self, q, state, support=None):
        key = sm.norm_label(q.target)
        self._searches.pop(key, None)
        self._searches[key] = {
            "query": q, "state": state, "support": support, "started": time.time(),
            "hits": deque(maxlen=3), "pending_support": None, "pending_count": 0,
            "last_event": time.time() if state == "tracking" else 0.0,
        }
        while len(self._searches) > MAX_SEARCHES:
            self._searches.popitem(last=False)

    def cancel(self, target=None):
        with self._lock:
            if not target:
                n = len(self._searches)
                self._searches.clear()
                return {"cancelled": n}
            key = sm.norm_label(target)
            gone = [k for k in self._searches if match_score(key, k) >= MATCH_THRESHOLD]
            for k in gone:
                del self._searches[k]
            return {"cancelled": len(gone)}

    # -- answering --------------------------------------------------------------------
    def _snapshot_uri(self, key):
        jpg = self.scene.snapshot_jpg(key)
        return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii") if jpg else None

    def ask(self, text):
        q = parse_query(text)
        reply = {"user_text": text, "intent": q.intent, "target": q.target, "outcome": None,
                 "relation": None, "distance_m": None, "bearing_deg": None, "snapshot": None, "searching": False}

        if q.intent == "unsupported":
            reply.update(outcome="unsupported",
                         reply="I can help you find things. Try asking: where are my headphones?")
            return reply
        if q.intent == "cancel":
            n = self.cancel(q.target)["cancelled"]
            if q.target:
                msg = f"Okay, I'll stop looking for {q.obj()}." if n else f"I wasn't looking for {q.obj()}."
            else:
                msg = "Okay, I stopped looking." if n else "I wasn't looking for anything."
            reply.update(outcome="cancelled", reply=msg)
            return reply

        now = time.time()
        with self._lock:
            latest = self._latest
        fresh = latest is not None and now - latest["ts"] <= LIVE_MAX_AGE_S

        # 1. In view right now?
        match = self._best_match(latest["sightings"].values(), q.target) if fresh else None
        if match is not None:
            with self._lock:
                self._watch(q, "tracking", match["relation"]["support"])
            return self._found(reply, q, "found_live", match, f"I found {q.obj()} {_where(match)}.")

        # 2. Seen earlier?
        remembered = self._best_match(self.scene.all(), q.target)
        if remembered is not None:
            with self._lock:
                self._watch(q, "searching")
            ago = _ago(now - remembered["ts"])
            lead = f"I don't see {q.them()} right now, but " if fresh else ""
            text = (f"{lead}I last saw {q.obj()} {_where(remembered, with_position=False)}, {ago}. "
                    f"I'll tell you when I see {q.them()} again.")
            reply = self._found(reply, q, "found_memory", remembered, text[0].upper() + text[1:])
            reply["searching"] = True
            reply["age_s"] = round(now - remembered["ts"], 1)
            return reply

        # 3. Ask the VLM to look for it in the whole current frame.
        if fresh and self.grounding:
            sighting = self._ground(q.target, latest)
            if sighting is not None:
                with self._lock:
                    self._watch(q, "tracking", sighting["relation"]["support"])
                return self._found(reply, q, "found_grounding", sighting,
                                   f"I think I see {q.obj()} {_where(sighting)}.")

        # 4. Not yet -- keep watching.
        with self._lock:
            self._watch(q, "searching")
        if latest is None:
            msg = (f"My camera isn't sending me anything yet, but I'll look for {q.obj()} "
                   f"as soon as it does.")
        else:
            msg = (f"I haven't seen {q.obj()} yet. Drive me around and I'll tell you "
                   f"as soon as I spot {q.them()}.")
        reply.update(outcome="not_found", reply=msg, searching=True)
        return reply

    def _found(self, reply, q, outcome, sighting, text):
        pos = sighting.get("position") if outcome != "found_memory" else None
        reply.update(
            outcome=outcome, reply=text, relation=sighting["relation"]["text"],
            matched_label=sighting["display"], source=sighting["source"],
            distance_m=pos["distance_m"] if pos else None, bearing_deg=pos["bearing_deg"] if pos else None,
            snapshot=self._snapshot_uri(sighting["key"]), searching=outcome == "found_memory",
        )
        return reply

    # -- VLM grounding fallback ----------------------------------------------------------
    def _ground(self, target, latest):
        """Asks Qwen3-VL to box `target` in the full latest frame, then double-checks the
        crop with a yes/no question (a 4B model will happily invent a box for something
        that isn't there). On success the result is recorded like any other sighting."""
        from OOD_object_name_generation.qwen3_generator import describe_images, generate_text

        im0 = latest["im0"]
        h, w = im0.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
        prompt = (f"Locate the {target} in this image. If you can see a {target}, reply with only JSON in "
                  f'the form {{"bbox_2d": [x1, y1, x2, y2]}}. If there is no {target} in the image, reply '
                  f"with only the word: none")
        try:
            raw = generate_text(pil, prompt, max_new_tokens=48)
        except Exception as e:  # never let a VLM hiccup turn into a 500 on the page
            print(f"[assistant] grounding failed: {e}")
            return None
        # The four numbers inside [...] -- not just any four numbers, or the "2" in the
        # "bbox_2d" key itself gets taken as x1.
        m = BBOX_RE.search(raw)
        if m is None:
            print(f"[assistant] grounding '{target}': {raw!r} -> not found")
            return None
        nums = [float(v) for v in m.groups()]
        x1, y1, x2, y2 = nums
        if max(nums) <= 1000:
            # Qwen3-VL answers in 0-1000 relative coordinates (Qwen2.5-VL used absolute
            # pixels; anything above 1000 can only be absolute, so that's handled too).
            x1, x2, y1, y2 = x1 * w / 1000, x2 * w / 1000, y1 * h / 1000, y2 * h / 1000
        x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
        y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
        if x2 - x1 < 6 or y2 - y1 < 6 or (x2 - x1) * (y2 - y1) > 0.6 * w * h:
            print(f"[assistant] grounding '{target}': {raw!r} -> implausible box, ignored")
            return None

        crop = pil.crop((int(x1), int(y1), int(x2), int(y2)))
        try:
            verdict = describe_images([crop], prompt=f"Is this a {target}? Answer yes or no.",
                                      max_new_tokens=2, batch_size=1)[0]
        except Exception as e:
            print(f"[assistant] grounding verification failed: {e}")
            return None
        print(f"[assistant] grounding '{target}': {raw!r} -> box {[round(v) for v in (x1, y1, x2, y2)]}, "
              f"verified={verdict!r}")
        if not verdict.startswith("yes"):
            return None

        found = {"label": target, "source": "grounding", "box": [x1, y1, x2, y2], "score": 1.0}
        objects = [o for o in latest["objects"] if match_score(target, o["label"]) < MATCH_THRESHOLD] + [found]
        sightings = sm.compute_sightings(objects, im0.shape, latest["depth"])
        sighting = sightings.get(sm.norm_label(target))
        if sighting is None:
            return None
        self.scene.record(latest["frame_id"], latest["ts"], im0, {sighting["key"]: sighting})
        return self.scene.get(sighting["key"])

    # -- page state ---------------------------------------------------------------------
    def state(self, since=0):
        now = time.time()
        with self._lock:
            latest = self._latest
            searches = [{
                "target": s["query"].target, "state": s["state"], "age_s": round(now - s["started"], 1),
            } for s in self._searches.values()]
            events = [e for e in self._events if e["id"] > since]
            last_id = self._next_id - 1
            highlights = []
            if latest is not None:
                h, w = latest["im_shape"]
                for s in self._searches.values():
                    m = self._best_match(latest["sightings"].values(), s["query"].target)
                    if m is not None:
                        x1, y1, x2, y2 = m["box"]
                        highlights.append({"target": s["query"].target, "label": m["display"],
                                           "box": [x1 / w, y1 / h, x2 / w, y2 / h]})
        camera = None
        if latest is not None:
            camera = {"image": latest["preview"], "resolution": latest["resolution"], "frame": latest["frame_id"],
                      "age_s": round(now - latest["ts"], 1), "source": latest["source"],
                      "depth": latest["depth"] is not None}
        return {"camera": camera, "highlights": highlights, "searches": searches,
                "scene": self.scene.entries(now), "events": events, "last_event_id": last_id}
