"""
Persistent, object-centric memory of user-taught objects: each object has a name and
several *views* (front / side / back / ...), each stored as an embedding plus the JPG it
came from.

Layout on disk (memory_dir):
    memory.json                 names, view metadata, embeddings, which embedder made them
    views/<object_id>/NNN.jpg   the view crops themselves

Keeping the JPGs means switching embedders (e.g. --embedder clip-b32 to compare) doesn't
throw the memory away: on load, views whose embeddings came from a different model are
re-embedded from their JPGs.

Matching: a query is compared with every stored view; an object's score is its best view's
cosine similarity (a remote seen from the side should match the remote's side view, not
the average of all its views). A match is accepted if the best object's score clears
`accept_thres` and beats the runner-up object by at least `margin`.

No Flask/app imports here, so another module (e.g. the VLM pipeline) can reuse it.
"""
import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from imgutils import encode_jpg, normalize_rows
from naming import name_key

DEFAULT_ACCEPT_THRES = 0.62
DEFAULT_MARGIN = 0.05
DEFAULT_MAX_VIEWS = 24
VIEW_MAX_DIM = 256
SCHEMA_VERSION = 1


def _now():
    return datetime.now(timezone.utc).isoformat()


class MultiViewMemory:
    def __init__(self, memory_dir, embedder, accept_thres=DEFAULT_ACCEPT_THRES, margin=DEFAULT_MARGIN,
                 max_views=DEFAULT_MAX_VIEWS):
        self.dir = Path(memory_dir)
        self.path = self.dir / "memory.json"
        self.embedder = embedder
        self.accept_thres = accept_thres
        self.margin = margin
        self.max_views = max_views
        self.version = 0  # bumped on every change, so a UI can tell when to refetch
        self._lock = threading.RLock()
        self._objects = []  # [{id, name, created_at, updated_at, next_view, views: [{file, embedding(np), origin, added_at}]}]
        self._V = None       # (n_views_total, dim) stacked embeddings, object-major order
        self._starts = None  # index of each object's first row in _V
        self._load()

    # -- persistence -------------------------------------------------------------
    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[memory] warning: could not read {self.path} ({e}); starting empty")
            return
        objects = []
        for obj in raw.get("objects", []):
            views = []
            for v in obj.get("views", []):
                if (self.dir / v["file"]).is_file():
                    views.append({**v, "embedding": np.asarray(v.get("embedding") or [], dtype=np.float32)})
                else:
                    print(f"[memory] warning: missing view file {v['file']}; dropping it")
            if views:
                objects.append({**obj, "views": views})
        self._objects = objects

        stale = raw.get("embedder") != self.embedder.model_id or any(
            v["embedding"].shape != (self.embedder.dim,) for o in objects for v in o["views"])
        if stale and objects:
            print(f"[memory] embeddings were made by {raw.get('embedder')!r}, re-embedding all views "
                  f"with {self.embedder.model_id!r} from their stored JPGs ...")
            for obj in objects:
                imgs = [cv2.imread(str(self.dir / v["file"])) for v in obj["views"]]
                embs = self.embedder.embed(imgs)
                for v, e in zip(obj["views"], embs):
                    v["embedding"] = e
            self._save()
        self._rebuild()
        n_views = sum(len(o["views"]) for o in self._objects)
        print(f"[memory] loaded {len(self._objects)} object(s), {n_views} view(s) from {self.path}")

    def reload(self, swap=None):
        """Forgets everything in RAM and reads the folder again. `swap` (e.g. a memory_backup
        call that moves the folder) runs first under the same lock, so no save can land in
        between; its return value is returned."""
        with self._lock:
            result = swap() if swap is not None else None
            self._objects = []
            self._load()
            self._rebuild()
            return result

    def _save(self):
        # Atomic write (tmp file + os.replace), same as ood_memory.OODMemory._save: a crash
        # mid-write never truncates the existing file.
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": SCHEMA_VERSION,
            "embedder": self.embedder.model_id,
            "dim": self.embedder.dim,
            "objects": [
                {**{k: v for k, v in o.items() if k != "views"},
                 "views": [{**v, "embedding": [round(float(x), 6) for x in v["embedding"]]} for v in o["views"]]}
                for o in self._objects
            ],
        }
        tmp = self.path.with_suffix(f".json.tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self.path)

    def _rebuild(self):
        self._objects = [o for o in self._objects if o["views"]]
        if not self._objects:
            self._V, self._starts = None, None
        else:
            self._V = np.stack([v["embedding"] for o in self._objects for v in o["views"]]).astype(np.float32)
            counts = [len(o["views"]) for o in self._objects]
            self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.version += 1

    # -- matching ----------------------------------------------------------------
    def match(self, queries, thres_offset=0.0):
        """queries: (M, dim) L2-normalized embeddings. Returns one dict per row:
        {object_id, name, score, second, accepted} for the best object (accepted says whether
        it passed threshold + margin), or None for every row when memory is empty."""
        queries = np.asarray(queries, dtype=np.float32)
        with self._lock:
            if self._V is None or len(queries) == 0:
                return [None] * len(queries)
            sims = queries @ self._V.T                                   # (M, n_views)
            per_obj = np.maximum.reduceat(sims, self._starts, axis=1)    # (M, n_objects): best view per object
            objects = [(o["id"], o["name"]) for o in self._objects]
        out = []
        for row in per_obj:
            order = np.argsort(-row)
            best = float(row[order[0]])
            second = float(row[order[1]]) if len(order) > 1 else -1.0
            accepted = best >= self.accept_thres + thres_offset and (len(order) == 1 or best - second >= self.margin)
            obj_id, name = objects[order[0]]
            out.append({"object_id": obj_id, "name": name, "score": best, "second": second, "accepted": accepted})
        return out

    # -- editing -----------------------------------------------------------------
    def _find(self, object_id):
        for o in self._objects:
            if o["id"] == object_id:
                return o
        return None

    def find_by_name(self, name):
        with self._lock:
            key = name_key(name)
            for o in self._objects:
                if name_key(o["name"]) == key:
                    return o["id"]
        return None

    def add_object(self, name, crops, embeddings, origin="ask"):
        """Creates a new object -- or, if one with the same name (case-insensitive) already
        exists, appends these views to it instead. Returns the object id."""
        name = (name or "").strip()
        if not name:
            raise ValueError("empty name")
        with self._lock:
            existing = self.find_by_name(name)
            if existing is None:
                obj = {"id": "obj_" + uuid.uuid4().hex[:8], "name": name, "created_at": _now(),
                       "updated_at": _now(), "next_view": 0, "views": []}
                self._objects.append(obj)
                object_id = obj["id"]
            else:
                object_id = existing
            self._add_views_locked(object_id, crops, embeddings, origin)
            return object_id

    def add_views(self, object_id, crops, embeddings, origins="capture"):
        with self._lock:
            return self._add_views_locked(object_id, crops, embeddings, origins)

    def _add_views_locked(self, object_id, crops, embeddings, origins):
        obj = self._find(object_id)
        if obj is None:
            raise KeyError(object_id)
        if isinstance(origins, str):
            origins = [origins] * len(crops)
        view_dir = self.dir / "views" / object_id
        view_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for crop_img, emb, origin in zip(crops, normalize_rows(embeddings), origins):
            if crop_img is None or crop_img.size == 0:
                continue
            rel = f"views/{object_id}/{obj['next_view']:03d}.jpg"
            (self.dir / rel).write_bytes(encode_jpg(crop_img, max_dim=VIEW_MAX_DIM, quality=90))
            obj["next_view"] += 1
            obj["views"].append({"file": rel, "embedding": emb, "origin": origin, "added_at": _now()})
            added += 1
        self._enforce_view_cap(obj)
        obj["updated_at"] = _now()
        self._rebuild()
        self._save()
        return added

    def _enforce_view_cap(self, obj):
        # Drop the most redundant view (highest similarity to some other view of the same
        # object) until under the cap -- keeps the viewpoints that add the most coverage.
        while len(obj["views"]) > self.max_views:
            E = np.stack([v["embedding"] for v in obj["views"]])
            sims = E @ E.T
            np.fill_diagonal(sims, -1.0)
            victim = int(np.argmax(sims.max(axis=1)))
            self._unlink(obj["views"].pop(victim)["file"])

    def _unlink(self, rel):
        try:
            (self.dir / rel).unlink()
        except OSError:
            pass

    def delete_view(self, object_id, file):
        with self._lock:
            obj = self._find(object_id)
            if obj is None:
                return False
            keep = [v for v in obj["views"] if v["file"] != file]
            if len(keep) == len(obj["views"]):
                return False
            self._unlink(file)
            obj["views"] = keep
            if not keep:
                shutil.rmtree(self.dir / "views" / object_id, ignore_errors=True)
            self._rebuild()
            self._save()
            return True

    def rename(self, object_id, name):
        name = (name or "").strip()
        with self._lock:
            obj = self._find(object_id)
            if obj is None or not name:
                return False
            obj["name"] = name
            obj["updated_at"] = _now()
            self._rebuild()
            self._save()
            return True

    def delete(self, object_id):
        with self._lock:
            obj = self._find(object_id)
            if obj is None:
                return False
            self._objects.remove(obj)
            shutil.rmtree(self.dir / "views" / object_id, ignore_errors=True)
            self._rebuild()
            self._save()
            return True

    # -- reading -----------------------------------------------------------------
    def embeddings_for(self, object_id):
        with self._lock:
            obj = self._find(object_id)
            if obj is None:
                return np.zeros((0, self.embedder.dim), dtype=np.float32)
            return np.stack([v["embedding"] for v in obj["views"]]).astype(np.float32)

    def get(self, object_id):
        with self._lock:
            obj = self._find(object_id)
            return None if obj is None else self._public(obj)

    def list_objects(self):
        with self._lock:
            return [self._public(o) for o in self._objects]

    @staticmethod
    def _public(obj):
        return {
            "id": obj["id"], "name": obj["name"], "created_at": obj["created_at"], "updated_at": obj["updated_at"],
            "n_views": len(obj["views"]),
            "views": [{"file": v["file"], "origin": v["origin"]} for v in obj["views"]],
        }
