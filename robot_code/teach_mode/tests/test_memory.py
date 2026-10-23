import numpy as np

from conftest import FakeEmbedder, make_crop
from multiview_memory import MultiViewMemory


def _add(mem, emb, name, oid, views):
    crops = [make_crop(oid, v, seed=v) for v in views]
    return mem.add_object(name, crops, emb.embed(crops))


def test_match_other_view_of_same_object(tmp_path, embedder):
    mem = MultiViewMemory(tmp_path, embedder)
    remote = _add(mem, embedder, "TV remote", oid=1, views=[0, 1])
    _add(mem, embedder, "teddy bear", oid=2, views=[0])
    (m,) = mem.match(embedder.embed([make_crop(1, view=5)]))
    assert m["accepted"] and m["object_id"] == remote and m["name"] == "TV remote"


def test_unknown_object_not_accepted(tmp_path, embedder):
    mem = MultiViewMemory(tmp_path, embedder)
    _add(mem, embedder, "TV remote", oid=1, views=[0])
    (m,) = mem.match(embedder.embed([make_crop(7)]))
    assert not m["accepted"]


def test_empty_memory_returns_none(tmp_path, embedder):
    mem = MultiViewMemory(tmp_path, embedder)
    assert mem.match(embedder.embed([make_crop(1)])) == [None]


def test_margin_rejects_ambiguous_match(tmp_path, embedder):
    mem = MultiViewMemory(tmp_path, embedder, margin=0.05)
    crop = [make_crop(1)]
    mem.add_object("a", crop, embedder.embed(crop))
    mem.add_object("b", crop, embedder.embed(crop))  # two objects that look identical
    (m,) = mem.match(embedder.embed(crop))
    assert m["score"] > 0.99 and not m["accepted"]


def test_same_name_merges_views(tmp_path, embedder):
    mem = MultiViewMemory(tmp_path, embedder)
    a = _add(mem, embedder, "TV remote", oid=1, views=[0])
    b = _add(mem, embedder, "tv  REMOTE", oid=1, views=[1, 2])
    assert a == b
    assert mem.get(a)["n_views"] == 3 and len(mem.list_objects()) == 1


def test_persists_and_reloads(tmp_path, embedder):
    mem = MultiViewMemory(tmp_path, embedder)
    oid = _add(mem, embedder, "TV remote", oid=1, views=[0, 1])
    again = MultiViewMemory(tmp_path, embedder)
    assert again.get(oid)["n_views"] == 2
    assert all((tmp_path / v["file"]).is_file() for v in again.get(oid)["views"])
    assert again.match(embedder.embed([make_crop(1, view=3)]))[0]["accepted"]


def test_reembeds_from_jpgs_when_embedder_changes(tmp_path, embedder):
    _add(MultiViewMemory(tmp_path, embedder), embedder, "TV remote", oid=1, views=[0, 1])
    other = FakeEmbedder(model_id="fake-v2")
    mem = MultiViewMemory(tmp_path, other)
    assert other.calls == 1  # re-embedded the stored views on load
    assert '"fake-v2"' in (tmp_path / "memory.json").read_text(encoding="utf-8")
    assert mem.match(other.embed([make_crop(1, view=4)]))[0]["accepted"]


def test_rename_delete_and_delete_view(tmp_path, embedder):
    mem = MultiViewMemory(tmp_path, embedder)
    oid = _add(mem, embedder, "remote", oid=1, views=[0, 1])
    assert mem.rename(oid, "TV remote") and mem.get(oid)["name"] == "TV remote"
    first = mem.get(oid)["views"][0]["file"]
    assert mem.delete_view(oid, first) and mem.get(oid)["n_views"] == 1
    assert not (tmp_path / first).exists()
    v0 = mem.version
    assert mem.delete(oid) and mem.get(oid) is None and mem.version > v0
    assert not (tmp_path / "views" / oid).exists()


def test_view_cap_drops_most_redundant(tmp_path, embedder):
    mem = MultiViewMemory(tmp_path, embedder, max_views=3)
    oid = _add(mem, embedder, "remote", oid=1, views=[0, 0, 1, 2])  # two identical front views
    obj = mem.get(oid)
    assert obj["n_views"] == 3
    E = mem.embeddings_for(oid)
    sims = E @ E.T
    np.fill_diagonal(sims, 0)
    assert sims.max() < 0.99  # the duplicate was the one dropped
