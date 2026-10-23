"""
Fresh-demo support for Module 1's memory folder: back it up and start empty, list the
backups, and put one back. Only folder moves -- no models, no Flask -- so both teach_app.py
(on the live memory, followed by MultiViewMemory.reload()) and the hub (while Module 1 is
stopped) use the same code.

    teach_memory/                         the live memory (memory.json, views/, negatives/)
    teach_memory_backups/<YYYYmmdd_HHMMSS>/   one folder per backup, same layout
"""
import json
import shutil
from datetime import datetime
from pathlib import Path


def backups_dir_for(memory_dir):
    memory_dir = Path(memory_dir)
    return memory_dir.parent / f"{memory_dir.name}_backups"


def summary(folder):
    """(object names, total views) of a memory folder, read from its memory.json."""
    try:
        with open(Path(folder) / "memory.json", "r", encoding="utf-8") as f:
            objects = json.load(f).get("objects", [])
    except (OSError, ValueError):
        return [], 0
    return [o.get("name", "?") for o in objects], sum(len(o.get("views", [])) for o in objects)


def is_empty(memory_dir):
    return not summary(memory_dir)[0]


def list_backups(memory_dir):
    """Newest first: [{name, objects: [names], views}]."""
    root = backups_dir_for(memory_dir)
    if not root.is_dir():
        return []
    out = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        names, n_views = summary(d)
        out.append({"name": d.name, "objects": names, "views": n_views})
    return out


def backup_and_clear(memory_dir):
    """Moves the memory folder into a new timestamped backup and leaves no memory behind.
    Returns the backup's name, or None if there was nothing to back up."""
    memory_dir = Path(memory_dir)
    if not memory_dir.exists():
        return None
    if is_empty(memory_dir):
        shutil.rmtree(memory_dir, ignore_errors=True)  # nothing worth keeping (e.g. only negatives)
        return None
    root = backups_dir_for(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest, n = root / name, 1
    while dest.exists():  # two resets within the same second
        n += 1
        dest = root / f"{name}_{n}"
    shutil.move(str(memory_dir), str(dest))
    return dest.name


def restore(memory_dir, name):
    """Backs up the current memory (if any), then makes backup `name` the live memory.
    Returns the name of the backup made of the current memory (or None)."""
    root = backups_dir_for(memory_dir)
    src = root / name
    if Path(name).name != name or not src.is_dir():  # only plain folder names from list_backups()
        raise ValueError(f"no backup named {name!r}")
    saved = backup_and_clear(memory_dir)
    shutil.move(str(src), str(memory_dir))
    return saved
