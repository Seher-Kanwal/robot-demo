"""
Inspect and edit the persistent OOD-object memory (see ood_memory.py).

    python robot_code/manage_memory.py                        # just list what's stored
    python robot_code/manage_memory.py --delete 0 2            # delete entries by index
    python robot_code/manage_memory.py --delete-name "lotion bottle"  # delete by name

If robot_web_app.py or robot_implementation.py is currently running, it holds its own
in-memory copy of this file and will overwrite these changes on its next remember() call
-- restart it afterward for a deletion to actually stick.
"""
import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
if str(FILE.parent) not in sys.path:
    sys.path.append(str(FILE.parent))

import ood_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=str, default=str(FILE.parent / "ood_memory.json"))
    parser.add_argument("--delete", type=int, nargs="+", metavar="INDEX",
                        help="delete one or more entries by index (see the listing below for indices)")
    parser.add_argument("--delete-name", type=str, metavar="NAME",
                        help="delete every entry whose name exactly matches NAME (case-insensitive)")
    args = parser.parse_args()

    mem = ood_memory.OODMemory(args.path)

    if args.delete_name:
        target = args.delete_name.strip().lower()
        indices = [e["index"] for e in mem.list_entries() if e["name"] == target]
        if not indices:
            print(f"no entries named {args.delete_name!r}")
        else:
            removed = mem.forget(indices)
            print(f"deleted {removed} entrie(s) named {args.delete_name!r}")

    if args.delete:
        removed = mem.forget(args.delete)
        print(f"deleted {removed} entrie(s) by index")

    entries = mem.list_entries()
    if not entries:
        print(f"{args.path} is empty")
        return
    print(f"{len(entries)} entrie(s) in {args.path}:")
    for e in entries:
        print(f"  [{e['index']}] {e['name']!r:30s} added {e['added_at']}")


if __name__ == "__main__":
    main()
