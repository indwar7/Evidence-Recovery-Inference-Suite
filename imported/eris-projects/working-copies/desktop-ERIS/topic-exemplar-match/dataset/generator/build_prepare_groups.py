"""
Embeds build/topic_map.json and build/hard_neighbors.json into prepare.py
(with --inline, replacing the existing _BLOB = (...) literal in place) or
into a standalone module. This is the only place the topic grouping lives;
it is never shipped in raw/.

Usage: python build_prepare_groups.py --topic-map ../../build/topic_map.json --neighbors ../../build/hard_neighbors.json --out ../../prepare.py --inline
"""
import argparse
import base64
import json
import zlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-map", default="../../build/topic_map.json")
    ap.add_argument("--neighbors", default="../../build/hard_neighbors.json")
    ap.add_argument("--out", default="../../prepare_groups.py")
    ap.add_argument("--max-neighbors", type=int, default=40)
    ap.add_argument("--inline", action="store_true", help="replace the _BLOB literal inside an existing prepare.py")
    args = ap.parse_args()

    tm = json.load(open(args.topic_map))
    hn = {k: v[: args.max_neighbors] for k, v in json.load(open(args.neighbors)).items()}
    comp = base64.b85encode(zlib.compress(json.dumps({"m": tm, "n": hn}, separators=(",", ":")).encode(), 9)).decode()
    lines = [comp[i : i + 100] for i in range(0, len(comp), 100)]
    blob = "_BLOB = (\n" + "\n".join(f'    "{l}"' for l in lines) + "\n)\n"
    if args.inline:
        src = open(args.out).read()
        a = src.index("_BLOB = (")
        b = src.index("\n)\n", a) + 3
        open(args.out, "w").write(src[:a] + blob + src[b:])
        print(f"inlined grouping into {args.out} ({len(tm)} works, {len(hn)} groups)")
        return
    src = (
        '"""\nTopic grouping used by prepare.py, embedded here rather than shipped in the\n'
        'raw archive so that no file in the data mount pairs an abstract with its\n'
        'topic. Decodes to {"m": {work_id: group}, "n": {group: [neighbour groups]}}.\n"""\n'
        "import base64\nimport json\nimport zlib\n\n_BLOB = (\n"
        + "\n".join(f'    "{l}"' for l in lines)
        + "\n)\n\n\ndef load():\n    return json.loads(zlib.decompress(base64.b85decode(_BLOB)).decode())\n"
    )
    open(args.out, "w").write(src)
    print(f"wrote {args.out} ({len(src) // 1024} KB, {len(tm)} works, {len(hn)} groups)")


if __name__ == "__main__":
    main()
