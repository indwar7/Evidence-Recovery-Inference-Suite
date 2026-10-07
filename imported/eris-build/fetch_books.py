import csv
import os
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

MIRRORS = [
    "https://mirror.csclub.uwaterloo.ca/gutenberg",
    "https://mirror.cs.odu.edu/gutenberg",
    "http://mirrors.xmission.com/gutenberg",
]
OUT_DIR = "raw_books"
os.makedirs(OUT_DIR, exist_ok=True)
PRINT_LOCK = threading.Lock()


def path_for(gid: str) -> str:
    digits = gid
    if len(digits) == 1:
        prefix = "0"
    else:
        prefix = "/".join(digits[:-1])
    return f"/{prefix}/{digits}"


def try_fetch(gid: str, mirror: str):
    candidates = [f"{gid}-0.txt", f"{gid}-8.txt", f"{gid}.txt"]
    base = f"{mirror}{path_for(gid)}"
    for name in candidates:
        url = f"{base}/{name}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (research corpus builder; "
                                        "contact abhay.indwar7@gmail.com)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    data = resp.read()
                    if len(data) > 2000:
                        return data
        except Exception:
            continue
    return None


def _fetch_one(args):
    idx, row = args
    gid = row["id"][2:]
    out_path = os.path.join(OUT_DIR, f"{gid}.txt")
    if os.path.exists(out_path):
        return row, True
    mirror = MIRRORS[idx % len(MIRRORS)]
    data = try_fetch(gid, mirror)
    if data is None:
        return row, False
    with open(out_path, "wb") as fh:
        fh.write(data)
    return row, True


def main():
    with open("candidate_books.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok, fail = 0, 0
    manifest = []
    done = 0
    # 3 workers, one per mirror, round-robin assignment -> each mirror sees
    # roughly sequential, unhurried traffic (politeness), not a burst.
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_fetch_one, (i, row)) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            row, success = fut.result()
            done += 1
            if success:
                ok += 1
                manifest.append(row)
            else:
                fail += 1
            if done % 20 == 0:
                with PRINT_LOCK:
                    print(f"progress: done={done}/{len(rows)} ok={ok} fail={fail}", flush=True)

    with open("fetched_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "author", "downloads"])
        w.writeheader()
        for row in manifest:
            w.writerow(row)

    print(f"DONE ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
