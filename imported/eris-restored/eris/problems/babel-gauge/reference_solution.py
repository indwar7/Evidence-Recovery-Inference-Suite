#!/usr/bin/env python3
"""reference_solution.py - reference solver for "Babel Gauge: Legacy Telemetry
Log Canonicalization".

Demonstrates that the task is solvable up to its information-theoretic ceiling.

Six legacy formats are induced from the training pairs. Three of them (F1, F5
and, for temperature/pressure, F3/F4) log values at a coarser precision than
the canonical record requires, and F3/F4/F5 print no seconds at all. Those
digits are absent from the input, so no solver can recover them: the best
possible strategy is to emit the most likely underlying value given the coarse
observation, learned from the training pairs.

Reads  ./dataset/public/{train,test,sites}.csv
Writes ./working/submission.csv

Deterministic; pandas only.
"""

import collections
import json
import os
import re

import pandas as pd

FIELDS = ["ts", "dev", "site", "temp_c", "pres_kpa", "status"]
TY = {"P": "PMP", "C": "CMP", "F": "FAN", "V": "VLV"}
T, DT = pd.Timestamp, pd.Timedelta


def observe(raw, off):
    """Parse a raw line into (format, exactly-known fields, coarse readings).

    Anything in `coarse` was rendered at reduced precision, so the exact
    canonical value cannot be derived from it arithmetically.
    """
    m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| ([PCFV]\w\w-\d{4}) @ "
                 r"([A-Z]{2,4}\d) \| T=(-?\d+)F P=(\d+)psi \| (OK|WARN|FAULT)$",
                 raw)
    if m:
        dt, dev, s, t, p, st = m.groups()
        ts = (T(dt.replace(" ", "T")) - DT(minutes=off[s]))
        return "F1", {"ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "dev": dev,
                      "site": s, "status": st}, {"tF": int(t), "pP": int(p)}

    m = re.match(r"^<([PCFV])(\d{4})\|([A-Z]{2,4}\d)\|(\d{14})\|t:(-?[\d.]+)\|"
                 r"p:([\d.]+)\|s:([012])>$", raw)
    if m:
        ty, n, s, dt, t, p, st = m.groups()
        ts = T(f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}T{dt[8:10]}:{dt[10:12]}:{dt[12:]}")
        return "F2", {"ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "dev": f"{TY[ty]}-{n}", "site": s,
                      "temp_c": f"{float(t):.1f}", "pres_kpa": f"{float(p):.1f}",
                      "status": {"0": "OK", "1": "WARN", "2": "FAULT"}[st]}, {}

    if "time=" in raw and "flag=" in raw:
        g = dict(re.findall(r"(\w+)=(\S+)", raw))
        dm = re.search(r"time=(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d)", raw)
        if dm and re.match(r"^[PCFV]\w\w\d{4}$", g.get("dev", "")) \
                and g.get("site") in off:
            dd, mm, yy, H, Mi = dm.groups()
            base = T(f"{yy}-{mm}-{dd}T{H}:{Mi}:00") - DT(minutes=off[g["site"]])
            return "F3", {"dev": f"{TY[g['dev'][0]]}-{g['dev'][3:]}",
                          "site": g["site"], "status": g["flag"]}, {
                "minute": base, "tC": int(g["temp"].rstrip("C")),
                "pK": int(g["pressure"].rstrip("kPa"))}

    m = re.match(r"^([PCFV])(\d{4})([A-Z0-9]+)\s+(\d{6})(\d{4})([+-]\d{3})"
                 r"(\d{5})([OA])$", raw)
    if m:
        ty, n, s, dd, hm, t, p, st = m.groups()
        base = T(f"20{dd[:2]}-{dd[2:4]}-{dd[4:6]}T{hm[:2]}:{hm[2:]}:00") \
            - DT(minutes=off[s])
        return "F4", {"dev": f"{TY[ty]}-{n}", "site": s,
                      }, {"minute": base, "tC": int(t), "pK": int(p),
                          "alarm": st == "A"}

    m = re.match(r"^([A-Z]{2,4}\d),([PCFV]\w\w-\d{4}),(\d\d)/(\d\d)/(\d\d) "
                 r"(\d\d):(\d\d)(AM|PM),(-?\d+),(\d+),(RUN|ALM)$", raw)
    if m:
        s, dev, mm, dd, yy, H, Mi, ap, t, p, st = m.groups()
        h = int(H) % 12 + (12 if ap == "PM" else 0)
        base = T(f"20{yy}-{mm}-{dd}T{h:02d}:{Mi}:00") - DT(minutes=off[s])
        return "F5", {"dev": dev, "site": s}, {
            "minute": base, "tF": int(t), "pP": int(p), "alarm": st == "ALM"}

    if raw.startswith("{") and raw.endswith("}"):
        try:
            j = json.loads(raw)
        except ValueError:
            return None, {}, {}
        return "F6", {"ts": T(j["e"], unit="s").strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "dev": j["d"], "site": j["st"],
                      "temp_c": f"{float(j['tc']):.1f}",
                      "pres_kpa": f"{float(j['pk']):.1f}",
                      "status": {"ok": "OK", "warn": "WARN",
                                 "fault": "FAULT"}[j["x"]]}, {}
    return None, {}, {}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pub = os.path.join(here, "dataset", "public")
    if not os.path.isdir(pub):
        pub = os.path.join(here, "prepared", "public")
    rd = lambda f: pd.read_csv(os.path.join(pub, f), keep_default_na=False)

    train, test, sites = rd("train.csv"), rd("test.csv"), rd("sites.csv")
    off = dict(zip(sites.site, sites.utc_offset_min))

    # ---- learn the most likely true value behind each coarse reading ----
    tab = collections.defaultdict(collections.Counter)
    alarm = collections.defaultdict(collections.Counter)
    for raw, canon in zip(train.raw_line, train.canonical):
        if canon == "REJECT":
            continue
        fmt, known, coarse = observe(raw, off)
        if fmt is None:
            continue
        tk = dict(p.split("=", 1) for p in canon.split("|"))
        for key in ("tF", "tC"):
            if key in coarse:
                tab[(key, coarse[key])][tk["temp_c"]] += 1
        for key in ("pP", "pK"):
            if key in coarse:
                tab[(key, coarse[key])][tk["pres_kpa"]] += 1
        if "minute" in coarse:
            tab[("sec", 0)][tk["ts"][-3:-1]] += 1
        if coarse.get("alarm"):
            # F4/F5 collapse WARN and FAULT into one alarm token. Which one it
            # was is not in the line, but it correlates with temperature.
            alarm[int(float(coarse.get("tC", coarse.get("tF", 0))) // 5)][
                tk["status"]] += 1
    best = {k: v.most_common(1)[0][0] for k, v in tab.items()}
    arule = {b: c.most_common(1)[0][0] for b, c in alarm.items()}
    amode = "FAULT"
    sec = int(best.get(("sec", 0), "0"))

    # ---- transduce the test lines ----
    out = []
    for raw in test.raw_line:
        fmt, known, coarse = observe(raw, off)
        if fmt is None:                     # grammar broken -> ingestion damage
            out.append("REJECT")
            continue
        rec = dict(known)
        for key, field in (("tF", "temp_c"), ("tC", "temp_c"),
                           ("pP", "pres_kpa"), ("pK", "pres_kpa")):
            if key in coarse:
                rec[field] = best.get((key, coarse[key]), "0.0")
        if "minute" in coarse:              # seconds were never transmitted
            rec["ts"] = (coarse["minute"] + DT(seconds=sec)) \
                .strftime("%Y-%m-%dT%H:%M:%SZ")
        if "alarm" in coarse:
            rec["status"] = "OK" if not coarse["alarm"] else arule.get(
                int(float(coarse.get("tC", coarse.get("tF", 0))) // 5), amode)
        out.append("|".join(f"{k}={rec.get(k, '')}" for k in FIELDS))

    work = os.path.join(here, "working")
    os.makedirs(work, exist_ok=True)
    sub = pd.DataFrame({"line_id": test.line_id, "output": out})
    sub.to_csv(os.path.join(work, "submission.csv"), index=False)
    print(f"wrote {len(sub):,} rows to working/submission.csv")


if __name__ == "__main__":
    main()
