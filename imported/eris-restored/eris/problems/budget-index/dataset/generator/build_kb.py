#!/usr/bin/env python3
"""build_kb.py -- rebuild `real_facts.json`, the knowledge base this challenge
ships, from its two upstream sources.

This is an AUTHORING script, not part of the solver pipeline: it is the only
step that touches the network, and it is run once to produce the snapshot that
`generate.py` then consumes offline. The challenge itself stays fully
self-contained.

Sources
-------
1. mledoze/countries -- the open country dataset that the REST Countries API is
   built from. Supplies the country name, ISO 3166-1 alpha-3 code, capital,
   region, land area, bordering-country list and official-language list.

       https://raw.githubusercontent.com/mledoze/countries/v4.0.0/countries.json

   Pinned to the immutable v4.0.0 tag. Licence: ODbL-1.0 (attribution +
   share-alike). The SHA-256 of the exact file used is recorded in the output.

2. World Bank -- World Development Indicators, indicator SP.POP.TOTL
   ("Population, total"). Supplies the population figure, pinned to a single
   reference year so the snapshot is reproducible.

       https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000&date=2024

   Licence: CC BY 4.0 (attribution).

Both endpoints are public and require no API key. (REST Countries' own v5 API
now requires authentication, which is why this reads the upstream dataset
directly instead.)

Output
------
`real_facts.json` -- a JSON array of country objects with exactly the fields
`name`, `cca3`, `capital`, `region`, `population`, `area`, `num_borders`,
`num_languages`, sorted by `cca3`. Only countries carrying every field are
kept, so no value in the shipped knowledge base is imputed or missing.

`real_facts.provenance.json` -- the exact source URLs, the reference year, the
input SHA-256 digests and the row counts, so the snapshot can be audited.

Usage:
    python build_kb.py [--out real_facts.json]
"""

import argparse
import hashlib
import json
import os
import urllib.request

COUNTRIES_URL = ("https://raw.githubusercontent.com/mledoze/countries/"
                 "v4.0.0/countries.json")
WB_YEAR = 2024
WB_URL = ("https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"
          f"?format=json&per_page=20000&date={WB_YEAR}")

COUNTRIES_LICENCE = "ODbL-1.0 (Open Data Commons Open Database License v1.0)"
WB_LICENCE = "CC BY 4.0"


def fetch(url):
    """Return (raw_bytes, sha256_hex)."""
    req = urllib.request.Request(url, headers={"User-Agent": "budget-index-kb"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    return raw, hashlib.sha256(raw).hexdigest()


def main(out_path):
    raw_c, sha_c = fetch(COUNTRIES_URL)
    countries = json.loads(raw_c.decode("utf-8"))
    print(f"mledoze/countries : {len(countries)} rows  sha256={sha_c[:16]}...")

    raw_w, sha_w = fetch(WB_URL)
    # the World Bank serves a UTF-8 BOM
    payload = json.loads(raw_w.decode("utf-8-sig"))
    pop_rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    pop = {r["countryiso3code"]: int(r["value"])
           for r in pop_rows
           if r.get("countryiso3code") and r.get("value")}
    print(f"World Bank {WB_YEAR}   : {len(pop)} populations  sha256={sha_w[:16]}...")

    facts, dropped = [], 0
    for c in countries:
        cca3 = c.get("cca3")
        caps = c.get("capital") or []
        borders = c.get("borders") or []
        langs = c.get("languages") or {}
        area = c.get("area")
        # some rows carry a placeholder empty capital (uninhabited territories)
        if not (cca3 and caps and caps[0].strip() and langs and area and area > 0):
            dropped += 1
            continue
        if cca3 not in pop:
            dropped += 1
            continue
        facts.append({
            "name": c["name"]["common"],
            "cca3": cca3,
            "capital": caps[0],
            "region": c.get("region", ""),
            "population": pop[cca3],
            "area": int(area),
            "num_borders": len(borders),
            "num_languages": len(langs),
        })
    facts.sort(key=lambda r: r["cca3"])
    print(f"kept {len(facts)} countries, dropped {dropped} "
          f"(missing capital / area / language / population)")

    # every field must be populated -- nothing imputed
    for r in facts:
        assert r["name"] and r["cca3"] and r["capital"]
        assert r["population"] > 0 and r["area"] > 0
        assert r["num_languages"] >= 1 and r["num_borders"] >= 0

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)

    prov = {
        "sources": [
            {"name": "mledoze/countries",
             "url": COUNTRIES_URL,
             "pinned_to": "tag v4.0.0",
             "sha256": sha_c,
             "rows": len(countries),
             "licence": COUNTRIES_LICENCE,
             "fields_supplied": ["name", "cca3", "capital", "region", "area",
                                 "num_borders", "num_languages"]},
            {"name": "World Bank World Development Indicators",
             "indicator": "SP.POP.TOTL (Population, total)",
             "url": WB_URL,
             "reference_year": WB_YEAR,
             "sha256": sha_w,
             "rows_with_value": len(pop),
             "licence": WB_LICENCE,
             "fields_supplied": ["population"]},
        ],
        "output_rows": len(facts),
        "derived_licence": (
            "ODbL-1.0 -- share-alike inherited from mledoze/countries; "
            "World Bank population values are CC BY 4.0 and are attributed above"),
    }
    prov_path = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                             "real_facts.provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    print(f"wrote {out_path} and {os.path.basename(prov_path)}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(here, "real_facts.json"))
    main(ap.parse_args().out)
