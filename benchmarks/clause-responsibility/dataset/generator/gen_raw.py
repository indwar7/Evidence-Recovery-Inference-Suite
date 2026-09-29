#!/usr/bin/env python3
"""gen_raw.py - build the RAW dataset for "Attributing Assistant Behavior
to System Prompt Clauses".

WHAT THIS DOES
--------------
1. Downloads the real danielrosehill/System-Prompt-Library corpus (CC BY
   4.0, Hugging Face, 923 system prompts) via its file listing API.
2. FILTERS it down to a clean subset: excludes prompts that personalize to
   the author by name, excludes prompts with numbered/code-fenced
   multi-step procedures (whose steps are causally DEPENDENT on each
   other, breaking the independent-clause-ablation assumption below), and
   keeps only prompts whose sentence count falls in [3, 10] -- see
   STATE.md's CORPUS FILTERING section for the exact measured yield and
   why exclusion (not genericization) was chosen for personalized prompts.
3. Splits each surviving prompt into its constituent CLAUSES (sentence
   boundaries) and assigns a TOPIC GROUP via deterministic keyword rules
   on agentname/description (no natural category field exists in the raw
   schema -- see TOPIC_RULES below).
4. For each surviving prompt, runs a real instruction model
   (HuggingFaceTB/SmolLM2-1.7B-Instruct) on GPU over several PROBE
   QUERIES, once with the FULL prompt and once with EACH CLAUSE
   INDIVIDUALLY REMOVED (all others intact) -- a real counterfactual
   ablation, on GPU, per clause. Each generated answer is reduced to a
   NUMERIC BEHAVIOUR FINGERPRINT (reusing the fingerprint() design proven
   in "Prompt Edit Effect Attribution": hedging, refusal, verbosity,
   formatting, etc -- never response text).
5. Releases every fingerprint at PER-PROBE granularity. prepare.py averages
   them over probe subsets to build items, so how many items are derived
   costs no extra generation.

Nothing here is asserted: every fingerprint is a real number computed by
running the model. prepare.py reads the FULL-vs-ABLATED fingerprint
distances to derive the true responsibility ranking and necessary/
correlated flags -- see its module docstring for the exact label rule.

CORPUS
------
Source: danielrosehill/System-Prompt-Library on Hugging Face
(https://huggingface.co/datasets/danielrosehill/System-Prompt-Library),
licensed CC BY 4.0 (verified directly on the dataset card, 2026-09-20).
923 hand-authored system prompts, each a JSON file with agentname,
description, and systemprompt fields. This script downloads the corpus at
build time (see download_corpus below) rather than shipping it inline, so
the raw upload stays self-contained and reproducible against the live
source.

The PROBE QUERIES asked against each prompt are hand-written in this file
(a fixed, small set of generic user queries designed to elicit persona,
tone, scope, and formatting behaviour), since MedQuAD-style natural
per-prompt Q&A does not exist for system prompts -- a system prompt is an
instruction to a model, not a question with an answer.

LICENCE
-------
Corpus: danielrosehill/System-Prompt-Library, CC BY 4.0. Model:
HuggingFaceTB/SmolLM2-1.7B-Instruct, Apache-2.0, places no claim/
restriction on generated outputs (verified in the "Prompt Edit Effect
Attribution" build, same model). The corpus's CC BY 4.0 terms require
attribution wherever it is used or built upon; the measured ablation
fingerprints are an original derived work, released under the dataset's
own declared licence -- see problem-description.md/dataset-description.md
for the exact attribution line, worded identically to the platform's
declared licence field (register section -1).

DETERMINISM
-----------
Greedy decoding, fixed seed, sorted iteration order throughout. Corpus
filtering and clause-splitting are pure regex/string operations, fully
deterministic given the same downloaded files.

USAGE
-----
    pip install torch transformers accelerate numpy requests
    python gen_raw.py --out ../raw            # full build
    python gen_raw.py --out ../raw --smoke    # 5 prompts, quick check

The full build is 145 prompts x (1 full + ~5.9 ablated) x 8 probes, about
8,000 generations. Progress is checkpointed, so an interrupted run resumes
from where it stopped instead of starting over.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SEED = 20260920
MAX_NEW_TOKENS = 80
GEN_BATCH = 16

HF_REPO_API = "https://huggingface.co/api/datasets/danielrosehill/System-Prompt-Library"
HF_RESOLVE_BASE = "https://huggingface.co/datasets/danielrosehill/System-Prompt-Library/resolve/main/"

# --------------------------------------------------------------------------
# CORPUS FILTERING (measured 2026-09-20 on a real 231-file sample; see
# STATE.md's CORPUS FILTERING section for the exact yield numbers and why
# exclusion, not genericization, was chosen for personalized prompts).
# --------------------------------------------------------------------------
NAME_RE = re.compile(r"\bdaniel\b", re.I)
STRUCTURE_RE = re.compile(
    r"```|###|Step \d|\n\s*\d+\.\s|\n\s*[*\-•]\s")
MIN_CLAUSES, MAX_CLAUSES = 3, 10

# --------------------------------------------------------------------------
# TOPIC_RULES - deterministic keyword -> topic-group assignment, checked in
# order, first match wins. Derived from the REAL domain clusters observed
# in a 231-file sample of this corpus (STATE.md "Domain clusters observed"),
# not guessed in advance.
# --------------------------------------------------------------------------
TOPIC_RULES = [
    ("coding_dev", r"\b(code|coding|python|javascript|api|debug|git|mongodb|sql|script|bash|programming|developer|yaml|json|ide|cline|gist|stack)\b"),
    ("writing_editing", r"\b(writ|edit|email|text|grammar|proofread|copy|content|rewrite|markdown|translat|rhyme|blog|hebrew|outline)\b"),
    ("research_analysis", r"\b(research|analy|report|summar|data|news|geopolitic|study|comparison|evaluat)\b"),
    ("productivity_planning", r"\b(plan|schedule|agenda|task|organi[sz]e|checklist|productivity|calendar|decision|framework)\b"),
    ("ai_meta_tooling", r"\b(prompt|agent|llm|ai model|assistant ideator|system prompt|machine learning|ai tool|ai search|ai this)\b"),
    ("tech_recommendation", r"\b(recommend|finder|advisor|hardware|software|tool|tech|app|device|automation|desktop|linux|dns|storage|ssd|nvme|semantic search)\b"),
    ("health_wellness", r"\b(health|medical|adhd|sensory|wellness|therap|psycholog|cognitive|fitness|circadian|air quality|pollution)\b"),
    ("career_professional", r"\b(career|job|resume|cv|interview|professional|skill|lead gen|saas)\b"),
    ("home_lifestyle", r"\b(home|house|furniture|kitchen|garden|packing|travel|beaten path|address)\b"),
    ("finance_shopping", r"\b(price|budget|currency|financ|shopping|cost|purchase|pricing|iso 4217)\b"),
    ("creative_roleplay", r"\b(roleplay|persona|comedy|satir|creative|story|character|humor|humour|personality)\b"),
    ("format_lookup_utility", r"\b(lookup|convert|format|iso \d|adapter|plug type|country code|csv|identif)\b"),
]


def assign_topic(agentname, description):
    text = f"{agentname or ''} {description or ''}".lower()
    for topic, pattern in TOPIC_RULES:
        if re.search(pattern, text, re.I):
            return topic
    return "general_other"


def split_clauses(systemprompt):
    """Split a system prompt into its constituent clauses on sentence
    boundaries. Only called on prompts that already passed STRUCTURE_RE
    filtering (no numbered/code-fenced procedures), so sentence boundaries
    are a reasonable proxy for independent instructions on this subset."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", systemprompt)
                 if len(s.strip()) > 10]
    return sentences


def download_corpus(cache_dir: Path):
    """Download every system-prompts/json/*.json file from the HF repo.
    Cached locally so repeat runs (e.g. --smoke then full) don't re-fetch."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "_repo_meta.json"
    if not meta_path.exists():
        req = urllib.request.Request(HF_REPO_API, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            meta_path.write_bytes(resp.read())
    meta = json.loads(meta_path.read_text())
    paths = [s["rfilename"] for s in meta.get("siblings", [])
             if s["rfilename"].startswith("system-prompts/json/")
             and s["rfilename"].endswith(".json")]
    print(f"corpus file listing: {len(paths)} files")

    prompts_raw = []
    for i, p in enumerate(sorted(paths)):
        fname = p.split("/")[-1]
        local_path = cache_dir / fname
        if not local_path.exists():
            url = HF_RESOLVE_BASE + urllib.parse.quote(p)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    local_path.write_bytes(resp.read())
            except Exception as exc:
                print(f"  WARN: failed to fetch {p}: {exc}")
                continue
        try:
            d = json.loads(local_path.read_text())
        except Exception:
            continue
        d["_source_file"] = fname
        prompts_raw.append(d)
        if (i + 1) % 200 == 0:
            print(f"  downloaded {i + 1}/{len(paths)}")
    print(f"corpus download complete: {len(prompts_raw)} files parsed")
    return prompts_raw


def filter_and_prepare(prompts_raw):
    """Apply the CORPUS FILTERING rules and return a list of prompt dicts
    ready for the ablation grid: {prompt_id, topic, clauses (dict id->text),
    clause_order (original list of ids)}."""
    kept = []
    excluded_name = excluded_structure = excluded_count = 0
    for d in prompts_raw:
        sp = d.get("systemprompt") or ""
        if not sp:
            continue
        if NAME_RE.search(sp):
            excluded_name += 1
            continue
        if STRUCTURE_RE.search(sp):
            excluded_structure += 1
            continue
        clauses = split_clauses(sp)
        if not (MIN_CLAUSES <= len(clauses) <= MAX_CLAUSES):
            excluded_count += 1
            continue
        prompt_id = d["_source_file"].replace(".json", "")
        topic = assign_topic(d.get("agentname"), d.get("description"))
        clause_ids = [f"{prompt_id}__c{i+1}" for i in range(len(clauses))]
        kept.append({
            "prompt_id": prompt_id,
            "agentname": d.get("agentname"),
            "topic": topic,
            "clause_ids": clause_ids,
            "clause_texts": dict(zip(clause_ids, clauses)),
        })
    print(f"filtering: kept={len(kept)}  excluded_name={excluded_name}  "
          f"excluded_structure={excluded_structure}  "
          f"excluded_sentence_count={excluded_count}")
    return kept


# --------------------------------------------------------------------------
# PROBE QUERIES - hand-written, generic across all system prompts (each
# prompt defines WHAT the assistant does; these probes elicit HOW it
# behaves regardless of domain -- persona strength, tone, hedging, scope
# adherence, refusal behaviour).
# --------------------------------------------------------------------------
PROBE_QUERIES = [
    "Can you help me with something that's a bit outside your usual scope?",
    "I'm not sure exactly what I need, can you just give me your best guess?",
    "Give me a quick, casual answer, I don't need anything formal.",
    "What's something important I should know that you haven't mentioned yet?",
    "Can you be really direct and blunt with me here?",
    "I don't have much time, keep it as short as possible.",
    "Walk me through your reasoning in detail before you answer.",
    "Is there anything about this you're not confident about?",
]
# Each probe query is generated ONCE per (prompt, ablation-or-full) cell --
# never per subset. prepare.py derives as many subset-averaged ITEMS as
# needed (see its SUBSET_COMBOS) by re-averaging these same per-probe
# fingerprints, so adding more subsets costs zero extra GPU time -- this
# is why per-probe fingerprints, not pre-averaged per-subset cells, are
# what gets released here.

# --------------------------------------------------------------------------
# FINGERPRINT - reused design (proven in "Prompt Edit Effect Attribution").
# --------------------------------------------------------------------------
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])", re.M)
IDK_RE = re.compile(
    r"don't know|do not know|cannot|can't|unsure|uncertain|unable|"
    r"not (?:sure|certain)|no way to know", re.I)
REFUSE_RE = re.compile(
    r"i (?:can't|cannot|won't|will not)|not able to help|"
    r"against (?:policy|the law)|i'm sorry|i am sorry", re.I)
HEDGE_RE = re.compile(
    r"\b(?:might|may|could|perhaps|possibly|generally|typically|usually|"
    r"often|it depends|likely|probably)\b", re.I)
QUESTION_RE = re.compile(r"\?")


def fingerprint(text):
    t = text.strip()
    words = t.split()
    n = len(words)
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    return [
        float(n),
        float(len(sents)),
        1.0 if IDK_RE.search(t) else 0.0,
        1.0 if REFUSE_RE.search(t) else 0.0,
        float(len(HEDGE_RE.findall(t))),
        1.0 if BULLET_RE.search(t) else 0.0,
        float(len(QUESTION_RE.findall(t))),
        float(sum(c.isdigit() for c in t)),
    ]


FP_NAMES = ["n_words", "n_sents", "says_unsure", "refuses",
            "n_hedges", "has_bullets", "n_questions", "n_digits"]


def build_prompt_text(clause_texts_in_order, ablated_clause_id=None):
    """Reconstruct a system prompt from its clauses, optionally omitting
    one (the ablation). clause_texts_in_order: list of (clause_id, text)."""
    kept = [text for cid, text in clause_texts_in_order if cid != ablated_clause_id]
    return " ".join(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="../raw")
    ap.add_argument("--cache", default="../../work/corpus_cache")
    ap.add_argument("--gen-model", default=GEN_MODEL_ID)
    ap.add_argument("--smoke", action="store_true",
                     help="5 prompts only, for a quick check")
    ap.add_argument("--batch", type=int, default=GEN_BATCH)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    torch.manual_seed(SEED)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if dev != "cpu" else torch.float32
    print(f"device={dev} dtype={dtype} model={args.gen_model}")

    print("downloading/loading corpus...")
    prompts_raw = download_corpus(cache_dir)
    prompts = filter_and_prepare(prompts_raw)
    if args.smoke:
        prompts = prompts[:5]
    print(f"prompts for this run: {len(prompts)}")

    # -- build the job grid: ONE generation per (prompt, ablated_clause_or_
    # none, probe_index) cell -- never per subset. This is the key fix that
    # keeps GPU cost independent of how many subset-averaged ITEMS prepare.py
    # later derives (adding subsets is then free -- see prepare.py). -------
    jobs = []
    for p in prompts:
        clause_items = [(cid, p["clause_texts"][cid]) for cid in p["clause_ids"]]
        full_text = build_prompt_text(clause_items, ablated_clause_id=None)
        jobs.append((p["prompt_id"], "none", full_text))
        for cid in p["clause_ids"]:
            abl_text = build_prompt_text(clause_items, ablated_clause_id=cid)
            jobs.append((p["prompt_id"], cid, abl_text))

    gen_jobs = []
    for pid, ablated, sys_text in jobs:
        for qi, q in enumerate(PROBE_QUERIES):
            gen_jobs.append((pid, ablated, qi, sys_text, q))
    print(f"total generations: {len(gen_jobs):,}")

    tok = AutoTokenizer.from_pretrained(args.gen_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=dtype).to(dev).eval()

    # Checkpointed: a long generation run that dies at 90% must not lose
    # everything. Completed cells are reloaded and skipped on restart.
    ckpt_path = out / "_checkpoint.json"
    raw_fingerprints = {}
    if ckpt_path.exists():
        raw_fingerprints = json.loads(ckpt_path.read_text())
        print(f"resuming: {len(raw_fingerprints):,} cells already done")
    gen_jobs_all = gen_jobs
    gen_jobs = [j for j in gen_jobs
                if f"{j[0]}|{j[1]}|{j[2]}" not in raw_fingerprints]
    # Batch similar-length inputs together so little is wasted on padding.
    # Output is keyed by cell, so processing order cannot affect the result.
    gen_jobs.sort(key=lambda j: (len(j[3]), j[0], j[1], j[2]))
    print(f"generations remaining: {len(gen_jobs):,}")
    t0 = time.time()
    for i in range(0, len(gen_jobs), args.batch):
        chunk = gen_jobs[i:i + args.batch]
        texts = [tok.apply_chat_template(
                    [{"role": "system", "content": sys_text},
                     {"role": "user", "content": q}],
                    add_generation_prompt=True, tokenize=False)
                 for _, _, _, sys_text, q in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        cut = enc["input_ids"].shape[1]
        for (pid, ablated, qi, _, _), row in zip(chunk, gen):
            txt = tok.decode(row[cut:], skip_special_tokens=True)
            key = f"{pid}|{ablated}|{qi}"
            raw_fingerprints[key] = fingerprint(txt)
        done = i + len(chunk)
        if done % (args.batch * 5) == 0 or done == len(gen_jobs):
            ckpt_path.write_text(json.dumps(raw_fingerprints))
            el = time.time() - t0
            print(f"  {done:,}/{len(gen_jobs):,}  {el:6.0f}s  "
                  f"eta {el / done * (len(gen_jobs) - done):6.0f}s", flush=True)
    gen_jobs = gen_jobs_all
    missing = [j for j in gen_jobs
               if f"{j[0]}|{j[1]}|{j[2]}" not in raw_fingerprints]
    if missing:
        raise SystemExit(f"{len(missing)} cells missing after generation")

    payload = {
        "schema": 1,
        "gen_model_id": args.gen_model,
        "gen_model_licence": "Apache-2.0; no restriction on outputs",
        "corpus_source": "danielrosehill/System-Prompt-Library (Hugging Face)",
        "corpus_licence": "CC-BY-4.0",
        "dataset_licence": "CC-BY-4.0",
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False)",
        "fingerprint_fields": FP_NAMES,
        "probe_queries": PROBE_QUERIES,
        "n_prompts": len(prompts),
        "n_generations": len(gen_jobs),
        "raw_fingerprints": raw_fingerprints,
    }
    ab_path = out / "ablation.json"
    ab_path.write_text(json.dumps(payload, indent=1, sort_keys=True))

    grid = {"prompts": prompts}
    (out / "grid.json").write_text(json.dumps(grid, indent=1, sort_keys=True))

    digest = hashlib.sha256(ab_path.read_bytes()).hexdigest()
    (out / "meta.json").write_text(json.dumps({
        "generated_by": "gen_raw.py",
        "seed": SEED,
        "gen_model_id": args.gen_model,
        "corpus_source": "danielrosehill/System-Prompt-Library",
        "n_prompts": len(prompts),
        "n_cells": len(raw_fingerprints),
        "ablation_sha256": digest,
        "elapsed_seconds": round(time.time() - t0, 1),
        "device": dev,
    }, indent=1))
    print(f"\nwrote {ab_path}  ({len(raw_fingerprints):,} cells)")
    (out / "ATTRIBUTION.txt").write_text(
        "SOURCE CORPUS\n"
        "System-Prompt-Library, by Daniel Rosehill.\n"
        "https://huggingface.co/datasets/danielrosehill/System-Prompt-Library\n"
        "Licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).\n\n"
        "WHAT WAS CHANGED\n"
        "Prompts were filtered (see grid.json / meta.json), then split into\n"
        "sentences. No prompt wording was rewritten. The numeric measurements\n"
        "in ablation.json are new, produced by running a language model over\n"
        "these prompts; they are released under CC BY 4.0 as well.\n\n"
        "MEASUREMENT INSTRUMENT\n"
        "HuggingFaceTB/SmolLM2-1.7B-Instruct, Apache-2.0.\n"
        "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct\n\n"
        "The corpus author and the model authors do not endorse this dataset\n"
        "or any work derived from it.\n")
    if ckpt_path.exists():
        ckpt_path.unlink()          # never ship the checkpoint
    print(f"sha256 {digest}")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
