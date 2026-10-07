#!/usr/bin/env python3
"""gen_raw.py - build the RAW dataset for "Predicting Retrieval Sufficiency
in RAG Pipelines".

WHAT THIS DOES
--------------
For a hand-written corpus of TOPICS (each a small knowledge base of short
passages) and hand-written QUERIES per topic, this script:

  1. Encodes every passage and every query with a real embedding model
     (BAAI/bge-small-en-v1.5) on GPU and computes real cosine-similarity
     retrieval: for each query, rank every passage IN THAT QUERY'S TOPIC'S
     CORPUS by similarity and record the actual measured rank of the
     query's designated gold passage (or that it is absent).
  2. Multiplies each query into several ITEMS by varying k (how many
     top-ranked passages are shown) -- see the ITEM MULTIPLIER comment
     below for exactly which queries get swept across which k values and
     why. Each item is a genuinely different retrieval-sufficiency
     situation, not a duplicated view of the same answer.
  3. Runs a real instruction model (HuggingFaceTB/SmolLM2-1.7B-Instruct) on
     GPU over (query + that item's top-k retrieved passages) and records a
     NUMERIC BEHAVIOUR FINGERPRINT of the actual generated answer (reusing
     the fingerprint() design proven in the "Prompt Edit Effect
     Attribution" build: hedge/refuse/IDK/length statistics, never the
     response text) -- one real generation per item, since a different k
     means a genuinely different context was actually shown to the model.
  4. Also measures, for every query, its cross-topic similarity to every
     OTHER topic's passages, to detect genuine cross-topic ambiguity by
     measurement rather than by asserting it at construction time.

Nothing here is asserted: every rank, every cross-topic similarity, and
every generation fingerprint is a real number computed by running the two
models. prepare.py reads these measurements to derive the gold decision
trace (diagnosis, action, reason_code) -- see its module docstring for the
label rule.

CORPUS AND QUERIES ARE HAND-WRITTEN IN THIS FILE, not model output. The
models are used only as instruments: the embedder produces similarity
rankings, the generator produces a real answer whose surface behaviour is
measured into a fixed-length numeric vector. No corpus text, query text, or
response text originates from a third party or from an LLM in the
"LLM-authored content" sense.

LICENCE
-------
Embedder: BAAI/bge-small-en-v1.5, MIT licence, ungated. Card states outputs
usable commercially with no restriction (verified 2026-09-16).
Generator: HuggingFaceTB/SmolLM2-1.7B-Instruct, Apache-2.0, ungated. Card
places no claim/restriction on generated outputs (verified in the
"Prompt Edit Effect Attribution" build, same model).
Corpus, queries, measured ranks, and generation fingerprints are original
work produced by this script and are released under the dataset's declared
licence, CC BY 4.0 -- attribution to both models above is included in
problem-description.md / dataset-description.md, worded identically to the
platform's declared licence field (register section -1).

DETERMINISM
-----------
Greedy decoding for the generator, fixed seeds, fixed dtype, sorted
iteration order throughout. Embeddings are deterministic given fixed
weights and inputs. Running this twice on the same hardware produces
byte-identical output; across different GPUs floating-point results can
differ marginally, and the shipped raw file is authoritative.

USAGE
-----
    pip install torch transformers sentence-transformers accelerate numpy
    python gen_raw.py --out ../raw            # full build
    python gen_raw.py --out ../raw --smoke    # 3 topics, quick check

On one A10G the full build is expected well under 1 hour: embedding 192
passages + 480 queries is near-instant, and the item multiplier produces
1,536 items, i.e. 1,536 short generations (~8-10 minutes at the per-prompt
throughput measured in the "Prompt Edit Effect Attribution" build, plus
model-load overhead).
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
SEED = 20260916
MAX_NEW_TOKENS = 80
GEN_BATCH = 16

# --------------------------------------------------------------------------
# ITEM MULTIPLIER - each hand-written query becomes MULTIPLE items by
# varying k (how many top-ranked passages are shown). This is only a real,
# non-duplicate multiplier for "gold_low_rank" queries: sweeping k across
# the gold passage's measured rank genuinely flips the diagnosis (k below
# the rank = insufficient, k at/above the rank = sufficient), producing a
# real per-item-different measured label. For "sufficient", "gold_absent",
# and "query_ambiguous" queries the diagnosis does not depend on k (gold is
# always top-ranked, always absent, or always cross-topic-confusable), so
# only a couple of k VARIANTS are kept as a mild stress check rather than
# a full sweep -- repeating the identical answer at every k would be the
# "repeated views share one answer" trap (see STATE.md item-definition
# discussion, same lesson as the "Prompt Edit Effect Attribution" build).
# --------------------------------------------------------------------------
K_SWEEP_GOLD_LOW_RANK = [2, 3, 4, 5, 6, 7]
K_VARIANTS_OTHER = [3, 5]

# --------------------------------------------------------------------------
# CORPUS - hand-written. Each topic is a small knowledge base. Every query
# in a topic names its own "gold_id" (the passage that actually answers it,
# a passage id from ANY topic's corpus, including "none" for gold_absent
# queries) and a CONSTRUCTION INTENT (the condition the query was written
# to trigger). The intent is a hypothesis, not the label -- prepare.py
# derives the actual label from the MEASURED rank, and an item whose
# measured rank contradicts its intent is relabeled by its true measured
# facts (see prepare.py LABEL RULE).
#
# Passage ids are unique across the WHOLE corpus (topic-prefixed), so
# cross-topic retrieval and cross-topic ambiguity are meaningful measurements.
# --------------------------------------------------------------------------
TOPICS = [
    {
        "topic_id": "leave_policy",
        "passages": {
            "leave_policy_p1": "Full-time employees accrue fifteen paid vacation days per calendar year, credited monthly at 1.25 days.",
            "leave_policy_p2": "Unused vacation days may be carried into the next year up to a maximum of five days; any remainder above five is forfeited on January 1.",
            "leave_policy_p3": "Sick leave is granted separately from vacation and does not require advance notice for absences under three consecutive days.",
            "leave_policy_p4": "Employees returning from more than ten consecutive sick days must submit a physician's clearance before resuming regular duties.",
            "leave_policy_p5": "Requests for vacation exceeding two consecutive weeks require manager approval at least thirty days in advance.",
            "leave_policy_p6": "Time taken for jury duty, for up to ten business days, is compensated at full pay and does not draw against vacation or sick balances.",
            "leave_policy_p7": "New parents are eligible for twelve weeks of paid leave, which may be split into two separate blocks within the first year.",
            "leave_policy_p8": "Bereavement leave provides up to five paid days for an immediate family member and two paid days for an extended relative.",
        },
        "queries": [
            {"text": "How many vacation days do full-time employees get per year?", "gold_id": "leave_policy_p1", "intent": "sufficient"},
            {"text": "What happens to vacation days I don't use by year end?", "gold_id": "leave_policy_p2", "intent": "sufficient"},
            {"text": "Do I need to warn my manager before calling in sick?", "gold_id": "leave_policy_p3", "intent": "sufficient"},
            {"text": "Is jury duty pay taken from my vacation balance?", "gold_id": "leave_policy_p6", "intent": "sufficient"},
            {"text": "How many weeks of paid leave do new parents receive?", "gold_id": "leave_policy_p7", "intent": "sufficient"},
            {"text": "How many paid days off are given for losing a close family member?", "gold_id": "leave_policy_p8", "intent": "sufficient"},
            {"text": "What paperwork clears me to come back after a long illness?", "gold_id": "leave_policy_p4", "intent": "gold_low_rank"},
            {"text": "How far ahead must I plan an extended holiday away from work?", "gold_id": "leave_policy_p5", "intent": "gold_low_rank"},
            {"text": "Can new parents split their time away into separate stretches?", "gold_id": "leave_policy_p7", "intent": "gold_low_rank"},
            {"text": "Is the allowance different depending on how closely related the deceased relative is?", "gold_id": "leave_policy_p8", "intent": "gold_low_rank"},
            {"text": "Does returning after a serious illness involve any sign-off from a doctor?", "gold_id": "leave_policy_p4", "intent": "gold_low_rank"},
            {"text": "Is a lengthy getaway something that needs advance clearance?", "gold_id": "leave_policy_p5", "intent": "gold_low_rank"},
            {"text": "Will my vacation balance affect my end-of-year bonus payout?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I transfer unused vacation days to a coworker?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the company offer a sabbatical after ten years of service?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Are religious holidays observed as automatic paid days off?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "How much time off do I get and how is it credited?", "gold_id": "leave_policy_p1", "intent": "query_ambiguous"},
            {"text": "What is the approval process for time away from work?", "gold_id": "leave_policy_p5", "intent": "query_ambiguous"},
            {"text": "What support is available around a major family life event?", "gold_id": "leave_policy_p7", "intent": "query_ambiguous"},
            {"text": "What happens with pay during a difficult personal circumstance?", "gold_id": "leave_policy_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "expense_policy",
        "passages": {
            "expense_policy_p1": "Meal expenses while travelling on company business are reimbursed up to sixty dollars per day with an itemized receipt.",
            "expense_policy_p2": "Reimbursement claims must be submitted within thirty days of the expense date or they will not be honored.",
            "expense_policy_p3": "Airfare should be booked at least fourteen days in advance through the approved travel portal whenever possible.",
            "expense_policy_p4": "Personal vehicle mileage for approved business travel is reimbursed at the current standard mileage rate of sixty-five cents per mile, published quarterly.",
            "expense_policy_p5": "Client entertainment expenses above two hundred dollars require pre-approval from a director before the event.",
            "expense_policy_p6": "Lost original receipts for amounts under seventy-five dollars may be substituted with a signed statement describing the expense, subject to finance review.",
            "expense_policy_p7": "Home internet costs for employees working remotely at least three days a week are reimbursed up to twenty-five dollars monthly.",
            "expense_policy_p8": "Corporate card statements must be reconciled in the expense system within seven days of the statement closing date.",
        },
        "queries": [
            {"text": "What is the daily cap on meal reimbursement while traveling?", "gold_id": "expense_policy_p1", "intent": "sufficient"},
            {"text": "How long do I have to file an expense claim?", "gold_id": "expense_policy_p2", "intent": "sufficient"},
            {"text": "How is mileage reimbursed for using my own car?", "gold_id": "expense_policy_p4", "intent": "sufficient"},
            {"text": "Do I need approval before taking a client to an expensive dinner?", "gold_id": "expense_policy_p5", "intent": "sufficient"},
            {"text": "How much of my home internet bill gets covered?", "gold_id": "expense_policy_p7", "intent": "sufficient"},
            {"text": "How soon after the statement closes must I reconcile my corporate card?", "gold_id": "expense_policy_p8", "intent": "sufficient"},
            {"text": "What do I do if I misplaced the paper receipt?", "gold_id": "expense_policy_p6", "intent": "gold_low_rank"},
            {"text": "When is the right time to lock in a flight for a work trip?", "gold_id": "expense_policy_p3", "intent": "gold_low_rank"},
            {"text": "Is there any support toward household connectivity costs for people working from home?", "gold_id": "expense_policy_p7", "intent": "gold_low_rank"},
            {"text": "What's the deadline for squaring up charges on my company card?", "gold_id": "expense_policy_p8", "intent": "gold_low_rank"},
            {"text": "What proof can stand in when a purchase slip has gone missing?", "gold_id": "expense_policy_p6", "intent": "gold_low_rank"},
            {"text": "How much lead time should booking air travel usually get?", "gold_id": "expense_policy_p3", "intent": "gold_low_rank"},
            {"text": "Can I expense a gift for a client's birthday?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Will finance reimburse a parking ticket incurred on a business trip?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a stipend for a home office desk setup?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can spouses join business trips at company expense?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the rules around spending money for work travel?", "gold_id": "expense_policy_p1", "intent": "query_ambiguous"},
            {"text": "What approvals do I need before spending company money?", "gold_id": "expense_policy_p5", "intent": "query_ambiguous"},
            {"text": "What ongoing costs does the company help cover for remote staff?", "gold_id": "expense_policy_p7", "intent": "query_ambiguous"},
            {"text": "What are my obligations around the company card?", "gold_id": "expense_policy_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "device_setup",
        "passages": {
            "device_setup_p1": "New laptops are pre-configured with 256-bit disk encryption enabled and cannot be disabled by the end user.",
            "device_setup_p2": "The corporate VPN client, which takes about five minutes to install, must be set up before accessing internal file shares from outside the office network.",
            "device_setup_p3": "Password resets for the single sign-on portal are self-service and take effect within two minutes.",
            "device_setup_p4": "Lost or stolen devices must be reported to IT security within one hour so the device can be remotely wiped.",
            "device_setup_p5": "Printer drivers for the third floor are installed automatically, taking under one minute, the first time a document is sent to that printer.",
            "device_setup_p6": "Software installation requests outside the approved catalog require a ticket reviewed within two business days.",
            "device_setup_p7": "Multi-factor authentication codes are valid for ninety seconds before a new code must be generated.",
            "device_setup_p8": "Docking stations for home use are loaned for a refundable fifty dollar deposit held on the employee's account.",
        },
        "queries": [
            {"text": "Is my new laptop's hard drive encrypted by default?", "gold_id": "device_setup_p1", "intent": "sufficient"},
            {"text": "What do I need installed to reach file shares from home?", "gold_id": "device_setup_p2", "intent": "sufficient"},
            {"text": "How fast does a self-service password reset apply?", "gold_id": "device_setup_p3", "intent": "sufficient"},
            {"text": "How long does it take to get software approved that isn't in the catalog?", "gold_id": "device_setup_p6", "intent": "sufficient"},
            {"text": "How long is a two-factor login code good for before it expires?", "gold_id": "device_setup_p7", "intent": "sufficient"},
            {"text": "Is there a deposit required to borrow a docking station?", "gold_id": "device_setup_p8", "intent": "sufficient"},
            {"text": "How urgently should I act if my company laptop goes missing?", "gold_id": "device_setup_p4", "intent": "gold_low_rank"},
            {"text": "Do I need to do anything special to print on the third floor?", "gold_id": "device_setup_p5", "intent": "gold_low_rank"},
            {"text": "Does the temporary code from an authenticator app time out quickly?", "gold_id": "device_setup_p7", "intent": "gold_low_rank"},
            {"text": "Can I take home a monitor stand and hub for occasional remote work?", "gold_id": "device_setup_p8", "intent": "gold_low_rank"},
            {"text": "What's the window for alerting security after equipment disappears?", "gold_id": "device_setup_p4", "intent": "gold_low_rank"},
            {"text": "Does the print queue on a certain floor need manual configuration?", "gold_id": "device_setup_p5", "intent": "gold_low_rank"},
            {"text": "Can I use my personal laptop to access internal systems?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does IT provide a loaner phone during international travel?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a way to dim the keyboard backlight remotely?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the company support dual-booting a different operating system?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "How do I get my computer set up and working properly?", "gold_id": "device_setup_p1", "intent": "query_ambiguous"},
            {"text": "What should I do about a security problem with my equipment?", "gold_id": "device_setup_p4", "intent": "query_ambiguous"},
            {"text": "What's involved in logging into company systems securely?", "gold_id": "device_setup_p7", "intent": "query_ambiguous"},
            {"text": "What accessories can I borrow for working outside the office?", "gold_id": "device_setup_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "onboarding",
        "passages": {
            "onboarding_p1": "New hires complete mandatory compliance training within their first two weeks of employment.",
            "onboarding_p2": "A designated buddy is assigned within the first two days to every new hire to answer day-to-day questions during the first month.",
            "onboarding_p3": "Payroll setup, including direct deposit details, must be completed within five business days of the start date to avoid a paper check on the first pay cycle.",
            "onboarding_p4": "Building access badges are issued on the first day and activated within two hours that same afternoon by security.",
            "onboarding_p5": "Benefits elections must be finalized within thirty days of the hire date or default coverage applies automatically.",
            "onboarding_p6": "Equipment requests submitted at least five days before the start date are typically ready and waiting at the new hire's desk.",
            "onboarding_p7": "A thirty-, sixty-, and ninety-day check-in is scheduled automatically between every new hire and their manager.",
            "onboarding_p8": "Emergency contact information must be entered into the HR system within the first three days of employment.",
        },
        "queries": [
            {"text": "How soon must compliance training be finished after starting?", "gold_id": "onboarding_p1", "intent": "sufficient"},
            {"text": "Who do I go to with basic questions in my first weeks?", "gold_id": "onboarding_p2", "intent": "sufficient"},
            {"text": "When does my building badge start working?", "gold_id": "onboarding_p4", "intent": "sufficient"},
            {"text": "What happens if I don't pick benefits in time?", "gold_id": "onboarding_p5", "intent": "sufficient"},
            {"text": "What check-ins are scheduled automatically with my manager early on?", "gold_id": "onboarding_p7", "intent": "sufficient"},
            {"text": "How soon do I need to add an emergency contact to the system?", "gold_id": "onboarding_p8", "intent": "sufficient"},
            {"text": "What do I need to sort out before my first paycheck arrives?", "gold_id": "onboarding_p3", "intent": "gold_low_rank"},
            {"text": "Will my desk already have a monitor if I ask early enough?", "gold_id": "onboarding_p6", "intent": "gold_low_rank"},
            {"text": "Is there a structured touchpoint schedule during my early months?", "gold_id": "onboarding_p7", "intent": "gold_low_rank"},
            {"text": "Who should be listed as a contact in case something happens to me at work?", "gold_id": "onboarding_p8", "intent": "gold_low_rank"},
            {"text": "What's required to make sure my salary lands correctly the first time?", "gold_id": "onboarding_p3", "intent": "gold_low_rank"},
            {"text": "Can gear I ask for ahead of time be sitting there when I arrive?", "gold_id": "onboarding_p6", "intent": "gold_low_rank"},
            {"text": "Will my previous employer's benefits transfer automatically?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a probation period before I'm a permanent employee?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the company send a welcome gift box before the start date?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is a background check redone annually after hiring?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What should I expect during my first days on the job?", "gold_id": "onboarding_p4", "intent": "query_ambiguous"},
            {"text": "What paperwork and setup do I need to take care of early on?", "gold_id": "onboarding_p3", "intent": "query_ambiguous"},
            {"text": "How does the company check in on how I'm settling in?", "gold_id": "onboarding_p7", "intent": "query_ambiguous"},
            {"text": "What personal information am I expected to provide right away?", "gold_id": "onboarding_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "returns_policy",
        "passages": {
            "returns_policy_p1": "Unopened items may be returned for a full refund within thirty days of the delivery date.",
            "returns_policy_p2": "Opened electronics are eligible for exchange only, not refund, within fourteen days of delivery.",
            "returns_policy_p3": "Return shipping is free, saving up to twelve dollars, when the return is due to a defect or an incorrect item shipped.",
            "returns_policy_p4": "Items marked final sale at checkout, typically discounted more than fifty percent, cannot be returned or exchanged under any circumstance.",
            "returns_policy_p5": "Refunds are issued to the original payment method within five to seven business days of the warehouse receiving the item.",
            "returns_policy_p6": "A return authorization number, valid for ten days once issued, must be requested online before shipping any item back to the warehouse.",
            "returns_policy_p7": "Store credit issued in place of a refund does not expire and can be combined with up to one additional promotional discount code.",
            "returns_policy_p8": "Seasonal holiday purchases made after November 1 may be returned through January 15 of the following year.",
        },
        "queries": [
            {"text": "How many days do I have to return something unopened for a refund?", "gold_id": "returns_policy_p1", "intent": "sufficient"},
            {"text": "Can I get a refund on an electronics item I already opened?", "gold_id": "returns_policy_p2", "intent": "sufficient"},
            {"text": "Do I pay for shipping if the wrong item was sent to me?", "gold_id": "returns_policy_p3", "intent": "sufficient"},
            {"text": "Can I return something that was marked final sale?", "gold_id": "returns_policy_p4", "intent": "sufficient"},
            {"text": "Does store credit ever expire if I choose that instead of cash back?", "gold_id": "returns_policy_p7", "intent": "sufficient"},
            {"text": "Is there an extended window for gifts bought around the holidays?", "gold_id": "returns_policy_p8", "intent": "sufficient"},
            {"text": "How long until I see my money back after sending an item in?", "gold_id": "returns_policy_p5", "intent": "gold_low_rank"},
            {"text": "What do I need to request before mailing a return package?", "gold_id": "returns_policy_p6", "intent": "gold_low_rank"},
            {"text": "Can a promo code be applied on top of a credit balance from a return?", "gold_id": "returns_policy_p7", "intent": "gold_low_rank"},
            {"text": "If I buy something as a present in December, when does the return clock run out?", "gold_id": "returns_policy_p8", "intent": "gold_low_rank"},
            {"text": "What's the typical turnaround before a refund posts back to my card?", "gold_id": "returns_policy_p5", "intent": "gold_low_rank"},
            {"text": "Is a code needed before I can drop a package back in the mail?", "gold_id": "returns_policy_p6", "intent": "gold_low_rank"},
            {"text": "Can I return a gift without the original receipt?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Do returns count against a loyalty rewards balance?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is curbside drop-off available for returns at physical stores?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I exchange an item for a completely different product category?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the rules for sending something back?", "gold_id": "returns_policy_p1", "intent": "query_ambiguous"},
            {"text": "How does the refund process actually work?", "gold_id": "returns_policy_p5", "intent": "query_ambiguous"},
            {"text": "What are my options besides getting cash back on a return?", "gold_id": "returns_policy_p7", "intent": "query_ambiguous"},
            {"text": "Does the time of year change how returns are handled?", "gold_id": "returns_policy_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "shipping_policy",
        "passages": {
            "shipping_policy_p1": "Standard shipping typically delivers within five to seven business days after the order is placed.",
            "shipping_policy_p2": "Expedited shipping guarantees delivery within two business days for an additional flat fee.",
            "shipping_policy_p3": "Orders over seventy-five dollars qualify for free standard shipping automatically at checkout.",
            "shipping_policy_p4": "International orders may be subject to customs duties, which can add up to twenty percent of the order value, collected by the carrier upon delivery.",
            "shipping_policy_p5": "A shipment is considered lost if tracking shows no movement for ten consecutive days.",
            "shipping_policy_p6": "Address changes requested within the first thirty minutes after an order ships still cannot be processed, and the package will go to the original address.",
            "shipping_policy_p7": "Split shipments occur automatically when items in one order ship from more than one warehouse.",
            "shipping_policy_p8": "Signature confirmation is required automatically on any single order totaling over three hundred dollars.",
        },
        "queries": [
            {"text": "How many business days does standard shipping usually take?", "gold_id": "shipping_policy_p1", "intent": "sufficient"},
            {"text": "How much extra does two-day shipping cost?", "gold_id": "shipping_policy_p2", "intent": "sufficient"},
            {"text": "What order total gets me free shipping?", "gold_id": "shipping_policy_p3", "intent": "sufficient"},
            {"text": "Will I owe customs fees on a package from another country?", "gold_id": "shipping_policy_p4", "intent": "sufficient"},
            {"text": "Why did my order arrive in more than one box?", "gold_id": "shipping_policy_p7", "intent": "sufficient"},
            {"text": "Will I have to sign for a large purchase when it arrives?", "gold_id": "shipping_policy_p8", "intent": "sufficient"},
            {"text": "At what point does a package officially count as missing?", "gold_id": "shipping_policy_p5", "intent": "gold_low_rank"},
            {"text": "Can I redirect a package after it has already left the warehouse?", "gold_id": "shipping_policy_p6", "intent": "gold_low_rank"},
            {"text": "Why would items from the same order not travel together?", "gold_id": "shipping_policy_p7", "intent": "gold_low_rank"},
            {"text": "Is someone required to be present to accept a pricier delivery?", "gold_id": "shipping_policy_p8", "intent": "gold_low_rank"},
            {"text": "How many days of stalled tracking before a package is treated as gone?", "gold_id": "shipping_policy_p5", "intent": "gold_low_rank"},
            {"text": "Once it's left the facility, can the drop-off location still be updated?", "gold_id": "shipping_policy_p6", "intent": "gold_low_rank"},
            {"text": "Does the carrier call before delivering a signature-required package?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is weekend delivery available in rural areas?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I schedule a specific delivery time window?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is carbon-neutral shipping offered as a checkout option?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "How does delivery timing work for my order?", "gold_id": "shipping_policy_p1", "intent": "query_ambiguous"},
            {"text": "What happens with a package after it leaves the warehouse?", "gold_id": "shipping_policy_p6", "intent": "query_ambiguous"},
            {"text": "Why might my order not come as one single package?", "gold_id": "shipping_policy_p7", "intent": "query_ambiguous"},
            {"text": "What extra steps apply to bigger or pricier orders at delivery?", "gold_id": "shipping_policy_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "warranty_policy",
        "passages": {
            "warranty_policy_p1": "Standard warranty coverage lasts twelve months from the original purchase date for manufacturing defects.",
            "warranty_policy_p2": "Extended warranty plans purchased at checkout add an additional twenty-four months of coverage.",
            "warranty_policy_p3": "Accidental damage, including drops and liquid exposure, is not covered under the standard warranty, though a paid accident protection add-on covers up to two incidents per year.",
            "warranty_policy_p4": "A proof of purchase, such as an order confirmation email, is required within fourteen days of filing to complete any warranty claim.",
            "warranty_policy_p5": "Warranty repairs are completed within ten business days of the item arriving at the service center.",
            "warranty_policy_p6": "Transferring a warranty to a new owner requires registering the transfer online within thirty days of resale.",
            "warranty_policy_p7": "A replacement unit ships within two business days for defects confirmed during the first thirty days of ownership.",
            "warranty_policy_p8": "Batteries and other consumable parts carry a shorter ninety-day coverage window separate from the main warranty term.",
        },
        "queries": [
            {"text": "How long does the standard warranty last?", "gold_id": "warranty_policy_p1", "intent": "sufficient"},
            {"text": "How much extra warranty time do I get if I buy the extended plan?", "gold_id": "warranty_policy_p2", "intent": "sufficient"},
            {"text": "Is water damage covered by the standard warranty?", "gold_id": "warranty_policy_p3", "intent": "sufficient"},
            {"text": "What proof do I need to make a warranty claim?", "gold_id": "warranty_policy_p4", "intent": "sufficient"},
            {"text": "How fast do I get a new unit if it's defective right out of the box?", "gold_id": "warranty_policy_p7", "intent": "sufficient"},
            {"text": "Is the battery covered for the same length of time as the rest of the device?", "gold_id": "warranty_policy_p8", "intent": "sufficient"},
            {"text": "How long does a repair take once it's received?", "gold_id": "warranty_policy_p5", "intent": "gold_low_rank"},
            {"text": "What do I do if I sell this item and the buyer wants the warranty?", "gold_id": "warranty_policy_p6", "intent": "gold_low_rank"},
            {"text": "If something's broken when it arrives, how quickly is a swap sent out?", "gold_id": "warranty_policy_p7", "intent": "gold_low_rank"},
            {"text": "Do wearable parts like batteries fall under a shorter protection period?", "gold_id": "warranty_policy_p8", "intent": "gold_low_rank"},
            {"text": "What's the usual timeline for getting a fixed item back?", "gold_id": "warranty_policy_p5", "intent": "gold_low_rank"},
            {"text": "Does coverage carry forward if the product changes hands?", "gold_id": "warranty_policy_p6", "intent": "gold_low_rank"},
            {"text": "Does the warranty cover a drop in resale value?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I get a loaner unit while mine is being repaired?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is warranty service available in a different country than where I bought it?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the warranty cover cosmetic scratches on the casing?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What's covered if something goes wrong with my purchase?", "gold_id": "warranty_policy_p1", "intent": "query_ambiguous"},
            {"text": "What do I need to do to get something fixed under warranty?", "gold_id": "warranty_policy_p4", "intent": "query_ambiguous"},
            {"text": "What happens if my item is faulty right from the start?", "gold_id": "warranty_policy_p7", "intent": "query_ambiguous"},
            {"text": "Are all parts of the product treated the same for coverage purposes?", "gold_id": "warranty_policy_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "subscription_billing",
        "passages": {
            "subscription_billing_p1": "Monthly subscriptions renew automatically on the same calendar day each month, with a reminder email sent three days prior, until cancelled.",
            "subscription_billing_p2": "Cancelling a subscription stops future renewals but does not refund any of the current thirty-day billing period.",
            "subscription_billing_p3": "Annual plans are billed once per year and include a fifteen percent discount versus monthly billing.",
            "subscription_billing_p4": "A failed payment triggers three retry attempts over seven days before the subscription is suspended.",
            "subscription_billing_p5": "Upgrading mid-cycle prorates the remaining days of the current plan against the new plan's cost, with the price difference charged within twenty-four hours.",
            "subscription_billing_p6": "Downgrading a plan takes effect at the start of the next billing cycle, up to thirty days later, not immediately.",
            "subscription_billing_p7": "A fourteen-day free trial is offered once per household and converts automatically to a paid plan unless cancelled.",
            "subscription_billing_p8": "Gift subscriptions are prepaid in full for a fixed twelve-month term and do not auto-renew when that term ends.",
        },
        "queries": [
            {"text": "When does my monthly subscription charge get renewed?", "gold_id": "subscription_billing_p1", "intent": "sufficient"},
            {"text": "Do I get money back for the rest of the month if I cancel?", "gold_id": "subscription_billing_p2", "intent": "sufficient"},
            {"text": "How much cheaper is the annual plan than paying monthly?", "gold_id": "subscription_billing_p3", "intent": "sufficient"},
            {"text": "When does a plan downgrade actually kick in?", "gold_id": "subscription_billing_p6", "intent": "sufficient"},
            {"text": "How long is the free trial before I get charged?", "gold_id": "subscription_billing_p7", "intent": "sufficient"},
            {"text": "Does a gifted subscription keep billing after the paid term is over?", "gold_id": "subscription_billing_p8", "intent": "sufficient"},
            {"text": "What happens after my card gets declined for the subscription?", "gold_id": "subscription_billing_p4", "intent": "gold_low_rank"},
            {"text": "How is the cost handled if I switch to a pricier plan partway through?", "gold_id": "subscription_billing_p5", "intent": "gold_low_rank"},
            {"text": "Is the trial period limited to one use per residence?", "gold_id": "subscription_billing_p7", "intent": "gold_low_rank"},
            {"text": "If someone buys me a subscription as a present, is it a one-time payment?", "gold_id": "subscription_billing_p8", "intent": "gold_low_rank"},
            {"text": "What's the retry process when a charge doesn't go through?", "gold_id": "subscription_billing_p4", "intent": "gold_low_rank"},
            {"text": "Does switching to a more expensive tier mean paying the full new price right away?", "gold_id": "subscription_billing_p5", "intent": "gold_low_rank"},
            {"text": "Can I share my subscription with a family member?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a student discount on the annual plan?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I pause billing temporarily instead of cancelling outright?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is a discount applied automatically for referring a friend?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "How does billing work for my plan over time?", "gold_id": "subscription_billing_p1", "intent": "query_ambiguous"},
            {"text": "What happens to my plan and price if I make a change?", "gold_id": "subscription_billing_p5", "intent": "query_ambiguous"},
            {"text": "What's the deal with trying the service before paying?", "gold_id": "subscription_billing_p7", "intent": "query_ambiguous"},
            {"text": "What are the options for giving someone else access to the service?", "gold_id": "subscription_billing_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "data_privacy",
        "passages": {
            "data_privacy_p1": "User account data is retained for two years after account closure before permanent deletion.",
            "data_privacy_p2": "Users may request a full export of their personal data through the account settings page, delivered within forty-eight hours.",
            "data_privacy_p3": "Third-party analytics providers receive only anonymized, aggregated usage statistics from groups of at least one hundred users, never personal identifiers.",
            "data_privacy_p4": "A data deletion request is processed within thirty days and confirmed to the user by email.",
            "data_privacy_p5": "Payment card details are tokenized within two seconds of entry and never stored in full on company servers.",
            "data_privacy_p6": "Marketing emails can be opted out of at any time via the unsubscribe link, which takes effect within ten days, without affecting account access.",
            "data_privacy_p7": "Location data is collected only while the mobile app is actively in use, purged after seven days, and never shared with advertisers.",
            "data_privacy_p8": "A security breach affecting personal data is disclosed to affected users within seventy-two hours of confirmation.",
        },
        "queries": [
            {"text": "How long is my data kept after I close my account?", "gold_id": "data_privacy_p1", "intent": "sufficient"},
            {"text": "How can I download a copy of my personal data?", "gold_id": "data_privacy_p2", "intent": "sufficient"},
            {"text": "Is my card number stored in full anywhere?", "gold_id": "data_privacy_p5", "intent": "sufficient"},
            {"text": "Will unsubscribing from emails lock me out of my account?", "gold_id": "data_privacy_p6", "intent": "sufficient"},
            {"text": "Does the app track my location when I'm not using it?", "gold_id": "data_privacy_p7", "intent": "sufficient"},
            {"text": "How quickly are users told if their data was exposed in a breach?", "gold_id": "data_privacy_p8", "intent": "sufficient"},
            {"text": "How long does it take to fully process a deletion request?", "gold_id": "data_privacy_p4", "intent": "gold_low_rank"},
            {"text": "What exactly do outside analytics companies see about me?", "gold_id": "data_privacy_p3", "intent": "gold_low_rank"},
            {"text": "Is my whereabouts ever passed along to ad companies?", "gold_id": "data_privacy_p7", "intent": "gold_low_rank"},
            {"text": "How fast do affected users get notified after a security incident is confirmed?", "gold_id": "data_privacy_p8", "intent": "gold_low_rank"},
            {"text": "What's the turnaround once I ask for my information to be erased?", "gold_id": "data_privacy_p4", "intent": "gold_low_rank"},
            {"text": "What kind of usage numbers get passed to outside measurement firms?", "gold_id": "data_privacy_p3", "intent": "gold_low_rank"},
            {"text": "Do you sell my browsing data to advertisers?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I see who has accessed my account recently?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is two-factor authentication mandatory for all accounts?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I restrict my data from being used to train new features?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What happens to my information over time?", "gold_id": "data_privacy_p1", "intent": "query_ambiguous"},
            {"text": "What control do I have over my personal information?", "gold_id": "data_privacy_p2", "intent": "query_ambiguous"},
            {"text": "How does the app handle sensitive information about where I am?", "gold_id": "data_privacy_p7", "intent": "query_ambiguous"},
            {"text": "What happens if something goes wrong with how my data is protected?", "gold_id": "data_privacy_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "remote_work",
        "passages": {
            "remote_work_p1": "Employees may work remotely up to three days per week with manager approval.",
            "remote_work_p2": "Fully remote positions require a documented home workspace inspection within the first thirty days.",
            "remote_work_p3": "Company equipment shipped to a remote address remains company property and must be returned within fourteen days of termination.",
            "remote_work_p4": "Remote employees must be reachable during core hours of ten to three in their local time zone.",
            "remote_work_p5": "International remote work beyond thirty consecutive days requires prior approval from both HR and the legal team due to tax implications.",
            "remote_work_p6": "Internet reimbursement for remote employees is capped at thirty dollars per month.",
            "remote_work_p7": "Remote employees are required to attend an in-person team gathering, typically lasting three days, at least twice per year.",
            "remote_work_p8": "A dedicated, distraction-free workspace of at least sixty square feet is required for any role approved as fully remote.",
        },
        "queries": [
            {"text": "How many days a week can I work from home?", "gold_id": "remote_work_p1", "intent": "sufficient"},
            {"text": "What are the required hours I need to be available remotely?", "gold_id": "remote_work_p4", "intent": "sufficient"},
            {"text": "How much is internet reimbursed for remote employees?", "gold_id": "remote_work_p6", "intent": "sufficient"},
            {"text": "Do I need approval to work from another country?", "gold_id": "remote_work_p5", "intent": "sufficient"},
            {"text": "How often do fully remote staff need to show up in person?", "gold_id": "remote_work_p7", "intent": "sufficient"},
            {"text": "Do I need a separate room set aside to qualify as fully remote?", "gold_id": "remote_work_p8", "intent": "sufficient"},
            {"text": "What happens to my equipment if I leave the company?", "gold_id": "remote_work_p3", "intent": "gold_low_rank"},
            {"text": "What's checked about my setup when I go fully remote?", "gold_id": "remote_work_p2", "intent": "gold_low_rank"},
            {"text": "Is periodic face-to-face attendance expected even for remote staff?", "gold_id": "remote_work_p7", "intent": "gold_low_rank"},
            {"text": "Does my home need a quiet, separate area just for work?", "gold_id": "remote_work_p8", "intent": "gold_low_rank"},
            {"text": "Who owns the gear that gets mailed to my house?", "gold_id": "remote_work_p3", "intent": "gold_low_rank"},
            {"text": "Does someone come check on my home setup before I go fully remote?", "gold_id": "remote_work_p2", "intent": "gold_low_rank"},
            {"text": "Can I expense a better chair for my home office?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a stipend for coworking space membership?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Are pets allowed to be visible during video calls?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a minimum internet speed requirement to work remotely?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the rules around working outside the office?", "gold_id": "remote_work_p1", "intent": "query_ambiguous"},
            {"text": "What do I need to sort out before going remote long-term?", "gold_id": "remote_work_p5", "intent": "query_ambiguous"},
            {"text": "How often does the team need to be physically together?", "gold_id": "remote_work_p7", "intent": "query_ambiguous"},
            {"text": "What does my home environment need to look like for this role?", "gold_id": "remote_work_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "performance_review",
        "passages": {
            "performance_review_p1": "Formal performance reviews are conducted twice yearly, in June and December, each taking about forty-five minutes.",
            "performance_review_p2": "Self-assessments must be submitted one week before the scheduled review meeting.",
            "performance_review_p3": "A rating of 'exceeds expectations' is required to be eligible for the annual merit pool, which distributes an average four percent raise.",
            "performance_review_p4": "Managers document at least one development goal per employee at every review cycle.",
            "performance_review_p5": "Disputing a review rating requires filing a written appeal with HR within fourteen days.",
            "performance_review_p6": "Peer feedback is collected anonymously from at least three colleagues before each cycle.",
            "performance_review_p7": "Employees promoted mid-cycle receive an off-schedule review sixty days after the promotion takes effect.",
            "performance_review_p8": "Ratings are calibrated across each department by a panel of at least three managers before being finalized.",
        },
        "queries": [
            {"text": "How often do formal performance reviews happen?", "gold_id": "performance_review_p1", "intent": "sufficient"},
            {"text": "When is my self-assessment due relative to the review?", "gold_id": "performance_review_p2", "intent": "sufficient"},
            {"text": "What rating do I need to qualify for a merit raise?", "gold_id": "performance_review_p3", "intent": "sufficient"},
            {"text": "How many colleagues give feedback before my review?", "gold_id": "performance_review_p6", "intent": "sufficient"},
            {"text": "When do I get reviewed if I was promoted outside the normal cycle?", "gold_id": "performance_review_p7", "intent": "sufficient"},
            {"text": "Is there a group that checks ratings are consistent across teams?", "gold_id": "performance_review_p8", "intent": "sufficient"},
            {"text": "How long do I have to challenge a rating I disagree with?", "gold_id": "performance_review_p5", "intent": "gold_low_rank"},
            {"text": "Does my manager have to set goals for me at each cycle?", "gold_id": "performance_review_p4", "intent": "gold_low_rank"},
            {"text": "Does getting a new title trigger an extra evaluation outside the normal schedule?", "gold_id": "performance_review_p7", "intent": "gold_low_rank"},
            {"text": "Is there a cross-check process so one manager isn't grading differently than another?", "gold_id": "performance_review_p8", "intent": "gold_low_rank"},
            {"text": "What's the process for pushing back on a score I think is unfair?", "gold_id": "performance_review_p5", "intent": "gold_low_rank"},
            {"text": "Am I guaranteed at least one growth objective written down each period?", "gold_id": "performance_review_p4", "intent": "gold_low_rank"},
            {"text": "Can I request a different manager to conduct my review?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Are performance ratings shared with future employers?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does a review ever result in a formal improvement plan?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can interns receive the same formal review as full-time staff?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "How does the review process work overall?", "gold_id": "performance_review_p1", "intent": "query_ambiguous"},
            {"text": "What happens if I don't agree with how I was reviewed?", "gold_id": "performance_review_p5", "intent": "query_ambiguous"},
            {"text": "What changes about my evaluation timeline if my role changes?", "gold_id": "performance_review_p7", "intent": "query_ambiguous"},
            {"text": "How is fairness maintained across different teams' evaluations?", "gold_id": "performance_review_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "travel_booking",
        "passages": {
            "travel_booking_p1": "Business travel must be booked at least seven days in advance through the corporate travel portal.",
            "travel_booking_p2": "Economy class is standard for flights under six hours; longer flights permit premium economy.",
            "travel_booking_p3": "A per diem of fifty dollars covers incidental expenses not otherwise reimbursed separately.",
            "travel_booking_p4": "Rental cars require prior approval at least three days ahead and must be booked through the preferred vendor list.",
            "travel_booking_p5": "Trip cancellations within forty-eight hours of departure require a manager's written justification.",
            "travel_booking_p6": "Frequent flyer miles earned on business trips belong to the traveling employee personally, even on the roughly six round trips taken per year.",
            "travel_booking_p7": "Hotel bookings are capped at two hundred dollars per night unless traveling to a designated high-cost city.",
            "travel_booking_p8": "Combining a business trip with personal travel is allowed for up to five additional days that are unpaid and self-funded.",
        },
        "queries": [
            {"text": "How far in advance must business flights be booked?", "gold_id": "travel_booking_p1", "intent": "sufficient"},
            {"text": "When am I allowed to book premium economy instead of economy?", "gold_id": "travel_booking_p2", "intent": "sufficient"},
            {"text": "How much is the daily incidental allowance while traveling?", "gold_id": "travel_booking_p3", "intent": "sufficient"},
            {"text": "Do I get to keep frequent flyer points from work trips?", "gold_id": "travel_booking_p6", "intent": "sufficient"},
            {"text": "What's the nightly hotel budget for a typical business trip?", "gold_id": "travel_booking_p7", "intent": "sufficient"},
            {"text": "Can I tack on personal vacation days to the end of a work trip?", "gold_id": "travel_booking_p8", "intent": "sufficient"},
            {"text": "What's required if I need to cancel a trip at the last minute?", "gold_id": "travel_booking_p5", "intent": "gold_low_rank"},
            {"text": "Where am I supposed to rent a car from for business travel?", "gold_id": "travel_booking_p4", "intent": "gold_low_rank"},
            {"text": "Does the nightly lodging limit change for pricier destinations?", "gold_id": "travel_booking_p7", "intent": "gold_low_rank"},
            {"text": "Am I on the hook for costs if I stay somewhere longer for leisure after work is done?", "gold_id": "travel_booking_p8", "intent": "gold_low_rank"},
            {"text": "What kind of sign-off is needed for a trip called off close to departure?", "gold_id": "travel_booking_p5", "intent": "gold_low_rank"},
            {"text": "Is there an approved list of companies for renting a vehicle?", "gold_id": "travel_booking_p4", "intent": "gold_low_rank"},
            {"text": "Can I extend a business trip for a personal vacation?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is travel insurance automatically included on business trips?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I book a train instead of flying for shorter trips?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a lounge access benefit for frequent business travelers?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the rules for arranging a work trip?", "gold_id": "travel_booking_p1", "intent": "query_ambiguous"},
            {"text": "What do I need to arrange transportation for business travel?", "gold_id": "travel_booking_p4", "intent": "query_ambiguous"},
            {"text": "What's the budget guidance for accommodations on a trip?", "gold_id": "travel_booking_p7", "intent": "query_ambiguous"},
            {"text": "What are the rules if I want to mix business and personal time on a trip?", "gold_id": "travel_booking_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "library_borrowing",
        "passages": {
            "library_borrowing_p1": "Standard books may be borrowed for three weeks with up to two renewals if no one else has requested them.",
            "library_borrowing_p2": "Overdue items accrue a fine of twenty-five cents per day up to a maximum of ten dollars per item.",
            "library_borrowing_p3": "Reference materials and rare collections, comprising about five percent of the catalog, may only be used inside the building and cannot be borrowed.",
            "library_borrowing_p4": "A lost item is billed at its replacement cost plus a fifteen dollar processing fee.",
            "library_borrowing_p5": "Interlibrary loan requests typically take seven to ten business days to arrive.",
            "library_borrowing_p6": "Borrowing privileges are suspended once unpaid fines exceed twenty dollars.",
            "library_borrowing_p7": "DVDs and audio materials circulate for one week only and cannot be renewed if a hold is waiting.",
            "library_borrowing_p8": "A library card is issued free to residents and must be renewed every three years with proof of address.",
        },
        "queries": [
            {"text": "How long can I keep a normal book before it's due?", "gold_id": "library_borrowing_p1", "intent": "sufficient"},
            {"text": "How much does it cost per day if a book is overdue?", "gold_id": "library_borrowing_p2", "intent": "sufficient"},
            {"text": "Can I take a rare archival document home?", "gold_id": "library_borrowing_p3", "intent": "sufficient"},
            {"text": "At what point do unpaid fines stop me from borrowing?", "gold_id": "library_borrowing_p6", "intent": "sufficient"},
            {"text": "How long can I keep a movie disc checked out?", "gold_id": "library_borrowing_p7", "intent": "sufficient"},
            {"text": "How often do I need to renew my library card?", "gold_id": "library_borrowing_p8", "intent": "sufficient"},
            {"text": "What do I owe if I never bring an item back?", "gold_id": "library_borrowing_p4", "intent": "gold_low_rank"},
            {"text": "How long should I expect to wait for a book from another branch?", "gold_id": "library_borrowing_p5", "intent": "gold_low_rank"},
            {"text": "Is extending the loan on a music CD possible if someone else wants it?", "gold_id": "library_borrowing_p7", "intent": "gold_low_rank"},
            {"text": "What do I need to bring to get my membership renewed?", "gold_id": "library_borrowing_p8", "intent": "gold_low_rank"},
            {"text": "What's the extra charge on top of replacement cost for a lost book?", "gold_id": "library_borrowing_p4", "intent": "gold_low_rank"},
            {"text": "How long does borrowing from a different branch's collection generally take?", "gold_id": "library_borrowing_p5", "intent": "gold_low_rank"},
            {"text": "Can I recommend a new book for the library to purchase?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a quiet study room I can reserve?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the library offer free printing for cardholders?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I access digital magazines with my library account?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the rules for borrowing items from the library?", "gold_id": "library_borrowing_p1", "intent": "query_ambiguous"},
            {"text": "What happens if something goes wrong with an item I borrowed?", "gold_id": "library_borrowing_p4", "intent": "query_ambiguous"},
            {"text": "What are the checkout terms for non-book media?", "gold_id": "library_borrowing_p7", "intent": "query_ambiguous"},
            {"text": "What do I need to do to keep using the library long-term?", "gold_id": "library_borrowing_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "gym_membership",
        "passages": {
            "gym_membership_p1": "Monthly memberships can be paused for up to two months per calendar year without cancellation fees.",
            "gym_membership_p2": "Guest passes are limited to two per member per month and require the member to be present.",
            "gym_membership_p3": "Cancelling a membership requires thirty days written notice submitted at the front desk or online.",
            "gym_membership_p4": "Personal training sessions expire six months after purchase if unused.",
            "gym_membership_p5": "Locker rentals are billed separately at ten dollars per month with a refundable key deposit.",
            "gym_membership_p6": "Members under eighteen require a parent or guardian co-signature on the membership agreement.",
            "gym_membership_p7": "A one-time enrollment fee of forty dollars applies to all new memberships except during promotional periods.",
            "gym_membership_p8": "Group fitness classes can be reserved up to seven days in advance and cancelled without penalty until two hours before start.",
        },
        "queries": [
            {"text": "How many months a year can I pause my membership?", "gold_id": "gym_membership_p1", "intent": "sufficient"},
            {"text": "How many guest passes can I use each month?", "gold_id": "gym_membership_p2", "intent": "sufficient"},
            {"text": "How much notice do I need to give to cancel?", "gold_id": "gym_membership_p3", "intent": "sufficient"},
            {"text": "Do unused personal training sessions ever expire?", "gold_id": "gym_membership_p4", "intent": "sufficient"},
            {"text": "Is there a one-time fee for signing up?", "gold_id": "gym_membership_p7", "intent": "sufficient"},
            {"text": "How far ahead can I book a spot in a fitness class?", "gold_id": "gym_membership_p8", "intent": "sufficient"},
            {"text": "What's the extra cost if I want a locker?", "gold_id": "gym_membership_p5", "intent": "gold_low_rank"},
            {"text": "What's needed on the paperwork for a minor joining the gym?", "gold_id": "gym_membership_p6", "intent": "gold_low_rank"},
            {"text": "Is there an upfront charge that's waived sometimes during special offers?", "gold_id": "gym_membership_p7", "intent": "gold_low_rank"},
            {"text": "How close to class time can I back out without being penalized?", "gold_id": "gym_membership_p8", "intent": "gold_low_rank"},
            {"text": "Is there a monthly charge for storage space at the facility?", "gold_id": "gym_membership_p5", "intent": "gold_low_rank"},
            {"text": "Does a teenage member need an adult's signature to join?", "gold_id": "gym_membership_p6", "intent": "gold_low_rank"},
            {"text": "Does the membership include access to group fitness classes?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a discount for signing up with a friend?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the gym offer childcare services during workouts?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a corporate wellness discount available?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the terms of my gym membership?", "gold_id": "gym_membership_p1", "intent": "query_ambiguous"},
            {"text": "What extra costs might come up with my membership?", "gold_id": "gym_membership_p5", "intent": "query_ambiguous"},
            {"text": "What should I know before signing up as a new member?", "gold_id": "gym_membership_p7", "intent": "query_ambiguous"},
            {"text": "How does scheduling work for the classes offered here?", "gold_id": "gym_membership_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "apartment_lease",
        "passages": {
            "apartment_lease_p1": "The standard lease term is twelve months with automatic month-to-month renewal after expiration.",
            "apartment_lease_p2": "A security deposit equal to one month's rent is refundable within thirty days of move-out, minus damages.",
            "apartment_lease_p3": "Subletting requires written landlord approval, granted or denied within ten days, and a background check on the new occupant.",
            "apartment_lease_p4": "Late rent incurs a five percent fee if not received within five days of the due date.",
            "apartment_lease_p5": "Tenants must give sixty days written notice before vacating at the end of the lease term.",
            "apartment_lease_p6": "Pets are permitted with a one-time two hundred dollar non-refundable pet fee per animal.",
            "apartment_lease_p7": "Guests staying longer than fourteen consecutive days must be added to the lease as an occupant.",
            "apartment_lease_p8": "Maintenance requests deemed non-emergency are addressed by staff within seventy-two hours of submission.",
        },
        "queries": [
            {"text": "How long is the standard lease term?", "gold_id": "apartment_lease_p1", "intent": "sufficient"},
            {"text": "When do I get my security deposit back after moving out?", "gold_id": "apartment_lease_p2", "intent": "sufficient"},
            {"text": "What fee applies if rent is a few days late?", "gold_id": "apartment_lease_p4", "intent": "sufficient"},
            {"text": "Is there a fee for having a pet in the apartment?", "gold_id": "apartment_lease_p6", "intent": "sufficient"},
            {"text": "How long can a visitor stay before they need to be added to the lease?", "gold_id": "apartment_lease_p7", "intent": "sufficient"},
            {"text": "How quickly does a routine repair request get handled?", "gold_id": "apartment_lease_p8", "intent": "sufficient"},
            {"text": "How much advance notice do I owe before I move out for good?", "gold_id": "apartment_lease_p5", "intent": "gold_low_rank"},
            {"text": "What's involved if I want someone else to take over my lease temporarily?", "gold_id": "apartment_lease_p3", "intent": "gold_low_rank"},
            {"text": "At what point does a long-term visitor count as an actual resident?", "gold_id": "apartment_lease_p7", "intent": "gold_low_rank"},
            {"text": "What's the typical wait for a non-urgent repair to be looked at?", "gold_id": "apartment_lease_p8", "intent": "gold_low_rank"},
            {"text": "How much heads-up is required before ending the lease at term's end?", "gold_id": "apartment_lease_p5", "intent": "gold_low_rank"},
            {"text": "Does someone need to be screened before staying in place of the original tenant?", "gold_id": "apartment_lease_p3", "intent": "gold_low_rank"},
            {"text": "Is renters insurance mandatory under this lease?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I paint the walls a different color?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is a parking spot included with every unit?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the lease allow smoking on the balcony?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the terms of my apartment lease?", "gold_id": "apartment_lease_p1", "intent": "query_ambiguous"},
            {"text": "What do I need to do before leaving the apartment?", "gold_id": "apartment_lease_p5", "intent": "query_ambiguous"},
            {"text": "What are the rules around having people stay with me?", "gold_id": "apartment_lease_p7", "intent": "query_ambiguous"},
            {"text": "How are issues in the unit supposed to get resolved?", "gold_id": "apartment_lease_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "insurance_claims",
        "passages": {
            "insurance_claims_p1": "Claims must be filed within sixty days of the incident to be eligible for coverage.",
            "insurance_claims_p2": "A deductible of five hundred dollars applies per claim before coverage begins.",
            "insurance_claims_p3": "Approved claims are paid out within ten business days of final documentation review.",
            "insurance_claims_p4": "Filing three or more claims within a policy year may result in a premium increase at renewal.",
            "insurance_claims_p5": "A claims adjuster typically contacts the policyholder within two business days of filing.",
            "insurance_claims_p6": "Disputed claim decisions can be appealed in writing within thirty days of the denial letter.",
            "insurance_claims_p7": "Supporting documentation, such as photographs or receipts, must be uploaded within fourteen days of filing.",
            "insurance_claims_p8": "A total loss determination is made when repair costs exceed seventy-five percent of the item's assessed value.",
        },
        "queries": [
            {"text": "How soon after an incident must I file a claim?", "gold_id": "insurance_claims_p1", "intent": "sufficient"},
            {"text": "What's the deductible amount before coverage kicks in?", "gold_id": "insurance_claims_p2", "intent": "sufficient"},
            {"text": "How quickly does an adjuster usually reach out?", "gold_id": "insurance_claims_p5", "intent": "sufficient"},
            {"text": "How long do I have to appeal if my claim is denied?", "gold_id": "insurance_claims_p6", "intent": "sufficient"},
            {"text": "How soon after filing do I need to upload photos or receipts?", "gold_id": "insurance_claims_p7", "intent": "sufficient"},
            {"text": "At what repair cost threshold is something declared a total loss?", "gold_id": "insurance_claims_p8", "intent": "sufficient"},
            {"text": "How long after approval until the payout actually arrives?", "gold_id": "insurance_claims_p3", "intent": "gold_low_rank"},
            {"text": "Does filing several claims in a year affect what I pay later?", "gold_id": "insurance_claims_p4", "intent": "gold_low_rank"},
            {"text": "Is there a deadline for submitting evidence to back up my claim?", "gold_id": "insurance_claims_p7", "intent": "gold_low_rank"},
            {"text": "At what point does the insurer just write the item off entirely instead of fixing it?", "gold_id": "insurance_claims_p8", "intent": "gold_low_rank"},
            {"text": "How much time passes between approval and actually getting paid?", "gold_id": "insurance_claims_p3", "intent": "gold_low_rank"},
            {"text": "Will my premium go up next year if I file too often?", "gold_id": "insurance_claims_p4", "intent": "gold_low_rank"},
            {"text": "Does the policy cover claims from a natural disaster?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I add a second driver to an existing claim?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is roadside assistance included with every policy?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I switch insurance providers mid-claim?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What's the process for handling a claim?", "gold_id": "insurance_claims_p1", "intent": "query_ambiguous"},
            {"text": "What happens with the money after a claim is filed?", "gold_id": "insurance_claims_p3", "intent": "query_ambiguous"},
            {"text": "What am I responsible for submitting to support my case?", "gold_id": "insurance_claims_p7", "intent": "query_ambiguous"},
            {"text": "How does the insurer decide whether to repair or replace something?", "gold_id": "insurance_claims_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "course_enrollment",
        "passages": {
            "course_enrollment_p1": "Course registration opens two weeks before each term and closes on the first day of classes.",
            "course_enrollment_p2": "Dropping a course after the second week, but before week ten, results in a 'W' grade on the transcript instead of removal.",
            "course_enrollment_p3": "Waitlisted students, up to a maximum of fifteen per course, are enrolled automatically as seats open, in the order they joined the waitlist.",
            "course_enrollment_p4": "Prerequisite requirements are checked automatically at registration and can be overridden only by department approval within five business days.",
            "course_enrollment_p5": "Full tuition refunds are issued only for drops completed within the first week of the term; after that, refunds drop to fifty percent.",
            "course_enrollment_p6": "Auditing a course costs twenty-five percent of standard tuition, requires instructor permission, and excludes the student from receiving a grade.",
            "course_enrollment_p7": "A maximum of eighteen credit hours may be taken per term without approval from an academic advisor.",
            "course_enrollment_p8": "Students on academic probation are limited to twelve credit hours until their standing is restored.",
        },
        "queries": [
            {"text": "When does registration open before a new term?", "gold_id": "course_enrollment_p1", "intent": "sufficient"},
            {"text": "What shows on my transcript if I drop late?", "gold_id": "course_enrollment_p2", "intent": "sufficient"},
            {"text": "How does the waitlist decide who gets in next?", "gold_id": "course_enrollment_p3", "intent": "sufficient"},
            {"text": "Do I need permission to sit in on a class without a grade?", "gold_id": "course_enrollment_p6", "intent": "sufficient"},
            {"text": "How many credit hours can I take without special sign-off?", "gold_id": "course_enrollment_p7", "intent": "sufficient"},
            {"text": "How many credits am I capped at if I'm on academic probation?", "gold_id": "course_enrollment_p8", "intent": "sufficient"},
            {"text": "By when must I drop to get all my tuition back?", "gold_id": "course_enrollment_p5", "intent": "gold_low_rank"},
            {"text": "What happens if I try to register without having taken the prerequisite?", "gold_id": "course_enrollment_p4", "intent": "gold_low_rank"},
            {"text": "Is there a ceiling on course load before an advisor needs to weigh in?", "gold_id": "course_enrollment_p7", "intent": "gold_low_rank"},
            {"text": "Does a poor academic standing restrict how many classes I can take?", "gold_id": "course_enrollment_p8", "intent": "gold_low_rank"},
            {"text": "What's the cutoff to still get my full money back on a dropped class?", "gold_id": "course_enrollment_p5", "intent": "gold_low_rank"},
            {"text": "Can a missing prerequisite be waived by someone in the department?", "gold_id": "course_enrollment_p4", "intent": "gold_low_rank"},
            {"text": "Can I enroll in two sections of the same course at once?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a fee for registering late?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I take a course pass/fail instead of for a letter grade?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Do online courses have a different registration deadline?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What's the process for signing up for classes?", "gold_id": "course_enrollment_p1", "intent": "query_ambiguous"},
            {"text": "What are my options if I want to leave a course?", "gold_id": "course_enrollment_p5", "intent": "query_ambiguous"},
            {"text": "What limits exist on how much I can take on in a term?", "gold_id": "course_enrollment_p7", "intent": "query_ambiguous"},
            {"text": "How does my academic record affect what I'm allowed to enroll in?", "gold_id": "course_enrollment_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "parking_permit",
        "passages": {
            "parking_permit_p1": "Annual resident parking permits cost forty-five dollars and expire on March 31 each year regardless of purchase date.",
            "parking_permit_p2": "A vehicle may be registered to only one parking permit at a time, and permits are non-transferable between vehicles.",
            "parking_permit_p3": "Visitor parking passes are valid for a maximum of seventy-two consecutive hours before requiring renewal.",
            "parking_permit_p4": "Unpaid parking citations exceeding one hundred dollars will block permit renewal until the balance is cleared.",
            "parking_permit_p5": "Permit holders receive one free replacement decal per year if the original is damaged during removal.",
            "parking_permit_p6": "Overnight street parking without a valid permit is limited to two occurrences per thirty-day period before towing is authorized.",
            "parking_permit_p7": "Reserved garage spaces require a separate waitlist application and typically open up within six to nine months.",
            "parking_permit_p8": "Electric vehicle charging spaces are restricted to permit holders actively charging, with a four-hour limit per session.",
        },
        "queries": [
            {"text": "How much does a yearly resident parking permit cost?", "gold_id": "parking_permit_p1", "intent": "sufficient"},
            {"text": "Can I use the same permit on two different cars?", "gold_id": "parking_permit_p2", "intent": "sufficient"},
            {"text": "How long is a visitor parking pass good for?", "gold_id": "parking_permit_p3", "intent": "sufficient"},
            {"text": "Will old unpaid tickets stop me from renewing my permit?", "gold_id": "parking_permit_p4", "intent": "sufficient"},
            {"text": "How long can I charge my electric car before I need to move it?", "gold_id": "parking_permit_p8", "intent": "sufficient"},
            {"text": "How long does the wait usually run for a reserved garage spot?", "gold_id": "parking_permit_p7", "intent": "sufficient"},
            {"text": "What if my permit sticker gets torn when I peel it off?", "gold_id": "parking_permit_p5", "intent": "gold_low_rank"},
            {"text": "How many nights can my car sit on the street without a sticker before it's hauled away?", "gold_id": "parking_permit_p6", "intent": "gold_low_rank"},
            {"text": "Is a free do-over sticker offered if mine gets ruined?", "gold_id": "parking_permit_p5", "intent": "gold_low_rank"},
            {"text": "What's the towing threshold for parking overnight without proper authorization?", "gold_id": "parking_permit_p6", "intent": "gold_low_rank"},
            {"text": "Is there a queue to join for a covered parking space?", "gold_id": "parking_permit_p7", "intent": "gold_low_rank"},
            {"text": "How many hours am I allotted at a charging spot before it's someone else's turn?", "gold_id": "parking_permit_p8", "intent": "gold_low_rank"},
            {"text": "Does the permit work in neighboring towns as well?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a discounted permit rate for senior citizens?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I get a refund on a permit if I sell my car mid-year?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Are motorcycles exempt from needing a permit?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What do I need to know about getting a permit for my car?", "gold_id": "parking_permit_p1", "intent": "query_ambiguous"},
            {"text": "What are the restrictions on parking overnight in the neighborhood?", "gold_id": "parking_permit_p6", "intent": "query_ambiguous"},
            {"text": "What's involved in getting access to covered or specialized parking?", "gold_id": "parking_permit_p7", "intent": "query_ambiguous"},
            {"text": "What happens if my sticker gets damaged or I have outstanding fines?", "gold_id": "parking_permit_p4", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "pet_adoption",
        "passages": {
            "pet_adoption_p1": "Adoption fees are one hundred twenty dollars for dogs and eighty dollars for cats, covering initial vaccinations.",
            "pet_adoption_p2": "All adopted animals must be spayed or neutered within thirty days if not already altered at the time of adoption.",
            "pet_adoption_p3": "A two-week trial period allows an adopter to return the animal for a full refund of the adoption fee.",
            "pet_adoption_p4": "Adopters must be at least twenty-one years old and provide government-issued photo identification.",
            "pet_adoption_p5": "Microchipping, valued at twenty dollars, is included in the adoption fee and is completed before the animal leaves the shelter.",
            "pet_adoption_p6": "A single home visit is required within seven days before finalizing the adoption of any animal listed as a special-needs case.",
            "pet_adoption_p7": "Adopted rabbits and other small animals come with a starter supply kit lasting approximately two weeks.",
            "pet_adoption_p8": "Post-adoption behavioral consultations are offered free of charge for the first ninety days after adoption.",
        },
        "queries": [
            {"text": "How much does it cost to adopt a dog?", "gold_id": "pet_adoption_p1", "intent": "sufficient"},
            {"text": "Do I have to get my new pet fixed after adopting?", "gold_id": "pet_adoption_p2", "intent": "sufficient"},
            {"text": "Can I bring an animal back if it doesn't work out?", "gold_id": "pet_adoption_p3", "intent": "sufficient"},
            {"text": "Is there a minimum age to adopt an animal?", "gold_id": "pet_adoption_p4", "intent": "sufficient"},
            {"text": "Is microchipping included when I adopt?", "gold_id": "pet_adoption_p5", "intent": "sufficient"},
            {"text": "Do small pets like rabbits come with any starter supplies?", "gold_id": "pet_adoption_p7", "intent": "sufficient"},
            {"text": "Does someone check out my house before I can take home a special-needs animal?", "gold_id": "pet_adoption_p6", "intent": "gold_low_rank"},
            {"text": "Is behavior coaching available at no cost after I take the animal home?", "gold_id": "pet_adoption_p8", "intent": "gold_low_rank"},
            {"text": "Is an in-person property check part of adopting a pet with extra medical needs?", "gold_id": "pet_adoption_p6", "intent": "gold_low_rank"},
            {"text": "How long after bringing my pet home can I still get free training advice?", "gold_id": "pet_adoption_p8", "intent": "gold_low_rank"},
            {"text": "How much food and bedding comes bundled with a small animal adoption?", "gold_id": "pet_adoption_p7", "intent": "gold_low_rank"},
            {"text": "What identification do I need to bring to prove I'm old enough?", "gold_id": "pet_adoption_p4", "intent": "gold_low_rank"},
            {"text": "Can I adopt a pet on behalf of a family member who lives elsewhere?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is pet insurance included with the adoption fee?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Are there discounted fees during holiday adoption events?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I foster an animal before committing to adopt it?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What should I expect to pay and receive when adopting?", "gold_id": "pet_adoption_p1", "intent": "query_ambiguous"},
            {"text": "What ongoing support is available after I take a pet home?", "gold_id": "pet_adoption_p8", "intent": "query_ambiguous"},
            {"text": "What extra steps apply for animals with special conditions?", "gold_id": "pet_adoption_p6", "intent": "query_ambiguous"},
            {"text": "What paperwork or requirements do I need to meet to adopt?", "gold_id": "pet_adoption_p4", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "software_licensing",
        "passages": {
            "software_licensing_p1": "A single-user license permits installation on up to two personal devices owned by the same license holder.",
            "software_licensing_p2": "Site licenses for organizations require a minimum purchase of twenty-five seats billed annually.",
            "software_licensing_p3": "License keys are deactivated automatically after ninety days of inactivity on all registered devices.",
            "software_licensing_p4": "Transferring a license to a new owner requires a written request submitted at least ten business days beforehand.",
            "software_licensing_p5": "Educational licenses are offered at a seventy percent discount and require annual re-verification of student status.",
            "software_licensing_p6": "Perpetual licenses include one year of updates; continued updates after that require a separate maintenance subscription.",
            "software_licensing_p7": "Offline activation codes remain valid for thirty days before requiring the device to reconnect and re-verify online.",
            "software_licensing_p8": "Violating the license agreement's usage terms results in immediate revocation without a refund of the remaining twelve-month term fees.",
        },
        "queries": [
            {"text": "How many devices can I install a single-user license on?", "gold_id": "software_licensing_p1", "intent": "sufficient"},
            {"text": "What's the minimum seat count for an organization-wide license?", "gold_id": "software_licensing_p2", "intent": "sufficient"},
            {"text": "How long can a license sit unused before it deactivates?", "gold_id": "software_licensing_p3", "intent": "sufficient"},
            {"text": "How big a discount do students get on a license?", "gold_id": "software_licensing_p5", "intent": "sufficient"},
            {"text": "Do perpetual licenses keep receiving updates forever?", "gold_id": "software_licensing_p6", "intent": "sufficient"},
            {"text": "How long does an offline activation code work before I need to reconnect?", "gold_id": "software_licensing_p7", "intent": "sufficient"},
            {"text": "How much advance notice is needed to move a license to someone else?", "gold_id": "software_licensing_p4", "intent": "gold_low_rank"},
            {"text": "What happens to my access if I break the terms of the agreement?", "gold_id": "software_licensing_p8", "intent": "gold_low_rank"},
            {"text": "Is there a lead time requirement before ownership of a license can change hands?", "gold_id": "software_licensing_p4", "intent": "gold_low_rank"},
            {"text": "Will breaking the usage rules get my access pulled without any money back?", "gold_id": "software_licensing_p8", "intent": "gold_low_rank"},
            {"text": "Does a student discount need to be reconfirmed each year?", "gold_id": "software_licensing_p5", "intent": "gold_low_rank"},
            {"text": "Is there a point where a one-time purchase stops including free updates?", "gold_id": "software_licensing_p6", "intent": "gold_low_rank"},
            {"text": "Can I get a refund if I stop using the software after a year?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a free trial available before purchasing a license?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the license include access to a mobile companion app?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can nonprofits apply for a special discounted license tier?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the limits on how and where I can install this software?", "gold_id": "software_licensing_p1", "intent": "query_ambiguous"},
            {"text": "What happens to my license under certain conditions over time?", "gold_id": "software_licensing_p3", "intent": "query_ambiguous"},
            {"text": "What are the consequences of not following the license rules?", "gold_id": "software_licensing_p8", "intent": "query_ambiguous"},
            {"text": "What ongoing obligations come with a discounted license type?", "gold_id": "software_licensing_p5", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "conference_registration",
        "passages": {
            "conference_registration_p1": "Early bird registration pricing of three hundred dollars is available until sixty days before the event start date.",
            "conference_registration_p2": "Cancellations made more than thirty days before the event receive a full refund minus a fifty dollar processing fee.",
            "conference_registration_p3": "Registered attendees may transfer their pass to a colleague at no charge up until one week before the event.",
            "conference_registration_p4": "Workshop add-ons, priced at twenty-five dollars each, must be selected at the time of registration and cannot be added after badges are printed.",
            "conference_registration_p5": "Student registration requires a valid enrollment verification and is priced at fifty percent of the standard attendee rate.",
            "conference_registration_p6": "Group registrations of five or more attendees from the same organization receive a fifteen percent discount.",
            "conference_registration_p7": "Badge pickup on-site closes one hour before the opening keynote and does not reopen until the first break.",
            "conference_registration_p8": "Virtual attendance passes include access to session recordings for ninety days after the event concludes.",
        },
        "queries": [
            {"text": "How much does early registration cost and when does it end?", "gold_id": "conference_registration_p1", "intent": "sufficient"},
            {"text": "What refund do I get if I cancel my registration?", "gold_id": "conference_registration_p2", "intent": "sufficient"},
            {"text": "Can I give my conference pass to someone else?", "gold_id": "conference_registration_p3", "intent": "sufficient"},
            {"text": "Do I need to prove I'm a student to get the discounted rate?", "gold_id": "conference_registration_p5", "intent": "sufficient"},
            {"text": "Is there a discount for registering as a group from one company?", "gold_id": "conference_registration_p6", "intent": "sufficient"},
            {"text": "How long can I watch recorded sessions after a virtual pass event ends?", "gold_id": "conference_registration_p8", "intent": "sufficient"},
            {"text": "Can I sign up for a workshop after my badge has already been printed?", "gold_id": "conference_registration_p4", "intent": "gold_low_rank"},
            {"text": "What time does the check-in desk stop serving people before the show starts?", "gold_id": "conference_registration_p7", "intent": "gold_low_rank"},
            {"text": "Is it too late to add optional sessions once badges are already made?", "gold_id": "conference_registration_p4", "intent": "gold_low_rank"},
            {"text": "When does the badge counter shut for the morning before things kick off?", "gold_id": "conference_registration_p7", "intent": "gold_low_rank"},
            {"text": "How late before the event can I hand off my registration to a coworker?", "gold_id": "conference_registration_p3", "intent": "gold_low_rank"},
            {"text": "What's the cutoff timing for getting most of my money back on a cancellation?", "gold_id": "conference_registration_p2", "intent": "gold_low_rank"},
            {"text": "Is parking included with a conference registration?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I pay for registration in installments?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a networking dinner included in the registration price?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Are meals provided for attendees during the conference?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What should I know about pricing and signing up for the event?", "gold_id": "conference_registration_p1", "intent": "query_ambiguous"},
            {"text": "What are my options if my plans change after registering?", "gold_id": "conference_registration_p2", "intent": "query_ambiguous"},
            {"text": "What's the process for picking up my materials at the venue?", "gold_id": "conference_registration_p7", "intent": "query_ambiguous"},
            {"text": "What do remote attendees get access to compared to in-person?", "gold_id": "conference_registration_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "car_rental",
        "passages": {
            "car_rental_p1": "Renters must be at least twenty-five years old, or pay a daily young-driver surcharge of fifteen dollars if between twenty-one and twenty-four.",
            "car_rental_p2": "A full tank of fuel must be returned at drop-off or a refueling charge of ninety dollars applies.",
            "car_rental_p3": "Collision damage waiver coverage adds twenty dollars per day and eliminates the standard one-thousand-dollar deductible.",
            "car_rental_p4": "Returning a vehicle more than twenty-nine minutes late triggers a full additional day's rental charge.",
            "car_rental_p5": "Additional drivers can be added for eight dollars per day each, up to a maximum of three extra drivers.",
            "car_rental_p6": "Crossing an international border with a rental vehicle requires written authorization submitted at least five days in advance.",
            "car_rental_p7": "Roadside assistance for a breakdown is dispatched within forty-five minutes in metro areas under the standard rental agreement.",
            "car_rental_p8": "Cancelling a reservation more than twenty-four hours before pickup incurs no penalty; later cancellations forfeit a one-day rate.",
        },
        "queries": [
            {"text": "Is there an extra charge for renters under twenty-five?", "gold_id": "car_rental_p1", "intent": "sufficient"},
            {"text": "What happens if I don't fill the tank before returning the car?", "gold_id": "car_rental_p2", "intent": "sufficient"},
            {"text": "How much does the damage waiver cost per day?", "gold_id": "car_rental_p3", "intent": "sufficient"},
            {"text": "How much does it cost to add another driver to the rental?", "gold_id": "car_rental_p5", "intent": "sufficient"},
            {"text": "How fast does roadside help usually arrive if the car breaks down in a city?", "gold_id": "car_rental_p7", "intent": "sufficient"},
            {"text": "Can I cancel my reservation without being charged anything?", "gold_id": "car_rental_p8", "intent": "sufficient"},
            {"text": "What's the grace period before a late return counts as a whole extra day?", "gold_id": "car_rental_p4", "intent": "gold_low_rank"},
            {"text": "Do I need paperwork ahead of time to drive the rental into another country?", "gold_id": "car_rental_p6", "intent": "gold_low_rank"},
            {"text": "How many minutes of leeway is given before a late drop-off gets billed extra?", "gold_id": "car_rental_p4", "intent": "gold_low_rank"},
            {"text": "Is advance clearance needed before taking the vehicle across a national border?", "gold_id": "car_rental_p6", "intent": "gold_low_rank"},
            {"text": "How soon before pickup can I back out without losing money?", "gold_id": "car_rental_p8", "intent": "gold_low_rank"},
            {"text": "Is emergency roadside help part of the standard agreement in cities?", "gold_id": "car_rental_p7", "intent": "gold_low_rank"},
            {"text": "Can I rent a car with a debit card instead of a credit card?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is a car seat available to rent for a toddler?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the rental include unlimited mileage?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a discount for returning the car to a different branch?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What extra fees might come up when I rent a car?", "gold_id": "car_rental_p1", "intent": "query_ambiguous"},
            {"text": "What should I know about timing when picking up or dropping off?", "gold_id": "car_rental_p4", "intent": "query_ambiguous"},
            {"text": "What support is available if something goes wrong with the vehicle?", "gold_id": "car_rental_p7", "intent": "query_ambiguous"},
            {"text": "What are my options if my travel plans fall through?", "gold_id": "car_rental_p8", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "utility_billing",
        "passages": {
            "utility_billing_p1": "Monthly utility bills are due fifteen days after the statement date, with a five percent late fee applied afterward.",
            "utility_billing_p2": "Budget billing averages the past twelve months of usage into equal monthly payments, reconciled once per year.",
            "utility_billing_p3": "Service disconnection for non-payment cannot occur until an account is at least sixty days past due.",
            "utility_billing_p4": "A security deposit equal to two months of average usage is required for customers without an established payment history.",
            "utility_billing_p5": "Disputed charges must be reported within forty-five days of the billing date to qualify for an adjustment review.",
            "utility_billing_p6": "Reconnection after a disconnection for non-payment requires the full past-due balance plus a seventy-five dollar reconnection fee.",
            "utility_billing_p7": "Paperless billing enrollment applies a one-time three dollar monthly credit for the first six months.",
            "utility_billing_p8": "Usage alerts can be configured to notify customers by text within fifteen minutes when consumption exceeds a self-set daily threshold.",
        },
        "queries": [
            {"text": "How many days after the statement is my utility bill due?", "gold_id": "utility_billing_p1", "intent": "sufficient"},
            {"text": "How does the equal-payment budget billing plan work?", "gold_id": "utility_billing_p2", "intent": "sufficient"},
            {"text": "How far behind can I fall before service actually gets cut off?", "gold_id": "utility_billing_p3", "intent": "sufficient"},
            {"text": "Is a deposit required for new customers with no billing history?", "gold_id": "utility_billing_p4", "intent": "sufficient"},
            {"text": "Is there a small credit for switching to paperless billing?", "gold_id": "utility_billing_p7", "intent": "sufficient"},
            {"text": "Can I get a text alert if I'm using more energy than usual in a day?", "gold_id": "utility_billing_p8", "intent": "sufficient"},
            {"text": "What does it cost to get turned back on after being shut off?", "gold_id": "utility_billing_p6", "intent": "gold_low_rank"},
            {"text": "How long do I have to flag a charge on my bill that looks wrong?", "gold_id": "utility_billing_p5", "intent": "gold_low_rank"},
            {"text": "What's owed beyond the overdue amount to restore service after a cutoff?", "gold_id": "utility_billing_p6", "intent": "gold_low_rank"},
            {"text": "Is there a window for questioning an unexpected line item on the statement?", "gold_id": "utility_billing_p5", "intent": "gold_low_rank"},
            {"text": "Does going paperless come with any small ongoing perk?", "gold_id": "utility_billing_p7", "intent": "gold_low_rank"},
            {"text": "Can consumption notifications be tailored to my own limit?", "gold_id": "utility_billing_p8", "intent": "gold_low_rank"},
            {"text": "Is there a discount for elderly or low-income customers?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I switch to a renewable energy plan at no extra cost?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the utility offer a referral bonus program?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a mobile app to remotely control my thermostat?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What are the terms around paying my utility bill on time?", "gold_id": "utility_billing_p1", "intent": "query_ambiguous"},
            {"text": "What happens if I fall behind on payments?", "gold_id": "utility_billing_p3", "intent": "query_ambiguous"},
            {"text": "What options exist for keeping track of or lowering my usage?", "gold_id": "utility_billing_p8", "intent": "query_ambiguous"},
            {"text": "What perks are tied to how I choose to receive my bill?", "gold_id": "utility_billing_p7", "intent": "query_ambiguous"},
        ],
    },
    {
        "topic_id": "home_warranty",
        "passages": {
            "home_warranty_p1": "Standard home warranty plans cover major systems and appliances for a twelve-month term renewable annually.",
            "home_warranty_p2": "A service call fee of seventy-five dollars is charged per visit regardless of whether a repair is completed.",
            "home_warranty_p3": "Pre-existing conditions identified during the initial home inspection are excluded from coverage for the first ninety days.",
            "home_warranty_p4": "Claims are typically assigned to a licensed contractor within forty-eight hours of being filed.",
            "home_warranty_p5": "Coverage caps for a single appliance replacement are set at two thousand dollars per contract term.",
            "home_warranty_p6": "Upgrading to the premium plan for an added two hundred dollars per year adds coverage for pools, spas, and well pumps not included in the standard plan.",
            "home_warranty_p7": "Cancelling within the first thirty days results in a full refund minus any service calls already used.",
            "home_warranty_p8": "Roof leak coverage is limited to a maximum of five hundred dollars per incident and excludes storm damage.",
        },
        "queries": [
            {"text": "How long does a standard home warranty term last?", "gold_id": "home_warranty_p1", "intent": "sufficient"},
            {"text": "How much do I pay for each service visit?", "gold_id": "home_warranty_p2", "intent": "sufficient"},
            {"text": "How soon is a contractor assigned after I file a claim?", "gold_id": "home_warranty_p4", "intent": "sufficient"},
            {"text": "What's the maximum payout for replacing one appliance?", "gold_id": "home_warranty_p5", "intent": "sufficient"},
            {"text": "What extra does the premium plan cover that the standard one doesn't?", "gold_id": "home_warranty_p6", "intent": "sufficient"},
            {"text": "How much is covered if my roof develops a leak?", "gold_id": "home_warranty_p8", "intent": "sufficient"},
            {"text": "Are issues found during the initial inspection covered right away?", "gold_id": "home_warranty_p3", "intent": "gold_low_rank"},
            {"text": "Can I get most of my money back if I cancel shortly after signing up?", "gold_id": "home_warranty_p7", "intent": "gold_low_rank"},
            {"text": "Is there a waiting period for problems that were already there before the plan started?", "gold_id": "home_warranty_p3", "intent": "gold_low_rank"},
            {"text": "What refund can I expect if I back out early in the contract?", "gold_id": "home_warranty_p7", "intent": "gold_low_rank"},
            {"text": "Does storm-related roof damage fall under the leak coverage limit?", "gold_id": "home_warranty_p8", "intent": "gold_low_rank"},
            {"text": "Is there a ceiling on how much a single broken appliance gets covered for?", "gold_id": "home_warranty_p5", "intent": "gold_low_rank"},
            {"text": "Does the warranty cover damage caused by a natural disaster?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Can I choose my own contractor instead of the assigned one?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Is there a discount for bundling with homeowners insurance?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "Does the plan cover detached structures like a shed?", "gold_id": "none", "intent": "gold_absent"},
            {"text": "What should I expect to pay under this home warranty?", "gold_id": "home_warranty_p2", "intent": "query_ambiguous"},
            {"text": "What isn't covered when I first sign up?", "gold_id": "home_warranty_p3", "intent": "query_ambiguous"},
            {"text": "What are my options if I change my mind soon after enrolling?", "gold_id": "home_warranty_p7", "intent": "query_ambiguous"},
            {"text": "What limits exist on payouts for different kinds of damage?", "gold_id": "home_warranty_p8", "intent": "query_ambiguous"},
        ],
    },
]


# --------------------------------------------------------------------------
# FINGERPRINT - reused design (proven in "Prompt Edit Effect Attribution"):
# a fixed-length numeric behaviour vector computed from the actual
# generated text. No response text is ever released.
# --------------------------------------------------------------------------
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])", re.M)
IDK_RE = re.compile(
    r"don't know|do not know|cannot|can't|unsure|uncertain|unable|"
    r"no way to know|not (?:mentioned|stated|specified|provided)|"
    r"no information|does not (?:say|mention|specify)", re.I)
HEDGE_RE = re.compile(
    r"\b(?:might|may|could|perhaps|possibly|generally|typically|usually|"
    r"often|it depends|likely|probably)\b", re.I)
CONTEXT_REF_RE = re.compile(
    r"\b(?:according to|the (?:passage|document|text|context) (?:says|states)|"
    r"based on the (?:passage|document|text|context))\b", re.I)
QUESTION_RE = re.compile(r"\?")


def fingerprint(text):
    """Map one generated answer to a fixed-length numeric behaviour vector.

    Every component is a property of HOW the model answered given its
    retrieved context, not a copy of the passage text, so it measures
    confidence/grounding behaviour without releasing the passage or the
    full response.
    """
    t = text.strip()
    words = t.split()
    n = len(words)
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    return [
        float(n),                                          # length in words
        float(len(sents)),                                 # sentence count
        1.0 if IDK_RE.search(t) else 0.0,                   # admits no answer
        float(len(HEDGE_RE.findall(t))),                    # hedging density
        1.0 if CONTEXT_REF_RE.search(t) else 0.0,           # cites context explicitly
        float(len(QUESTION_RE.findall(t))),                 # asks back
        float(sum(c.isdigit() for c in t)),                 # digits used
        float(BULLET_RE.search(t) is not None),             # list structure
    ]


FP_NAMES = ["n_words", "n_sents", "says_no_answer", "n_hedges",
            "cites_context", "n_questions", "n_digits", "has_bullets"]


def _embed(model, texts, batch_size=64):
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)


def _cosine_rank(query_vec, passage_ids, passage_vecs, gold_id):
    """Real cosine similarity ranking of passage_ids against query_vec.
    Returns (sorted_ids_by_similarity_desc, rank_of_gold_or_None,
    similarities_dict)."""
    sims = {pid: float(query_vec @ passage_vecs[pid]) for pid in passage_ids}
    ranked = sorted(passage_ids, key=lambda pid: sims[pid], reverse=True)
    rank = (ranked.index(gold_id) + 1) if gold_id in ranked else None
    return ranked, rank, sims


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="../raw")
    ap.add_argument("--gen-model", default=GEN_MODEL_ID)
    ap.add_argument("--embed-model", default=EMBED_MODEL_ID)
    ap.add_argument("--smoke", action="store_true",
                     help="3 topics only, for a quick check")
    ap.add_argument("--batch", type=int, default=GEN_BATCH)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if dev != "cpu" else torch.float32
    print(f"device={dev} dtype={dtype} gen_model={args.gen_model} "
          f"embed_model={args.embed_model}")

    topics = TOPICS[:3] if args.smoke else TOPICS

    # -- build the whole-corpus passage index (ALL topics, always, so
    # cross-topic ambiguity and gold_absent are measured against the full
    # released corpus, never a subset). ----------------------------------
    all_passages = {}
    for t in topics:
        all_passages.update(t["passages"])
    passage_ids_sorted = sorted(all_passages)
    print(f"topics={len(topics)}  total passages={len(passage_ids_sorted)}")

    embed_model = SentenceTransformer(args.embed_model, device=dev)
    passage_vecs_arr = _embed(embed_model,
                               [all_passages[pid] for pid in passage_ids_sorted])
    passage_vecs = {pid: passage_vecs_arr[i]
                    for i, pid in enumerate(passage_ids_sorted)}

    # -- build query list -------------------------------------------------
    query_jobs = []
    for t in topics:
        for qi, q in enumerate(t["queries"]):
            query_jobs.append((t["topic_id"], qi, q["text"], q["gold_id"],
                                q["intent"]))
    query_texts = [q[2] for q in query_jobs]
    query_vecs_arr = _embed(embed_model, query_texts)
    print(f"queries={len(query_jobs)}")

    # -- real retrieval: rank WITHIN the query's own topic (the natural RAG
    # setting: retrieve from the topic's knowledge base), and separately
    # measure cross-topic similarity for ambiguity detection. This computes
    # the FULL ranking once per query; k-specific top_k slices are taken
    # per ITEM below, so re-ranking never needs to be repeated per k. -----
    query_facts = {}
    for (topic_id, qi, text, gold_id, intent), qvec in zip(query_jobs, query_vecs_arr):
        own_ids = sorted(next(t for t in topics if t["topic_id"] == topic_id)["passages"])
        ranked, rank, sims = _cosine_rank(qvec, own_ids, passage_vecs, gold_id)

        other_ids = [pid for pid in passage_ids_sorted if pid not in own_ids]
        other_sims = {pid: float(qvec @ passage_vecs[pid]) for pid in other_ids}
        best_other = max(other_sims, key=other_sims.get) if other_sims else None
        best_own = max(sims, key=sims.get)

        qkey = f"{topic_id}|{qi}"
        query_facts[qkey] = {
            "topic_id": topic_id,
            "query_index": qi,
            "text": text,
            "gold_id": gold_id,
            "construction_intent": intent,
            "ranked_ids": ranked,
            "sims": sims,
            "gold_rank_in_topic": rank,
            "best_own_topic_score": round(sims[best_own], 6),
            "best_other_topic_id": best_other,
            "best_other_topic_score": (round(other_sims[best_other], 6)
                                        if best_other else None),
        }

    # -- item multiplication: each query becomes one item per k in its
    # intent's k-list (see ITEM MULTIPLIER comment above). ----------------
    items = {}
    for qkey, qf in sorted(query_facts.items()):
        k_list = (K_SWEEP_GOLD_LOW_RANK if qf["construction_intent"] == "gold_low_rank"
                  else K_VARIANTS_OTHER)
        n_avail = len(qf["ranked_ids"])
        for k in k_list:
            k_eff = min(k, n_avail)
            item_key = f"{qkey}|k{k_eff}"
            if item_key in items:
                continue  # k_eff collapsed onto an already-emitted k, skip dup
            top_k = qf["ranked_ids"][:k_eff]
            items[item_key] = {
                "topic_id": qf["topic_id"],
                "query_index": qf["query_index"],
                "query_text": qf["text"],
                "gold_id": qf["gold_id"],
                "construction_intent": qf["construction_intent"],
                "k": k_eff,
                "top_k_ids": top_k,
                "top_k_scores": [round(qf["sims"][pid], 6) for pid in top_k],
                "gold_rank_in_topic": qf["gold_rank_in_topic"],
                "best_own_topic_score": qf["best_own_topic_score"],
                "best_other_topic_id": qf["best_other_topic_id"],
                "best_other_topic_score": qf["best_other_topic_score"],
            }
    print(f"items after k-multiplication: {len(items):,}")

    # -- real generation: run the generator over (query + top_k retrieved
    # passage texts) and fingerprint the actual output, ONE generation per
    # ITEM (not per query), since a different k means a different context
    # actually shown to the generator and therefore a genuinely different
    # answer to measure. ---------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.gen_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=dtype).to(dev).eval()

    gen_jobs = []
    for item_key, it in sorted(items.items()):
        context = "\n".join(f"- {all_passages[pid]}" for pid in it["top_k_ids"])
        prompt = (f"Context:\n{context}\n\nQuestion: {it['query_text']}\n"
                  "Answer using only the context above. If the context does "
                  "not contain the answer, say so.")
        gen_jobs.append((item_key, prompt))

    fingerprints = {}
    t0 = time.time()
    for i in range(0, len(gen_jobs), args.batch):
        chunk = gen_jobs[i:i + args.batch]
        texts = [tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True, tokenize=False)
                 for _, p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        with torch.no_grad():
            gen = gen_model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False,
                                      pad_token_id=tok.pad_token_id)
        cut = enc["input_ids"].shape[1]
        for (item_key, _), row in zip(chunk, gen):
            txt = tok.decode(row[cut:], skip_special_tokens=True)
            fingerprints[item_key] = fingerprint(txt)
        done = i + len(chunk)
        if done % (args.batch * 10) == 0 or done == len(gen_jobs):
            el = time.time() - t0
            print(f"  {done:,}/{len(gen_jobs):,}  {el:6.0f}s  "
                  f"eta {el / done * (len(gen_jobs) - done):6.0f}s", flush=True)

    payload = {
        "schema": 1,
        "gen_model_id": args.gen_model,
        "embed_model_id": args.embed_model,
        "gen_model_licence": "Apache-2.0; no restriction on outputs",
        "embed_model_licence": "MIT; no restriction on outputs",
        "dataset_licence": "CC-BY-4.0",
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=False)",
        "k_sweep_gold_low_rank": K_SWEEP_GOLD_LOW_RANK,
        "k_variants_other": K_VARIANTS_OTHER,
        "fingerprint_fields": FP_NAMES,
        "topics": [t["topic_id"] for t in topics],
        "n_passages": len(passage_ids_sorted),
        "n_queries": len(query_jobs),
        "n_items": len(items),
        "items": items,
        "generation_fingerprints": fingerprints,
    }
    fp_path = out / "retrieval.json"
    fp_path.write_text(json.dumps(payload, indent=1, sort_keys=True))

    # ship the corpus/grid definition alongside, for reproducibility
    grid = {"topics": topics}
    (out / "grid.json").write_text(json.dumps(grid, indent=1, sort_keys=True))

    digest = hashlib.sha256(fp_path.read_bytes()).hexdigest()
    (out / "meta.json").write_text(json.dumps({
        "generated_by": "gen_raw.py",
        "seed": SEED,
        "gen_model_id": args.gen_model,
        "embed_model_id": args.embed_model,
        "n_queries": len(query_jobs),
        "n_passages": len(passage_ids_sorted),
        "n_items": len(items),
        "retrieval_sha256": digest,
        "elapsed_seconds": round(time.time() - t0, 1),
        "device": dev,
    }, indent=1))
    print(f"\nwrote {fp_path}  ({len(items):,} items, "
          f"{len(fingerprints):,} generation fingerprints)")
    print(f"sha256 {digest}")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
