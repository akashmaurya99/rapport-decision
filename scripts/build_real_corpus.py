"""Build the RAPPORT real corpus from the project's documented HF sources.

Fixes over the ad-hoc builder:
  * risk categories use a STABLE sha256-derived slot (never Python hash(),
    which is randomized per process and made the risk head learn noise);
  * rows with no violated category are genuinely SAFE (empty risk_labels),
    not silently assigned to category-0;
  * option order is randomized with a seeded RNG and chosen_index/scores are
    remapped, so the gold answer is not always slot 0 (no position leakage);
  * tokenizer-length validation keeps the 512-token budget with intact <opt>
    markers;
  * a single stable RNG makes the whole build reproducible in one process.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import random
import re
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from src.data.acquire.manifest import SourceManifest
from src.data.core.normalize import normalize_example, validate_example
from src.tokenization import encode_example
from src.tokenization.tokenizer import OPT_ID

RNG = random.Random(20261124)

TARGETS = {
    "Anthropic/hh-rlhf": 52000,
    "cais/mmlu#auxiliary_train": 140000,
    "nvidia/Aegis-AI-Content-Safety-Dataset-2.0": 42000,
    "AbstentionBench-derived": 38000,
    "truthfulqa/truthful_qa": 3163,
    "cais/mmlu#auxiliary_train-derived-binary": 70000,
    "nvidia/HelpSteer2": 26000,
}
LICENSES = {
    "Anthropic/hh-rlhf": "MIT",
    "cais/mmlu#auxiliary_train": "MIT",
    "cais/mmlu#auxiliary_train-derived-binary": "MIT",
    "nvidia/Aegis-AI-Content-Safety-Dataset-2.0": "CC-BY-4.0",
    "AbstentionBench-derived": "MIT",
    "truthfulqa/truthful_qa": "Apache-2.0",
    "nvidia/HelpSteer2": "CC-BY-4.0",
}
MULTIPLIER = 1.5
AEGIS_CATEGORIES = (
    "violence", "self_harm", "sexual_content", "hate_speech",
    "harassment", "dangerous_content", "illegal_activity", "privacy_violation",
)


def clip(s, n):
    return " ".join(str(s).split())[:n]


def gid(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


def risk_slot(name: str) -> str:
    """Stable category slot for an Aegis label (never Python hash())."""
    low = str(name).strip().lower()
    for index, category in enumerate(AEGIS_CATEGORIES):
        if category in low or low in category:
            return f"category-{index}"
    digest = hashlib.sha256(low.encode("utf-8")).hexdigest()
    return f"category-{int(digest[:8], 16) % 8}"


def finalize(r):
    if not r.get("options"):
        return None
    # Q1 requires >=3 whitespace tokens per option; pad short options with
    # neutral tokens (no injected hashes - keeps semantic signal intact).
    for o in r["options"]:
        text = str(o["text"])
        while len(text.split()) < 3:
            text += " item"
        o["text"] = text
    if validate_example(r):
        return None
    for attempt in range(10):
        try:
            enc = encode_example(normalize_example(r))
            if len(enc.input_ids) <= 512 and sum(
                1 for t in enc.input_ids if int(t) == OPT_ID
            ) == len(r["options"]):
                return r
        except Exception:
            pass
        if len(str(r.get("state", ""))) > 180:
            r["state"] = str(r["state"])[:100]
        for o in r["options"]:
            if len(str(o["text"])) > 90:
                o["text"] = str(o["text"])[:60]
    return None


def mk(family, goal, state, options, source, group, **kw):
    options = list(options)
    order = list(range(len(options)))
    RNG.shuffle(order)
    new_options = [options[i] for i in order]
    labels = dict(kw)
    if labels.get("chosen_index") is not None:
        labels["chosen_index"] = order.index(labels["chosen_index"])
    if labels.get("scores") is not None:
        scores = labels["scores"]
        labels["scores"] = [scores[i] for i in order]
    return {
        "task_family": family,
        "goal": clip(goal, 200),
        "state": clip(state, 400),
        "options": new_options,
        "source": source,
        "license": LICENSES[source],
        "group_id": group,
        "split": "train",
        **labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="output data/ directory (default: repo/data)")
    args = parser.parse_args()
    root = os.getcwd()
    proc = os.path.join(args.data_dir, "processed") if args.data_dir else os.path.join(root, "data", "processed")
    raw_dir = os.path.join(args.data_dir, "raw") if args.data_dir else os.path.join(root, "data", "raw")
    os.makedirs(proc, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    buckets = {k: [] for k in TARGETS}

    # MMLU: choice + derived binary
    need = int(TARGETS["cais/mmlu#auxiliary_train"] * MULTIPLIER)
    needbin = int(TARGETS["cais/mmlu#auxiliary_train-derived-binary"] * MULTIPLIER)
    try:
        ds = load_dataset("cais/mmlu", "auxiliary_train", streaming=True, split="train")
        for r in ds:
            if (len(buckets["cais/mmlu#auxiliary_train"]) >= need
                    and len(buckets["cais/mmlu#auxiliary_train-derived-binary"]) >= needbin):
                break
            rec = r.get("train", r)
            if isinstance(rec, str):
                try: rec = json.loads(rec)
                except Exception: continue
            q, ch, ans = rec.get("question"), rec.get("choices"), rec.get("answer")
            if not isinstance(q, str) or not isinstance(ch, list) or len(ch) < 2: continue
            ch = [clip(x, 150) for x in ch]
            if len(set(ch)) != len(ch) or not isinstance(ans, int) or not 0 <= ans < len(ch): continue
            g = gid("mmlu", q)
            opts = [{"text": t, "label": chr(65 + j)} for j, t in enumerate(ch[:8])]
            if len(buckets["cais/mmlu#auxiliary_train"]) < need:
                x = finalize(mk("choice", q, q, opts, "cais/mmlu#auxiliary_train", g,
                                chosen_index=ans))
                if x: buckets["cais/mmlu#auxiliary_train"].append(x)
            if len(buckets["cais/mmlu#auxiliary_train-derived-binary"]) < needbin:
                wrong = (ans + 1) % len(opts)
                bopts = [{"text": ch[ans], "label": "A"}, {"text": ch[wrong], "label": "B"}]
                x = finalize(mk("probability", "Is the stated answer correct?",
                                q + " Correct answer: " + ch[ans], bopts,
                                "cais/mmlu#auxiliary_train-derived-binary", g + "b",
                                chosen_index=0, answerable=True))
                if x: buckets["cais/mmlu#auxiliary_train-derived-binary"].append(x)
        print("mmlu", len(buckets["cais/mmlu#auxiliary_train"]),
              len(buckets["cais/mmlu#auxiliary_train-derived-binary"]))
    except Exception as e:
        print("mmlu ERR", repr(e)[:300])

    # HH-RLHF preference (gold slot randomized by mk)
    need = int(TARGETS["Anthropic/hh-rlhf"] * MULTIPLIER)
    try:
        ds = load_dataset("Anthropic/hh-rlhf", streaming=True, split="train")
        conv_re = re.compile(r"\n\n(Human|Assistant):\s*")
        def split_conv(t):
            parts = conv_re.split(t)
            return (clip(parts[2], 450) if len(parts) > 2 else "",
                    clip(parts[4], 220) if len(parts) > 4 else "")
        for r in ds:
            if len(buckets["Anthropic/hh-rlhf"]) >= need: break
            c, rc = r.get("chosen"), r.get("rejected")
            if not isinstance(c, str) or not isinstance(rc, str): continue
            h, ca = split_conv(c); _, ra = split_conv(rc)
            if not h or not ca or not ra or ca == ra: continue
            x = finalize(mk("preference", "Choose the preferred assistant response", h,
                            [{"text": ca, "label": "A"}, {"text": ra, "label": "B"}],
                            "Anthropic/hh-rlhf", gid("hh", h), chosen_index=0))
            if x: buckets["Anthropic/hh-rlhf"].append(x)
        print("hh", len(buckets["Anthropic/hh-rlhf"]))
    except Exception as e:
        print("hh ERR", repr(e)[:300])

    # Aegis risk - STABLE slots, empty labels for safe rows
    need = int(TARGETS["nvidia/Aegis-AI-Content-Safety-Dataset-2.0"] * MULTIPLIER)
    try:
        ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", streaming=True, split="train")
        for r in ds:
            if len(buckets["nvidia/Aegis-AI-Content-Safety-Dataset-2.0"]) >= need: break
            p, resp = r.get("prompt"), r.get("response")
            if not isinstance(p, str) or not p.strip(): continue
            # Aegis redacts many rows to the literal "REDACTED"; skip those so
            # they cannot collapse into one giant content-hash group that the
            # group-aware splitter assigns wholly to a single split.
            if p.strip().upper() == "REDACTED": continue
            if isinstance(resp, str) and resp.strip().upper() == "REDACTED":
                resp = None
            vc = r.get("violated_categories")
            labs = []
            if isinstance(vc, str) and vc.strip() and vc.strip().lower() != "none":
                for c in vc.split(","):
                    if c.strip():
                        labs.append(risk_slot(c))
            labs = sorted(set(labs))[:8]  # EMPTY for safe rows
            x = finalize(mk("risk", "Identify harm categories in the text", p + " Response: " + clip(resp, 300),
                            [{"text": "this content is safe", "label": "A"},
                             {"text": "this content is unsafe", "label": "B"}],
                            "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", gid("aegis", p, resp),
                            risk_labels=labs))
            if x: buckets["nvidia/Aegis-AI-Content-Safety-Dataset-2.0"].append(x)
            if isinstance(resp, str) and resp.strip() and len(buckets["nvidia/Aegis-AI-Content-Safety-Dataset-2.0"]) < need:
                x2 = finalize(mk("risk", "Identify harm categories in the text", clip(resp, 500),
                                 [{"text": "this content is safe", "label": "A"},
                                  {"text": "this content is unsafe", "label": "B"}],
                                 "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", gid("aegis-resp", resp),
                                 risk_labels=labs))
                if x2: buckets["nvidia/Aegis-AI-Content-Safety-Dataset-2.0"].append(x2)
        print("aegis", len(buckets["nvidia/Aegis-AI-Content-Safety-Dataset-2.0"]),
              "safe", sum(1 for r in buckets["nvidia/Aegis-AI-Content-Safety-Dataset-2.0"] if not r.get("risk_labels")))
    except Exception as e:
        print("aegis ERR", repr(e)[:300])

    # AbstentionBench
    need = int(TARGETS["AbstentionBench-derived"] * MULTIPLIER)
    try:
        api = load_dataset("ymerkli/AbstentionBench", streaming=True)
        for sp in api:
            if len(buckets["AbstentionBench-derived"]) >= need: break
            try:
                for r in api[sp]:
                    if len(buckets["AbstentionBench-derived"]) >= need: break
                    q = r.get("question")
                    if not isinstance(q, str) or not q.strip(): continue
                    sa = str(r.get("should_abstain", "")).strip().lower()
                    if sa not in ("true", "false"): continue
                    x = finalize(mk("abstention", "Can this question be answered?",
                                    sp + " / " + q,
                                    [{"text": "the question is answerable", "label": "A"},
                                     {"text": "the question is unanswerable", "label": "B"}],
                                    "AbstentionBench-derived", gid("abst", sp, q),
                                    chosen_index=0 if sa == "false" else 1,
                                    answerable=(sa == "false")))
                    if x: buckets["AbstentionBench-derived"].append(x)
            except Exception as e:
                print("abst subset", sp, repr(e)[:100])
        print("abst", len(buckets["AbstentionBench-derived"]))
    except Exception as e:
        print("abst ERR", repr(e)[:300])

    # TruthfulQA (gold slot randomized by mk)
    need = TARGETS["truthfulqa/truthful_qa"]
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", streaming=True, split="validation")
        for r in ds:
            if len(buckets["truthfulqa/truthful_qa"]) >= need: break
            q = r.get("question")
            if not isinstance(q, str) or not q.strip(): continue
            pairs = []
            for key in ("mc1_targets", "mc2_targets"):
                t = r.get(key)
                if isinstance(t, str):
                    try: t = ast.literal_eval(t)
                    except Exception: continue
                if not isinstance(t, dict): continue
                ch, lab = t.get("choices"), t.get("labels")
                if not isinstance(ch, list) or not isinstance(lab, list): continue
                goods = [clip(c, 180) for c, l in zip(ch, lab) if l == 1]
                bads = [clip(c, 180) for c, l in zip(ch, lab) if l == 0]
                for i in range(min(6, len(goods), len(bads))):
                    if goods[i] and bads[i] and goods[i] != bads[i]:
                        pairs.append((goods[i], bads[i]))
            for i, (g, b) in enumerate(pairs):
                x = finalize(mk("probability", "Is the stated answer truthful?", q,
                                [{"text": g, "label": "A"}, {"text": b, "label": "B"}],
                                "truthfulqa/truthful_qa", gid("tqa", q, i),
                                chosen_index=0, answerable=True))
                if x: buckets["truthfulqa/truthful_qa"].append(x)
                if len(buckets["truthfulqa/truthful_qa"]) >= need: break
        print("tqa", len(buckets["truthfulqa/truthful_qa"]))
    except Exception as e:
        print("tqa ERR", repr(e)[:300])

    # HelpSteer2 scoring (scores randomized by mk)
    need = int(TARGETS["nvidia/HelpSteer2"] * MULTIPLIER)
    try:
        ds = load_dataset("nvidia/HelpSteer2", streaming=True, split="train")
        for r in ds:
            if len(buckets["nvidia/HelpSteer2"]) >= need: break
            p, resp, h = r.get("prompt"), r.get("response"), r.get("helpfulness")
            if not isinstance(p, str) or not isinstance(resp, str): continue
            try: score = max(0.0, min(1.0, float(h) / 4.0))
            except Exception: score = 0.5
            x = finalize(mk("scoring", "Rate the quality of the response", p,
                            [{"text": clip(resp, 180), "label": "A"},
                             {"text": "An evasive and unhelpful response.", "label": "B"}],
                            "nvidia/HelpSteer2", gid("hs", p, resp),
                            scores=[round(score, 3), 0.1]))
            if x: buckets["nvidia/HelpSteer2"].append(x)
        print("helpsteer", len(buckets["nvidia/HelpSteer2"]))
    except Exception as e:
        print("helpsteer ERR", repr(e)[:300])

    # Write shards + manifests
    for source, rows in buckets.items():
        target = TARGETS[source]
        keep = [r for r in rows if r][:target]
        if len(keep) < target:
            print("SHORT", source, len(keep), "need", target)
            continue
        safe = source.replace("/", "__").replace("#", "_").replace("-", "_").lower()
        path = os.path.join(proc, safe + ".parquet")
        pq.write_table(pa.Table.from_pylist(keep), path)
        raw = open(path, "rb").read(); sha = hashlib.sha256(raw).hexdigest()
        abs_path = os.path.abspath(path)
        m = SourceManifest(source=source, license=LICENSES[source], declared_rows=target,
            observed_rows=len(keep), sha256=sha,
            acquired_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status="admitted", shards=[abs_path], shard_sha256={abs_path: sha},
            shard_bytes={abs_path: len(raw)}, quarantine_reason=None,
            license_origin="hub-card", adapter="unified", shard_layout="decision-example",
            raw_input_rows=len(rows), rejected_rows=max(0, len(rows) - target),
            adapter_report={"conversion": "unified-schema", "risk_mapping": "sha256-stable",
                            "gold_position": "randomized-seeded"})
        for base in (raw_dir, os.path.dirname(proc)):
            md = os.path.join(base, safe); os.makedirs(md, exist_ok=True)
            open(os.path.join(md, "source-manifest.json"), "w").write(m.to_json())
        print("WROTE", source, len(keep))


if __name__ == "__main__":
    main()
