"""Inspect train/validation token loss with a local RoBERTa tokenizer JSON."""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from tool_call_auditor.modeling import load_split
from tool_call_auditor.modeling.serialization import _observation_text, select_observations
from tool_call_auditor.modeling.transformer_input import (
    CRITICAL_ARGUMENTS, _encode_with_head_tail, encode_model_input, serialize_action,
)


class LocalTokenizer:
    def __init__(self, path):
        self.backend = Tokenizer.from_file(str(path))
        self.cls_token_id = self.backend.token_to_id("<s>")
        self.sep_token_id = self.backend.token_to_id("</s>")
        self.pad_token_id = self.backend.token_to_id("<pad>")

    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=False)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/splits/dataset_502_v1"))
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = LocalTokenizer(args.tokenizer_json)
    summaries = {}
    for split in ("train", "validation"):
        inspected = []
        for row in load_split(args.data_dir / f"{split}.jsonl"):
            m = row.model_input
            observations = m["observations_before_call"]
            selected = select_observations(observations)
            fields = {
                "request": ("[USER REQUEST]\n" + m["original_user_request"], 96, .15),
                "observation": ("[UNTRUSTED OBSERVATIONS]\n" + "\n---\n".join(_observation_text(o) for o in selected), 256, .5),
                "action": ("[PROPOSED ACTION]\n" + serialize_action(m["proposed_action"]), 156, .5),
            }
            encoded = encode_model_input(tokenizer, m)
            info = {"candidate_id": row.candidate_id, "label": row.label,
                    "observations_total": len(observations), "observations_selected": len(selected),
                    "used_tokens": sum(encoded["attention_mask"])}
            for name, (text, budget, fraction) in fields.items():
                ids = tokenizer.encode(text)
                info[f"{name}_original_tokens"] = len(ids)
                info[f"{name}_truncated"] = len(ids) > budget
            action_ids = _encode_with_head_tail(tokenizer, fields["action"][0], 156, tail_fraction=.5)
            decoded = tokenizer.decode(action_ids)
            missing = {}
            for key, value in m["proposed_action"]["arguments"].items():
                if key in CRITICAL_ARGUMENTS:
                    lost = [v for v in strings(value) if v.strip() and v not in decoded]
                    if lost:
                        missing[key] = lost
            info["critical_strings_absent_from_decoded_action"] = missing
            # Development-only review material: show the actual request/action
            # text consumed by the frozen encoder, not annotation explanations.
            info["decoded_request"] = tokenizer.decode(_encode_with_head_tail(tokenizer, fields["request"][0], 96, tail_fraction=.15))
            info["decoded_action"] = decoded
            inspected.append(info)
        summaries[split] = {
            "rows": len(inspected),
            "truncation_counts": {field: sum(r[f"{field}_truncated"] for r in inspected)
                                  for field in fields},
            "rows_with_missing_critical_strings": sum(bool(r["critical_strings_absent_from_decoded_action"]) for r in inspected),
            "rows_with_observations_dropped": sum(r["observations_total"] > r["observations_selected"] for r in inspected),
        }
        (args.output_dir / f"{split}_input_audit.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inspected), encoding="utf-8")
    report = {
        "test_read_or_evaluated": False,
        "tokenizer_json_sha256": hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest(),
        "tokenizer_source": str(args.tokenizer_json),
        "budgets": {"request": 96, "observation": 256, "action": 156},
        "note": "Uses the downloaded tokenizer.json backend, not original run artifacts. Missing critical strings are a review flag, not proof of model error.",
        "splits": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
