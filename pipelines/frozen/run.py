"""Critique the existing course baseline, revise with Qwen, export full answers."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROJECT = Path.home() / "RAG/author_alignrag_course"
EXPECTED = {
    "baseline": "2385069c6061e2cd78631a8f5f59c3936355af60ed2ba3323d86a88a492662c3",
    "test": "132adfbd00b394a293b5eb6fc1f8195471fbfcf0e9a796a2bdbcb905cb860bcb",
    "sample": "7415d29a4ad892ea75a6c56a2bca3dab3740d03ff4e4cdd166e6e690ff90d751",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_results(path, test):
    rows = read_json(path)
    if len(rows) != len(test):
        raise ValueError(f"Wrong output row count: {path}")
    for row, item in zip(rows, test):
        if str(row["id"]) != str(item["id"]) or row["question"] != item["question"]:
            raise ValueError(f"Output ID/question mismatch: {path}")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError(f"Empty output for ID {item['id']}: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "full"])
    parser.add_argument("--baseline", type=Path, default=PROJECT / "course/artifacts/full/qwen/2WikiMultiHopQA/result_rationale_cot.json")
    parser.add_argument("--test", type=Path, default=PROJECT / "course_data/2WikiMultiHopQA/test.json")
    parser.add_argument("--sample", type=Path, default=PROJECT / "course_data/2WikiMultiHopQA/sample_submission.csv")
    parser.add_argument("--qwen", type=Path, default=Path.home() / "RAG/models_verified/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llama", type=Path, default=Path.home() / "RAG/models_verified/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite an existing run")
    parser.add_argument("--check-only", action="store_true", help="Verify inputs without loading models or creating outputs")
    args = parser.parse_args()
    for name in ("baseline", "test", "sample", "qwen", "llama"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name, expected in EXPECTED.items():
        if sha(getattr(args, name)) != expected:
            raise ValueError(f"{name} differs from the submitted baseline experiment")
    if args.baseline.name != "result_rationale_cot.json" or args.baseline.parent.name != "2WikiMultiHopQA":
        raise ValueError("Baseline must use the author's output directory layout")
    test = read_json(args.test)
    if len(test) != 12576 or any(r.get("answers") is not None for r in test):
        raise ValueError("Expected the 12576-row unlabeled course test set")
    validate_results(args.baseline, test)
    with args.sample.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "answer"]:
            raise ValueError("Unexpected sample submission columns")
        ids = [row["id"] for row in reader]
    if ids != [str(r["id"]) for r in test] or len(set(ids)) != len(ids):
        raise ValueError("Sample IDs must match unique test IDs in order")
    source = read_json(ROOT / "source_manifest.json")
    for name, expected in source["files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Packaged code changed: {name}")
    print("INPUTS_OK: exact original baseline, test set and sample; 12576 IDs/questions aligned.", flush=True)
    if args.check_only:
        print("Model availability and GPU inference have NOT been checked.", flush=True)
        return
    for name in ("qwen", "llama"):
        if not (getattr(args, name) / "config.json").is_file():
            raise FileNotFoundError(f"Missing {name} model: {getattr(args, name)}")
    import torch
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Expected 4 visible GPUs; set CUDA_VISIBLE_DEVICES=0,1,2,3")
    output = args.output.expanduser().resolve() if args.output else ROOT / "artifacts" / args.mode
    versions = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")}
    output.mkdir(parents=True, exist_ok=False)
    selected = test[:5] if args.mode == "smoke" else test
    common = [
        sys.executable, "-u", "eval/inference_vllm.py",
        "--data_path", str(args.test), "--dataset_name", "2WikiMultiHopQA",
        "--model_name_or_path", str(args.qwen), "--critique_model", str(args.llama),
        "--config_file", "configs/eval.yaml", "--type", "cot", "--iters", "1",
        "--n_docs", "5", "--seed", "42", "--max_tokens", "5120", "--max_model_len", "32768",
        "--output_dir", str(args.baseline.parent.parent), "--critique_model_path", str(output),
    ]
    if args.mode == "smoke":
        common += ["--max_instances", "5"]
    manifest = {
        "status": "running", "mode": args.mode, "rows": len(selected),
        "method": "Frozen LLaMA critique, one Qwen refinement, full rationale export",
        "started_utc": datetime.now(timezone.utc).isoformat(), "inputs_sha256": EXPECTED,
        "source": source, "versions": versions, "qwen": str(args.qwen), "llama": str(args.llama),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(4)],
        "sampling": {"qwen": {"temperature": 0.7, "top_p": 0.8}, "llama": {"temperature": 0.6, "top_p": 0.9}},
        "commands": [], "completed_phases": [],
    }
    manifest_path = output / "run_manifest.json"
    write_json(manifest_path, manifest)
    try:
        for phase, filename in [(2, "result_critic_rationale_cot.json"), (3, "result_refine_cot.json")]:
            command = common + ["--phase", str(phase)]
            manifest["commands"].append(command)
            write_json(manifest_path, manifest)
            print(f"START_PHASE_{phase}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            result_path = output / "2WikiMultiHopQA" / filename
            rows = validate_results(result_path, selected)
            manifest["completed_phases"].append(phase)
            manifest[filename + "_sha256"] = sha(result_path)
            write_json(manifest_path, manifest)
            print(f"PHASE_{phase}_COMPLETE rows={len(rows)}", flush=True)
        name = "preview_qwen_llama_full_rationale.csv" if args.mode == "smoke" else "submission_qwen_llama_full_rationale.csv"
        csv_path = output / name
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "answer"])
            writer.writerows((str(r["id"]), r["rationale"]) for r in rows)
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            exported = list(reader)
        if len(exported) != len(rows) or any(x != {"id": str(r["id"]), "answer": r["rationale"]} for x, r in zip(exported, rows)):
            raise ValueError("CSV round-trip validation failed")
        if sha(args.baseline) != EXPECTED["baseline"]:
            raise ValueError("Original baseline changed during this run")
        manifest.update(status="complete", output=str(csv_path), csv_sha256=sha(csv_path),
                        answer_format="full_rationale_verbatim", preview_only=args.mode == "smoke")
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    print(f"FROZEN_CLM_COMPLETE rows={len(rows)} csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
