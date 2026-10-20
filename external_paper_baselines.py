#!/usr/bin/env python3
"""External paper-baseline integration without relabelling local ablations.

Standard library only. Does not modify SJPA-TR or vendor third-party implementations.
Training of C2Former/SuperYOLO uses upstream scripts in their own environment.
HBB baselines must NEVER be ranked against OBB AP from this repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "PAPER_BASELINES_ADDITIONAL.json"


def load_registry():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unknown paper baseline registry schema")
    return payload["methods"]


def get_method(key):
    methods = load_registry()
    if key not in methods:
        raise ValueError(f"Unknown baseline {key!r}; choose {', '.join(methods)}")
    return methods[key]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return p.stdout.strip()


def verify_repo(key, path):
    method = get_method(key)
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for entry in method["source_required"]:
        if not (path / entry).is_file():
            raise FileNotFoundError(f"{key} is missing official upstream file: {entry}")
    head = git_value(path, "rev-parse", "HEAD")
    remote = git_value(path, "remote", "get-url", "origin")
    # A fork may be used experimentally, but not described as official paper code.
    expected = method["source"].removesuffix(".git").casefold()
    actual = remote.removesuffix(".git").casefold()
    if actual not in {expected, method["source"].casefold()}:
        raise ValueError(f"Unexpected Git origin for {key}: {remote}; expected {method['source']}")
    return {
        "baseline": key, "source_paper": method["title"],
        "upstream_repository": method["source"], "source_commit": head,
        "prediction_geometry": method["prediction_geometry"],
        "verified_official_files": method["source_required"],
        "verified_scope": "source paths, origin and commit only; model training NOT verified",
    }


def fetch(key, workspace):
    method = get_method(key)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    destination = workspace / key
    if not destination.exists():
        subprocess.run(["git", "clone", "--depth", "1", method["source"], str(destination)], check=True)
    receipt = verify_repo(key, destination)
    receipt_path = workspace / f"{key}.upstream_source_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def build_upstream_command(key, python, config, output, checkpoint=None, phase="train"):
    entry = get_method(key)
    templates = entry.get("official_train_command" if phase == "train" else "official_test_command")
    if not templates:
        raise ValueError(f"No independently confirmed official {phase} CLI for {key}; follow upstream README")
    replacement = {
        "{python}": str(python), "{config}": str(config), "{work_dir}": str(output),
        "{checkpoint}": str(checkpoint), "{predictions}": str(output),
    }
    return [replacement.get(part, part) for part in templates]


def run_upstream(key, workspace, config, out, phase, checkpoint, execute):
    method = get_method(key)
    source = Path(workspace).resolve() / key
    receipt = verify_repo(key, source)
    config = Path(config).resolve()
    if not config.is_file():
        raise FileNotFoundError(f"A prepared and VERIFIED official data-aware config is required: {config}")
    if phase == "test" and (not checkpoint or not Path(checkpoint).is_file()):
        raise FileNotFoundError("Testing requires an actual upstream checkpoint")
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_upstream_command(key, sys.executable, config, out, checkpoint, phase)
    # For SuperYOLO the --data option must refer to its data/SRvedai.yaml with exact VEDAI paths;
    # for C2Former the config must specify the entire matched paired data+pretrained weights.
    payload = {
        **receipt, "phase": phase, "command": cmd,
        "selected_config": str(config), "selected_config_sha256": sha256(config),
        "execution_requested": execute, "status": "not_started",
        "fairness_notice": "Verify identical splits, class labels, modality pairings and geometry/evaluator before merging metrics.",
    }
    receipt_name = out.parent / f"{key}.{phase}.attempt.json"
    if execute:
        payload["status"] = "running"
        receipt_name.write_text(json.dumps(payload, indent=2) + "\n")
        result = subprocess.run(cmd, cwd=source, check=False)
        payload["status"] = "command_finished" if result.returncode == 0 else "command_failed"
        payload["exit_code"] = result.returncode
        receipt_name.write_text(json.dumps(payload, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"Upstream command failed (exit {result.returncode}). See {receipt_name}")
    else:
        receipt_name.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parse_names(data):
    import yaml
    payload = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    names = payload.get("names")
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
    if not isinstance(names, list) or not names:
        raise ValueError("The dataset YAML must supply an ordered nonempty class 'names' list/dict")
    if len({str(n) for n in names}) != len(names):
        raise ValueError("Dataset class names must be unique")
    return payload, [str(name) for name in names]


def convert_obb_line_to_hbb(line, class_count):
    """YOLO OBB: class x1 y1 ... x4 y4 -> horizontal YOLO class xc yc w h."""
    fields = line.split()
    if len(fields) != 9:
        raise ValueError(f"Expected class + 8 normalized OBB coordinates, got {len(fields)} columns")
    cl = int(fields[0])
    if str(cl) != fields[0] or not 0 <= cl < class_count:
        raise ValueError(f"Invalid class {fields[0]!r} for {class_count} categories")
    coords = [float(x) for x in fields[1:]]
    if not all(0 <= x <= 1 for x in coords):
        raise ValueError("OBB labels must be normalized into [0,1]")
    xs = coords[::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid zero-area oriented bounding box")
    return f"{cl} {(xmin+xmax)/2:.8f} {(ymin+ymax)/2:.8f} {xmax-xmin:.8f} {ymax-ymin:.8f}"


def prepare_hbb(data_yaml, output, copy_images=False):
    """Create auditable HBB data view; not an upstream-specific dataset adapter.

    This makes no claim of geometry-equivalent scores; retrain and re-evaluate
    EVERY algorithm with this common HBB protocol before comparing HBB mAP.
    """
    src, names = parse_names(data_yaml)
    if src.get("source", {}).get("task") not in (None, "obb"):
        raise ValueError("Refusing to convert a non-OBB dataset")
    base = Path(src.get("path", Path(data_yaml).parent))
    if not base.is_absolute():
        base = Path(data_yaml).resolve().parent / base
    base = base.resolve()
    output = Path(output).resolve()
    if output == base or base in output.parents:
        raise ValueError("Refusing to write HBB output inside the OBB source directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must not already contain files: {output}")
    # Fail fast on bad annotations and modality mismatch before touching output.
    inventory = []
    by_split = {}
    for split in ("train", "val", "test"):
        if split not in src:
            continue
        rgb_dir = base / src[split]
        ir_dir = base / src[f"{split}_ir"]
        label_dir = base / src[f"{split}_labels"]
        if not rgb_dir.is_dir() or not ir_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing paired {split} images or labels")
        rgb_files = [p for p in rgb_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        if not rgb_files:
            raise ValueError(f"Empty split: {split}")
        key_set = set()
        for rgb_path in sorted(rgb_files):
            relative = rgb_path.relative_to(rgb_dir)
            key = relative.with_suffix("").as_posix()
            if key in key_set:
                raise ValueError(f"Duplicate RGB basename in {split}: {key}")
            key_set.add(key)
            ir_path = ir_dir / relative
            lbl_path = label_dir / relative.with_suffix(".txt")
            if not ir_path.is_file() or not lbl_path.is_file():
                raise FileNotFoundError(f"Missing IR pair or label for {rgb_path}")
            converted = [convert_obb_line_to_hbb(l, len(names)) for l in lbl_path.read_text().splitlines() if l.strip()]
            inventory.append((split, relative, rgb_path, ir_path, lbl_path, converted))
        by_split[split] = key_set
    splits = list(by_split)
    for idx, a in enumerate(splits):
        for b in splits[idx+1:]:
            dup = by_split[a] & by_split[b]
            if dup:
                raise ValueError(f"Train/validation/test leakage via duplicate filenames: {a} vs {b}: {list(dup)[:2]}")
    output.mkdir(parents=True, exist_ok=True)
    import yaml
    hbb_conf = {"path": str(output), "names": {idx: name for idx, name in enumerate(names)}, "source": {"task": "detect", "converted_from": str(data_yaml), "geometry": "hbb"}}
    fingerprints = hashlib.sha256()
    for split, relative, rgb, ir, label, lines in inventory:
        for mode, file in (("rgb", rgb), ("ir", ir)):
            dest = output / mode / "images" / split / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if copy_images:
                shutil.copy2(file, dest)
            else:
                try:
                    os.link(file, dest)
                except OSError:
                    shutil.copy2(file, dest)
            fingerprints.update(f"{split}/{mode}/{relative}|{file.stat().st_size}\n".encode())
        lab = output / "labels" / split / relative.with_suffix(".txt")
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        fingerprints.update(f"{split}/{relative}:".encode() + label.read_bytes())
    for split in splits:
        hbb_conf[split] = f"rgb/images/{split}"
        hbb_conf[f"{split}_ir"] = f"ir/images/{split}"
        hbb_conf[f"{split}_labels"] = f"labels/{split}"
    (output / "data_hbb.yaml").write_text(yaml.safe_dump(hbb_conf, sort_keys=False), encoding="utf-8")
    manifest = {
        "status": "converted_not_trained", "geometry": "hbb", "num_pairs": len(inventory),
        "split_counts": {split: len(keys) for split, keys in by_split.items()},
        "dataset_contract": "A common HBB re-evaluation of all models is required for fair ranking",
        "image_integrity": "source filename + byte count only (NOT a cryptographic full-image hash)",
        "source_label_and_image_metadata_sha256": fingerprints.hexdigest(),
        "source_dataset_yaml_sha256": sha256(data_yaml),
    }
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def compare_run_receipts(paths, out):
    """Only compare observed upstream results with identical evaluation contracts."""
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if len(rows) < 2:
        raise ValueError("Need >=2 measured evaluation receipts for comparison")
    expected = None
    records = []
    for row in rows:
        name = row.get("method")
        # The proposed model and ETFNet are internal and have their own framework.
        if name not in load_registry() and name not in ("sjpa_tr_proposal", "etfnet_tgf_paper"):
            raise ValueError(f"Unknown model: {name}")
        if row.get("status") != "completed" or row.get("source_kind") != "measured":
            raise ValueError(f"Not an observed completed run: {name}")
        if row.get("geometry") not in ("obb", "hbb"):
            raise ValueError("Must explicitly label geometry as obb or hbb")
        for required in ("dataset_sha256", "split_sha256", "evaluator", "seed", "mAP50", "mAP50_95", "prediction_log", "checkpoint"):
            if required not in row:
                raise ValueError(f"Missing required evidence {required}: {name}")
        if not Path(row["prediction_log"]).is_file() or not Path(row["checkpoint"]).is_file():
            raise FileNotFoundError(f"Missing checkpoint/prediction file required for {name}")
        contract = (row["dataset_sha256"], row["split_sha256"], row["evaluator"], row["geometry"], row["seed"])
        if expected is not None and contract != expected:
            raise ValueError("Cannot combine nonmatching dataset, split, metric geometry, evaluator, or seed")
        expected = contract
        for field in ("mAP50", "mAP50_95"):
            value = float(row[field])
            if not (0 <= value <= 1):
                raise ValueError(f"{field} must be decimal fraction in [0,1], not percentage")
        records.append({field: row[field] for field in ("method", "geometry", "seed", "mAP50", "mAP50_95")})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {"status": "comparison_receipts_consistent_but_not_independently_verified", "rows": len(rows), "csv": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser("list")
    f = commands.add_parser("fetch")
    f.add_argument("--method", required=True, choices=[*load_registry(), "all"])
    f.add_argument("--workspace", type=Path, required=True)
    v = commands.add_parser("verify")
    v.add_argument("--method", required=True, choices=load_registry())
    v.add_argument("--repo", type=Path, required=True)
    r = commands.add_parser("upstream-command")
    r.add_argument("--method", required=True, choices=load_registry())
    r.add_argument("--workspace", type=Path, required=True)
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--phase", choices=("train", "test"), default="train")
    r.add_argument("--checkpoint", type=Path)
    r.add_argument("--execute", action="store_true", help="Actually run upstream command; otherwise only write plan")
    h = commands.add_parser("prepare-hbb")
    h.add_argument("--data", type=Path, required=True)
    h.add_argument("--output", type=Path, required=True)
    h.add_argument("--copy-images", action="store_true")
    c = commands.add_parser("compare")
    c.add_argument("--receipt", nargs="+", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "list":
        for key, obj in load_registry().items():
            print(f"{key:18s} {obj['prediction_geometry'].upper():3s}  {obj['venue']}")
            print(f"    {obj['paper_url']} | {obj['source']}")
    elif args.cmd == "fetch":
        keys = load_registry() if args.method == "all" else [args.method]
        for key in keys:
            print(json.dumps(fetch(key, args.workspace), indent=2))
    elif args.cmd == "verify":
        print(json.dumps(verify_repo(args.method, args.repo), indent=2))
    elif args.cmd == "upstream-command":
        print(json.dumps(run_upstream(args.method, args.workspace, args.config,
                                      args.out, args.phase, args.checkpoint, args.execute), indent=2))
    elif args.cmd == "prepare-hbb":
        print(json.dumps(prepare_hbb(args.data, args.output, args.copy_images), indent=2))
    elif args.cmd == "compare":
        print(json.dumps(compare_run_receipts(args.receipt, args.out), indent=2))


if __name__ == "__main__":
    main()
