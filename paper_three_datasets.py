"""One-command, DOI-traceable preparation of three paired RGB/IR benchmarks.

This tool ONLY prepares real datasets. It never synthesizes images or labels and
never replaces an unavailable dataset with a different benchmark. All sources
require the user's license acceptance. The final registry shows paper DOIs,
annotation provenance, split counts, and the exact file paths used for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper_benchmark_dataset import prepare as create_heldout
from ultralytics.data.public_rgb_ir import get_public_dataset, prepare_public_dataset
from ultralytics.data.rgb_ir_check import validate_dataset

CORE = ("vedai", "dronevehicle", "m3fd")
GEOMETRY = {"vedai": "native_obb", "dronevehicle": "native_obb", "m3fd": "hbb_axis_aligned_polygons"}


def run(names=CORE, *, output="datasets", accept_license=False, archives=None,
        quick_limit=None, split_seed=991, test_fraction=0.20,
        validation_fingerprint="metadata", build_heldout=True):
    if not accept_license:
        raise PermissionError("Review each source license and pass --accept-license")
    names = tuple(names)
    if not names or len(names) != len(set(names)) or any(name not in CORE for name in names):
        raise ValueError(f"Choose unique datasets from {', '.join(CORE)}")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    archives = archives or {}
    if quick_limit is not None and quick_limit < 20 and build_heldout:
        raise ValueError("For a source-train holdout use --quick-limit >=20, or --no-heldout")
    records = []
    for name in names:
        spec = get_public_dataset(name)
        print(f"\n=== {name} | paper DOI: {spec.citation_doi} ===", flush=True)
        source = prepare_public_dataset(
            name, out, variant="512" if name == "vedai" else "default",
            task="obb", accept_license=True, archive=archives.get(name),
            limit=quick_limit, split_seed=split_seed,
        )
        source = Path(source).resolve()
        final = source
        if build_heldout:
            final = create_heldout(source, out / name / "paper_heldout",
                                   test_fraction=test_fraction, split_seed=split_seed,
                                   uniform_jpeg=True)
        report = validate_dataset(str(final), task="obb", fingerprint=validation_fingerprint)
        if not report.get("ok"):
            raise RuntimeError(f"Strict RGB/IR validation failed for {name}: {report.get('errors')}")
        dataset_meta = json.loads((source.parent / "SOURCE_MANIFEST.json").read_text())
        splits = dataset_meta["statistics"]["splits"]
        if build_heldout:
            split_manifest = json.loads((final.parent / "heldout_manifest.json").read_text())
            splits = split_manifest["counts"]
        if any(not splits.get(part) for part in ("train", "val", "test") if build_heldout):
            raise RuntimeError(f"Missing independent dataset split for {name}: {splits}")
        records.append({
            "dataset": name, "paper_title": spec.title, "paper_doi": spec.citation_doi,
            "paper_url": "https://doi.org/" + spec.citation_doi,
            "dataset_deposit_doi": spec.dataset_doi,
            "homepage": spec.homepage,
            "source_download_url": spec.download_url,
            "source_mirror_not_official": name == "dronevehicle",
            "annotation_geometry": GEOMETRY[name],
            "annotation_policy": "RGB-side source polygons shared across paired images" if name == "dronevehicle" else "original source annotation",
            "source_yaml": str(source), "ready_yaml": str(final), "splits": splits,
            "source_manifest_sha256": hashlib.sha256((source.parent / "SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
            "validated": True,
        })
        print(f"READY: {name}: {final} | {splits}", flush=True)
    payload = {
        "benchmark": "three_original_papers_RGB_IR_dataset_support",
        "datasets": records, "seed": split_seed, "test_fraction": test_fraction if build_heldout else None,
        "metrics": "OBB polygon detection; M3FD polygons represent source horizontal boxes",
        "claims": "local adapter validation; author-model GPU training has not been verified by this command",
    }
    manifest = out / "THREE_DATASET_BENCHMARK.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nPrepared {len(records)} real published datasets; manifest: {manifest}", flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=CORE, default=list(CORE))
    p.add_argument("--output", default="datasets")
    p.add_argument("--accept-license", action="store_true", help="You must review each dataset license")
    p.add_argument("--archive", action="append", default=[], metavar="NAME=PATH",
                   help="Optional local official source for an unavailable provider; repeat per dataset")
    p.add_argument("--quick-limit", type=int, default=None, help="Per-split source limit for local testing only")
    p.add_argument("--split-seed", type=int, default=991)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--fingerprint", choices=("none", "metadata", "sha256"), default="metadata")
    p.add_argument("--no-heldout", action="store_true", help="Only dataset conversion, not paper evaluation")
    args = p.parse_args()
    archives = {}
    for token in args.archive:
        if "=" not in token: p.error("--archive must be NAME=PATH")
        name, path = token.split("=", 1)
        if name not in CORE or not Path(path).exists():p.error(f"Bad local archive: {token}")
        archives[name] = path
    run(args.datasets, output=args.output, accept_license=args.accept_license,
        archives=archives, quick_limit=args.quick_limit,
        split_seed=args.split_seed, test_fraction=args.test_fraction,
        validation_fingerprint=args.fingerprint, build_heldout=not args.no_heldout)


if __name__ == "__main__":
    main()
