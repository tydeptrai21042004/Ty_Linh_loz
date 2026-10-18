"""GitHub single-level 50-file layout: verify, restore, or repack source files.

The packed files are the *committed* source of truth. Materialization regenerates
ignored original working paths so all existing imports, tests and YAML paths work.
No network access or external dependencies are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "LAYOUT_MANIFEST.json"


def _load():
    info = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if info.get("schema") != 1 or info.get("depth") != 1 or info.get("limit") != 50:
        raise RuntimeError("Unknown or invalid flat package manifest")
    originals, packed = set(), set()
    for row in info["files"]:
        for key in ("original", "packed"):
            value = row[key]
            part = PurePosixPath(value)
            if part.is_absolute() or ".." in part.parts or "." in part.parts or len(part.parts)<1:
                raise ValueError(f"Unsafe path in manifest: {value!r}")
        if len(PurePosixPath(row["packed"]).parts) != 2:
            raise ValueError("Pack files must be exactly one directory deep")
        if row["original"] in originals or row["packed"] in packed:
            raise ValueError("Duplicate path in flat manifest")
        originals.add(row["original"])
        packed.add(row["packed"])
    return info


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    """Check SHA-256, packed file inventory, directory depth and <=50 files."""
    info = _load()
    errors=[]
    packs={}
    expected=set()
    for row in info["files"]:
        name=row["packed"]
        path=ROOT / name
        expected.add(name)
        pack=name.split("/",1)[0]
        packs[pack]=packs.get(pack,0)+1
        if pack[0].isdigit():
            errors.append(f"Numeric directory prefix disallowed: {pack}")
        if not path.is_file():
            errors.append(f"Missing: {name}")
        elif _sha(path)!=row["sha256"] or path.stat().st_size!=row["bytes"]:
            errors.append(f"Changed without sync/reseal: {name}")
    for pack,count in packs.items():
        if count>50:
            errors.append(f"Over 50 packed files: {pack}: {count}")
        pack_dir=ROOT/pack
        if not pack_dir.is_dir():
            errors.append(f"Missing pack folder: {pack}")
        else:
            for path in pack_dir.rglob("*"):
                if path.is_dir(): errors.append(f"Nested directory not allowed: {path.relative_to(ROOT)}")
                elif path.is_file() and path.relative_to(ROOT).as_posix() not in expected:
                    errors.append(f"Unlisted packed file: {path.relative_to(ROOT)}")
    print(f"Flat-layout audit: {len(info['files'])} files, {len(packs)} top-level folders; max {max(packs.values(),default=0)} files/folder.")
    if errors:
        for error in errors[:30]:print("ERROR",error,file=sys.stderr)
        raise RuntimeError(f"Flat layout verification failed: {len(errors)} errors")
    return info


def materialize(force=False):
    """Re-create original Python package/test/config layout from flat source packs.

    On repeated calls, identical files are skipped; locally edited working files
    are protected until the user explicitly passes force=True.
    """
    info = verify()
    created=0
    for row in info["files"]:
        packed = ROOT / row["packed"]
        target = ROOT / row["original"]
        if target.exists():
            if target.is_file() and _sha(target)==row["sha256"]:
                continue
            if not force:
                # Local modifications are preserved. Use `sync` to commit their bytes.
                continue
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(packed,target)
        created+=1
    return created


def sync():
    """Copy edits in regenerated working source back to committed flat files."""
    info = _load()
    changed=0
    for row in info["files"]:
        target=ROOT/row["original"]
        packed=ROOT/row["packed"]
        if not target.is_file():
            continue
        digest=_sha(target)
        if digest != _sha(packed):
            shutil.copy2(target,packed)
            changed+=1
        row["sha256"] = digest
        row["bytes"] = packed.stat().st_size
    MANIFEST.write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    verify()
    return changed


def reseal():
    """Recalculate hashes after editing files directly in a flat source pack."""
    info=_load()
    for row in info["files"]:
        packed=ROOT/row["packed"]
        row["sha256"]=_sha(packed)
        row["bytes"]=packed.stat().st_size
    MANIFEST.write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    verify()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",nargs="?",choices=("restore","check","sync","reseal"),default="restore")
    p.add_argument("--force",action="store_true",help="For restore: overwrite edited generated files")
    args=p.parse_args()
    if args.command=="check":verify()
    elif args.command=="restore":
        print(f"Restored {materialize(force=args.force)} original-path files.")
    elif args.command=="sync":
        print(f"Synchronized {sync()} edited files back into flat packs.")
    elif args.command=="reseal":reseal()

if __name__=="__main__":main()
