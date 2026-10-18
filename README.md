# ETFNet SJPA-TR — paper-grounded baseline and ablation release

This repository retains **one independently published comparison baseline**:
[ETFNet–TGF](https://doi.org/10.1016/j.inffus.2026.104658), originating from the
[authors' official ETFNet code](https://github.com/lansimtech/ETFNet).
Other experiment variants are **ablation/control studies** or **our proposed SJPA-TR**,
**not claimed as independent paper reproductions**. The arbitrary RGB-only comparisons
and obsolete SJPA prototype have been removed.

## Getting started (flat GitHub layout)

The ZIP intentionally uses folders of **one level only**, **no numeric prefixes** and
**no more than 50 files per folder**. Run the safe local source restoration before tests:

```bash
python layout_manager.py check
python layout_manager.py restore
pip install -e .
python tools/audit_baselines.py --forward --imgsz 64
python -m pytest -q
```

The restored original Python package tree is ignored by Git. After editing a restored
file, run `python layout_manager.py sync` before committing. Do not upload both the
flat packs and the materialized nested source tree.

## Experiment roles

- **Published-paper reproduction**: `etfnet_tgf_paper` (6-channel ETFNet CAFEM+TGF);
  official YAML graph and module arguments manually checked against the authors' source.
- **Ablation/control**: `dual_concat`, `dual_add`, `cafem_only`, `tgf_only`,
  `goci_no_spatial`, `sjpa_no_reliability`, `rtpf_safety_ablation`.
- **Proposal**: `sjpa_tr_proposal` (6-channel SJPA-TR corrected configuration).

See `BASELINE_AUDIT.md` for provenance, comparability caveats and complete mapping.
The `MPCRF` YAML exists only to preserve its unit/architecture test; it is **not** a
benchmark paper baseline and is excluded from Kaggle's comparison matrix.

The Kaggle runner is `kaggle/ETFNet_Paper_Baseline_Ablations_Kaggle.py` after restore.
The zip-capable script can also clone the GitHub flat package directly. See
`KAGGLE_QUICKSTART.md` and `PUBLIC_DATASETS.md`.

**Reproduction scope:** Full public-dataset training, checkpoint comparison, and
published mAP equivalence are not established by CPU smoke tests. No results are
invented. All models must be trained on identical datasets/splits/seeds before
paper-quality numerical claims.

AGPL-3.0 license preserved in `LICENSE.txt`.
