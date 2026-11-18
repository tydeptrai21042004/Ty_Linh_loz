# CURRENT WORKFLOW (2026-11-18) — 3 datasets / 111 training jobs

Read **[RESEARCH_111_RUN_PLAN.md](RESEARCH_111_RUN_PLAN.md)** and **[KAGGLE_QUICKSTART.md](KAGGLE_QUICKSTART.md)**. 3 DOI-linked datasets (VEDAI, DroneVehicle, M3FD); 13 one-seed hyperparameter trials, 9 three-seed main paper comparisons, 15 three-seed MR-SJPA ablations **per dataset**. No hyperparameter confirmation stage. Author-source baselines only (ETFNet, C²Former), with no fake method substitution. The included legacy sections below describe earlier smaller-run workflows and do not supersede the current plan. Full Kaggle GPU and public archive execution remain unverified.

**Latest paper-backed benchmark:** See [PAPER_BENCHMARK_E2E.md](PAPER_BENCHMARK_E2E.md) and [KAGGLE_QUICKSTART.md](KAGGLE_QUICKSTART.md). Main matrix is only MR-SJPA versus the ORIGINAL ETFNet author source; C²Former runs through its ORIGINAL MMRotate framework in an optional separately prepared environment. Three paired seeds, independent held-out OBB test, shared polygon AP101, statistical receipts, and automatic VEDAI acquisition are implemented. GPU/cloud training remains unverified. No custom fusion-control variant is called a published-paper baseline.

# MR-SJPA — RGB–IR OBB detection

The **only new proposed model** is **Moment-Preserving Residual Spatial Joint
Procrustes Alignment (MR-SJPA)**. It retains the dual-stream YOLO OBB detector
and substitutes one fusion layer. No synthetic fusion choice is reported as an
independent research-paper baseline.

## Genuine paper comparison

| Method | Role | Author code | Status |
|---|---|---|---|
| MR-SJPA | Proposal | this repository | CPU module/graph tested; accuracy unmeasured |
| ETFNet (Information Fusion 2026) | Published paper baseline | https://github.com/lansimtech/ETFNet | Kaggle author-source runner supplied; accuracy unmeasured |
| C²Former (IEEE TGRS 2024) | Published OBB paper baseline | https://github.com/yuanmaoxun/C2Former | Independent MMRotate runner; requires matching dataset adapter/environment; accuracy unmeasured |

The Kaggle script evaluates the first two only. C²Former requires a separate
MMRotate environment; the repository has a fail-closed helper in
`official_baseline_runs.py` to verify and execute an adapted official config.
Do not pool results from incompatible HBB/OBB tasks or dataset splits.

The old custom `Concat`, `Add`, `CAFEM-only`, `TGF-only`, `GOCI`, and `RTPF`
comparisons have been removed **from the benchmark**, not relabeled as papers.
A few internal reference modules and unit-test fixtures remain solely for
regression checks and are not baseline entries.

## Run

```bash
python layout_manager.py restore
pip install -e . --no-deps
pytest -q
python tools/audit_baselines.py --imgsz 64
python etfnet_cli.py train --data /path/to/paired_obb.yaml --epochs 100 --batch 4
```

Kaggle: upload `ETFNet_MR_SJPA_PaperOnly_fixed.zip` and run
`kaggle/PaperOnly_ETFNet_MR_SJPA_Kaggle.py` with GPU and Internet on.
It fetches **the real ETFNet author repository**; if the author code cannot be
trained on the prepared paired dataset, the run fails without generating a
fake baseline score. Set `ETFNET_OFFICIAL_COMMIT` to a verified full SHA for
locked experiments. See `PAPER_ONLY_PROTOCOL.md` for caveats and instructions.

## Mathematical contract

For paired feature maps and `Z=concat(RGB,IR)`, the output is `Z+Δ` with
`||Δ||_F <= ρ ||Z||_F` per sample. The candidate transforms use per-sample,
point-sampled covariance with eigenvalue regularization; shifts are scored on the valid spatial
overlap and applied once at full resolution; residual projection preserves the
original representation. The residual weights start at zero and learn from the
backpropagated detection loss.

The proposal has **not** been shown to improve mAP. Per-sample SVD/eigh is
currently PyTorch-only; ONNX/TensorRT equivalence is **not certified**.

License: see `LICENSE.txt`.
