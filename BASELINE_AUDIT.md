# Audited baseline provenance and ablations

Only an independently published **paper reproduction candidate** is called a *paper
baseline*. Other retained options are *ablation controls*, even when the relevant
operator (concat, add, CAFEM, TGF, GOCI, RTPF) has existing literature.

**Paper**: Thi Lan Nguyen, Cao Truong Tran, Dinh Tan Nguyen, “ETFNet: An efficient
transformer-based RGB–IR fusion network for UAV object detection,” *Information Fusion*,
DOI **10.1016/j.inffus.2026.104658**.

**Authors' source code**: https://github.com/lansimtech/ETFNet

**Official configuration**:
https://raw.githubusercontent.com/lansimtech/ETFNet/main/ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_TGF.yaml

The local `etfnet_P2_CAFEM_TGF.yaml` was compared at the layer graph/module argument
level to the released official model YAML (six-channel input, CAFEM, TGF arguments,
Add2 residual reconstruction and OBB head). Local `scale: s` is an explicit model
scale selector consistent with the official `scales.s` settings. This is
**architecture-level provenance, not proof that reproduced metrics or weights match**.

| Run identifier | Role | Local YAML | Scope |
|---|---|---|---|
| `etfnet_tgf_paper` | Paper reproduction candidate | `etfnet_P2_CAFEM_TGF.yaml` | ETFNet full CAFEM + TGF graph |
| `dual_concat` | Ablation/control | `etfnet_dualstream_noCAFEM_noTGF.yaml` | Matched-backbone direct concat |
| `dual_add` | Ablation/control | `noCAFEM_noTGF_add.yaml` | Fusion operator substitution, not parameter matched |
| `cafem_only` | Ablation/control | `etfnet_CAFEM_noTGF.yaml` | ETFNet component ablation |
| `tgf_only` | Ablation/control | `etfnet_noCAFEM_TGF.yaml` | ETFNet component ablation |
| `goci_no_spatial` | Ablation/control | `etfnet_P2_CAFEM_GOCI.yaml` | Spatial-score removal study; different backbone from proposal |
| `sjpa_no_reliability` | Ablation/control | Generated from corrected SJPA-TR YAML | Exactly disables SJPA reliability bool |
| `rtpf_safety_ablation` | Ablation/control | `etfnet_P2_C3k2_RTPF.yaml` | Matched-backbone fusion operator control |
| `sjpa_tr_proposal` | Proposed method | `etfnet_P2_C3k2_SJPA_TR_Corrected.yaml` | Final proposed SJPA-TR |

Seven ablations and one proposal accompany one paper baseline. All use paired
six-channel RGB–IR input and an OBB head. **Not all ablations are single-component,
parameter-matched changes**; only `dual_concat`, `sjpa_no_reliability` and
`rtpf_safety_ablation` have the matching reference stem and head described in the
audit code. Report parameter count/GFLOPs and architecture differences for all.

## Deliberately removed

- `rgb_p3p5` (`noCAFEM_noTGF.yaml`): custom RGB reference, not supported as a
  faithful recreation of an independently published full baseline;
- `rgb_yolo11_p2p5` (`etfnet_yolo11.yaml`): modified RGB-only YOLO architecture,
  not verified equivalent to the official paper or released YOLO11 OBB model;
- `sjpa_cafem_legacy` (`etfnet_P2_CAFEM_SJPA.yaml`): obsolete internal prototype,
  not an independent published baseline or isolated full-proposal ablation;
- additional unused architecture YAMLs not present in the reproducible benchmark.

The test-only MPCRF YAML is retained for mathematical/architecture verification,
not included in benchmark result tables.

## What is and is not verified

`python tools/audit_baselines.py --forward --imgsz 64` validates channel counts,
OBB heads, expected fusion modules, matched reference graph constraints, model
construction and an FP32 CPU forward pass. It does **not** establish dataset
convergence, published mAP reproduction, original checkpoint weight equivalence,
or statistical differences across seeds.

The paper's own benchmark datasets are DroneVehicle, VTUAV-Det and
RGBTDronePerson; experiments on VEDAI are a new cross-dataset evaluation, **not**
a claim of reproducing the paper's published dataset metrics.

## Verified local model smoke results

All models below were constructed and run once in eval/FP32 on a `1×6×64×64`
synthetic tensor on CPU. Parameters count trainable and nontrainable parameters
without dataset-specific classifier remapping. This does not reproduce the paper's
mAP, training setup, full-resolution performance or release weights.

| Model | Evidence group | Model parameters | CPU forward |
|---|---|---:|---|
| `etfnet_tgf_paper` | `paper_reproduction` | 10,540,023 | PASS |
| `dual_concat` | `ablation` | 11,205,570 | PASS |
| `dual_add` | `ablation` | 11,309,186 | PASS |
| `cafem_only` | `ablation` | 9,981,746 | PASS |
| `tgf_only` | `ablation` | 12,699,623 | PASS |
| `goci_no_spatial` | `ablation` | 9,981,746 | PASS |
| `sjpa_no_reliability` | `ablation` | 11,205,570 | PASS |
| `rtpf_safety_ablation` | `ablation` | 11,205,571 | PASS |
| `sjpa_tr_proposal` | `proposal` | 11,205,570 | PASS |

The source ZIP passes 112/112 existing pytest cases after restoration.
