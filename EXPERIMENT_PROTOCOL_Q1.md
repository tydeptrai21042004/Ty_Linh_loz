# Experimental protocol — MR-SJPA with genuine published baselines

## Main comparison

- MR-SJPA: this repository, one RGB/IR OBB fusion proposal.
- ETFNet: Nguyen et al., Information Fusion (2026), DOI
  10.1016/j.inffus.2026.104658; **train original author-source code**.
- C²Former: Yuan & Wei, IEEE TGRS (2024), DOI
  10.1109/TGRS.2024.3376819; **train original MMRotate/S²ANet code**.

Use the same paired OBB dataset, split, class map and evaluation definition.
Results using a different detection head or environment must disclose those
conditions. VEDAI is a cross-dataset experiment, not a reproduction of
ETFNet's original published DroneVehicle/VTUAV/RGBTDronePerson numbers.

## Design verification versus results

- CPU tests establish code invariants only: finite numerical paths,
  zero-initialization, per-sample trust-radius bound and shift recovery.
- GPU training, native-oriented test annotations, real mAP, latency and
  export equivalence remain to be empirically established.
- Do not merge horizontal-box scores into the OBB table.
- Report all random seeds and exact checkpoint hashes; select on validation
  and evaluate on a distinct test set only if a genuine held-out test exists.

## Suggested report columns

Model, paper DOI, upstream commit, dataset revision, train/val/test split,
OBB evaluator, input size, epoch count, seed, mAP50, mAP50–95, FPS,
parameters, GPU memory. Empty data is blank, never zero or invented.

See PAPER_ONLY_PROTOCOL.md for acceptance requirements.
