# Verification report — 2026-10-20

- Uploaded project: `Ty_Linh_loz-main.zip`; original source and SJPA-TR implementation retained unchanged.
- Added four independently documented upstream methods; CFT cited correctly as **2021 preprint**, not a journal article.
- Source contracts inspected using author-released source: C²Former `mmrotate/datasets/dronevehicle.py`, SuperYOLO `utils/datasets.py` and `train.py`, ICAFusion `test.py`, CFT `README.md` and train/test scripts.
- Converter tests cover C²Former dual `.jpg` naming and `*_tir.txt`, MMRotate child class count; ICAFusion validation/test separation; SuperYOLO true 512 and 1024 image tiers and exact author loader filename; mismatched pair rejection; CFT transformerx3 guard.
- Common HBB evaluator tests cover perfect AP, wrong-class/unpatched output, false-positive AP penalty, heldout-ID checks, SuperYOLO ID remapping, empty prediction rejection, receipt sha256 protection, OBB–HBB mismatch rejection, ICAFusion I/O patch backup.
- **19 integration/offline tests passed** under Python in this environment; no external repo GPU training, CUDA install, author's checkpoints, real project dataset AP, or full Q1 paper-result reproduction were performed here.
- Caveat: because the original baseline authors use separate libraries, version incompatibilities can appear; each baseline needs pinned upstream environment and genuine paired dataset. Independent end-to-end verification is pending, particularly MMRotate OBB result export and SuperYOLO HR training source.
- The ICAFusion test-output patch is deliberately limited to file locations and output box/class formatting; detector model and NMS are unchanged.
- Common HBB AP101 is a documented shared evaluator, not the official COCOeval algorithm. Native author mAP numbers and cross-architecture AP101 must not be mixed in the same table.
