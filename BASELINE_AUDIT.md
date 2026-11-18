# Paper-only baseline audit

Local proposal: MR-SJPA; local CPU OBB graph validation is available with
`python tools/audit_baselines.py --imgsz 64`.

Published OBB comparison candidates: **ETFNet** (Nguyen et al. 2026,
10.1016/j.inffus.2026.104658; official authors' repository) and **C²Former**
(Yuan & Wei 2024, 10.1109/TGRS.2024.3376819; official authors' repository).
Neither is declared accuracy-reproduced without a completed official-source
training and evaluation run. The Kaggle runner supports ETFNet author-source
training; C²Former is a separate MMRotate workflow.

Former Concat, Add, CAFEM-only, TGF-only, GOCI, RTPF and no-reliability graphs
are not research-paper baselines and are excluded from benchmark reports.
