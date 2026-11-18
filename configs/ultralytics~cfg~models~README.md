# OBB model configurations

Main proposed architecture:
`etfnet/etfnet_P2_C3k2_SJPA_TR_Corrected.yaml` — now uses `MR_SJPA`.

`etfnet/etfnet_P2_CAFEM_TGF.yaml` is retained for local graph inspection;
**the reported ETFNet paper baseline must be trained from the original authors'**
https://github.com/lansimtech/ETFNet repository, as in the Kaggle script.

`MPCRF`/`Concat` configs exist only for regression tests, not paper-baseline
entries. Unsupported custom variants have been removed.
