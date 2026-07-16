# UK headlines results (frozen UK test)

| System | UK test accuracy | UK test macro-F1 | UK test micro-F1 |
|---|---:|---:|---:|
| NB @ AG | 0.8467 | 0.8351 | 0.8467 |
| DistilBERT @ AG (before) | 0.8812 | 0.8695 | 0.8812 |
| NB @ UK | 0.8812 | 0.8744 | 0.8812 |
| DistilBERT UK only (no AG) | 0.9464 | 0.9432 | 0.9464 |
| DistilBERT UK fine-tuned (after) | 0.9579 | 0.9554 | 0.9579 |

Macro-F1 delta (AG→UK after − AG before): **+0.0858**.
Micro-F1 delta (AG→UK after − AG before): **+0.0766**.
Macro-F1 delta (AG→UK after − UK only): **+0.0122**.
Micro-F1 delta (AG→UK after − UK only): **+0.0115**.

Notes:
- Same frozen UK test set (seed 42, stratified 70/15/15) for all systems.
- UK only = DistilBERT fine-tuned from base on UK train (no AG Stage 1).
- AG→UK after = continued fine-tune from the AG Stage-1 checkpoint.
- AG `World` was mapped to `politics` (imperfect but intentional).
- BBC `entertainment` rows were dropped.
- For single-label multi-class, micro-F1 often equals accuracy; both are reported for lect2 comparison.
