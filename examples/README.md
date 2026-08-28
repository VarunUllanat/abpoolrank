# Examples

## Your own predictions

One subdirectory per predictor:

```
predictions/
  boltz2/          pose_0.cif  pose_1.cif
  protenix/        sample_00.cif  sample_01.cif
  my_new_model/    a.pdb  b.pdb
```

```bash
cofoldrank select predictions/ --workers 8 --csv ranking.csv
```

```python
from cofoldrank import PoolSelector

result = PoolSelector(workers=8).rank("predictions/")
print(result.best.path)
result.to_csv("ranking.csv")
```

Three things to get right:

* **Chain naming** -- antibody chains default to `H`/`L`, everything else is the antigen. Pass
  `--antibody-chains` / `antibody_chains=(...)` if your files differ; a VHH pool uses just `H`.
* **At least two poses per predictor**, or the source-balanced feature degenerates (the API warns).
* **At least two predictors**, or there is no cross-source agreement to exploit (the API warns).

## Runnable demo

```bash
python examples/make_demo_pool.py            # writes examples/demo_pool/
cofoldrank select examples/demo_pool --workers 8
```

`make_demo_pool.py` copies real predictions from the study into the layout above and prints each pose's
DockQ, plus what each predictor's own confidence picks from all 100 of its samples. On the default
target the selector picks a DockQ 0.790 pose out of 10 structures; the best any predictor's confidence
manages is 0.739 out of 100.

Use `--target` for a different complex and `--per-source` to change the pool size.

## Scripts

| File | Does |
|---|---|
| `rank_predictions.py` | ranks a directory and prints the table, per-predictor shortlist and features |
| `make_demo_pool.py` | builds a demo pool from the study's predictions (needs the study data) |
