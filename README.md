# cofoldrank

Pick the best predicted antibody--antigen structure out of a pool of predictions from several co-folding
predictors, using pose geometry alone -- no confidence scores, no predictor identity, no reference
structure.

![Success@1 against the number of predicted structures per target](assets/figure1b.png)

Ten structures drawn two at a time from five predictors (teal) reach **55.8%** medium-or-better
Success@1 (DockQ >= 0.49) on 103 post-2021 complexes -- above every individual predictor ranked by its
own confidence at 100 structures (faint lines, 26--52%). The dotted segment below ten structures is one
pose per predictor, where the selector's source-balanced feature degenerates. Bands are marginal
target-bootstrap 95% intervals; every arm is scored on the same targets, so their overlap does not by
itself imply no difference. Method and evaluation are described in the manuscript.

## Install

```bash
pip install -e .            # or: pip install -e '.[dev]' to run the tests
```

Three trained selectors (~2.6 MB in total) ship with the package -- nothing to download. See
[Choosing a model](#choosing-a-model); `both` is the default. Use your own checkpoint with
`$COFOLDRANK_SELECTOR` or `PoolSelector(checkpoint=...)`.

## Run

Put each predictor's structures in its own subdirectory. The names are arbitrary; they only tell the
selector which poses share an origin.

```
predictions/
  boltz2/        pose_0.cif  pose_1.cif
  protenix/      sample_00.cif  sample_01.cif
  my_new_model/  a.pdb  b.pdb
```

```bash
cofoldrank select predictions/ --workers 8 --csv ranking.csv
```

```python
from cofoldrank import PoolSelector

result = PoolSelector(workers=8).rank("predictions/")
print(result.best.path)
```

Options: `--model` (see below), `--antibody-chains` (default `H,L`; all other chains are the antigen),
`--max-per-source`, `--top`, `--csv`. Accepts `.cif`, `.mmcif`, `.pdb`, `.ent`.

### Choosing a model

| `--model` / `model=` | Trained on | Use for |
|---|---|---|
| `both` (default) | paired antibodies and VHHs (608 complexes) | any antibody type; the selector reported in the manuscript |
| `paired-only` | paired heavy + light chain antibodies only (411) | paired antibodies |
| `vhh-only` | VHHs / nanobodies only (197) | VHHs, mainly when pooling many structures per target |

```bash
cofoldrank select predictions/ --model vhh-only
```

```python
result = PoolSelector(model="paired-only").rank("predictions/")
```

All three use the same features and settings; only the training targets differ. On the 103 post-2021
test complexes, scored only on targets of the matching type:

- **Paired (62 targets):** `paired-only` and `both` are indistinguishable at every pool size (56.3% vs
  56.1% at 5 predictors x 2 structures).
- **VHH (41 targets):** `vhh-only` is lower at 5 x 2 (53.5% vs 55.2%), mostly because it has less training
  data. It is higher with large pools (61.4% vs 57.9% at 5 x 20). That gain rests on a handful of targets
  and has not been confirmed on an independent cohort.

If unsure, keep `both`. A model named with `--model` is always used as named; `$COFOLDRANK_SELECTOR` only
replaces the default.

Use **at least two poses per predictor**. With one, the source-balanced feature reduces to plain
pose-similarity support and the selector warns.

## Results

Selector 1.1.0 is trained on 608 of the 614 training complexes. The 1.0.0 checkpoint had silently used only
264, because whole targets with any unlabelled source were dropped; the trainer now reports every skipped
target. On the same 103 test complexes and pool draws, 1.1.0 scores 55.8% at 5x2 against 55.4% for 1.0.0.

Example output, from `cofoldrank select examples/demo_pool`:

```
pool: 10 poses from 5 sources (boltz2=2, esmfold2_fast=2, opendde=2, promera=2, protenix_v2=2)

best: predictions/protenix_v2/pose_1.pdb
      source=protenix_v2  score=0.6811

top 5:
    1. +0.6811  protenix_v2        pose_1.pdb
    2. +0.6781  protenix_v2        pose_0.pdb
    ...
```

Use `result.best.path` -- that is the structure to take forward. The score is a within-pool ordering,
not a quality estimate: it is comparable between poses in the same pool and meaningless across pools.

`result.table` (or `--csv`) gives the full ranking with the four features behind it:

| rank | source | score | path | pss_pct | srcbal_pct | epi_pct | log_pool |
|---|---|---|---|---|---|---|---|

`result.candidates` is the same ordering as `Candidate` objects, `result.top(k)` the first `k`, and
`result.warnings` anything questionable about the pool (poses that failed to parse, sources with too few
poses).

See `examples/` for a runnable demo.
