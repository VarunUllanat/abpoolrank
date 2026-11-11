# cofoldrank

Pick the best predicted antibody--antigen structure out of a pool of predictions from several co-folding
predictors, using pose geometry alone (no confidence scores, no predictor identity, no reference
structure).

![Success@1 against the number of predicted structures per target](assets/figure1b.png)

## Install

```bash
pip install -e .            
```

## Run

Put each predictor's structures in its own subdirectory. The predictor names are arbitrary; they only tell the
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
| `vhh-only` | VHHs / nanobodies only (197) | VHHs |

```bash
cofoldrank select predictions/ --model vhh-only
```

```python
result = PoolSelector(model="paired-only").rank("predictions/")
```

Use **at least two poses per predictor**. With one, the source-balanced feature reduces to plain
pose-similarity support and the selector warns.

## Results


Example output:

```
pool: 15 poses from 5 sources

best: predictions/protenix_v2/pose_1.pdb
      source=protenix_v2  score=0.6811

top 5:
    1. +0.6811  protenix_v2        pose_1.pdb
    2. +0.6781  protenix_v2        pose_0.pdb
    ...
```

Use `result.best.path` -- that is the structure to take forward. 

`result.table` (or `--csv`) gives the full ranking with the four features behind it:

| rank | source | score | path | pss_pct | srcbal_pct | epi_pct | log_pool |
|---|---|---|---|---|---|---|---|

