# Accuracy Sprint v7A results: H/M measurement-cube development gate

Date: 2026-04-30

## Summary

v7A completed the planned first-stage upgrade from scalar features to measurement-cube features using the existing v6c raw hits/metadata. No Geant4 rerun, shadow seeds, or final seeds were used.

The v7A development gate failed. This means the current v6c raw observation set, even when represented as a multi-source/multi-detector spatial measurement cube and trained with hard-negative variants, does not provide enough stable H/M separation.

This is not a reason to return to blind scalar-feature tuning. It is the planned trigger for v7B: a new hard-negative Geant4 matrix with stronger physical sampling.

## Run Status

- export script: `analysis/export_measurement_cube_v7a.py`
- training script: `analysis/train_hm_v7a.py`
- output directory: `results/accuracy_v3/v7a_hm_measurement_cube/`
- tensor shape: `(360, 18, 2, 8, 8, 7)`
- samples: `360`
- cube features: `16128`
- model features with thickness: `16129`
- validation samples: `120`
- validation support per class: `60`
- shadow/final used: `false`

## Gate Result

Observed best development metrics:

- selected method: `HardNegativeXGBoost`
- selected round: `2`
- H/M min recall: `0.5166666666666667`
- pairwise H/M min recall: `0.5166666666666667`
- Hematite recall: `0.5666666666666667`
- Magnetite recall: `0.5166666666666667`
- validation support per class: `60`
- gate passed: `false`

Required thresholds:

- H/M min recall >= `0.75`
- pairwise H/M min recall >= `0.72`
- Hematite recall >= `0.72`
- Magnetite recall >= `0.72`
- validation support per class >= `60`

## Interpretation

v7A successfully tested the representation upgrade but did not improve the H/M bottleneck. The best score is below the v6c scalar-feature plateau, which suggests that raw spatial cube features from the existing v6c observations are too sparse or too weakly calibrated to separate Hematite/Magnetite by themselves.

The failure remains a direct mutual confusion:

- Hematite support `60`, recall `0.5666666666666667`, misses `26`, all predicted as Magnetite.
- Magnetite support `60`, recall `0.5166666666666667`, misses `29`, all predicted as Hematite.

View ablation did not reveal a hidden strong source of signal. The best view-level H/M min recall stayed below the gate:

- all views: `0.43333333333333335`
- transmission only: `0.3333333333333333`
- side scatter only: `0.4166666666666667`
- high energy only: `0.4666666666666667`
- normal wide only: `0.36666666666666664`

## Decision

v7A has completed its role:

- It preserved the paper narrative that traditional scalar-feature classifiers were a valid baseline but insufficient for fine-grained H/M separation.
- It tested whether richer representation alone could rescue the existing v6c data.
- It failed the preregistered gate after the planned repeat training allowance.

The next phase is v7B, not more v7A tuning. v7B should generate a new hard-negative Geant4 matrix with higher photon budget, more seeds, more incidence angles, and more informative scatter geometry.

Do not run shadow or final from v7A.
