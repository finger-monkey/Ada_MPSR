# AdaMPSR



## Run

### Coupled PDE regression

```bash
python adaptive_coupled_symbolic_regression.py
```

This script fits coupled temperature and displacement fields using adaptive decoupling.

### SOFC battery case

```bash
cd battery_case
python AdaMPSR_SOFC_test.py
```

This script discovers symbolic expressions for the O2 and N2 fields in the SOFC case.


### Ablation Experiment

```bash
cd ablation
python Structural_entropy_ablation.py
```

This script is an experiment on the ablation of structural entropy.
