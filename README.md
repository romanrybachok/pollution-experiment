# Pollution Experiment

Deterministic–stochastic modelling of surface-water contamination by pesticides.

Supplementary source code for the article *"Deterministic-stochastic modelling of
surface water contamination by pesticides: software implementation and computational
experiment"* (Rybachok R., Pasichnyk R., West Ukrainian National University).

The code couples a **deterministic** pesticide-transport model with a **stochastic**
Monte-Carlo shell that propagates input uncertainty and estimates the probability of
exceeding the environmental quality standard (MAC-EQS).

The deterministic chain integrates five processes:

1. pesticide degradation — first-order kinetics;
2. sorption — linear isotherm with a retardation factor;
3. surface runoff — the SCS-CN method;
4. retention in a vegetated buffer zone — an empirical exponential model;
5. in-stream transport — the 1-D convection–diffusion equation (explicit
   finite-difference scheme under the CFL condition).

## Modules

| File | Description |
|------|-------------|
| `agro_pollution_model.py` | Deterministic core. Domain classes (`Pesticide`, `SoilProfile`, `AgriculturalField`, `River`, `BufferZone`), the `SimulationEngine`, and the `Visualizer`. |
| `monte_carlo_experiment.py` | Stochastic shell. `UncertainParameter`, `MonteCarloEngine` (N independent trials), Spearman-based sensitivity analysis, and `MonteCarloVisualizer`. Run as a script to reproduce the experiment. |
| `build_class_diagram.py` | Generates the UML class diagram used as Fig. 1 of the article. |

## Requirements

Python 3.10+ and the packages in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

## Reproducing the experiment

```bash
python monte_carlo_experiment.py
```

This runs the atrazine / Cherkasy-oblast scenario (N = 2000 iterations, 7 uncertain
parameters), prints the summary statistics and risk assessment, and saves the Monte-Carlo
report and convergence plots to `images/`. The stochastic run completes in well under a
second on a laptop-class CPU (~0.02–0.03 s for N = 2000).

To regenerate the class diagram:

```bash
python build_class_diagram.py   # writes output/class_diagram.png
```

## Citation

If you use this code, please cite the article above.

## License

No license has been assigned yet; please contact the authors regarding reuse.
