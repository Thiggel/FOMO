# Statistical test coverage

## Paired seed tests available

The retained result files support paired tests for the following claims.

* Full schedule MoCo v3 with ViT S transfer and representation metrics
* One cycle and three cycle MoCo v3 diagnostics
* MAE diagnostic
* ImageNet 100 LT fine tuning with 1 percent and 10 percent labels
* CIFAR 10 LT Base, TS, BRIDGE, and BRIDGE plus TS factorial
* Selected FLUX setting comparisons
* Adaptive, dense placebo, and top tail feedback controls
* TADA and BRIDGE plus TADA

These are reported in `statistical_audit.md` and `statistical_audit.json`.

## Only aggregate tests available

The raw per-seed values are not present in the retained result store for the following reported tables. A conservative Welch sensitivity analysis was reconstructed from the reported mean, sample standard deviation, and three runs per condition.

* Main SimCLR versus BRIDGE source averages and individual downstream cells
* Five stage selector ablation
* One repair oracle, uniform, extreme score, TADA, and distillation diagnostics
* Some generator comparisons involving SD3

These are reported in `summary_only_audit.md`, `main_table_all_cells.md`, and the corresponding JSON files. They must not be represented as paired tests unless the original seed files are recovered.

## No valid significance test currently possible

* VOC segmentation has one reported run per condition
* Full schedule DINO is incomplete
* Guidance image equality was checked for one fixed anchor and generation seed
* Runtime and generated image fractions are deterministic accounting values rather than repeated stochastic outcomes
* Several k, cutoff, strength, and candidate multiplier results are reported only as ranges without a complete retained per-seed table
* One-cycle duplication and conventional augmentation are reported without three retained seed values

These results can be described as diagnostics but not as statistically significant.
