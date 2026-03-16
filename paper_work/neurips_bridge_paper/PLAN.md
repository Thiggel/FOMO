# NeurIPS Paper Plan

## Main-paper budget
- Abstract: 0.3 pages
- Introduction + related work: 1.7 pages
- Method: 1.5 pages
- Experimental setup: 0.9 pages
- Results: 3.0 pages
- Discussion / conclusion: 0.4 pages
- Figures / tables reserve: 1.2 pages
- Target total: <= 9 content pages

## Main-paper assets
- Figure 1: `figures/mode_window_selection.pdf`
  - purpose: explain the new mode-window selector vs. the old top-tail ablation
- Table 1: `tables/main_baselines_linear.tex`
  - purpose: balanced vs. imbalanced vs. BRIDGE on ImageNet-100-LT
- Table 2: `tables/main_sota_average.tex`
  - purpose: compact main-paper comparison across all five source datasets
- Table 3: `tables/main_ablation_summary.tex`
  - purpose: summarize the key pretraining / generation / selection ablations

## Main-paper narrative
1. Motivate imbalance as a data-geometry problem, not only an optimization problem.
2. Present BRIDGE as a cyclic data-repair pipeline.
3. Explain why the old top-tail selector is too brittle.
4. Make the histogram-mode selector the core methodological update.
5. Emphasize broad transfer across five source datasets, not only ImageNet-100-LT.
6. Use appendix for exhaustive tables, qualitative samples, and implementation details.

## Appendix contents
- Full linear and kNN baseline tables
- Full linear and kNN SOTA tables
- Full ablation tables for pretraining, generation, selection, cycles, and architecture
- Generated-sample overview figure
- Class-distribution figure
- Additional OOD histogram figure
- Pseudocode and implementation notes

## Explicit exclusions
- No ImageNet-1k experiments
- No comparison to new untested post-project methods in the paper body
