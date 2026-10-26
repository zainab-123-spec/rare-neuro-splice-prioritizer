| Model | Split | Subset | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- |
| Distance rule | fixed rule | all variants | 0.851 (0.839–0.864) | 0.882 (0.871–0.892) | 0.186 |
| ±1/±2 dinucleotide rule | fixed rule | all variants | 0.858 (0.849–0.867) | 0.824 (0.811–0.837) | 0.110 |
| Distance logistic | gene held out | all variants | 0.520 (0.504–0.537) | 0.484 (0.464–0.505) | 0.258 |
| Logistic regression | gene held out | all variants | 0.964 (0.959–0.968) | 0.955 (0.948–0.961) | 0.063 |
| HistGradientBoosting | gene held out | all variants | 0.974 (0.970–0.977) | 0.966 (0.961–0.971) | 0.059 |
| GradientBoosting | gene held out | all variants | 0.975 (0.971–0.978) | 0.966 (0.961–0.971) | 0.057 |
| Logistic regression | random split | all variants | 0.973 (0.969–0.977) | 0.967 (0.961–0.972) | 0.055 |
| HistGradientBoosting | random split | all variants | 0.988 (0.985–0.990) | 0.981 (0.975–0.985) | 0.039 |
| GradientBoosting | random split | all variants | 0.986 (0.984–0.989) | 0.979 (0.974–0.983) | 0.043 |
| Logistic regression, motif features | gene held out | ±3..±50 bp | 0.909 (0.889–0.928) | 0.706 (0.664–0.745) | 0.095 |
| HistGradientBoosting, motif features | gene held out | ±3..±50 bp | 0.931 (0.914–0.948) | 0.784 (0.746–0.820) | 0.062 |
| HistGradientBoosting, full features | gene held out | ±3..±50 bp | 0.932 (0.915–0.946) | 0.771 (0.728–0.809) | 0.062 |
| ±1/±2 rule on the hard subset | fixed rule | ±3..±50 bp | 0.500 (0.500–0.500) | 0.123 (0.111–0.135) | 0.123 |
| Distance rule on the hard subset | fixed rule | ±3..±50 bp | 0.590 (0.547–0.636) | 0.349 (0.300–0.402) | 0.103 |
