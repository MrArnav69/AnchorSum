# Significance Testing Results

We performed a paired **Wilcoxon signed-rank test** to determine if the differences between configurations are statistically significant. The Wilcoxon test is appropriate here because it handles non-normal instance-level distributions and paired observations (same documents, different configurations).

### Test 1: Full (Tmax=1) vs Base — SummaCConv
- **Mean Difference**: 0.7327 vs 0.7146
- **p-value**: `4.49e-28`
- **Conclusion**: Highly statistically significant. AnchorSum (Tmax=1) definitively improves SummaCConv over the Base model.

### Test 2: Full (Tmax=1) vs Base — BARTScore s→d
- **Mean Difference**: -4.0194 vs -4.0385
- **p-value**: `4.01e-04`
- **Conclusion**: Statistically significant. The modest improvement in BARTScore s→d is robust across the dataset.

### Test 3: Tmax=2 vs Tmax=1 — SummaCConv
- **Mean Difference**: 0.9182 vs 0.7327
- **p-value**: `5.68e-81`
- **Conclusion**: Extremely significant. The inflation of the SummaCConv score due to verifier exploitation in the second revision cycle is a pervasive phenomenon.

### Test 4: Tmax=2 vs Tmax=1 — BARTScore s→d
- **Mean Difference**: -6.5849 vs -4.0194
- **p-value**: `2.68e-83` (Wilcoxon statistic: 0.0)
- **Conclusion**: Extremely significant (every single instance deteriorated). The collapse of BARTScore s→d during the second revision cycle is absolute and systematic.

### Test 5: no_entity vs Base — SummaCConv
- **Mean Difference**: 0.7121 vs 0.7146
- **p-value**: `2.91e-05`
- **Conclusion**: Highly significant. This confirms that the removal of the Entity Guard actively degrades faithfulness below the unaugmented base model, rather than just failing to improve it.

### Test 6: Full vs no_nli — BARTScore s→d
- **Mean Difference**: -4.0194 vs -4.0274
- **p-value**: `3.71e-02`
- **Conclusion**: Statistically significant. This proves the complementarity claim: while `no_nli` matches `Full` on SummaCConv, BARTScore effectively discriminates between them, showing that the `Full` pipeline is significantly better at preserving register.

---

*Note: The raw CSV format of these results has been saved in this directory at `wilcoxon_results.csv`.*
