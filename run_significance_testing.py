import os
import pandas as pd
from scipy.stats import wilcoxon

def run_tests():
    # Define paths
    base_dir = "/Users/mrarnav69/Documents/AnchorSum"
    summac_base_path = os.path.join(base_dir, "Results", "Component Ablation ", "summac_final_results", "summac_detailed_base.csv")
    summac_full_path = os.path.join(base_dir, "Results", "Component Ablation ", "summac_final_results", "summac_detailed_full.csv")
    summac_noentity_path = os.path.join(base_dir, "Results", "Component Ablation ", "summac_final_results", "summac_detailed_no_entity.csv")
    summac_t2_path = os.path.join(base_dir, "Results", "Revision Depth", "summac_full_revisions_2_results", "summac_detailed_full_revisions_2.csv")

    bart_base_path = os.path.join(base_dir, "Results", "Component Ablation ", "bartscore_simple_results", "bartscore_detailed_base.csv")
    bart_full_path = os.path.join(base_dir, "Results", "Component Ablation ", "bartscore_simple_results", "bartscore_detailed_full.csv")
    bart_nonli_path = os.path.join(base_dir, "Results", "Component Ablation ", "bartscore_simple_results", "bartscore_detailed_no_nli.csv")
    bart_t2_path = os.path.join(base_dir, "Results", "Revision Depth", "bartscore_full_revisions_2_results", "bartscore_detailed_full_revisions_2.csv")

    # Output directory
    output_dir = os.path.join(base_dir, "Significance_Testing")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Helper function to run Wilcoxon
    def run_wilcoxon(name, path1, path2, col, test_type="two-sided"):
        df1 = pd.read_csv(path1).sort_values("id").reset_index(drop=True)
        df2 = pd.read_csv(path2).sort_values("id").reset_index(drop=True)
        
        # Ensure alignment
        assert (df1["id"] == df2["id"]).all(), f"IDs do not align for {name}"
        
        x = df1[col]
        y = df2[col]
        
        stat, p = wilcoxon(x, y, alternative=test_type)
        mean1 = x.mean()
        mean2 = y.mean()
        
        print(f"--- {name} ---")
        print(f"Mean 1: {mean1:.4f}, Mean 2: {mean2:.4f}")
        print(f"Wilcoxon statistic: {stat}, p-value: {p:.4e}")
        print()
        
        results.append({
            "Test": name,
            "Mean 1": mean1,
            "Mean 2": mean2,
            "Statistic": stat,
            "P-Value": p
        })

    # Test 1: Full (Tmax=1) vs Base — SummaCConv
    run_wilcoxon("Test 1: Full (Tmax=1) vs Base - SummaCConv", summac_full_path, summac_base_path, "summac_score")

    # Test 2: Full (Tmax=1) vs Base — BARTScore s→d
    # 'bartscore_sum2doc' maps to s->d based on the means in the paper
    run_wilcoxon("Test 2: Full (Tmax=1) vs Base - BARTScore s->d", bart_full_path, bart_base_path, "bartscore_sum2doc")

    # Test 3: Tmax=2 vs Tmax=1 — SummaCConv
    run_wilcoxon("Test 3: Tmax=2 vs Tmax=1 - SummaCConv", summac_t2_path, summac_full_path, "summac_score")

    # Test 4: Tmax=2 vs Tmax=1 — BARTScore s→d
    run_wilcoxon("Test 4: Tmax=2 vs Tmax=1 - BARTScore s->d", bart_t2_path, bart_full_path, "bartscore_sum2doc")

    # Test 5: no_entity vs Base — SummaCConv
    run_wilcoxon("Test 5: no_entity vs Base - SummaCConv", summac_noentity_path, summac_base_path, "summac_score", test_type="less")

    # Test 6: Full vs no_nli — BARTScore s→d
    run_wilcoxon("Test 6: Full vs no_nli - BARTScore s->d", bart_full_path, bart_nonli_path, "bartscore_sum2doc", test_type="greater")

    # Save results
    results_df = pd.DataFrame(results)
    out_file = os.path.join(output_dir, "wilcoxon_results.csv")
    results_df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    run_tests()
