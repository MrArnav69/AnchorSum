from ablation_base_runner import run_experiment

if __name__ == "__main__":
    run_experiment("full", ablation_flags={'nli': True, 'entity': True, 'revision': True})
