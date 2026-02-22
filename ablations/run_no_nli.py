from ablation_base_runner import run_experiment

if __name__ == "__main__":
    run_experiment("no_nli", ablation_flags={'nli': False, 'entity': True, 'revision': True})
