from ablation_base_runner import run_experiment

if __name__ == "__main__":
    run_experiment("base", ablation_flags={'nli': False, 'entity': False, 'revision': False})
