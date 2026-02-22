from ablation_base_runner import run_experiment

if __name__ == "__main__":
    run_experiment("no_entity", ablation_flags={'nli': True, 'entity': False, 'revision': True})
