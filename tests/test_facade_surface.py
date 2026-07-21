import homopt.experiments as experiments


def test_top_level_facade_exports_experiment_infrastructure_only():
    assert experiments.ExperimentResult is not None
    assert experiments.ExperimentContext is not None
    assert experiments.EXPERIMENT_FAMILIES
    assert experiments.SCRIPT_EXPERIMENTS
    assert callable(experiments.default_script_output_dir)
    assert callable(experiments.record_script_run)
    assert callable(experiments.run_and_record)
    assert callable(experiments.load_result)
    assert callable(experiments.save_result)
    assert not hasattr(experiments, "qcqp_inn_experiment")
