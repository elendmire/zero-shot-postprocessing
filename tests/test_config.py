from zeropp.config import load_experiment_config, load_data_config


def test_experiment_config_quantile_levels():
    cfg = load_experiment_config()
    assert cfg.quantile_levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_experiment_config_data_size_axis():
    cfg = load_experiment_config()
    assert cfg.data_size_days == [0, 30, 90, 365, 1095, "full"]


def test_experiment_config_seeds_present():
    cfg = load_experiment_config()
    assert len(cfg.seeds) >= 1
    assert all(isinstance(s, int) for s in cfg.seeds)


def test_experiment_config_data_size_sweep_seeds():
    cfg = load_experiment_config()
    assert cfg.data_size_sweep_seeds == [0, 1, 2, 3, 4]


def test_data_config_targets():
    cfg = load_data_config()
    assert cfg.targets == ["t2m", "w10"]
    assert cfg.split_name == "euppbench_default"
