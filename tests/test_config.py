from pathlib import Path

from life_satisfaction.config import load_config


def test_publication_config_is_locked():
    config, raw = load_config(Path("configs/paper.yaml"))
    assert config.test_size == 0.20
    assert config.random_seed_split == 20
    assert config.random_seed_preprocessing == 21
    assert config.random_seed_undersampling is None
    assert config.rfecv_folds == 5
    assert config.feature_importance_threshold == 0.0094
    assert set(raw) == {"pipeline"}
