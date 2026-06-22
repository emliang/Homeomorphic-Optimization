"""Training loops and checkpoint helpers for learning-based routes."""

from .adversarial import train_adversarial_model
from .cache import build_training_payload, load_or_train_mapping, save_training_record, training_time_from_record
from .inn import (
    build_inn_runtime_args,
    load_or_train_inn_mapping,
    train_mdh_mapping,
    unsupervised_training_mdh,
)
from .ipnn import load_or_train_ipnn_mapping, train_ipnn_mapping
from .mapping import (
    build_learning_mapping_payload,
    load_or_train_learning_mapping,
    normalize_mapping_type,
    prepare_learning_mapping,
)
from .predictor import (
    build_predictor_training_payload,
    load_or_train_decision_predictor,
    prepare_decision_predictor,
    train_decision_predictor,
)

__all__ = [
    "build_learning_mapping_payload",
    "build_inn_runtime_args",
    "build_predictor_training_payload",
    "build_training_payload",
    "load_or_train_decision_predictor",
    "load_or_train_learning_mapping",
    "load_or_train_inn_mapping",
    "load_or_train_ipnn_mapping",
    "load_or_train_mapping",
    "normalize_mapping_type",
    "prepare_learning_mapping",
    "prepare_decision_predictor",
    "save_training_record",
    "train_adversarial_model",
    "train_decision_predictor",
    "train_ipnn_mapping",
    "train_mdh_mapping",
    "training_time_from_record",
    "unsupervised_training_mdh",
]
