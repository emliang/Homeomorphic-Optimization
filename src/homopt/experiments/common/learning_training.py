"""Learning-model training cache helpers."""

from __future__ import annotations

from homopt.learning.decision_predictor_training import train_decision_predictor
from homopt.models import load_decision_predictor, save_decision_predictor

from .inn_training import _load_or_train_mapping


def build_predictor_training_payload(problem_args, model_args, train_args, *, ensure_results_save_freq=False):
    payload = {
        **model_args,
        "n_samples": int(train_args["n_samples"]),
        "batch_size": int(train_args["batch_size"]),
        "total_iteration": int(train_args["total_iteration"]),
    }
    if "pre_training" in train_args:
        payload["pre_training"] = int(train_args["pre_training"])
    if ensure_results_save_freq and "resultsSaveFreq" not in payload:
        payload["resultsSaveFreq"] = int(train_args["total_iteration"]) + 1
    return {
        "problem_config": problem_args,
        "predictor_model_config": payload,
    }


def load_or_train_decision_predictor(data, args, save_dir, *, retrain=False):
    return _load_or_train_mapping(
        data=data,
        args=args,
        save_dir=save_dir,
        retrain=retrain,
        model_filename="decision_predictor.pt",
        record_filename="decision_predictor_training_record.npy",
        load_fn=load_decision_predictor,
        save_fn=save_decision_predictor,
        train_fn=train_decision_predictor,
    )


__all__ = ["build_predictor_training_payload", "load_or_train_decision_predictor"]
