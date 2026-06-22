"""Model namespace."""

from .adversarial import AdaptiveCNN, ModelTrainer, ResidualBlock, build_model
from .condition import MLP, Mixer, PINN, QuadMixer, ResBlock, build_condition_encoder, normalize_condition_encoder_type
from .inn import EMA, INN, initialize_inn_as_identity, load_inn_mapping, save_inn_mapping
from .local_stretch import combine_local_stretch_objective, local_log_stretch_loss
from .cvxinn import (
    CubeGaugeCvxINN,
    CvxINN,
    SphereReparam,
    SphereReparamGaugeCvxINN,
    TanhGaugeCvxINN,
    cube_compactify,
    cube_decompactify,
    radial_compactify,
    radial_decompactify,
    train_cvxinn_min_distortion,
)
from .ipnn import (
    IPNN,
    NoiseModule,
    load_ipnn_mapping,
    save_ipnn_mapping,
)
from .nn import (
    DecisionPredictorNet,
    load_decision_predictor,
    save_decision_predictor,
)

__all__ = [
    'AdaptiveCNN',
    'CubeGaugeCvxINN',
    'CvxINN',
    'DecisionPredictorNet',
    'EMA',
    'INN',
    'IPNN',
    'MLP',
    'ModelTrainer',
    'Mixer',
    'NoiseModule',
    'PINN',
    'QuadMixer',
    'ResidualBlock',
    'ResBlock',
    'SphereReparam',
    'SphereReparamGaugeCvxINN',
    'TanhGaugeCvxINN',
    'build_condition_encoder',
    'build_model',
    'combine_local_stretch_objective',
    'cube_compactify',
    'cube_decompactify',
    'initialize_inn_as_identity',
    'load_decision_predictor',
    'load_inn_mapping',
    'load_ipnn_mapping',
    'local_log_stretch_loss',
    'normalize_condition_encoder_type',
    'radial_compactify',
    'radial_decompactify',
    'save_decision_predictor',
    'save_inn_mapping',
    'save_ipnn_mapping',
    'train_cvxinn_min_distortion',
]
