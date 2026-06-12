"""Base interfaces for optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
from typing import Any

import torch

from homopt.utils import cast_tensors_to_dtype, move_tensors_to_device


@dataclass
class ProblemInstance:
    """One concrete optimization instance inside a sampled problem family."""

    family: str
    instance_id: int
    input_data: Any
    objective_data: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ProblemInstanceBatch:
    """Batch of problem-instance input parameters and optional objective data."""

    family: str
    inputs: Any
    objectives: Any = None
    n_instances: int = 1
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.n_instances)

    def get_instance(self, index):
        return ProblemInstance(
            family=self.family,
            instance_id=int(index),
            input_data=_index_instance_payload(self.inputs, index),
            objective_data=_index_instance_payload(self.objectives, index),
            metadata=dict(self.metadata),
        )


class BaseProblem(ABC):
    """Unified interface for deterministic optimization problems."""

    nvar: int
    n_constraints: int | None = None
    name: str = "problem"
    has_eq: bool = False
    is_parametric: bool = False

    @abstractmethod
    def objective_x(self, x):
        """Return objective value(s) at x."""

    @abstractmethod
    def constraint_x(self, x, clip=True):
        """Return constraint violation(s) at x."""

    def objective(self, x):
        """Convenience wrapper around objective_x."""
        return self.objective_x(x)

    def project(self, x):
        """Optional projection hook used by projection-based optimizers."""
        return x

    def sample_instances(self, n_samples, seed=2025, **kwargs):
        """Optional parametric instance sampler."""
        del kwargs
        raise NotImplementedError(f"{type(self).__name__} does not implement sample_instances().")

    def sample_instance_batch(self, n_instances=1, seed=2025, **kwargs):
        del n_instances, seed, kwargs
        return _make_singleton_instance_batch(getattr(self, "name", type(self).__name__))

    def bind_instance(self, instance):
        """Bind a sampled instance payload to a concrete deterministic problem view."""
        del instance
        return self

    def with_runtime(self, device=None, dtype=None):
        """Unified runtime setter for device and dtype."""
        if device is not None:
            if hasattr(self, "to_device"):
                self.to_device(device)
            else:
                move_tensors_to_device(self, device)
        if dtype is not None:
            if hasattr(self, "to_dtype"):
                self.to_dtype(dtype)
            else:
                cast_tensors_to_dtype(self, dtype)
        return self

    def is_feasible(self, x, tol=1e-6):
        cons = self.constraint_x(x, clip=False)
        return cons.max() <= tol


class ParametricProblemBase(ABC):
    """Unified interface for parametric families with contract `(input_params, y)`."""

    nvar: int
    n_constraints: int | None = None
    name: str = "parametric_problem"
    has_eq: bool = False
    is_parametric: bool = True

    @abstractmethod
    def objective_xy(self, input_params, y):
        """Return objective value(s) for instance input parameters and candidate decision y."""

    @abstractmethod
    def constraint_residual_xy(self, input_params, y, clip=True):
        """Return constraint residual(s) for instance input parameters and candidate decision y."""

    def project_xy(self, input_params, y):
        """Optional projection/refinement hook for learning routes."""
        del input_params
        return y

    def sample_instances(self, n_samples, seed=2025, **kwargs):
        del kwargs
        raise NotImplementedError(f"{type(self).__name__} does not implement sample_instances().")

    def sample_instance_batch(self, n_instances=1, seed=2025, **kwargs):
        sampled = _sample_problem_instances(self, n_instances, seed=seed, **kwargs)
        return _coerce_problem_instance_batch(self, sampled, n_instances=n_instances, seed=seed)

    def bind_instance(self, instance):
        payload = instance.input_data if isinstance(instance, ProblemInstance) else instance
        objective_payload = instance.objective_data if isinstance(instance, ProblemInstance) else None
        if hasattr(self, "build_instance"):
            return self.build_instance(payload, objective_data=objective_payload)
        raise NotImplementedError(f"{type(self).__name__} does not implement bind_instance().")

    def with_runtime(self, device=None, dtype=None):
        _set_problem_runtime(self, device=device, dtype=dtype)
        return self

    def is_feasible_xy(self, input_params, y, tol=1e-6):
        cons = normalize_constraint_violation(self.constraint_residual_xy(input_params, y, clip=False))
        if not isinstance(cons, torch.Tensor):
            cons = torch.as_tensor(cons)
        return cons.max() <= tol


class StateBackedParametricProblemBase(ParametricProblemBase):
    """Parametric-family base for problems whose current instance is stored in object state.

    This is useful for stateful families that already expose a deterministic
    `objective_x/constraint_x` view over the current internal instance data, but
    need to participate in the shared `(input_params, y)` learning-route
    contract without re-implementing the same pass-through methods in each
    subclass.
    """

    def objective_xy(self, input_params, y):
        del input_params
        return self.objective_x(y)

    def constraint_residual_xy(self, input_params, y, clip=True):
        del input_params
        return _call_with_optional_clip(self.constraint_x, y, clip=clip)

    def project_xy(self, input_params, y):
        del input_params
        if hasattr(self, "project"):
            return self.project(y)
        return y

    def bind_instance(self, instance):
        if hasattr(self, "build_instance"):
            return super().bind_instance(instance)
        return self


def _set_problem_runtime(problem, device=None, dtype=None):
    if device is not None:
        if hasattr(problem, "to_device"):
            problem.to_device(device)
        else:
            move_tensors_to_device(problem, device)
    if dtype is not None:
        if hasattr(problem, "to_dtype"):
            problem.to_dtype(dtype)
        else:
            cast_tensors_to_dtype(problem, dtype)


def _leading_batch_dim(payload):
    if payload is None:
        return None
    if isinstance(payload, ProblemInstanceBatch):
        return int(payload.n_instances)
    if torch.is_tensor(payload):
        return int(payload.shape[0]) if payload.ndim > 0 else 1
    if isinstance(payload, (list, tuple)):
        if not payload:
            return 0
        inferred = [_leading_batch_dim(item) for item in payload]
        inferred = [count for count in inferred if count is not None]
        return inferred[0] if inferred else len(payload)
    return None


def _infer_instance_count(inputs, objectives=None):
    for payload in (inputs, objectives):
        count = _leading_batch_dim(payload)
        if count is not None:
            return int(count)
    return 1


def _index_instance_payload(payload, index):
    if payload is None:
        return None
    if torch.is_tensor(payload):
        return payload[index]
    if isinstance(payload, (list, tuple)):
        return type(payload)(_index_instance_payload(item, index) for item in payload)
    return payload


def _make_singleton_instance_batch(family, *, metadata=None):
    payload = {"deterministic": True}
    if metadata:
        payload.update(dict(metadata))
    return ProblemInstanceBatch(
        family=str(family),
        inputs=None,
        objectives=None,
        n_instances=1,
        metadata=payload,
    )


def _coerce_problem_instance_batch(problem, sampled, *, n_instances, seed=2025):
    if isinstance(sampled, ProblemInstanceBatch):
        metadata = dict(sampled.metadata)
        metadata.setdefault("seed", int(seed))
        metadata.setdefault("requested_n_instances", int(n_instances))
        return ProblemInstanceBatch(
            family=sampled.family,
            inputs=sampled.inputs,
            objectives=sampled.objectives,
            n_instances=sampled.n_instances,
            metadata=metadata,
        )
    if isinstance(sampled, tuple) and len(sampled) == 2:
        inputs, objectives = sampled
    else:
        inputs, objectives = sampled, None
    return ProblemInstanceBatch(
        family=getattr(problem, "name", type(problem).__name__),
        inputs=inputs,
        objectives=objectives,
        n_instances=_infer_instance_count(inputs, objectives),
        metadata={
            "seed": int(seed),
            "requested_n_instances": int(n_instances),
        },
    )


def sample_problem_instance_batch(problem, *, n_instances=1, seed=2025, **kwargs):
    if hasattr(problem, "sample_instance_batch"):
        return problem.sample_instance_batch(n_instances=n_instances, seed=seed, **kwargs)
    return _make_singleton_instance_batch(getattr(problem, "name", type(problem).__name__))


def bind_problem_instance(problem, instance):
    if not hasattr(problem, "bind_instance"):
        return problem
    return problem.bind_instance(instance)


def bind_singleton_problem_instance(problem, *, seed=2025, **kwargs):
    if not hasattr(problem, "sample_instance_batch") or not hasattr(problem, "bind_instance"):
        return problem
    instance_batch = sample_problem_instance_batch(problem, n_instances=1, seed=seed, **kwargs)
    return bind_problem_instance(problem, instance_batch.get_instance(0))


def _sample_problem_instances(problem, n_samples, seed=2025, **kwargs):
    if hasattr(problem, "sample_instances"):
        try:
            return problem.sample_instances(n_samples, seed=seed, **kwargs)
        except NotImplementedError:
            pass
    if hasattr(problem, "generate_problem_samples_torch"):
        return problem.generate_problem_samples_torch(n_samples=n_samples, seed=seed, **kwargs)
    raise NotImplementedError(f"{type(problem).__name__} does not implement a sampling routine.")


def _signature_or_none(fn):
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _accepts_keyword(fn, name):
    signature = _signature_or_none(fn)
    if signature is None:
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    param = signature.parameters.get(name)
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _call_with_optional_clip(fn, *args, clip=True):
    if _accepts_keyword(fn, "clip"):
        return fn(*args, clip=clip)
    return fn(*args)


class TensorRuntimeMixin:
    """Shared runtime helper for problem classes that only store tensor attributes."""

    def to_device(self, device):
        device = torch.device(device)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        self.device = device
        return self


class _DelegatingProblemAdapterBase:
    """Shared metadata/runtime/delegation logic for problem adapters."""

    def __init__(self, problem, name=None, *, default_parametric=False):
        self.problem = problem
        self.name = name or type(problem).__name__
        self.nvar = getattr(problem, "nvar", None)
        self.n_constraints = getattr(problem, "n_constraints", None)
        self.has_eq = bool(getattr(problem, "has_eq", False))
        self.is_parametric = bool(getattr(problem, "is_parametric", default_parametric))

    def sample_instances(self, n_samples, seed=2025, **kwargs):
        return _sample_problem_instances(self.problem, n_samples, seed=seed, **kwargs)

    def sample_instance_batch(self, n_instances=1, seed=2025, **kwargs):
        if hasattr(self.problem, "sample_instance_batch"):
            return self.problem.sample_instance_batch(n_instances=n_instances, seed=seed, **kwargs)
        try:
            sampled = _sample_problem_instances(self.problem, n_instances, seed=seed, **kwargs)
        except NotImplementedError:
            return _make_singleton_instance_batch(self.name)
        return _coerce_problem_instance_batch(
            self.problem,
            sampled,
            n_instances=n_instances,
            seed=seed,
        )

    def with_runtime(self, device=None, dtype=None):
        _set_problem_runtime(self.problem, device=device, dtype=dtype)
        return self

    def bind_instance(self, instance):
        if hasattr(self.problem, "bind_instance"):
            return self.problem.bind_instance(instance)
        return self.problem

    def __getattr__(self, name):
        return getattr(self.problem, name)


class FixedProblemParametricAdapter(_DelegatingProblemAdapterBase, ParametricProblemBase):
    """Adapt a fixed deterministic problem into a degenerate parametric family."""

    def __init__(self, problem, name=None):
        super().__init__(problem, name=name, default_parametric=True)

    def objective_xy(self, input_params, y):
        del input_params
        if hasattr(self.problem, "objective_x"):
            return self.problem.objective_x(y)
        if hasattr(self.problem, "objective"):
            return self.problem.objective(y)
        raise AttributeError(f"{type(self.problem).__name__} is missing objective_x/objective.")

    def constraint_residual_xy(self, input_params, y, clip=True):
        del input_params
        if hasattr(self.problem, "constraint_x"):
            return _call_with_optional_clip(self.problem.constraint_x, y, clip=clip)
        if hasattr(self.problem, "constraint_residual"):
            return _call_with_optional_clip(self.problem.constraint_residual, y, clip=clip)
        raise AttributeError(f"{type(self.problem).__name__} is missing a constraint interface.")

    def project_xy(self, input_params, y):
        del input_params
        if hasattr(self.problem, "project"):
            return self.problem.project(y)
        return y

    def sample_instance_batch(self, n_instances=1, seed=2025, **kwargs):
        del n_instances, seed, kwargs
        return _make_singleton_instance_batch(self.name)

    def bind_instance(self, instance):
        del instance
        return self.problem


def normalize_constraint_violation(cons):
    """Standardize constraint tensors to `B x m` layout when possible."""
    if isinstance(cons, torch.Tensor):
        if cons.ndim == 0:
            return cons.view(1, 1)
        if cons.ndim == 1:
            return cons.view(-1, 1)
    return cons
