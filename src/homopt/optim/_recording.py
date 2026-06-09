"""Shared utilities for iterative optimizer trajectories."""

from __future__ import annotations

import torch


class IterationRecorder:
    """Collect shared optimizer histories without changing optimizer semantics."""

    def __init__(self, *, track_decisions=False, extra_track_names=()):
        self.track_decisions = bool(track_decisions)
        self.objective_traj = []
        self.violation_traj = []
        self.per_iter_time = []
        self._decision_trajectory = [] if self.track_decisions else None
        self._extra_trajectories = {name: [] for name in extra_track_names}
        self._best_violation = None
        self._best_objective = None
        self._best_decision = None
        self._best_iteration = None

    def record_state(self, objective, violation, decision=None, **extra_tracks):
        current_iteration = len(self.violation_traj)
        self.objective_traj.append(objective.detach())
        self.violation_traj.append(violation.detach())
        violation_scalar = violation.detach().reshape(-1).max()
        if self._best_violation is None or violation_scalar < self._best_violation:
            self._best_violation = violation_scalar.detach().clone()
            self._best_objective = objective.detach().reshape(-1).mean().clone()
            self._best_decision = None if decision is None else decision.detach().clone()
            self._best_iteration = current_iteration
        if self.track_decisions and decision is not None:
            self._decision_trajectory.append(decision.detach())
        for name, values in extra_tracks.items():
            if name in self._extra_trajectories and values is not None:
                self._extra_trajectories[name].append(values.detach())

    def append_iter_time(self, iter_time):
        self.per_iter_time.append(float(iter_time))

    def finalize(self):
        payload = {
            "decision_trajectory": [],
            "objective_traj": torch.cat(self.objective_traj, axis=0),
            "violation_traj": torch.cat(self.violation_traj, axis=0),
            "per_iter_time": self.per_iter_time,
            "best_decision": self._best_decision,
            "best_objective": self._best_objective,
            "best_violation": self._best_violation,
            "best_iteration": self._best_iteration,
        }
        if self.track_decisions and self._decision_trajectory:
            payload["decision_trajectory"] = torch.cat(self._decision_trajectory, axis=0)
        for name, values in self._extra_trajectories.items():
            payload[name] = torch.cat(values, axis=0) if values else []
        return payload


__all__ = ["IterationRecorder"]
