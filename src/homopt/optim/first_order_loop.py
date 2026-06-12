"""Shared first-order iteration loop used by multiple optimizers."""

from __future__ import annotations

from time import perf_counter

from tqdm import tqdm

from homopt.optim.recording import IterationRecorder


def _scalar_mean(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def run_first_order_loop(
    *,
    initial_state,
    initial_eval,
    update_backend,
    compute_gradient,
    apply_step,
    evaluate_state,
    max_iterations,
    max_running_time,
    initial_learning_rate,
    stepsize_rule,
    lr_decay,
    min_lr,
    convergence_threshold,
    verbose=False,
    progress_disable=False,
    track_decisions=False,
    extra_track_names=(),
    convergence_metric=None,
    objective_change_threshold=None,
    min_iterations=1,
    require_feasible_for_objective_change=False,
):
    """Run a shared first-order loop while preserving optimizer-specific hooks."""

    recorder = IterationRecorder(track_decisions=track_decisions, extra_track_names=extra_track_names)
    recorder.record_state(
        initial_eval["objective"],
        initial_eval["violation"],
        decision=initial_eval.get("decision"),
        **{name: initial_eval.get(name) for name in extra_track_names},
    )

    state = initial_state
    prev_state = initial_state
    learning_rate = initial_learning_rate
    elapsed_time = 0.0

    for iteration in tqdm(range(max_iterations), disable=progress_disable):
        start_time = perf_counter()
        grad = compute_gradient(state, prev_state, iteration)
        update = update_backend.step(grad)
        lr = learning_rate
        if stepsize_rule == "diminish":
            lr = max(lr / (iteration + 1) ** 0.5, min_lr)
        next_state = apply_step(state, update, lr, prev_state, iteration).detach()

        iter_time = perf_counter() - start_time
        recorder.append_iter_time(iter_time)
        elapsed_time += iter_time

        eval_payload = evaluate_state(next_state)
        recorder.record_state(
            eval_payload["objective"],
            eval_payload["violation"],
            decision=eval_payload.get("decision"),
            **{name: eval_payload.get(name) for name in extra_track_names},
        )

        if verbose and iteration % 1000 == 0:
            obj_scalar = _scalar_mean(eval_payload["objective"])
            print(f"Iteration {iteration}, Objective: {obj_scalar:.6f}")

        if stepsize_rule == "adaptive" and recorder.objective_traj[-1].mean() > recorder.objective_traj[-2].mean():
            learning_rate = max(learning_rate * lr_decay, min_lr)

        prev_state = state.detach()
        state = next_state
        if elapsed_time > max_running_time:
            break
        if convergence_metric is None:
            continue
        if str(convergence_metric).lower() == "objective_change":
            if iteration + 1 < int(min_iterations):
                continue
            if require_feasible_for_objective_change and _scalar_mean(eval_payload["violation"]) > convergence_threshold:
                continue
            threshold = convergence_threshold if objective_change_threshold is None else objective_change_threshold
            objective_change = abs(_scalar_mean(recorder.objective_traj[-1]) - _scalar_mean(recorder.objective_traj[-2]))
            if objective_change <= float(threshold):
                break
            continue
        raise ValueError(f"Unsupported convergence_metric: {convergence_metric}")

    return state, recorder.finalize(), learning_rate


__all__ = ["run_first_order_loop"]
