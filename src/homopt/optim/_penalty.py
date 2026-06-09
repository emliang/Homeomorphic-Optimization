"""Shared outer-loop scaffolding for penalty and augmented Lagrangian optimizers."""

from __future__ import annotations

from time import perf_counter

from tqdm import tqdm

from homopt.optim._recording import IterationRecorder
from homopt.optim._verbose import (
    VERBOSE_COLUMNS,
    as_float as _as_float,
    format_table_cell as _format_table_cell,
    format_value as _format_value,
    write_header as _write_verbose_header,
    write_row as _write_verbose_row,
)


def _verbose_columns():
    return list(VERBOSE_COLUMNS)


def _print_outer_iteration_header(extra_metrics):
    del extra_metrics
    _write_verbose_header(tqdm.write)


def _print_outer_iteration_metrics(
    *,
    outer_iter,
    eval_payload,
    error,
    outer_lr,
    dual_state,
    iter_time,
    elapsed_time,
    extra_metrics,
):
    dual_norm = dual_state.detach().norm() if hasattr(dual_state, "detach") else None
    extra_metrics = extra_metrics or {}
    eq_violation = (
        eval_payload.get("eq_violation")
        if eval_payload.get("eq_violation") is not None
        else eval_payload.get("equality_violation", eval_payload.get("violation"))
    )
    ineq_violation = (
        eval_payload.get("ineq_violation")
        if eval_payload.get("ineq_violation") is not None
        else eval_payload.get("inequality_violation", 0.0)
    )
    row = {
        "outer": outer_iter + 1,
        "objective": eval_payload.get("objective"),
        "eq_violation": eq_violation,
        "ineq_violation": ineq_violation,
        "learning_rate": outer_lr,
        "dual_norm": dual_norm,
        "penalty": extra_metrics.get("penalty"),
        "lag_gap": extra_metrics.get("lag_gap"),
        "iter_time": iter_time,
    }
    del error, elapsed_time
    _write_verbose_row(tqdm.write, **row)


def run_penalty_outer_loop(
    *,
    initial_state,
    initial_eval,
    initial_dual_state,
    run_inner_loop,
    evaluate_state,
    update_dual_state,
    update_learning_rate,
    should_stop,
    augment_eval_payload=None,
    outer_iterations,
    max_running_time,
    initial_learning_rate,
    outer_stepsize_rule,
    verbose=False,
    progress_disable=False,
    track_decisions=False,
    record_best_decision=False,
    extra_track_names=(),
    verbose_metrics=None,
    verbose_interval=50,
):
    """Run a shared outer loop while delegating optimizer-specific inner logic."""

    recorder = IterationRecorder(track_decisions=track_decisions, extra_track_names=extra_track_names)
    should_record_decision = bool(track_decisions or record_best_decision)
    recorder.record_state(
        initial_eval["objective"],
        initial_eval["violation"],
        decision=initial_eval.get("decision") if should_record_decision else None,
        **{name: initial_eval.get(name) for name in extra_track_names},
    )

    state = initial_state
    dual_state = initial_dual_state
    learning_rate = initial_learning_rate
    elapsed_time = 0.0
    error = None
    inner_iter_time = []
    printed_verbose_header = False
    verbose_interval = max(1, int(verbose_interval))

    for outer_iter in tqdm(range(outer_iterations), disable=progress_disable):
        outer_start_time = perf_counter()
        outer_lr = learning_rate
        if outer_stepsize_rule == "diminish":
            outer_lr /= (outer_iter + 1) ** 0.5

        inner_start_time = perf_counter()
        state, error = run_inner_loop(state, dual_state, outer_lr, outer_iter)
        inner_iter_time.append(perf_counter() - inner_start_time)

        eval_payload = evaluate_state(state)
        if augment_eval_payload is not None:
            extra_eval = augment_eval_payload(state, dual_state, eval_payload)
            if extra_eval:
                eval_payload.update(extra_eval)
        iter_time = perf_counter() - outer_start_time
        recorder.append_iter_time(iter_time)
        elapsed_time += iter_time
        recorder.record_state(
            eval_payload["objective"],
            eval_payload["violation"],
            decision=eval_payload.get("decision") if should_record_decision else None,
            **{name: eval_payload.get(name) for name in extra_track_names},
        )

        if verbose and (outer_iter == 0 or (outer_iter + 1) % verbose_interval == 0):
            extra_metrics = verbose_metrics(state, dual_state) if verbose_metrics is not None else {}
            if not printed_verbose_header:
                _print_outer_iteration_header(extra_metrics)
                printed_verbose_header = True
            _print_outer_iteration_metrics(
                outer_iter=outer_iter,
                eval_payload=eval_payload,
                error=error,
                outer_lr=outer_lr,
                dual_state=dual_state,
                iter_time=iter_time,
                elapsed_time=elapsed_time,
                extra_metrics=extra_metrics,
            )

        if should_stop(eval_payload, error, outer_iter):
            break

        dual_state = update_dual_state(dual_state, state, outer_iter)
        learning_rate = update_learning_rate(learning_rate, recorder, outer_iter)
        if elapsed_time > max_running_time:
            break

    payload = recorder.finalize()
    payload["outer_iter_time"] = list(payload["per_iter_time"])
    payload["inner_iter_time"] = inner_iter_time
    return state, dual_state, payload, learning_rate, error


__all__ = ["run_penalty_outer_loop"]
