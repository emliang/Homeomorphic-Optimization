"""Shared outer-loop scaffolding for penalty and augmented Lagrangian optimizers."""

from __future__ import annotations

from time import perf_counter

from tqdm import tqdm

from homopt.optim.core.recording import IterationRecorder
from homopt.optim.core.verbose import write_iteration_row as _write_verbose_iteration_row


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

    for outer_iter in tqdm(range(outer_iterations), disable=progress_disable or not verbose):
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

        if verbose:
            should_write_verbose = outer_iter == 0 or (outer_iter + 1) % verbose_interval == 0
            extra_metrics = verbose_metrics(state, dual_state) if should_write_verbose and verbose_metrics is not None else None
            printed_verbose_header = _write_verbose_iteration_row(
                tqdm.write,
                enabled=True,
                printed_header=printed_verbose_header,
                iteration=outer_iter,
                interval=verbose_interval,
                eval_payload=eval_payload,
                learning_rate=outer_lr,
                dual_state=dual_state,
                iter_time=iter_time,
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
