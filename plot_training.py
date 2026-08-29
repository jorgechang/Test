#!/usr/bin/env python3
"""Live per-run plots for the v13.45 ANYmal camera-pose experiments.

The script can be run once or as a lightweight sidecar.  In live mode it polls
metrics.jsonl but only redraws after new completed episodes arrive, so training
is never blocked by matplotlib.  It accepts both the older r2dreamer
``train_return`` naming and the newer console-style ``episode/score`` naming.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Training can be appending the final line while we read it.
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def rolling(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    buf: list[float] = []
    acc = 0.0
    for value in values:
        value = float(value)
        buf.append(value)
        acc += value
        if len(buf) > window:
            acc -= buf.pop(0)
        out.append(acc / len(buf))
    return out


def first_key(rows: list[dict], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if any(finite_number(row.get(key)) for row in rows):
            return key
    return None


def series(rows: list[dict], key: str | None) -> tuple[list[float], list[float]]:
    if key is None:
        return [], []
    xs, ys = [], []
    for row in rows:
        if finite_number(row.get("step")) and finite_number(row.get(key)):
            xs.append(float(row["step"]))
            ys.append(float(row[key]))
    return xs, ys


def episode_series(rows: list[dict], key: str | None) -> tuple[list[float], list[float]]:
    """Return a per-episode curve indexed by completed episode number."""
    if key is None:
        return [], []
    xs, ys = [], []
    for episode_index, row in enumerate(rows, start=1):
        if finite_number(row.get(key)):
            xs.append(float(episode_index))
            ys.append(float(row[key]))
    return xs, ys


def episode_rows(rows: list[dict]) -> tuple[list[dict], str | None]:
    key = first_key(rows, ("train_return", "episode/score", "episode_score"))
    if key is None:
        return [], None
    return [row for row in rows if finite_number(row.get(key))], key


def alias_key(rows: list[dict], name: str) -> str | None:
    # Environment log channels have appeared both bare and under episode/ in
    # different Dreamer forks.  Accept both without silently dropping a plot.
    return first_key(rows, (name, f"episode/{name}"))


def episode_difference_series(
    rows: list[dict], minuend_name: str, subtrahend_name: str
) -> tuple[list[float], list[float]]:
    """Per-episode max(minuend - subtrahend, 0), aligned on the same row."""
    a_key = alias_key(rows, minuend_name)
    b_key = alias_key(rows, subtrahend_name)
    if a_key is None or b_key is None:
        return [], []
    xs, ys = [], []
    for episode_index, row in enumerate(rows, start=1):
        if finite_number(row.get(a_key)) and finite_number(row.get(b_key)):
            xs.append(float(episode_index))
            ys.append(max(float(row[a_key]) - float(row[b_key]), 0.0))
    return xs, ys


def wall_exposure_series(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Percent of episode steps spent inside the non-terminal wall guard."""
    wall_key = alias_key(rows, "log_wall_collision")
    length_key = first_key(rows, ("episode/length", "train_length", "episode_length"))
    if wall_key is None or length_key is None:
        return [], []
    xs, ys = [], []
    for episode_index, row in enumerate(rows, start=1):
        if finite_number(row.get(wall_key)) and finite_number(row.get(length_key)):
            length = max(float(row[length_key]), 1.0)
            xs.append(float(episode_index))
            ys.append(100.0 * float(row[wall_key]) / length)
    return xs, ys


def tolerance_occupancy_series(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Percent of episode steps inside the precise camera-pose tolerance."""
    tol_key = alias_key(rows, "log_in_tolerance_steps")
    length_key = first_key(rows, ("episode/length", "train_length", "episode_length"))
    if tol_key is None or length_key is None:
        return [], []
    xs, ys = [], []
    for episode_index, row in enumerate(rows, start=1):
        if finite_number(row.get(tol_key)) and finite_number(row.get(length_key)):
            length = max(float(row[length_key]), 1.0)
            xs.append(float(episode_index))
            ys.append(100.0 * float(row[tol_key]) / length)
    return xs, ys


def episode_ratio_series(
    rows: list[dict], numerator_name: str, denominator_name: str
) -> tuple[list[float], list[float]]:
    """Return 100*numerator/denominator for episode-summed diagnostic counts."""
    num_key = alias_key(rows, numerator_name)
    den_key = alias_key(rows, denominator_name)
    if num_key is None or den_key is None:
        return [], []
    xs, ys = [], []
    for episode_index, row in enumerate(rows, start=1):
        if finite_number(row.get(num_key)) and finite_number(row.get(den_key)):
            den = float(row[den_key])
            if den > 0.0:
                xs.append(float(episode_index))
                ys.append(100.0 * float(row[num_key]) / den)
    return xs, ys


def rolling_count_ratio_series(
    rows: list[dict], numerator_name: str, denominator_name: str, window: int
) -> tuple[list[float], list[float]]:
    """Rolling 100*sum(numerator)/sum(denominator) for sparse count channels."""
    num_key = alias_key(rows, numerator_name)
    den_key = alias_key(rows, denominator_name)
    if num_key is None or den_key is None:
        return [], []
    xs: list[float] = []
    ys: list[float] = []
    num_buf: list[float] = []
    den_buf: list[float] = []
    num_sum = 0.0
    den_sum = 0.0
    for episode_index, row in enumerate(rows, start=1):
        num = float(row[num_key]) if finite_number(row.get(num_key)) else 0.0
        den = float(row[den_key]) if finite_number(row.get(den_key)) else 0.0
        num_buf.append(num)
        den_buf.append(den)
        num_sum += num
        den_sum += den
        if len(num_buf) > window:
            num_sum -= num_buf.pop(0)
            den_sum -= den_buf.pop(0)
        if den_sum > 0.0:
            xs.append(float(episode_index))
            ys.append(100.0 * num_sum / den_sum)
    return xs, ys


def atomic_savefig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, dpi=dpi)
    tmp.replace(path)


def save_line(
    path: Path,
    title: str,
    ylabel: str,
    curves: list[tuple[list[float], list[float], str]],
    *,
    zero: bool = False,
    xlabel: str = "Environment steps",
) -> bool:
    curves = [(x, y, label) for x, y, label in curves if x and y]
    if not curves:
        return False
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for x, y, label in curves:
        ax.plot(x, y, label=label)
    if zero:
        ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if len(curves) > 1:
        ax.legend()
    fig.tight_layout()
    atomic_savefig(fig, path)
    plt.close(fig)
    return True


def metric_curve(rows: list[dict], candidates: tuple[str, ...], label: str):
    key = first_key(rows, candidates)
    x, y = series(rows, key)
    return x, y, label


def plot_once(logdir: Path, window: int) -> tuple[list[Path], int]:
    metrics = logdir / "metrics.jsonl"
    rows = read_jsonl(metrics)
    if not rows:
        raise FileNotFoundError(f"No readable metrics yet: {metrics}")

    episodes, return_key = episode_rows(rows)
    outdir = logdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if episodes and return_key:
        x, y = episode_series(episodes, return_key)
        p = outdir / "episode_return.png"
        if save_line(
            p,
            "Episode return",
            "Return",
            [(x, y, "Raw"), (x, rolling(y, window), f"Rolling {window}")],
            zero=True,
            xlabel="Completed episode",
        ):
            written.append(p)

        # Success rate.
        success_key = alias_key(episodes, "log_success")
        x, y = episode_series(episodes, success_key)
        if y:
            p = outdir / "success_rate.png"
            if save_line(
                p,
                "Rolling success rate",
                "Success (%)",
                [(x, [100.0 * value for value in rolling(y, window)], f"Rolling {window}")],
                xlabel="Completed episode",
            ):
                written.append(p)

        # Position diagnostics.
        pos_curves = []
        for key, label in (
            ("log_initial_position_error", "Initial"),
            ("log_min_position_error", "Minimum reached"),
            ("log_final_position_error", "Final"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                pos_curves.append((xk, rolling(yk, window), label))
        p = outdir / "position_error.png"
        if save_line(p, "Camera position error", "Error (m)", pos_curves, xlabel="Completed episode"):
            written.append(p)

        # Final target-camera yaw diagnostics.
        yaw_curves = []
        for key, label in (
            ("log_initial_yaw_error_deg", "Initial"),
            ("log_min_yaw_error_deg", "Minimum reached"),
            ("log_final_yaw_error_deg", "Final"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                yaw_curves.append((xk, rolling(yk, window), label))
        p = outdir / "yaw_error.png"
        if save_line(p, "Final camera-yaw error", "Error (deg)", yaw_curves, xlabel="Completed episode"):
            written.append(p)

        # Bearing-to-target-position diagnostics. This is the new far-field
        # orientation diagnostic in v13.14 and is important for diagnosing turning.
        bearing_curves = []
        for key, label in (
            ("log_initial_bearing_error_deg", "Initial"),
            ("log_min_bearing_error_deg", "Minimum reached"),
            ("log_final_bearing_error_deg", "Final"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                bearing_curves.append((xk, rolling(yk, window), label))
        p = outdir / "bearing_error.png"
        if save_line(p, "Bearing to target position", "Error (deg)", bearing_curves, xlabel="Completed episode"):
            written.append(p)

        # v13.14: quantify the wandering/backslide visible in simulation.
        # Zero means the episode finished at its best position; larger values mean
        # it approached the target and then threw that progress away.
        x_back, y_back = episode_difference_series(
            episodes, "log_final_position_error", "log_min_position_error"
        )
        p = outdir / "position_backslide.png"
        if save_line(
            p,
            "Position progress lost after closest approach",
            "Final - minimum error (m)",
            [(x_back, rolling(y_back, window), f"Rolling {window}")],
            zero=True,
            xlabel="Completed episode",
        ):
            written.append(p)

        angle_backslide = []
        for final_name, min_name, label in (
            ("log_final_bearing_error_deg", "log_min_bearing_error_deg", "Bearing"),
            ("log_final_yaw_error_deg", "log_min_yaw_error_deg", "Goal yaw"),
        ):
            xa, ya = episode_difference_series(episodes, final_name, min_name)
            if ya:
                angle_backslide.append((xa, rolling(ya, window), label))
        p = outdir / "angular_backslide.png"
        if save_line(
            p,
            "Angular progress lost after best approach",
            "Final - minimum error (deg)",
            angle_backslide,
            zero=True,
            xlabel="Completed episode",
        ):
            written.append(p)

        # Coupled SE(2) precision: does the policy achieve good position and yaw
        # at the same instant, or alternate between them?
        yaw_at_pos_curves = []
        for key, label in (
            ("log_min_yaw_error_deg", "Best yaw anywhere"),
            ("log_yaw_at_min_position_deg", "Yaw at closest position"),
            ("log_final_yaw_error_deg", "Final yaw"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                yaw_at_pos_curves.append((xk, rolling(yk, window), label))
        p = outdir / "yaw_at_closest_position.png"
        if save_line(
            p,
            "Yaw error when camera position is closest",
            "Yaw error (deg)",
            yaw_at_pos_curves,
            xlabel="Completed episode",
        ):
            written.append(p)

        pos_at_yaw_curves = []
        for key, label in (
            ("log_min_position_error", "Best position anywhere"),
            ("log_position_at_min_yaw", "Position at best yaw"),
            ("log_final_position_error", "Final position"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                pos_at_yaw_curves.append((xk, rolling(yk, window), label))
        p = outdir / "position_at_best_yaw.png"
        if save_line(
            p,
            "Camera position error when yaw is best",
            "Position error (m)",
            pos_at_yaw_curves,
            xlabel="Completed episode",
        ):
            written.append(p)

        base_curves = []
        for key, label in (
            ("log_final_base_position_error", "Base distance to arrow base"),
            ("log_final_abs_longitudinal_error", "|Longitudinal|"),
            ("log_final_abs_lateral_error", "|Lateral|"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                base_curves.append((xk, rolling(yk, window), label))
        p = outdir / "base_goal_error.png"
        if save_line(
            p,
            "Final robot-base error in arrow coordinates",
            "Error (m)",
            base_curves,
            xlabel="Completed episode",
        ):
            written.append(p)

        # Physical terminal outcomes. Success is intentionally non-terminal in
        # v13.14 and is plotted separately as "ever reached precise pose".
        term_curves = []
        for key, label in (
            ("log_timeout", "Timeout"),
            ("log_fall", "Fall"),
            ("log_room_exit", "Room exit"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                term_curves.append(
                    (xk, [100.0 * value for value in rolling(yk, window)], label)
                )
        p = outdir / "termination_rates.png"
        if save_line(p, "Physical episode endings", "Rate (%)", term_curves, xlabel="Completed episode"):
            written.append(p)

        x_wall, y_wall = wall_exposure_series(episodes)
        p = outdir / "wall_guard_exposure.png"
        if save_line(
            p,
            "Time spent near outer wall guard",
            "Episode steps in guard (%)",
            [(x_wall, rolling(y_wall, window), f"Rolling {window}")],
            zero=True,
            xlabel="Completed episode",
        ):
            written.append(p)

        x_tol, y_tol = tolerance_occupancy_series(episodes)
        p = outdir / "tolerance_occupancy.png"
        if save_line(
            p,
            "Time inside precise camera-pose tolerance",
            "Episode steps in tolerance (%)",
            [(x_tol, rolling(y_tol, window), f"Rolling {window}")],
            zero=True,
            xlabel="Completed episode",
        ):
            written.append(p)

        # Longest precise-pose dwell achieved in each episode.
        xh, yh = episode_series(episodes, alias_key(episodes, "log_max_hold_steps"))
        p = outdir / "max_hold_steps.png"
        if save_line(
            p,
            "Longest consecutive precise-pose dwell",
            "Consecutive steps",
            [(xh, rolling(yh, window), f"Rolling {window}")],
            zero=True,
            xlabel="Completed episode",
        ):
            written.append(p)

        # First 5-step precise success time, only for episodes that reached it.
        fs_key = alias_key(episodes, "log_first_success_step")
        xfs, yfs = [], []
        if fs_key is not None:
            for episode_index, row in enumerate(episodes, start=1):
                if finite_number(row.get(fs_key)) and float(row[fs_key]) > 0.0:
                    xfs.append(float(episode_index)); yfs.append(float(row[fs_key]))
        p = outdir / "first_precision_step.png"
        if save_line(
            p,
            "First precise 5-step hold",
            "Episode step",
            [(xfs, yfs, "Successful episodes")],
            xlabel="Completed episode",
        ):
            written.append(p)

        # Does the actor actually condition its far-field actions on the target?
        # For |bearing| > 20 deg, random 11-action behavior is roughly 27% toward,
        # 27% away, and 45% no-yaw because the two strafes add no yaw. Learning should push toward up and away down.
        alignment_curves = []
        for numerator, label in (
            ("log_turn_toward_target", "Turn toward target"),
            ("log_turn_away_from_target", "Turn away from target"),
            ("log_no_yaw_when_turn_needed", "No yaw while turn needed"),
        ):
            xa, ya = episode_ratio_series(
                episodes, numerator, "log_target_turn_opportunities"
            )
            if ya:
                alignment_curves.append((xa, rolling(ya, window), label))
        p = outdir / "target_turn_alignment.png"
        if save_line(
            p,
            "Far-field action alignment with target bearing",
            "Share of turn-needed steps (%)",
            alignment_curves,
            xlabel="Completed episode",
        ):
            written.append(p)

        # Near-target FINAL-yaw action alignment. This is distinct from the
        # far-field bearing plot above and diagnoses final-yaw action choice.
        near_yaw_curves = []
        for numerator, label in (
            ("log_yaw_correcting_near", "Yaw-correcting"),
            ("log_yaw_away_near", "Yaw-away"),
            ("log_no_yaw_near", "No yaw"),
        ):
            xn, yn = episode_ratio_series(
                episodes, numerator, "log_near_yaw_opportunities"
            )
            if yn:
                near_yaw_curves.append((xn, rolling(yn, window), label))
        p = outdir / "near_target_yaw_alignment.png"
        if save_line(
            p,
            "Near-target action alignment with final yaw",
            "Share of near/wrong-yaw steps (%)",
            near_yaw_curves,
            xlabel="Completed episode",
        ):
            written.append(p)

        xa, ya = episode_ratio_series(
            episodes, "log_forward_when_target_ahead", "log_target_ahead_opportunities"
        )
        p = outdir / "target_forward_alignment.png"
        if save_line(
            p,
            "Forward motion when target position is ahead",
            "Forward-command share (%)",
            [(xa, rolling(ya, window), f"Rolling {window}")],
            xlabel="Completed episode",
        ):
            written.append(p)

        # Exact v13.20 state-reward decomposition.
        reward_curves = []
        for key, label in (
            ("log_reward_proximity", "Proximity state reward"),
            ("log_reward_in_tolerance", "Precise-tolerance occupancy reward"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                reward_curves.append((xk, rolling(yk, window), label))
        p = outdir / "reward_components.png"
        if save_line(
            p,
            "Episode reward components",
            "Cumulative contribution",
            reward_curves,
            zero=True,
            xlabel="Completed episode",
        ):
            written.append(p)

        # Goal-sampler health and heading-dependent rejection bias.
        sampler_curves = []
        for key, label in (
            ("log_goal_sample_attempts", "Sampling attempts"),
            ("log_goal_sample_reject_wall", "Wall/footprint rejects"),
            ("log_goal_sample_reject_object", "Object rejects"),
            ("log_goal_sample_reject_start", "Start-on-arrow rejects"),
        ):
            xk, yk = episode_series(episodes, alias_key(episodes, key))
            if yk:
                sampler_curves.append((xk, rolling(yk, window), label))
        p = outdir / "goal_sampling_counts.png"
        if save_line(
            p,
            "Goal sampling diagnostics",
            "Count per episode",
            sampler_curves,
            zero=True,
            xlabel="Completed episode",
        ):
            written.append(p)

        heading_curves = []
        for heading_bin in range(8):
            low = -180 + 45 * heading_bin
            high = low + 45
            xh, yh = rolling_count_ratio_series(
                episodes,
                f"log_goal_heading_rejections_bin_{heading_bin}",
                f"log_goal_heading_attempts_bin_{heading_bin}",
                window,
            )
            if yh:
                heading_curves.append((xh, yh, f"[{low},{high}) deg"))
        p = outdir / "goal_sampling_rejection_by_heading.png"
        if save_line(
            p,
            "Goal-sampler rejection rate by desired base heading",
            "Rejected candidates (%)",
            heading_curves,
            xlabel="Completed episode",
        ):
            written.append(p)

        # Mean geometric state error used by the dense reward.
        xge, yge = [], []
        _gk = alias_key(episodes, "log_geometric_error")
        _lk = first_key(episodes, ("episode/length", "train_length", "episode_length"))
        if _gk is not None and _lk is not None:
            for _ei, _row in enumerate(episodes, start=1):
                if finite_number(_row.get(_gk)) and finite_number(_row.get(_lk)) and float(_row[_lk]) > 0:
                    xge.append(float(_ei)); yge.append(float(_row[_gk]) / float(_row[_lk]))
        p = outdir / "geometric_error.png"
        if save_line(
            p,
            "Mean state-reward geometric error",
            "Mean equivalent error (m)",
            [(xge, rolling(yge, window), f"Rolling {window}")],
            xlabel="Completed episode",
        ):
            written.append(p)

        # Per-episode action counts -> percentages.
        action_names = (
            "forward", "backward", "left_arc", "right_arc", "coarse_left",
            "coarse_right", "stop", "fine_left", "fine_right",
            "strafe_left", "strafe_right",
        )
        action_keys = [alias_key(episodes, f"log_action_{name}") for name in action_names]
        action_x: list[float] = []
        action_rows: list[list[float]] = []
        for episode_index, row in enumerate(episodes, start=1):
            vals = []
            for key in action_keys:
                vals.append(float(row.get(key, 0.0)) if key and finite_number(row.get(key)) else 0.0)
            total = sum(vals)
            if total > 0:
                action_x.append(float(episode_index))
                action_rows.append([value / total for value in vals])
        if action_rows:
            curves = []
            for index, label in enumerate(action_names):
                values = [100.0 * row[index] for row in action_rows]
                curves.append((action_x, rolling(values, window), label.replace("_", " ")))
            p = outdir / "action_usage.png"
            if save_line(p, "Policy action usage", "Action share (%)", curves, xlabel="Completed episode"):
                written.append(p)

    # Dreamer losses: recognize the actual slash-separated names printed by
    # NM512/r2dreamer, plus simple aliases for nearby forks.
    image_curve = metric_curve(rows, ("train/loss/image", "loss/image", "image_loss"), "Image")
    p = outdir / "image_loss.png"
    if save_line(p, "Image reconstruction loss", "Loss", [image_curve]):
        written.append(p)

    wm_specs = (
        (("train/loss/dyn", "loss/dyn", "dyn_loss"), "Dynamics"),
        (("train/loss/rep", "loss/rep", "rep_loss"), "Representation"),
        (("train/loss/proprio", "loss/proprio", "proprio_loss"), "Proprioception"),
        (("train/loss/goal_vec", "loss/goal_vec", "goal_vec_loss", "train/loss/observation", "loss/observation"), "Goal vector"),
        (("train/loss/rew", "loss/rew", "reward_loss"), "Reward"),
        (("train/loss/con", "loss/con", "continuation_loss"), "Continuation"),
    )
    wm_curves = [metric_curve(rows, keys, label) for keys, label in wm_specs]
    p = outdir / "world_model_losses.png"
    if save_line(p, "World-model losses", "Loss", wm_curves, zero=True):
        written.append(p)

    weighted_specs = (
        (("train/loss_weighted/image", "loss_weighted/image"), "Image weighted"),
        (("train/loss_weighted/proprio", "loss_weighted/proprio"), "Proprioception weighted"),
        (("train/loss_weighted/goal_vec", "loss_weighted/goal_vec", "train/loss_weighted/observation", "loss_weighted/observation"), "Goal vector weighted"),
        (("train/loss_weighted/rew", "loss_weighted/rew"), "Reward weighted"),
        (("train/loss_weighted/dyn", "loss_weighted/dyn"), "Dynamics weighted"),
        (("train/loss_weighted/rep", "loss_weighted/rep"), "Representation weighted"),
    )
    weighted_curves = [metric_curve(rows, keys, label) for keys, label in weighted_specs]
    p = outdir / "weighted_world_model_losses.png"
    if save_line(p, "Weighted world-model loss contributions", "Weighted loss", weighted_curves, zero=True):
        written.append(p)

    control_specs = (
        (("train/loss/policy", "loss/policy", "policy_loss", "actor_loss"), "Policy"),
        (("train/loss/value", "loss/value", "value_loss"), "Value"),
        (("train/loss/repval", "loss/repval", "repval_loss"), "Rep-value"),
    )
    control_curves = [metric_curve(rows, keys, label) for keys, label in control_specs]
    p = outdir / "actor_value_losses.png"
    if save_line(p, "Policy and value losses", "Loss", control_curves, zero=True):
        written.append(p)

    policy_specs = (
        (("train/action_entropy", "action_entropy"), "Action entropy"),
        (("train/adv_std", "adv_std"), "Advantage std"),
    )
    policy_curves = [metric_curve(rows, keys, label) for keys, label in policy_specs]
    p = outdir / "policy_diagnostics.png"
    if save_line(p, "Policy exploration diagnostics", "Value", policy_curves, zero=True):
        written.append(p)

    # ReturnEMA quantiles explain whether the normalizer floor at 1.0 is active.
    q05 = metric_curve(rows, ("train/ret_005", "ret_005"), "Return EMA q05")
    q95 = metric_curve(rows, ("train/ret_095", "ret_095"), "Return EMA q95")
    q05_key = first_key(rows, ("train/ret_005", "ret_005"))
    q95_key = first_key(rows, ("train/ret_095", "ret_095"))
    xs_spread, ys_spread, ys_scale = [], [], []
    if q05_key is not None and q95_key is not None:
        for row in rows:
            if finite_number(row.get("step")) and finite_number(row.get(q05_key)) and finite_number(row.get(q95_key)):
                spread = float(row[q95_key]) - float(row[q05_key])
                xs_spread.append(float(row["step"]))
                ys_spread.append(spread)
                ys_scale.append(max(spread, 1.0))
    p = outdir / "return_ema_scale.png"
    if save_line(
        p,
        "ReturnEMA quantiles, spread, and actor scale",
        "Return",
        [q05, q95, (xs_spread, ys_spread, "q95-q05"), (xs_spread, ys_scale, "max(spread,1)")],
        zero=True,
    ):
        written.append(p)

    entropy_specs = (
        (("train/dyn_entropy", "dyn_entropy"), "Prior / dynamics entropy"),
        (("train/rep_entropy", "rep_entropy"), "Posterior / representation entropy"),
    )
    entropy_curves = [metric_curve(rows, keys, label) for keys, label in entropy_specs]
    p = outdir / "rssm_entropies.png"
    if save_line(p, "RSSM prior/posterior entropies", "Entropy", entropy_curves):
        written.append(p)

    value_specs = (
        (("train/ret_replay_mean", "ret_replay_mean"), "Replay return"),
        (("train/value_replay_mean", "value_replay_mean"), "Value"),
        (("train/slow_value_replay_mean", "slow_value_replay_mean"), "Slow value"),
    )
    value_curves = [metric_curve(rows, keys, label) for keys, label in value_specs]
    p = outdir / "return_value.png"
    if save_line(p, "Imagined return and value", "Value", value_curves, zero=True):
        written.append(p)

    # Any gradient-norm metrics the repo may expose.
    numeric_keys = sorted(
        {key for row in rows for key, value in row.items() if key != "step" and finite_number(value)}
    )
    grad_curves = []
    for key in [key for key in numeric_keys if "grad_norm" in key.lower()][:12]:
        xk, yk = series(rows, key)
        if yk:
            grad_curves.append((xk, yk, key))
    p = outdir / "gradient_norms.png"
    if save_line(p, "Gradient norms", "Norm", grad_curves):
        written.append(p)

    # Compact terminal snapshot.
    summary = [
        f"metrics: {metrics}",
        f"rows: {len(rows)}",
        f"completed train episodes: {len(episodes)}",
    ]
    if episodes and return_key:
        for candidates, label, percent in (
            ((return_key,), "return", False),
            (("log_success", "episode/log_success"), "success", True),
            (("log_final_position_error", "episode/log_final_position_error"), "final position m", False),
            (("log_min_position_error", "episode/log_min_position_error"), "min position m", False),
            (("log_final_bearing_error_deg", "episode/log_final_bearing_error_deg"), "final bearing deg", False),
            (("log_final_yaw_error_deg", "episode/log_final_yaw_error_deg"), "final yaw deg", False),
            (("log_yaw_at_min_position_deg", "episode/log_yaw_at_min_position_deg"), "yaw at min position deg", False),
            (("log_position_at_min_yaw", "episode/log_position_at_min_yaw"), "position at min yaw m", False),
            (("log_final_base_position_error", "episode/log_final_base_position_error"), "final base position m", False),
            (("log_final_abs_longitudinal_error", "episode/log_final_abs_longitudinal_error"), "final |longitudinal| m", False),
            (("log_final_abs_lateral_error", "episode/log_final_abs_lateral_error"), "final |lateral| m", False),
        ):
            key = first_key(episodes, candidates)
            _, vals = series(episodes, key)
            if vals:
                value = rolling(vals, window)[-1]
                if percent:
                    value *= 100.0
                    summary.append(f"rolling {window} {label}: {value:.2f}%")
                else:
                    summary.append(f"rolling {window} {label}: {value:.4f}")

    if episodes:
        _, pos_backslide = episode_difference_series(
            episodes, "log_final_position_error", "log_min_position_error"
        )
        if pos_backslide:
            summary.append(
                f"rolling {window} position backslide m: {rolling(pos_backslide, window)[-1]:.4f}"
            )
        _, wall_exposure = wall_exposure_series(episodes)
        if wall_exposure:
            summary.append(
                f"rolling {window} wall guard exposure: {rolling(wall_exposure, window)[-1]:.2f}%"
            )
        _, tol_occ = tolerance_occupancy_series(episodes)
        if tol_occ:
            summary.append(
                f"rolling {window} precise tolerance occupancy: {rolling(tol_occ, window)[-1]:.2f}%"
            )
        _, max_hold = episode_series(episodes, alias_key(episodes, "log_max_hold_steps"))
        if max_hold:
            summary.append(
                f"rolling {window} max hold steps: {rolling(max_hold, window)[-1]:.2f}"
            )
        _, turn_toward = episode_ratio_series(
            episodes, "log_turn_toward_target", "log_target_turn_opportunities"
        )
        if turn_toward:
            summary.append(
                f"rolling {window} turn-toward-target: {rolling(turn_toward, window)[-1]:.2f}%"
            )
        _, forward_ahead = episode_ratio_series(
            episodes, "log_forward_when_target_ahead", "log_target_ahead_opportunities"
        )
        if forward_ahead:
            summary.append(
                f"rolling {window} forward-when-target-ahead: {rolling(forward_ahead, window)[-1]:.2f}%"
            )

    if ys_spread:
        summary.append(f"latest ReturnEMA q95-q05 spread: {ys_spread[-1]:.4f}")
        summary.append(f"latest ReturnEMA actor scale: {max(ys_spread[-1], 1.0):.4f}")

    action_entropy_key = first_key(rows, ("train/action_entropy", "action_entropy"))
    _, entropy = series(rows, action_entropy_key)
    if entropy:
        summary.append(f"latest action entropy: {entropy[-1]:.4f}")
    adv_std_key = first_key(rows, ("train/adv_std", "adv_std"))
    _, adv_std = series(rows, adv_std_key)
    if adv_std:
        summary.append(f"latest advantage std: {adv_std[-1]:.4f}")

    # Append the three retained checkpoint states to the run summary.
    try:
        checkpoint_state = json.loads(
            (logdir / "checkpoint_summary.json").read_text(encoding="utf-8")
        )
    except Exception:
        checkpoint_state = {}
    for key, label in (
        ("latest", "latest checkpoint"),
        ("best_success", "best-success checkpoint"),
        ("best_reward", "best-reward checkpoint"),
    ):
        data = checkpoint_state.get(key) or {}
        if not data:
            continue
        parts = [f"{label}: step={data.get('step', '?')}"]
        selection = data.get("selection") or {}
        if finite_number(selection.get("success_rate")):
            parts.append(f"success={100.0 * float(selection['success_rate']):.2f}%")
        if finite_number(selection.get("mean_return")):
            parts.append(f"return={float(selection['mean_return']):.3f}")
        summary.append(" | ".join(parts))

    tmp_summary = outdir / "latest_summary.tmp.txt"
    tmp_summary.write_text("\n".join(summary) + "\n", encoding="utf-8")
    tmp_summary.replace(outdir / "latest_summary.txt")
    return written, len(episodes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, type=Path)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--watch-episodes", action="store_true")
    ap.add_argument("--every-episodes", type=int, default=1)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    # Backward-compatible v13.6 switches.
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=None)
    args = ap.parse_args()
    if args.window < 1:
        raise ValueError("--window must be >= 1")
    if args.every_episodes < 1:
        raise ValueError("--every-episodes must be >= 1")

    live = args.watch_episodes or args.watch
    poll = args.poll_interval if args.interval is None else args.interval
    logdir = args.logdir.expanduser().resolve()
    last_plotted_episodes = -1

    while True:
        try:
            rows = read_jsonl(logdir / "metrics.jsonl")
            episodes, _ = episode_rows(rows)
            count = len(episodes)
            should_plot = not live
            if live and count > 0:
                should_plot = (
                    last_plotted_episodes < 0
                    or count - last_plotted_episodes >= args.every_episodes
                )
            if should_plot:
                written, plotted_count = plot_once(logdir, args.window)
                last_plotted_episodes = plotted_count
                print(
                    f"[LIVE PLOTS] episodes={plotted_count} wrote={len(written)} "
                    f"dir={logdir / 'plots'}",
                    flush=True,
                )
        except FileNotFoundError as exc:
            if not live:
                raise
            if last_plotted_episodes < 0:
                print(f"[LIVE PLOTS] {exc}", flush=True)
        except Exception as exc:
            # Never kill training because a diagnostic plot failed. The sidecar
            # stays alive and retries on the next completed episode.
            print(f"[LIVE PLOTS][WARNING] {type(exc).__name__}: {exc}", flush=True)
            if not live:
                raise

        if not live:
            break
        time.sleep(max(float(poll), 0.5))


if __name__ == "__main__":
    main()
