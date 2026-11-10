#!/usr/bin/env python3
"""Regenerate revision tables, intervals and plots from saved raw NPZ files only."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import load_json, save_json, write_rows
from .metrics import CONDITIONS, bootstrap_weights, routing_vectors, run_metrics, summarize_runs, validate_run


_REPORT_REQUIRED_ARRAYS = frozenset({
    "image_ids", "labels", "alpha_clean", "alpha_adv",
    "expert_probs_clean", "expert_probs_adv", "linf", "l2", "queries",
    "first_success_query", *("probs_" + condition for condition in CONDITIONS),
})

# These arrays are inexpensive but, when present, let validate_run retain the
# same pairing/route and transfer checks as the original report implementation.
# Deliberately exclude cached backbone/token features and logits: a sparse
# query-snapshot archive can contain about 1 GB of them, yet no report metric
# or figure consumes them.
_REPORT_OPTIONAL_ARRAYS = frozenset({
    "image_ids_clean", "image_ids_adv", "route_clean", "route_adv",
    "dominant_expert_clean", "dominant_expert_adv", "router_asr",
    "top1_flip_rate", "surrogate_route_clean", "surrogate_route_adv",
    "surrogate_success",
})
_REPORT_ARRAYS = _REPORT_REQUIRED_ARRAYS | _REPORT_OPTIONAL_ARRAYS


def _load(path: Path, metadata: Mapping[str, Any] | None = None):
    """Load only arrays consumed by the report, never cached model features.

    NPZ files are ZIP archives, so indexing an individual member does not
    materialize the other members.  This keeps the report bounded by one
    replicate group instead of decompressing the 154-GB snapshot collection
    into RAM merely to calculate scalar summaries.
    """
    meta = dict(metadata) if metadata is not None else load_json(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key in _REPORT_ARRAYS}
    validate_run(arrays, meta)
    return arrays, meta


def _grouped_paths(paths: list[Path], *, snapshots: bool = False) -> dict[tuple, list[tuple[Path, dict]]]:
    """Index small JSON metadata first, without opening the large NPZ payloads."""
    grouped: dict[tuple, list[tuple[Path, dict]]] = {}
    for path in paths:
        meta = load_json(path.with_suffix(".json"))
        grouped.setdefault(_group_key(meta, snapshots=snapshots), []).append((path, meta))
    return grouped


def _group_key(meta: dict, *, snapshots: bool = False) -> tuple:
    return tuple(meta.get(key) for key in ("dataset", "model", "architecture", "attack", "target", "epsilon", "surrogate_acquisition_n")) + ((meta.get("requested_query_budget"),) if snapshots else ())


def _group_labels(key: tuple, snapshot: bool = False) -> dict:
    names = ["dataset", "model", "architecture", "attack", "target", "epsilon", "surrogate_acquisition_n"] + (["requested_query_budget"] if snapshot else [])
    return dict(zip(names, key))


def _summarize(paths: list[Path], replicates: int, seed: int) -> tuple[list[dict], list[dict]]:
    summary: list[dict] = []; per_run: list[dict] = []
    # Raw outputs and query snapshots are separate calls to this function, so
    # either every path is a snapshot or none is.  Group from JSON sidecars
    # before loading a single NPZ so no full experiment collection is retained.
    snapshots = bool(paths and "requested_query_budget" in load_json(paths[0].with_suffix(".json")))
    grouped = _grouped_paths(paths, snapshots=snapshots)
    for key, items in grouped.items():
        is_snapshot = snapshots
        first_ids = first_y = None
        weights = None
        point_rows = []; bootstrap_rows = []
        for path, indexed_meta in items:
            data, meta = _load(path, indexed_meta)
            if first_ids is None:
                # Copies prevent the first (potentially large) NPZ dictionary
                # from remaining alive while the rest of the group is read.
                first_ids = np.asarray(data["image_ids"]).copy()
                first_y = np.asarray(data["labels"]).copy()
                weights = bootstrap_weights(len(first_ids), replicates, seed)
            elif not np.array_equal(first_ids, data["image_ids"]) or not np.array_equal(first_y, data["labels"]):
                raise ValueError(f"Cannot aggregate group with different image pairing: {key}")
            point = run_metrics(data, include_experts=True)
            boot = run_metrics(data, weights=weights, include_experts=True)
            point_rows.append({metric: float(np.asarray(value)[0]) for metric, value in point.items()})
            bootstrap_rows.append({metric: np.asarray(value) for metric, value in boot.items()})
            labels = _group_labels(key, is_snapshot)
            for metric, value in point_rows[-1].items():
                per_run.append({**labels, "seed": meta.get("seed"), "raw_file": path.name, "metric": metric, "value": value})
            # `bootstrap_rows` contains only small scalar-vector summaries;
            # releasing the archive data here keeps memory independent of the
            # number and payload size of NPZ inputs.
            del data, point, boot
        metrics = sorted(set().union(*(row.keys() for row in point_rows)))
        labels = _group_labels(key, is_snapshot)
        for metric in metrics:
            values = np.asarray([row.get(metric, np.nan) for row in point_rows], dtype=float)
            boots = np.stack([row.get(metric, np.full(replicates, np.nan)) for row in bootstrap_rows])
            estimate = asdict(summarize_runs(values, boots))
            summary.append({**labels, "metric": metric, **estimate})
    return summary, per_run


_MATCHED_TARGET_CONTRASTS = (
    # name, source metric, multiplier on (router-targeted - fused-targeted),
    # effect definition, and the interpretation of a positive estimate.
    ("router_flip_rate_excess", "routing/router_asr", 1.0,
     "router-targeted minus fused-output-targeted", "more target-router top-1 flips"),
    ("routing_l1_shift_excess", "routing/routing_l1_shift", 1.0,
     "router-targeted minus fused-output-targeted", "larger L1 routing-weight shift"),
    ("routing_l2_shift_excess", "routing/routing_l2_shift", 1.0,
     "router-targeted minus fused-output-targeted", "larger L2 routing-weight shift"),
    ("final_router_margin_reduction", "routing/final_router_margin", -1.0,
     "fused-output-targeted minus router-targeted", "a lower final clean-route margin"),
    ("fused_probability_shift_excess", "AA/probability_shift", 1.0,
     "router-targeted minus fused-output-targeted", "larger fused-probability shift"),
    ("macro_f1_degradation_excess", "AA/macro_f1", -1.0,
     "fused-output-targeted minus router-targeted", "lower fused macro-F1"),
    ("map_degradation_excess", "AA/map", -1.0,
     "fused-output-targeted minus router-targeted", "lower fused mAP"),
    ("macro_auroc_degradation_excess", "AA/macro_auroc", -1.0,
     "fused-output-targeted minus router-targeted", "lower fused macro AUROC"),
    ("bce_increase_excess", "AA/bce", 1.0,
     "router-targeted minus fused-output-targeted", "larger fused BCE"),
)


def _matched_target_contrasts(paths: list[Path], replicates: int, seed: int) -> list[dict]:
    """Compare router- and fused-targeted attacks on matched seed/image pairs.

    The two attack targets are not independent samples: for each seed they use
    the same immutable manifest. The table therefore reuses one image-level
    bootstrap draw for both outputs before differencing them, then aggregates
    the resulting seed-level contrasts. Partial queues are allowed; only
    completed seed pairs are included and their count is written explicitly.
    """
    fields = ("dataset", "model", "architecture", "attack", "epsilon", "surrogate_acquisition_n")
    # Keep paths and their small metadata sidecars here.  Holding the loaded
    # arrays for both targets of every seed previously retained all raw runs.
    grouped: dict[tuple, dict[str, dict[int, tuple[Path, dict]]]] = {}
    for path in paths:
        meta = load_json(path.with_suffix(".json"))
        target = meta.get("target")
        if target not in ("router", "fused"):
            continue
        key = tuple(meta.get(name) for name in fields)
        by_target = grouped.setdefault(key, {"router": {}, "fused": {}})
        run_seed = int(meta["seed"])
        if run_seed in by_target[target]:
            raise ValueError(f"Duplicate {target}-targeted seed in matched contrast group: {key}")
        by_target[target][run_seed] = (path, meta)

    rows: list[dict] = []
    for key, targets in grouped.items():
        router, fused = targets["router"], targets["fused"]
        matched_seeds = sorted(set(router) & set(fused))
        if not matched_seeds:
            continue
        points: dict[str, list[float]] = {name: [] for name, *_ in _MATCHED_TARGET_CONTRASTS}
        bootstraps: dict[str, list[np.ndarray]] = {name: [] for name, *_ in _MATCHED_TARGET_CONTRASTS}
        query_budget = None
        for run_seed in matched_seeds:
            router_path, router_indexed_meta = router[run_seed]
            fused_path, fused_indexed_meta = fused[run_seed]
            router_data, router_meta = _load(router_path, router_indexed_meta)
            fused_data, fused_meta = _load(fused_path, fused_indexed_meta)
            if not np.array_equal(router_data["image_ids"], fused_data["image_ids"]) or not np.array_equal(router_data["labels"], fused_data["labels"]):
                raise ValueError(f"Matched target runs use different images or labels: {key}, seed={run_seed}")
            for setting in ("query_budget", "pgd_steps", "pgd_restarts"):
                if router_meta.get(setting) != fused_meta.get(setting):
                    raise ValueError(f"Matched target runs differ in {setting}: {key}, seed={run_seed}")
            if query_budget is None:
                query_budget = router_meta.get("query_budget")
            weights = bootstrap_weights(len(router_data["image_ids"]), replicates, seed)
            router_point = run_metrics(router_data, include_experts=False)
            fused_point = run_metrics(fused_data, include_experts=False)
            router_boot = run_metrics(router_data, weights=weights, include_experts=False)
            fused_boot = run_metrics(fused_data, weights=weights, include_experts=False)
            for name, metric, multiplier, _, _ in _MATCHED_TARGET_CONTRASTS:
                if metric not in router_point or metric not in fused_point:
                    raise KeyError(f"Missing matched-target metric {metric}")
                points[name].append(float(multiplier * (np.asarray(router_point[metric])[0] - np.asarray(fused_point[metric])[0])))
                bootstraps[name].append(multiplier * (np.asarray(router_boot[metric]) - np.asarray(fused_boot[metric])))
            del router_data, fused_data, router_point, fused_point, router_boot, fused_boot
        labels = dict(zip(fields, key))
        for name, metric, _, definition, positive_interpretation in _MATCHED_TARGET_CONTRASTS:
            estimate = asdict(summarize_runs(np.asarray(points[name]), np.stack(bootstraps[name])))
            rows.append({**labels, "contrast": name, "source_metric": metric,
                         "effect_definition": definition, "positive_interpretation": positive_interpretation,
                         "matched_query_budget": query_budget, "router_target_runs_available": len(router),
                         "fused_target_runs_available": len(fused), "matched_seeds": ";".join(map(str, matched_seeds)),
                         **estimate})
    return rows


def _plot_matched_target_contrasts(rows: list[dict], plots: Path) -> None:
    """Draw compact, matched-control effect panels with their paired CIs."""
    shown = ("router_flip_rate_excess", "routing_l1_shift_excess", "routing_l2_shift_excess",
             "final_router_margin_reduction", "macro_f1_degradation_excess", "fused_probability_shift_excess")
    labels = {name: interpretation for name, _, _, _, interpretation in _MATCHED_TARGET_CONTRASTS}
    fields = ("dataset", "model", "architecture", "attack", "epsilon", "surrogate_acquisition_n")
    groups = sorted({tuple(row.get(field) for field in fields) for row in rows})
    for group in groups:
        entries = {row["contrast"]: row for row in rows if tuple(row.get(field) for field in fields) == group}
        if not any(name in entries and np.isfinite(entries[name]["mean"]) for name in shown):
            continue
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
        for ax, name in zip(axes.flat, shown):
            entry = entries.get(name)
            if entry is None or not np.isfinite(entry["mean"]):
                ax.axis("off")
                continue
            low = max(0.0, entry["mean"] - entry["ci_low"]) if np.isfinite(entry["ci_low"]) else 0.0
            high = max(0.0, entry["ci_high"] - entry["mean"]) if np.isfinite(entry["ci_high"]) else 0.0
            ax.bar([0], [entry["mean"]], yerr=[[low], [high]], capsize=4, color="#4C78A8")
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_xticks([])
            ax.set_title(labels[name], fontsize=9)
            ax.set_ylabel("matched effect")
        fig.suptitle("Router-targeted excess effect (positive has the stated adverse interpretation)\n" + " / ".join(map(str, group[:4])))
        fig.tight_layout()
        tag = "__".join(_slug(value) for value in group)
        fig.savefig(plots / f"matched_target_contrast_{tag}.png", dpi=180)
        fig.savefig(plots / f"matched_target_contrast_{tag}.pdf")
        plt.close(fig)


def _surrogate_scaling_rows(summary: list[dict], paths: list[Path]) -> list[dict]:
    """Pivot transfer-PGD summaries into the required acquisition-scaling table."""
    fields = ("dataset", "model", "architecture", "attack", "target", "epsilon", "surrogate_acquisition_n")
    estimates = {(tuple(row.get(field) for field in fields), row["metric"]): row for row in summary}
    groups: dict[tuple, list[dict]] = {}
    for path in paths:
        # The scalar estimates above have already validated every raw NPZ.
        # This pivot only needs acquisition metadata, so never decompress the
        # cached features in a transfer archive a second time.
        meta = load_json(path.with_suffix(".json"))
        if meta.get("attack") != "transfer_pgd" or meta.get("surrogate_acquisition_n") is None:
            continue
        key = tuple(meta.get(field) for field in fields)
        groups.setdefault(key, []).append(meta)
    metrics = (
        ("clean_surrogate_target_routing_agreement", "routing/surrogate_clean_routing_agreement"),
        ("surrogate_attack_success_rate", "routing/surrogate_attack_success_rate"),
        ("target_router_flip_rate", "routing/router_asr"),
        ("target_flip_given_surrogate_flip", "routing/transfer_given_surrogate_success"),
        ("AA_macro_f1", "AA/macro_f1"), ("AA_map", "AA/map"), ("AA_macro_auroc", "AA/macro_auroc"),
    )
    rows: list[dict] = []
    for key, metas in sorted(groups.items()):
        labels = dict(zip(fields, key))
        def single(name: str):
            values = {meta.get(name) for meta in metas}
            if len(values) != 1:
                raise ValueError(f"Inconsistent {name} within transfer scaling group: {key}")
            return values.pop()
        row = {**labels, "n_independent_surrogate_runs": len(metas),
               "surrogate_nominal_acquisition_queries": single("surrogate_nominal_acquisition_queries"),
               "surrogate_shared_acquisition_queries": single("surrogate_shared_acquisition_queries"),
               "target_queries_during_generation": single("target_queries_during_generation"),
               "surrogate_training_seeds": ";".join(map(str, sorted(int(meta["seed"]) for meta in metas)))}
        for column, metric in metrics:
            estimate = estimates.get((key, metric))
            for suffix in ("mean", "run_sd", "ci_low", "ci_high", "n_runs", "n_valid_runs", "n_bootstrap", "n_valid_bootstrap"):
                row[f"{column}_{suffix}"] = np.nan if estimate is None else estimate[suffix]
        rows.append(row)
    return rows


def _plot_surrogate_scaling(rows: list[dict], plots: Path) -> None:
    fields = ("dataset", "model", "architecture", "target", "epsilon")
    groups = sorted({tuple(row.get(field) for field in fields) for row in rows})
    series = (("clean_surrogate_target_routing_agreement", "Clean surrogate–target routing agreement"),
              ("target_router_flip_rate", "Target router flip rate"),
              ("AA_macro_f1", "Target fused macro-F1 (AA)"))
    for group in groups:
        subset = sorted((row for row in rows if tuple(row.get(field) for field in fields) == group),
                        key=lambda row: row["surrogate_acquisition_n"])
        fig, axes = plt.subplots(1, len(series), figsize=(4.5 * len(series), 3.8), squeeze=False)
        for ax, (column, title) in zip(axes[0], series):
            valid = [row for row in subset if np.isfinite(row[f"{column}_mean"])]
            if not valid:
                ax.text(.5, .5, "Not applicable", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                continue
            x = np.asarray([row["surrogate_acquisition_n"] for row in valid], dtype=float)
            y = np.asarray([row[f"{column}_mean"] for row in valid], dtype=float)
            lo = np.asarray([max(0.0, row[f"{column}_mean"] - row[f"{column}_ci_low"]) if np.isfinite(row[f"{column}_ci_low"]) else 0.0 for row in valid])
            hi = np.asarray([max(0.0, row[f"{column}_ci_high"] - row[f"{column}_mean"]) if np.isfinite(row[f"{column}_ci_high"]) else 0.0 for row in valid])
            ax.errorbar(x, y, yerr=np.vstack((lo, hi)), marker="o", capsize=3)
            ax.set_xscale("log")
            ax.set_xlabel("Acquired clean target labels (N)")
            ax.set_title(title)
            ax.grid(alpha=.3)
        fig.suptitle("Transfer-surrogate acquisition scaling — " + " / ".join(map(str, group[:4])))
        fig.tight_layout()
        tag = "__".join(_slug(value) for value in group)
        fig.savefig(plots / f"surrogate_scaling_{tag}.png", dpi=180)
        fig.savefig(plots / f"surrogate_scaling_{tag}.pdf")
        plt.close(fig)


def _plot_attribution(rows: list[dict], plots: Path) -> None:
    chosen = [r for r in rows if r["metric"] in ("attribution/routing_recovery/macro_f1", "attribution/routing_recovery/map", "attribution/routing_recovery/macro_auroc")]
    if not chosen: return
    groups = sorted({(r["dataset"], r["model"], r["attack"], r["target"], r["epsilon"]) for r in chosen})
    names = ["Macro-F1", "mAP", "Macro AUROC"]; metric = {"Macro-F1": "attribution/routing_recovery/macro_f1", "mAP": "attribution/routing_recovery/map", "Macro AUROC": "attribution/routing_recovery/macro_auroc"}
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    for ax, group in zip(axes[0], groups):
        values = []
        for name in names:
            entry = next((r for r in chosen if (r["dataset"], r["model"], r["attack"], r["target"], r["epsilon"]) == group and r["metric"] == metric[name]), None)
            values.append((np.nan, np.nan, np.nan) if entry is None else (entry["mean"], entry["ci_low"], entry["ci_high"]))
        means = np.asarray([x[0] for x in values]); err = np.asarray([[x[0]-x[1] for x in values], [x[2]-x[0] for x in values]])
        ax.bar(range(len(names)), means, yerr=err, capsize=4); ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right"); ax.set_ylabel("AC − AA (routing recovery)")
        ax.set_title(" / ".join(map(str, group[:4])))
    fig.tight_layout(); fig.savefig(plots / "routing_recovery.png", dpi=180); fig.savefig(plots / "routing_recovery.pdf"); plt.close(fig)


def _plot_query_curve(rows: list[dict], plots: Path) -> None:
    chosen = [r for r in rows if r.get("requested_query_budget") is not None and r["metric"] in ("routing/router_asr", "AA/macro_f1")]
    if not chosen: return
    groups = sorted({(r["dataset"], r["model"], r["attack"], r["target"], r["epsilon"], r["metric"]) for r in chosen})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group in groups:
        subset = sorted((r for r in chosen if tuple(r[k] for k in ("dataset", "model", "attack", "target", "epsilon", "metric")) == group), key=lambda r: r["requested_query_budget"])
        x = [r["requested_query_budget"] for r in subset]; y = [r["mean"] for r in subset]
        ax.plot(x, y, marker="o", label=" / ".join(map(str, group[:5])) + " — " + group[-1])
    ax.set_xlabel("Requested target-query budget"); ax.set_ylabel("Metric"); ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(plots / "query_budget_curves.png", dpi=180); fig.savefig(plots / "query_budget_curves.pdf"); plt.close(fig)


def _slug(value: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _plot_routing_geometry(paths: list[Path], plots: Path) -> None:
    """Distribution and transition figures pooled over independent attack runs.

    This is descriptive: the tables retain the independent-run mean, SD and
    paired-bootstrap intervals. Pooling is used only to show the geometry of
    individual image changes without claiming that repeated seeds are new images.
    """
    groups = _grouped_paths(paths)
    for key, items in groups.items():
        first_meta = items[0][1]
        vectors: dict[str, list[np.ndarray]] = {}
        clean_alphas: list[np.ndarray] = []
        adv_alphas: list[np.ndarray] = []
        cc_probabilities: list[np.ndarray] = []
        aa_probabilities: list[np.ndarray] = []
        labels = None
        for path, indexed_meta in items:
            data, _ = _load(path, indexed_meta)
            if labels is None:
                labels = np.asarray(data["labels"]).copy()
            for name, values in routing_vectors(data).items():
                # Some routing vectors are views into the archive arrays.
                # Copy the compact metric vector so the large archive can be
                # discarded immediately after this replicate.
                vectors.setdefault(name, []).append(np.asarray(values).copy())
            clean_alphas.append(np.asarray(data["alpha_clean"]).copy())
            adv_alphas.append(np.asarray(data["alpha_adv"]).copy())
            cc_probabilities.append(np.asarray(data["probs_CC"]).copy())
            aa_probabilities.append(np.asarray(data["probs_AA"]).copy())
            del data
        vectors = {name: np.concatenate(values) for name, values in vectors.items()}
        # Keep epsilon and surrogate-acquisition size in filenames.  Omitting
        # them silently overwrote sensitivity/scaling plots that share the same
        # dataset, architecture, attack, and target.
        label = "__".join(_slug(x) for x in key)
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        fields = [("clean_route_margin_change", "Clean-route margin change", "#4C78A8"),
                  ("routing_l1_shift", "L1 routing-weight shift", "#F58518"),
                  ("routing_l2_shift", "L2 routing-weight shift", "#E45756"),
                  ("entropy_change", "Normalized entropy change", "#72B7B2"),
                  ("max_probability_change", "Maximum routing probability change", "#54A24B"),
                  ("linf", "Realized L∞ perturbation", "#B279A2"),
                  ("l2", "Realized L2 perturbation", "#9D755D")]
        for ax, (name, title, color) in zip(axes.flat, fields):
            ax.hist(vectors[name][np.isfinite(vectors[name])], bins=35, color=color, alpha=.85)
            if "change" in name: ax.axvline(0, color="black", linewidth=.7)
            ax.set_title(title); ax.set_ylabel("Image–run count")
        for ax in axes.flat[len(fields):]:
            ax.axis("off")
        fig.suptitle(f"Routing geometry — {first_meta['dataset']} / {first_meta['model']} / {first_meta['attack']} / {first_meta['target']}")
        fig.tight_layout(); fig.savefig(plots / f"routing_geometry_{label}.png", dpi=180); fig.savefig(plots / f"routing_geometry_{label}.pdf"); plt.close(fig)
        alpha_c = np.concatenate(clean_alphas); alpha_a = np.concatenate(adv_alphas)
        transitions = np.zeros((alpha_c.shape[1], alpha_c.shape[1]), dtype=int)
        np.add.at(transitions, (alpha_c.argmax(1), alpha_a.argmax(1)), 1)
        fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(transitions / np.maximum(transitions.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues")
        names = first_meta["expert_names"]; ax.set_xticks(range(len(names)), names, rotation=30, ha="right"); ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Adversarial dominant route"); ax.set_ylabel("Clean dominant route")
        for i in range(len(names)):
            for j in range(len(names)):
                rate = transitions[i, j] / max(transitions[i].sum(), 1)
                ax.text(j, i, str(transitions[i, j]), ha="center", va="center", color="white" if float(rate) > .5 else "black")
        fig.colorbar(image, ax=ax, label="Row-normalized transition rate"); fig.tight_layout(); fig.savefig(plots / f"route_transitions_{label}.png", dpi=180); fig.savefig(plots / f"route_transitions_{label}.pdf"); plt.close(fig)
        try:
            from sklearn.metrics import precision_recall_curve, roc_curve
            cc = np.mean(np.stack(cc_probabilities), axis=0); aa = np.mean(np.stack(aa_probabilities), axis=0)
            classes = first_meta["class_names"]; fig, axes = plt.subplots(2, len(classes), figsize=(4 * len(classes), 7), squeeze=False)
            for c, name in enumerate(classes):
                if np.unique(labels[:, c]).size < 2:
                    axes[0, c].text(.5, .5, "Undefined: one label state", ha="center", va="center", transform=axes[0, c].transAxes)
                    axes[1, c].text(.5, .5, "Undefined: one label state", ha="center", va="center", transform=axes[1, c].transAxes)
                    axes[0, c].set_title(name); axes[1, c].set_title(name)
                    continue
                for probabilities, style, line_label in ((cc, "-", "CC"), (aa, "--", "AA")):
                    pr, re, _ = precision_recall_curve(labels[:, c], probabilities[:, c]); fp, tp, _ = roc_curve(labels[:, c], probabilities[:, c])
                    axes[0, c].plot(re, pr, style, label=line_label); axes[1, c].plot(fp, tp, style, label=line_label)
                axes[0, c].set_title(name); axes[0, c].set_xlabel("Recall"); axes[0, c].set_ylabel("Precision"); axes[1, c].set_xlabel("False-positive rate"); axes[1, c].set_ylabel("True-positive rate"); axes[1, c].plot([0, 1], [0, 1], ":", color="grey"); axes[0, c].legend()
            fig.suptitle("Classwise fused prediction curves (mean probability across attack seeds)"); fig.tight_layout(); fig.savefig(plots / f"pr_roc_{label}.png", dpi=180); fig.savefig(plots / f"pr_roc_{label}.pdf"); plt.close(fig)
        except ImportError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261029)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args(); root = args.output_dir.resolve(); tables = root / "tables"; plots = root / "plots"; tables.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    raw = sorted((root / "raw_runs").glob("*.npz")); query = sorted((root / "query_budgets").glob("*.npz"))
    if not args.include_smoke:
        raw = [p for p in raw if not load_json(p.with_suffix(".json")).get("smoke", False)]
        query = [p for p in query if not load_json(p.with_suffix(".json")).get("smoke", False)]
    if not raw:
        raise FileNotFoundError(f"No completed raw runs under {root / 'raw_runs'}")
    summary, per_run = _summarize(raw, args.bootstrap_replicates, args.bootstrap_seed)
    query_summary, query_per_run = _summarize(query, args.bootstrap_replicates, args.bootstrap_seed) if query else ([], [])
    matched_contrasts = _matched_target_contrasts(raw, args.bootstrap_replicates, args.bootstrap_seed)
    surrogate_scaling = _surrogate_scaling_rows(summary, raw)
    write_rows(tables / "metrics_summary.csv", summary); write_rows(tables / "metrics_per_run.csv", per_run)
    write_rows(tables / "query_budget_results.csv", query_summary); write_rows(tables / "query_budget_per_run.csv", query_per_run)
    write_rows(tables / "matched_target_contrasts.csv", matched_contrasts)
    write_rows(tables / "surrogate_scaling_results.csv", surrogate_scaling)
    _plot_attribution(summary, plots); _plot_query_curve(query_summary, plots)
    _plot_matched_target_contrasts(matched_contrasts, plots); _plot_surrogate_scaling(surrogate_scaling, plots)
    _plot_routing_geometry(raw, plots)
    save_json(tables / "report_metadata.json", {"bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.bootstrap_seed,
                                                  "resampling_unit": "paired image_id", "raw_runs": len(raw), "query_snapshot_runs": len(query),
                                                  "matched_target_contrast_rows": len(matched_contrasts), "surrogate_scaling_rows": len(surrogate_scaling),
                                                  "input_only": "All values in this report were regenerated from NPZ raw outputs; no model or image data were loaded."})
    print(f"Wrote {tables / 'metrics_summary.csv'} and {plots}")


if __name__ == "__main__": main()
