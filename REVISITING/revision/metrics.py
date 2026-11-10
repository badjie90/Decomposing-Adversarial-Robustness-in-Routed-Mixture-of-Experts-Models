"""Auditable metrics and paired image bootstrap for multilabel routed models.

Macro metrics use the complete, fixed class set. An unobservable class is NaN;
it is never silently removed from a macro average. Bootstrap weights are image
multiplicities, reused for every condition, expert and independent attack run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

CONDITIONS = ("CC", "CA", "AC", "AA")
HIGHER_IS_BETTER = ("macro_f1", "map", "macro_auroc")
LOWER_IS_BETTER = ("bce", "probability_shift")


def _require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def entropy_and_collapse(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized Shannon entropy and the specified maximum-route collapse score.

    ``collapse`` is max_k alpha_k, exactly as defined in the manuscript and
    revision instructions.  It is deliberately not a concentration-index
    substitute such as sum(alpha**2): those metrics have different ranges and
    led to the earlier invalid entropy/collapse comparisons.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    _require(alpha.ndim == 2 and alpha.shape[1] >= 2, "Need at least two experts")
    _require(np.isfinite(alpha).all(), "Routing weights are not finite")
    _require(np.all(alpha >= -1e-8) and np.all(alpha <= 1 + 1e-8), "Routing weights outside [0,1]")
    _require(np.allclose(alpha.sum(1), 1, atol=1e-6, rtol=0), "Routing weights must sum to one")
    safe = np.clip(alpha, 0, 1)
    entropy = -(safe * np.log(np.maximum(safe, np.finfo(float).tiny))).sum(1) / np.log(alpha.shape[1])
    collapse = safe.max(1)
    _require(np.all((entropy >= -1e-8) & (entropy <= 1 + 1e-8)), "Normalized entropy out of bounds")
    _require(np.all((collapse >= 1 / alpha.shape[1] - 1e-8) & (collapse <= 1 + 1e-8)), "Collapse out of bounds")
    # Shannon entropy is at least min-entropy, including after averaging by
    # Jensen's inequality.  This catches impossible reported mean pairs.
    _require(np.all(entropy + 1e-7 >= -np.log(collapse) / np.log(alpha.shape[1])), "Impossible entropy/collapse pair")
    return entropy, collapse


def validate_entropy_collapse(entropy: float, collapse: float, experts: int) -> None:
    """Necessary feasibility checks also valid for means over images."""
    _require(experts >= 2, "Need at least two experts")
    _require(-1e-8 <= entropy <= 1 + 1e-8, "Normalized entropy out of bounds")
    _require(1 / experts - 1e-8 <= collapse <= 1 + 1e-8, "Collapse out of bounds")
    _require(entropy + 1e-7 >= -np.log(collapse) / np.log(experts), "Impossible mean entropy/collapse pair")


def validate_run(data: Mapping[str, np.ndarray], metadata: Mapping | None = None) -> None:
    """Reject mismatched IDs, invalid intervention outputs and budget violations."""
    metadata = metadata or {}
    required = ("image_ids", "labels", "alpha_clean", "alpha_adv", "expert_probs_clean", "expert_probs_adv", "linf", "l2", "queries", "first_success_query") + tuple("probs_" + c for c in CONDITIONS)
    _require(all(key in data for key in required), "Missing raw arrays: " + ", ".join(k for k in required if k not in data))
    ids, y = np.asarray(data["image_ids"]).astype(str), np.asarray(data["labels"])
    _require(ids.ndim == 1 and len(ids) > 0 and len(set(ids)) == len(ids), "Image IDs must be nonempty and unique")
    _require(y.ndim == 2 and y.shape[0] == len(ids), "Labels and image IDs disagree")
    _require(np.isin(y, [0, 1]).all(), "Labels must be binary")
    for key in ("image_ids_clean", "image_ids_adv"):
        if key in data:
            _require(np.array_equal(ids, np.asarray(data[key]).astype(str)), "Clean/adversarial image pairing mismatch")
    for suffix in ("clean", "adv"):
        alpha = np.asarray(data["alpha_" + suffix])
        _require(alpha.shape[0] == len(ids), "Routing/image count mismatch")
        entropy, collapse = entropy_and_collapse(alpha)
        validate_entropy_collapse(float(entropy.mean()), float(collapse.mean()), alpha.shape[1])
        for route_key in ("route_" + suffix, "dominant_expert_" + suffix):
            if route_key in data:
                _require(np.array_equal(data[route_key], alpha.argmax(1)), "Reported route differs from argmax(alpha)")
        expert = np.asarray(data["expert_probs_" + suffix])
        _require(expert.shape == (len(ids), alpha.shape[1], y.shape[1]), "Expert output shape mismatch")
        _require(np.isfinite(expert).all() and np.all((expert >= 0) & (expert <= 1)), "Invalid expert probabilities")
    for condition in CONDITIONS:
        probs = np.asarray(data["probs_" + condition])
        alpha = data["alpha_clean" if condition[1] == "C" else "alpha_adv"]
        expert = data["expert_probs_clean" if condition[0] == "C" else "expert_probs_adv"]
        _require(probs.shape == y.shape and np.isfinite(probs).all(), "Invalid fused probability shape/values")
        _require(np.all((probs >= 0) & (probs <= 1)), "Fused probabilities outside [0,1]")
        # CSA-GMoE fuses expert FEATURES before its learned object head. Its
        # separate expert classifiers are diagnostics, not mixture components.
        # Only a model explicitly declaring probability fusion permits this
        # reconstruction without the saved features and checkpoint.
        if metadata.get("fusion_scheme") == "probability_mixture":
            expected = np.einsum("nk,nkc->nc", alpha, expert)
            _require(np.allclose(probs, expected, atol=2e-6, rtol=1e-5), "Incorrect controlled fusion for " + condition)
    if "intervention_validated" in metadata:
        _require(bool(metadata["intervention_validated"]), "Controlled routing injection was not validated")
    for key in ("linf", "l2", "queries", "first_success_query"):
        arr = np.asarray(data[key])
        _require(arr.shape == (len(ids),) and np.isfinite(arr).all(), "Invalid per-image " + key)
        _require(np.all(arr >= (-1 if key == "first_success_query" else 0)), "Negative " + key)
    _require(np.all(data["l2"] + 1e-6 >= data["linf"]), "L2 is smaller than L_inf")
    if "epsilon" in metadata:
        _require(np.all(data["linf"] <= float(metadata["epsilon"]) + 2e-6), "Perturbation exceeds L_inf budget")
    for key in ("queries", "first_success_query"):
        _require(np.all(np.asarray(data[key]) == np.floor(data[key])), "Query counts must be integers")
    _require(np.all(data["first_success_query"] <= data["queries"]), "First success exceeds total queries")
    if metadata.get("query_budget") is not None and str(metadata.get("attack", "")).lower() not in ("transfer", "pgd", "transfer_pgd"):
        _require(np.all(data["queries"] <= int(metadata["query_budget"])), "Actual queries exceed configured budget")
    flip = float(np.mean(np.argmax(data["alpha_clean"], 1) != np.argmax(data["alpha_adv"], 1)))
    for source in (metadata, data):
        for key in ("router_asr", "top1_flip_rate"):
            if key in source:
                _require(np.allclose(source[key], flip, atol=1e-8, rtol=0), "router_asr must equal top1_flip_rate from clean top-1 routes")


def bootstrap_weights(n: int, replicates: int, seed: int = 2026) -> np.ndarray:
    """Multinomial paired bootstrap, resampling image IDs with replacement."""
    _require(n > 0 and replicates >= 0, "Invalid bootstrap dimensions")
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=replicates).astype(np.int32)


def _divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=np.asarray(b) > 0)


def weighted_prediction_metrics(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray | None = None, reference: np.ndarray | None = None, threshold: float = .5, chunk_size: int = 128) -> dict[str, np.ndarray]:
    """Exact empirical multilabel metrics for many bootstrap weight vectors.

    AP and ROC AUC handle tied scores exactly. No sklearn fit/call is required per
    bootstrap replicate. Returned vectors have one entry per weight vector.
    Classwise keys use stable integer indices (e.g., class_0/ap).
    """
    y, p = np.asarray(labels, dtype=np.float64), np.asarray(probs, dtype=np.float64)
    _require(y.shape == p.shape and y.ndim == 2 and len(y) > 0, "Prediction/label shape mismatch")
    _require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Invalid probabilities")
    _require(np.isin(y, [0, 1]).all(), "Labels must be binary")
    w = np.ones((1, len(y))) if weights is None else np.asarray(weights)
    if w.ndim == 1:
        w = w[None, :]
    _require(w.ndim == 2 and w.shape[1] == len(y) and np.all(w >= 0), "Invalid bootstrap weights")
    nrep, nclass = len(w), y.shape[1]
    result = {k: np.full(nrep, np.nan) for k in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER)}
    class_values = {k: np.full((nrep, nclass), np.nan) for k in ("f1", "ap", "auroc")}
    pred = p >= threshold
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean(1)
    shift = np.zeros(len(y)) if reference is None else np.abs(p - reference).mean(1)
    rank_cache = []
    for c in range(nclass):
        order = np.argsort(-p[:, c], kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(p[order, c]) != 0), len(y) - 1]
        rank_cache.append((order, ends))
    for start in range(0, nrep, chunk_size):
        end = min(start + chunk_size, nrep)
        wc = w[start:end].astype(np.float64, copy=False)
        mass = wc.sum(1)
        # einsum avoids costly multithreaded BLAS setup for these short matrices.
        pos = np.einsum("rn,nc->rc", wc, y)
        tp = np.einsum("rn,nc->rc", wc, y * pred)
        pred_pos = np.einsum("rn,nc->rc", wc, pred)
        f1 = _divide(2 * tp, pos + pred_pos)
        f1[pos == 0] = np.nan
        class_values["f1"][start:end] = f1
        result["bce"][start:end] = _divide(np.einsum("rn,n->r", wc, loss), mass)
        result["probability_shift"][start:end] = _divide(np.einsum("rn,n->r", wc, shift), mass)
        for c in range(nclass):
            order, ends = rank_cache[c]
            wy = wc[:, order] * y[order, c]
            wn = wc[:, order] * (1 - y[order, c])
            cp, cn = np.cumsum(wy, axis=1)[:, ends], np.cumsum(wn, axis=1)[:, ends]
            dp, dn = np.diff(cp, prepend=0, axis=1), np.diff(cn, prepend=0, axis=1)
            precision = np.divide(cp, cp + cn, out=np.zeros_like(cp), where=cp + cn > 0)
            class_values["ap"][start:end, c] = _divide(np.sum(precision * dp, axis=1), cp[:, -1])
            pairs = np.sum(dp * (cn[:, -1, None] - cn + .5 * dn), axis=1)
            class_values["auroc"][start:end, c] = _divide(pairs, cp[:, -1] * cn[:, -1])
    for key, ck in (("macro_f1", "f1"), ("map", "ap"), ("macro_auroc", "auroc")):
        result[key] = class_values[ck].mean(1)  # Deliberately propagate missing classes.
    for ck, values in class_values.items():
        for c in range(nclass):
            result[f"class_{c}/{ck}"] = values[:, c]
    return result


def prediction_metrics(labels: np.ndarray, probs: np.ndarray, reference: np.ndarray | None = None) -> dict[str, float]:
    return {key: float(val[0]) for key, val in weighted_prediction_metrics(labels, probs, reference=reference).items()}


def routing_vectors(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-image routing geometry, with margins measured in probability units."""
    clean, adv = np.asarray(data["alpha_clean"]), np.asarray(data["alpha_adv"])
    ec, cc = entropy_and_collapse(clean)
    ea, ca = entropy_and_collapse(adv)
    route_c, route_a = clean.argmax(1), adv.argmax(1)
    margin_c = np.sort(clean, axis=1)[:, -1] - np.sort(clean, axis=1)[:, -2]
    margin_a = np.sort(adv, axis=1)[:, -1] - np.sort(adv, axis=1)[:, -2]
    competitor = adv.copy()
    competitor[np.arange(len(adv)), route_c] = -np.inf
    clean_route_margin_adv = adv[np.arange(len(adv)), route_c] - competitor.max(1)
    flip = (route_c != route_a).astype(float)
    result = {
        "router_asr": flip, "top1_flip_rate": flip,
        "routing_l1_shift": np.abs(adv - clean).sum(1), "routing_l2_shift": np.linalg.norm(adv - clean, axis=1),
        "entropy_clean": ec, "entropy_adv": ea, "entropy_change": ea - ec,
        "collapse_clean": cc, "collapse_adv": ca, "collapse_change": ca - cc,
        "max_probability_clean": clean.max(1), "max_probability_adv": adv.max(1), "max_probability_change": adv.max(1) - clean.max(1),
        "top2_margin_clean": margin_c, "top2_margin_adv": margin_a, "top2_margin_change": margin_a - margin_c,
        # The final decision margin of the clean dominant route against its
        # strongest adversarial competitor. It is negative exactly when the
        # top-1 route flips, so this explicit name is retained in every table.
        "final_router_margin": clean_route_margin_adv,
        "clean_route_margin_adv": clean_route_margin_adv,
        "clean_route_margin_change": clean_route_margin_adv - margin_c,
        "linf": np.asarray(data["linf"]), "l2": np.asarray(data["l2"]), "queries": np.asarray(data["queries"]),
        "ever_success_rate": (np.asarray(data["first_success_query"]) >= 0).astype(float),
    }
    if "surrogate_route_clean" in data:
        sr = np.asarray(data["surrogate_route_clean"])
        result["surrogate_clean_routing_agreement"] = (sr == route_c).astype(float)
        if "surrogate_route_adv" in data:
            result["surrogate_attack_success_rate"] = (sr != data["surrogate_route_adv"]).astype(float)
    elif "surrogate_success" in data:
        result["surrogate_attack_success_rate"] = np.asarray(data["surrogate_success"], dtype=float)
    return result


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _divide(np.einsum("rn,n->r", weights, values), weights.sum(1))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact numpy-median convention for integer image multiplicities."""
    order = np.argsort(values, kind="stable")
    vals, w = np.asarray(values)[order], np.asarray(weights)[:, order]
    cs, total = np.cumsum(w, axis=1), w.sum(1)
    a, b = np.maximum((total - 1) // 2, 0), total // 2
    lo = (cs <= a[:, None]).sum(1).clip(max=len(vals) - 1)
    hi = (cs <= b[:, None]).sum(1).clip(max=len(vals) - 1)
    return np.where(total > 0, (vals[lo] + vals[hi]) / 2, np.nan)


def run_metrics(data: Mapping[str, np.ndarray], weights: np.ndarray | None = None, include_experts: bool = True) -> dict[str, np.ndarray]:
    """All scalar statistics and paired attribution contrasts from one raw run."""
    n = len(data["image_ids"])
    w = np.ones((1, n), dtype=int) if weights is None else np.asarray(weights)
    result = {}
    for condition in CONDITIONS:
        m = weighted_prediction_metrics(data["labels"], data["probs_" + condition], w, reference=data["probs_CC"])
        result.update({condition + "/" + key: val for key, val in m.items()})
    for metric in HIGHER_IS_BETTER:
        for name, first, second in (("delta_full", "CC", "AA"), ("delta_route_only", "CC", "CA"), ("delta_feature_only", "CC", "AC"), ("routing_recovery", "AC", "AA")):
            result[f"attribution/{name}/{metric}"] = result[first + "/" + metric] - result[second + "/" + metric]
    # BCE is lower-is-better, so label its contrasts as increases.  Do not call
    # these quantities deltas defined for higher-is-better performance metrics.
    for name, first, second in (("full", "CC", "AA"), ("route_only", "CC", "CA"), ("feature_only", "CC", "AC")):
        result[f"attribution/bce_increase_{name}"] = result[second + "/bce"] - result[first + "/bce"]
    result["attribution/bce_reduction_routing_recovery"] = result["AA/bce"] - result["AC/bce"]
    if include_experts:
        for expert in range(data["expert_probs_clean"].shape[1]):
            for state in ("clean", "adv"):
                m = weighted_prediction_metrics(data["labels"], data["expert_probs_" + state][:, expert], w, reference=data["expert_probs_clean"][:, expert])
                result.update({f"expert_{expert}/{state}/" + key: val for key, val in m.items() if not key.startswith("class_")})
            for metric in HIGHER_IS_BETTER:
                result[f"expert_{expert}/degradation/{metric}"] = result[f"expert_{expert}/clean/{metric}"] - result[f"expert_{expert}/adv/{metric}"]
            result[f"expert_{expert}/bce_increase"] = result[f"expert_{expert}/adv/bce"] - result[f"expert_{expert}/clean/bce"]
    vectors = routing_vectors(data)
    result.update({"routing/" + key: weighted_mean(val, w) for key, val in vectors.items()})
    result["routing/queries_median"] = weighted_median(np.asarray(data["queries"]), w)
    first = np.asarray(data["first_success_query"])
    success = first >= 0
    result["routing/first_success_query_median"] = weighted_median(first, w * success)
    if "surrogate_attack_success_rate" in vectors:
        success = vectors["surrogate_attack_success_rate"]
        result["routing/transfer_given_surrogate_success"] = _divide(np.einsum("rn,n->r", w, success * vectors["top1_flip_rate"]), np.einsum("rn,n->r", w, success))
    return result


@dataclass(frozen=True)
class Estimate:
    mean: float
    run_sd: float
    ci_low: float
    ci_high: float
    n_runs: int
    n_valid_runs: int
    n_bootstrap: int
    n_valid_bootstrap: int


def summarize_runs(points: np.ndarray, bootstraps: np.ndarray, confidence: float = .95) -> Estimate:
    """Mean across runs; sample SD; percentile CI of paired image bootstrap.

    Rows of bootstraps are runs and columns are the SAME image resamples. The CI
    conditions on fitted models and observed attack seeds; it does not bootstrap
    model training runs or pretend that run-image pairs are independent images.
    """
    p, b = np.asarray(points, dtype=float), np.asarray(bootstraps, dtype=float)
    _require(p.ndim == 1 and b.ndim == 2 and b.shape[0] == len(p), "Run/bootstrap shape mismatch")
    mean = float(p.mean())
    sd = float(p.std(ddof=1)) if len(p) > 1 and np.isfinite(p).all() else float("nan")
    means = b.mean(0)
    valid = means[np.isfinite(means)]
    ci = np.quantile(valid, [(1 - confidence) / 2, (1 + confidence) / 2]) if len(valid) else [np.nan, np.nan]
    return Estimate(mean, sd, float(ci[0]), float(ci[1]), len(p), int(np.isfinite(p).sum()), b.shape[1], len(valid))
