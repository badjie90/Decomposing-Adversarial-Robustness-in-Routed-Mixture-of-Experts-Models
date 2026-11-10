import json
from pathlib import Path

import numpy as np
import torch

from revision.adapters import controlled_conditions
from revision.attacks import hopskipjump_attack, square_attack
from revision.data import expert_names_for
from revision.io import save_npz_and_metadata
from revision.jobs import expected_path, job_artifact_paths, job_is_complete, quarantine_incomplete_artifacts
from revision.metrics import entropy_and_collapse, run_metrics, validate_run
from revision.report import (_REPORT_REQUIRED_ARRAYS, _load, _matched_target_contrasts,
                             _plot_matched_target_contrasts, _plot_surrogate_scaling,
                             _summarize, _surrogate_scaling_rows)
from revision.run import _append_semantic_contexts, _semantic_context_buffers
from revision.sparse_model import SparseVisionMoE


def _raw(n=8):
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, (n, 3), dtype=np.int8)
    # Ensure every class has both states for strict AP/AUC tests.
    labels[:2] = [[0, 0, 0], [1, 1, 1]]
    ac = rng.dirichlet((1, 1, 1), n); aa = rng.dirichlet((1, 1, 1), n)
    ec = rng.random((n, 3, 3)); ea = rng.random((n, 3, 3))
    cc = rng.random((n, 3)); ca = rng.random((n, 3)); acp = rng.random((n, 3)); aap = rng.random((n, 3))
    return {"image_ids": np.array([f"id{i}" for i in range(n)]), "labels": labels, "alpha_clean": ac, "alpha_adv": aa,
            "expert_probs_clean": ec, "expert_probs_adv": ea, "probs_CC": cc, "probs_CA": ca, "probs_AC": acp, "probs_AA": aap,
            "linf": np.full(n, 8/255), "l2": np.full(n, .1), "queries": np.full(n, 10), "first_success_query": np.full(n, -1)}


def test_metric_invariants_and_attribution():
    raw = _raw(); validate_run(raw, {"epsilon": 8/255, "attack": "square", "query_budget": 10, "intervention_validated": True})
    values = run_metrics(raw)
    assert "attribution/routing_recovery/macro_f1" in values
    entropy, collapse = entropy_and_collapse(raw["alpha_clean"])
    assert np.all((entropy >= 0) & (entropy <= 1))
    assert np.all((collapse >= 1/3) & (collapse <= 1))


def test_sparse_cached_swaps_match_native_forward():
    model = SparseVisionMoE(image_size=32, patch_size=8, embed_dim=32, num_heads=4, num_experts=4, top_k=2, depth=1, dropout=0).eval()
    x = torch.rand(2, 3, 32, 32); clean = model.forward_components(x); adv = model.forward_components(torch.rand(2, 3, 32, 32))
    conditions = controlled_conditions(model, clean, adv)
    assert torch.allclose(conditions["CC"], clean["fused_logits"])
    assert torch.allclose(conditions["AA"], adv["fused_logits"])
    assert torch.allclose(model.predict_sparse(x), clean["fused_logits"], atol=1e-6)


def test_sparse_runs_retain_dataset_contexts_not_token_expert_names():
    """CSA-selected metadata stays meaningful for the sparse control."""
    model = SparseVisionMoE(image_size=32, patch_size=8, embed_dim=32, num_heads=4,
                            num_experts=4, top_k=2, depth=1, dropout=0).eval()
    contexts = _semantic_context_buffers("bdd100k")
    assert set(contexts) == set(expert_names_for("bdd100k"))
    assert not set(contexts) & set(f"token_expert_{i}" for i in range(model.num_experts))
    _append_semantic_contexts(contexts, {"weather": "clear", "scene": "city", "time": "daytime"})
    assert contexts == {"weather": ["clear"], "scene": ["city"], "time": ["daytime"]}
    nu_contexts = _semantic_context_buffers("nuimages")
    _append_semantic_contexts(nu_contexts, {"location": "urban", "illumination": "day", "motion": "moving"})
    assert nu_contexts == {"location": ["urban"], "illumination": ["day"], "motion": ["moving"]}


def _toy_target(x):
    first = x.mean((1, 2, 3))
    return {"router_logits": torch.stack((first, -first, torch.zeros_like(first)), 1),
            "fused_logits": torch.stack((first, -first, first * .5), 1)}


def test_attacks_obey_budget_and_pixel_bound():
    x = torch.full((1, 3, 8, 8), .5)
    result = square_attack(_toy_target, x, epsilon=.1, query_budget=7, seed=5, checkpoints=(2, 7))
    assert result.queries == 7 and (result.x_adv - x).abs().max() <= .100001
    result = hopskipjump_attack(_toy_target, x, epsilon=.1, query_budget=7, seed=5, checkpoints=(2, 7), steps=2, initial_gradient_evals=2, max_gradient_evals=2, gradient_batch_size=1, max_initialization=2)
    assert result.queries <= 7 and (result.x_adv - x).abs().max() <= .100001


def test_hopskipjump_queue_path_uses_raw_artifact_name(tmp_path):
    row = {"dataset": "nuimages", "model": "csa", "kind": "attack", "attack": "hopskipjump",
           "target": "router", "seed": 42, "epsilon": 8 / 255, "acquisition_n": ""}
    assert "__hsj__" in expected_path(row, tmp_path).name


def test_worker_quarantines_partial_artifacts_before_retry(tmp_path):
    """A disk-full zero-byte NPZ must never be mistaken for a completed run."""
    row = {"dataset": "nuimages", "model": "csa", "kind": "attack", "attack": "square",
           "target": "router", "seed": "42", "epsilon": str(8 / 255), "acquisition_n": ""}
    checkpoints = (2, 4)
    raw = expected_path(row, tmp_path)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"")
    assert not job_is_complete(row, tmp_path, checkpoints)
    quarantined = quarantine_incomplete_artifacts(row, tmp_path, checkpoints)
    assert quarantined is not None and not raw.exists()
    assert (quarantined / "raw_runs" / raw.name).read_bytes() == b""

    expected_attack = "hsj" if row["attack"] == "hopskipjump" else row["attack"]
    for path in job_artifact_paths(row, tmp_path, checkpoints):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            np.savez_compressed(path, image_ids=np.array(["image-1"]))
        elif path.suffix == ".json":
            requested = int(path.stem.rsplit("__Q-", 1)[1]) if "__Q-" in path.stem else None
            metadata = {"dataset": row["dataset"], "model": row["model"], "attack": expected_attack,
                        "target": row["target"], "seed": int(row["seed"]), "epsilon": float(row["epsilon"])}
            if requested is not None:
                metadata["requested_query_budget"] = requested
            path.write_text(json.dumps(metadata) + "\n")
        else:
            path.write_text("image_id\nimage-1\n")
    assert job_is_complete(row, tmp_path, checkpoints)

    # A nonempty but malformed artifact must be rerun, too.  This covers the
    # cases that a simple exists()/size check would incorrectly skip.
    raw.with_suffix(".json").write_text("not-json\n")
    assert not job_is_complete(row, tmp_path, checkpoints)
    raw.with_suffix(".json").write_text(json.dumps({"dataset": row["dataset"], "model": row["model"],
        "attack": expected_attack, "target": row["target"], "seed": int(row["seed"]),
        "epsilon": float(row["epsilon"])}) + "\n")
    np.savez_compressed(raw, wrong=np.array([1]))
    assert not job_is_complete(row, tmp_path, checkpoints)
    np.savez_compressed(raw, image_ids=np.array(["image-1"]))
    raw.with_suffix(".csv").write_text("image_id\n")
    assert not job_is_complete(row, tmp_path, checkpoints)
    raw.with_suffix(".csv").write_text("image_id\nimage-1\n")
    snapshot = tmp_path / "query_budgets" / f"{raw.stem}__Q-2.json"
    snapshot.write_text(json.dumps({"dataset": row["dataset"], "model": row["model"], "attack": expected_attack,
        "target": row["target"], "seed": int(row["seed"]), "epsilon": float(row["epsilon"]),
        "requested_query_budget": 4}) + "\n")
    assert not job_is_complete(row, tmp_path, checkpoints)


def test_report_matched_controls_and_transfer_scaling(tmp_path):
    """The dedicated tables retain pairing and acquisition-cost metadata."""
    def write(name, data, **meta):
        base = {"dataset": "toy", "model": "toy_moe", "architecture": "toy", "epsilon": 8 / 255,
                "intervention_validated": True, "fusion_scheme": "feature_then_learned_head"}
        base.update(meta)
        path = tmp_path / (name + ".npz")
        save_npz_and_metadata(path, data, base)
        return path

    paths = []
    for seed in (1, 2):
        paths.append(write(f"router-square-{seed}", _raw(), attack="square", target="router", seed=seed, query_budget=10))
        paths.append(write(f"fused-square-{seed}", _raw(), attack="square", target="fused", seed=seed, query_budget=10))
    contrasts = _matched_target_contrasts(paths, replicates=10, seed=9)
    assert len(contrasts) == 9
    assert all(row["n_runs"] == 2 and row["matched_seeds"] == "1;2" for row in contrasts)

    transfer = []
    for seed in (1, 2):
        data = _raw()
        data["surrogate_route_clean"] = data["alpha_clean"].argmax(1).astype(np.int16)
        data["surrogate_route_adv"] = data["alpha_adv"].argmax(1).astype(np.int16)
        transfer.append(write(f"transfer-{seed}", data, attack="transfer_pgd", target="router", seed=seed,
                              query_budget=None, surrogate_acquisition_n=50,
                              surrogate_nominal_acquisition_queries=50,
                              surrogate_shared_acquisition_queries=2000,
                              target_queries_during_generation=0))
    summary, _ = _summarize(transfer, replicates=10, seed=9)
    scaling = _surrogate_scaling_rows(summary, transfer)
    assert len(scaling) == 1
    assert scaling[0]["surrogate_nominal_acquisition_queries"] == 50
    assert scaling[0]["n_independent_surrogate_runs"] == 2
    _plot_matched_target_contrasts(contrasts, tmp_path)
    _plot_surrogate_scaling(scaling, tmp_path)
    assert list(tmp_path.glob("matched_target_contrast_*.png"))
    assert list(tmp_path.glob("surrogate_scaling_*.png"))


def test_report_loads_metrics_not_cached_feature_tensors(tmp_path):
    """A report must not decompress unused model features from each NPZ."""
    data = _raw()
    # These mirror the large cached tensors carried by sparse-model snapshots.
    data["features_clean"] = np.ones((8, 256), dtype=np.float32)
    data["expert_features_adv"] = np.ones((8, 3, 256), dtype=np.float32)
    path = tmp_path / "run.npz"
    metadata = {"dataset": "toy", "model": "toy_moe", "architecture": "toy", "attack": "square",
                "target": "router", "epsilon": 8 / 255, "seed": 1, "query_budget": 10,
                "intervention_validated": True, "fusion_scheme": "feature_then_learned_head"}
    save_npz_and_metadata(path, data, metadata)
    loaded, _ = _load(path)
    assert _REPORT_REQUIRED_ARRAYS <= loaded.keys()
    assert "features_clean" not in loaded
    assert "expert_features_adv" not in loaded
    assert "routing/router_asr" in run_metrics(loaded)
