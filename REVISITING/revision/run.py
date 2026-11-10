#!/usr/bin/env python3
"""Create fixed cohorts and execute one auditable attack job at a time.

This module intentionally does not submit jobs or start tmux.  Each command is
restartable and writes one immutable raw artifact, which makes it safe to launch
independent seed/attack jobs on separate GPUs.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch

from .adapters import controlled_conditions, load_csa_model
from .attacks import hopskipjump_attack, square_attack
from .data import PixelDataset, expert_names_for, load_metadata, split_audit, stratified_indices
from .io import canonical_float, load_json, run_stem, save_json, save_npz_and_metadata, sha256_file, write_rows
from .metrics import validate_run
from .sparse_model import load_sparse_model
from .surrogates import AcquisitionData, acquire_surrogate_targets, surrogate_diagnostics, train_surrogate, transfer_pgd


from .paths import ROOT, output_dir, metadata_dir, csa_run

DEFAULT_OUTPUT = output_dir()
DEFAULT_METADATA = {dataset: metadata_dir(dataset) for dataset in ("bdd100k", "nuimages")}
DEFAULT_CSA = {dataset: csa_run(dataset) for dataset in ("bdd100k", "nuimages")}


def _torch_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this process. Use --device cpu or run on the GPU server.")
    return device


def _model_and_info(args: argparse.Namespace, device: torch.device):
    if args.model == "csa":
        run_dir = Path(args.run_dir or DEFAULT_CSA[args.dataset])
        return load_csa_model(args.dataset, str(run_dir), str(args.metadata_dir), str(device), args.checkpoint_name)
    if args.model == "sparse":
        if not args.sparse_checkpoint:
            raise ValueError("--sparse-checkpoint is required for --model sparse")
        return load_sparse_model(args.sparse_checkpoint, str(device))
    raise ValueError(args.model)


def _stable_seed(seed: int, image_id: str) -> int:
    digest = hashlib.blake2b(image_id.encode("utf-8"), digest_size=8).digest()
    return (int(seed) ^ int.from_bytes(digest, "little")) % (2**63 - 1)


def _semantic_context_buffers(dataset: str) -> dict[str, list[str]]:
    """Allocate dataset metadata contexts independently of model expert names.

    CSA's image experts happen to be named after these semantic metadata
    fields.  The sparse control instead names its routed units
    ``token_expert_*``.  Keeping the labels tied to the dataset makes the
    fixed cohort comparable across architectures and avoids manufacturing
    token-expert context columns that do not exist in the source metadata.
    """
    return {name: [] for name in expert_names_for(dataset)}


def _append_semantic_contexts(buffers: dict[str, list[str]], item_contexts: Mapping[str, str]) -> None:
    """Append one image's context labels, requiring the known dataset schema."""
    expected, observed = set(buffers), set(item_contexts)
    if observed != expected:
        raise AssertionError(f"Unexpected context fields: expected {sorted(expected)}, got {sorted(observed)}")
    for name in buffers:
        buffers[name].append(str(item_contexts[name]))


def _load_manifest(path: str | Path) -> dict:
    result = load_json(path)
    if not isinstance(result, dict) or "rows" not in result or "image_ids" not in result:
        raise ValueError(f"Invalid manifest: {path}")
    if len(result["rows"]) != len(result["image_ids"]):
        raise ValueError("Manifest rows and image IDs disagree")
    if len(set(result["image_ids"])) != len(result["image_ids"]):
        raise ValueError("Manifest contains duplicate image IDs")
    return result


def _components(model: torch.nn.Module, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        components = model.forward_components(pixels)
        conditions = controlled_conditions(model, components, components)
    if not torch.allclose(conditions["CC"], components["fused_logits"], atol=2e-6, rtol=1e-5):
        raise AssertionError("CC does not reproduce the original fused forward pass")
    return components


def _attack_target(model: torch.nn.Module, target_kind: str):
    def target(pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        values = model.attack_outputs(pixels, target_kind)
        return {"router_logits": values} if target_kind == "router" else {"fused_logits": values}
    return target


def _append_component(buffers: dict[str, list[np.ndarray]], clean: Mapping[str, torch.Tensor], adv: Mapping[str, torch.Tensor], model: torch.nn.Module) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        interventions = controlled_conditions(model, clean, adv)
    cc_error = float((interventions["CC"] - clean["fused_logits"]).abs().max().cpu())
    aa_error = float((interventions["AA"] - adv["fused_logits"]).abs().max().cpu())
    if max(cc_error, aa_error) > 2e-5:
        raise AssertionError(f"Controlled injection does not reproduce native forward: CC={cc_error}, AA={aa_error}")
    for name, value in (("alpha_clean", clean["alpha"]), ("alpha_adv", adv["alpha"]),
                        ("router_logits_clean", clean["router_logits"]), ("router_logits_adv", adv["router_logits"]),
                        ("expert_probs_clean", clean["expert_logits"].sigmoid()), ("expert_probs_adv", adv["expert_logits"].sigmoid()),
                        ("expert_logits_clean", clean["expert_logits"]), ("expert_logits_adv", adv["expert_logits"]),
                        ("features_clean", clean["features"]), ("features_adv", adv["features"]),
                        ("expert_features_clean", clean["expert_features"]), ("expert_features_adv", adv["expert_features"]),
                        ("probs_CC", interventions["CC"].sigmoid()), ("probs_CA", interventions["CA"].sigmoid()),
                        ("probs_AC", interventions["AC"].sigmoid()), ("probs_AA", interventions["AA"].sigmoid())):
        buffers.setdefault(name, []).append(value.detach().cpu().float().numpy())
    for key in ("token_alpha", "token_router_logits"):
        if key in clean:
            buffers.setdefault(key + "_clean", []).append(clean[key].detach().cpu().float().numpy())
            buffers.setdefault(key + "_adv", []).append(adv[key].detach().cpu().float().numpy())
    return {"cc_forward_max_abs_error": cc_error, "aa_forward_max_abs_error": aa_error}


def _finalize_arrays(buffers: Mapping[str, list[np.ndarray]], *, image_ids: list[str], labels: list[np.ndarray], contexts: Mapping[str, list[str]], queries: list[int], first: list[int], linf: list[float], l2: list[float], feature_dtype: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "image_ids": np.asarray(image_ids, dtype=str), "labels": np.stack(labels).astype(np.int8),
        "queries": np.asarray(queries, dtype=np.int32), "first_success_query": np.asarray(first, dtype=np.int32),
        "linf": np.asarray(linf, dtype=np.float32), "l2": np.asarray(l2, dtype=np.float32),
    }
    for key, chunks in buffers.items():
        value = np.concatenate(chunks, axis=0)
        arrays[key] = value.astype(np.float16 if feature_dtype == "float16" and ("features" in key or key.startswith("token_")) else np.float32)
    arrays["image_ids_clean"] = arrays["image_ids"].copy()
    arrays["image_ids_adv"] = arrays["image_ids"].copy()
    arrays["route_clean"] = arrays["alpha_clean"].argmax(1).astype(np.int16)
    arrays["route_adv"] = arrays["alpha_adv"].argmax(1).astype(np.int16)
    flip = float(np.mean(arrays["route_clean"] != arrays["route_adv"]))
    # Store both names intentionally. validate_run asserts their equality with
    # the per-image clean-reference flip event before any table is generated.
    arrays["router_asr"] = np.asarray(flip, dtype=np.float64)
    arrays["top1_flip_rate"] = np.asarray(flip, dtype=np.float64)
    arrays.update({"context_" + key: np.asarray(values, dtype=str) for key, values in contexts.items()})
    return arrays


def _per_image_rows(arrays: Mapping[str, np.ndarray], class_names: list[str], expert_names: list[str]) -> list[dict]:
    rows = []
    for i, image_id in enumerate(arrays["image_ids"]):
        row: dict[str, Any] = {"image_id": str(image_id), "queries": int(arrays["queries"][i]),
                               "first_success_query": int(arrays["first_success_query"][i]),
                               "linf": float(arrays["linf"][i]), "l2": float(arrays["l2"][i]),
                               "clean_route": expert_names[int(arrays["alpha_clean"][i].argmax())],
                               "adv_route": expert_names[int(arrays["alpha_adv"][i].argmax())]}
        for context in sorted(key for key in arrays if key.startswith("context_")):
            row[context] = str(arrays[context][i])
        for k, name in enumerate(expert_names):
            row["alpha_clean_" + name] = float(arrays["alpha_clean"][i, k])
            row["alpha_adv_" + name] = float(arrays["alpha_adv"][i, k])
        for c, name in enumerate(class_names):
            row["label_" + name] = int(arrays["labels"][i, c])
            for condition in ("CC", "CA", "AC", "AA"):
                row[f"prob_{condition}_{name}"] = float(arrays["probs_" + condition][i, c])
        rows.append(row)
    return rows


def cmd_audit_splits(args: argparse.Namespace) -> None:
    audit = split_audit(args.dataset, str(args.metadata_dir))
    audit["purpose"] = "Pre-run split audit. A historical checkpoint must not be used for final claims if this reports overlap."
    out = Path(args.output_dir) / "audits" / f"{args.dataset}_split_audit.json"
    save_json(out, audit)
    print(json.dumps(audit, indent=2))
    print(f"Wrote {out}")


def cmd_make_manifest(args: argparse.Namespace) -> None:
    device = _torch_device(args.device)
    model, info = _model_and_info(args, device)
    all_rows, bundle = load_metadata(args.dataset, str(args.metadata_dir), "test_fixed")
    if args.max_candidates:
        if not 1 <= args.max_candidates <= len(all_rows):
            raise ValueError("--max-candidates must be within the fixed test-set size")
        all_rows = all_rows[:args.max_candidates]
    dataset = PixelDataset(all_rows, info["image_size"], args.dataset, bundle)
    missing = dataset.validate_paths()
    if missing["n_missing_or_ambiguous"]:
        raise FileNotFoundError(json.dumps(missing, indent=2))
    routes = []
    for start in range(0, len(dataset), args.batch_size):
        pixels = torch.stack([dataset[i]["image"] for i in range(start, min(len(dataset), start + args.batch_size))]).to(device)
        with torch.no_grad():
            routes.extend(model.forward_components(pixels)["alpha"].argmax(1).cpu().tolist())
    routes_np = np.asarray(routes, dtype=np.int64)
    selected = stratified_indices(routes_np, args.n_images, args.seed, args.stratification, dataset.image_ids)
    rows = [copy.deepcopy(all_rows[int(i)]) for i in selected]
    ids = [dataset.image_ids[int(i)] for i in selected]
    route_counts = {info["expert_names"][int(route)]: int((routes_np[selected] == route).sum()) for route in np.unique(routes_np[selected])}
    subset_rows = []
    # Context strata come from dataset annotations, while route labels come
    # from the selected model.  They coincide for CSA but intentionally differ
    # for the token-routed sparse control.
    for field in expert_names_for(args.dataset):
        values = [dataset[i]["contexts"][field] for i in selected]
        for value in sorted(set(values)):
            idx = [j for j, item in enumerate(values) if item == value]
            record = {"context": field, "subset": value, "n_images": len(idx), "small_subset": len(idx) < args.minimum_subset}
            for c, name in enumerate(info["class_names"]):
                record["positive_" + name] = int(sum(rows[j][("traffic_sign_present" if c == 2 else name + "_present")] for j in idx))
            subset_rows.append(record)
    manifest = {"schema_version": 1, "dataset": args.dataset, "created_by": "revision.run make-manifest",
                "selection_model": {key: value for key, value in info.items() if key not in ("metadata", "config")},
                "n_images": len(rows), "seed": args.seed, "stratification": args.stratification,
                "reference_clean_route_counts": route_counts, "class_names": info["class_names"], "expert_names": info["expert_names"],
                "minimum_subset": args.minimum_subset, "image_ids": ids, "rows": rows,
                "smoke": bool(args.max_candidates)}
    path = Path(args.manifest or (Path(args.output_dir) / "manifests" / f"{args.dataset}_n{args.n_images}_seed{args.seed}.json"))
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists: {path}; use --overwrite only after checking it.")
    save_json(path, manifest)
    write_rows(path.with_name(path.stem + "_subset_counts.csv"), subset_rows)
    print(f"Wrote fixed {len(rows)}-image manifest: {path}")
    print(json.dumps({"route_counts": route_counts, "subset_count_file": str(path.with_name(path.stem + "_subset_counts.csv"))}, indent=2))


def _raw_metadata(args: argparse.Namespace, info: Mapping[str, Any], manifest: Mapping[str, Any], *, attack: str, target: str, epsilon: float, seed: int, query_budget: int | None, errors: Mapping[str, float], extra: Mapping[str, Any] | None = None) -> dict:
    payload = {"schema_version": 1, "dataset": args.dataset, "model": args.model, "architecture": info["architecture"],
               "routing_unit": info.get("routing_unit", "image"), "class_names": info["class_names"], "expert_names": info["expert_names"],
               "attack": attack, "target": target, "epsilon": float(epsilon), "epsilon_pixel_255": float(epsilon * 255),
               "seed": int(seed), "query_budget": query_budget, "bootstrap_resampling_unit": "image_id",
               "fused_prediction_definition": "learned head on alpha-weighted expert features; not an average of expert probabilities",
               "fusion_scheme": "feature_then_learned_head", "intervention_validated": True, "intervention_max_abs_error": dict(errors),
               "manifest_n_images": len(manifest["image_ids"]), "manifest_sha256": sha256_file(args.manifest),
               "checkpoint_path": info.get("checkpoint_path"), "checkpoint_sha256": info.get("checkpoint_sha256"),
               "preprocessing": info.get("perturbation_space"), "device": str(args.device), "python": sys.version,
               "torch": torch.__version__, "smoke": bool(args.smoke or manifest.get("smoke", False))}
    if extra:
        payload.update(extra)
    return payload


def _evaluate_attack(args: argparse.Namespace, model: torch.nn.Module, info: Mapping[str, Any], manifest: Mapping[str, Any], attack: str, target_kind: str, epsilon: float, seed: int, query_budget: int, *, transfer_surrogate=None, transfer_extra: Mapping[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict, dict[int, tuple[dict[str, np.ndarray], dict]]]:
    device = _torch_device(args.device)
    _, bundle = load_metadata(args.dataset, str(args.metadata_dir), split=None)
    dataset = PixelDataset(manifest["rows"], info["image_size"], args.dataset, bundle)
    if dataset.image_ids != manifest["image_ids"]:
        raise AssertionError("Manifest row ordering changed")
    buffers: dict[str, list[np.ndarray]] = {}
    identifiers: list[str] = []
    labels: list[np.ndarray] = []
    # These are dataset attributes (for example BDD100K weather/scene/time),
    # rather than routing-unit names.  This distinction matters for the custom
    # sparse token-MoE control, whose experts are called token_expert_0, etc.
    contexts = _semantic_context_buffers(args.dataset)
    queries: list[int] = []; first: list[int] = []; linf: list[float] = []; l2: list[float] = []
    surrogate_route_clean: list[int] = []; surrogate_route_adv: list[int] = []; surrogate_success: list[bool] = []
    errors = {"cc_forward_max_abs_error": 0.0, "aa_forward_max_abs_error": 0.0}
    snapshot_buffers: dict[int, dict[str, list[np.ndarray]]] = {q: {} for q in args.query_checkpoints} if attack != "transfer_pgd" else {}
    snapshot_quer: dict[int, list[int]] = {q: [] for q in snapshot_buffers}; snapshot_first: dict[int, list[int]] = {q: [] for q in snapshot_buffers}
    snapshot_linf: dict[int, list[float]] = {q: [] for q in snapshot_buffers}; snapshot_l2: dict[int, list[float]] = {q: [] for q in snapshot_buffers}
    with torch.no_grad():
        pass
    target_fn = _attack_target(model, target_kind)
    for index in range(len(dataset)):
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(device, dtype=torch.float32)
        clean = _components(model, image)
        image_seed = _stable_seed(seed, item["image_id"])
        if attack == "square":
            result = square_attack(target_fn, image, epsilon=epsilon, query_budget=query_budget, target_kind=target_kind,
                                   seed=image_seed, checkpoints=tuple(args.query_checkpoints), p_init=args.square_p_init,
                                   schedule_queries=max(args.query_checkpoints), stop_on_success=False)
        elif attack == "hsj":
            result = hopskipjump_attack(target_fn, image, epsilon=epsilon, query_budget=query_budget, target_kind=target_kind,
                                        seed=image_seed, checkpoints=tuple(args.query_checkpoints), steps=args.hsj_steps,
                                        initial_gradient_evals=args.hsj_initial_gradient_evals,
                                        max_gradient_evals=args.hsj_max_gradient_evals, gradient_batch_size=args.hsj_gradient_batch_size,
                                        max_initialization=args.hsj_max_initialization)
        elif attack == "transfer_pgd":
            if transfer_surrogate is None:
                raise ValueError("transfer attack requires a trained surrogate")
            result = transfer_pgd(transfer_surrogate, image, epsilon=epsilon, steps=args.pgd_steps, restarts=args.pgd_restarts,
                                  seed=image_seed, target_kind=target_kind)
            surrogate_route_clean.append(int(result.metadata.get("surrogate_clean_route", -1)))
            surrogate_route_adv.append(int(result.metadata.get("surrogate_adv_route", -1)))
            surrogate_success.append(bool(result.metadata.get("surrogate_success", False)))
        else:
            raise ValueError(attack)
        adv = _components(model, result.x_adv.to(device))
        local_errors = _append_component(buffers, clean, adv, model)
        for key in errors:
            errors[key] = max(errors[key], local_errors[key])
        identifiers.append(item["image_id"]); labels.append(item["labels"].numpy())
        _append_semantic_contexts(contexts, item["contexts"])
        delta = (result.x_adv.cpu() - image.detach().cpu()).float()
        queries.append(int(result.queries)); first.append(-1 if result.first_success_query is None else int(result.first_success_query))
        linf.append(float(delta.abs().max())); l2.append(float(delta.flatten(1).norm(dim=1)[0]))
        if attack != "transfer_pgd":
            for q in args.query_checkpoints:
                state = result.snapshots[q]
                qadv = _components(model, state["x_adv"].to(device))
                _append_component(snapshot_buffers[q], clean, qadv, model)
                qdelta = (state["x_adv"] - image.detach().cpu()).float()
                snapshot_quer[q].append(int(state["queries"])); snapshot_first[q].append(-1 if state["first_success_query"] is None else int(state["first_success_query"]))
                snapshot_linf[q].append(float(qdelta.abs().max())); snapshot_l2[q].append(float(qdelta.flatten(1).norm(dim=1)[0]))
        if (index + 1) % args.progress_every == 0 or index + 1 == len(dataset):
            print(f"[{attack}/{target_kind}/seed={seed}] {index + 1}/{len(dataset)} images", flush=True)
    arrays = _finalize_arrays(buffers, image_ids=identifiers, labels=labels, contexts=contexts, queries=queries, first=first, linf=linf, l2=l2, feature_dtype=args.feature_dtype)
    if transfer_surrogate is not None and target_kind == "router":
        # These fields are calculated exclusively on the surrogate while PGD is
        # generated; they add no target-model calls.
        arrays["surrogate_route_clean"] = np.asarray(surrogate_route_clean, dtype=np.int16)
        arrays["surrogate_route_adv"] = np.asarray(surrogate_route_adv, dtype=np.int16)
        arrays["surrogate_success"] = np.asarray(surrogate_success, dtype=bool)
    validate_run(arrays, {"epsilon": epsilon, "attack": attack, "query_budget": None if attack == "transfer_pgd" else query_budget,
                          "fusion_scheme": "feature_then_learned_head", "intervention_validated": True})
    snapshots: dict[int, tuple[dict[str, np.ndarray], dict]] = {}
    for q, snap in snapshot_buffers.items():
        snap_arrays = _finalize_arrays(snap, image_ids=identifiers, labels=labels, contexts=contexts, queries=snapshot_quer[q], first=snapshot_first[q], linf=snapshot_linf[q], l2=snapshot_l2[q], feature_dtype="float32")
        validate_run(snap_arrays, {"epsilon": epsilon, "attack": attack, "query_budget": query_budget,
                                   "fusion_scheme": "feature_then_learned_head", "intervention_validated": True})
        snapshots[q] = (snap_arrays, {"requested_query_budget": q, "actual_queries_are_per_image": True})
    return arrays, errors, snapshots


def cmd_attack(args: argparse.Namespace) -> None:
    device = _torch_device(args.device)
    model, info = _model_and_info(args, device)
    manifest = _load_manifest(args.manifest)
    attack = "hsj" if args.attack == "hopskipjump" else args.attack
    if not args.query_checkpoints or any(q < 1 or q > args.query_budget for q in args.query_checkpoints):
        raise ValueError("Every --query-checkpoints value must be between 1 and --query-budget.")
    if sorted(set(args.query_checkpoints)) != list(args.query_checkpoints):
        raise ValueError("--query-checkpoints must be strictly increasing and unique.")
    arrays, errors, snapshots = _evaluate_attack(args, model, info, manifest, attack, args.target, args.epsilon, args.seed, args.query_budget)
    stem = run_stem(args.dataset, args.model, attack, args.target, args.epsilon, args.seed)
    raw = Path(args.output_dir) / "raw_runs" / stem
    metadata = _raw_metadata(args, info, manifest, attack=attack, target=args.target, epsilon=args.epsilon, seed=args.seed,
                             query_budget=args.query_budget, errors=errors, extra={"query_checkpoints": args.query_checkpoints,
                             "attack_parameters": {"square_p_init": args.square_p_init, "hsj_steps": args.hsj_steps,
                             "hsj_initial_gradient_evals": args.hsj_initial_gradient_evals, "hsj_max_gradient_evals": args.hsj_max_gradient_evals,
                             "hsj_gradient_batch_size": args.hsj_gradient_batch_size, "hsj_max_initialization": args.hsj_max_initialization}})
    npz, _ = save_npz_and_metadata(raw.with_suffix(".npz"), arrays, metadata, overwrite=args.overwrite)
    write_rows(raw.with_suffix(".csv"), _per_image_rows(arrays, info["class_names"], info["expert_names"]))
    for q, (snap_arrays, extra) in snapshots.items():
        spath = Path(args.output_dir) / "query_budgets" / (stem + f"__Q-{q}.npz")
        snap_meta = dict(metadata, **extra, snapshot=True)
        save_npz_and_metadata(spath, snap_arrays, snap_meta, overwrite=args.overwrite)
    print(f"Wrote {npz}")


def _acquisition_bank(args: argparse.Namespace, model, info: Mapping[str, Any], manifest: Mapping[str, Any], device: torch.device) -> AcquisitionData:
    bank_path = Path(args.bank_path or (Path(args.output_dir) / "surrogate_banks" / f"{args.dataset}_{args.model}_n{args.bank_max}_seed{args.bank_seed}.pt"))
    bank_path.parent.mkdir(parents=True, exist_ok=True)

    # Several GPU workers can reach their first transfer job at once.  The bank
    # is deliberately shared, so serialise its initial target-model acquisition:
    # without this lock, each worker could spend the 2,000 target queries and
    # concurrently corrupt the same .pt file.  The lock is held only while the
    # bank is absent; independent surrogate training remains parallel.
    lock_path = bank_path.with_suffix(bank_path.suffix + ".lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if bank_path.exists():
            saved = torch.load(bank_path, map_location="cpu", weights_only=False)
            if (saved["dataset"] != args.dataset or saved["model"] != args.model or
                    saved["checkpoint_sha256"] != info.get("checkpoint_sha256") or
                    saved.get("bank_seed") != args.bank_seed):
                raise ValueError("Existing surrogate bank was built for a different model, dataset, or bank seed.")
            return AcquisitionData(saved["images"], saved["router_labels"], saved["fused_probabilities"], int(saved["target_queries"]),
                                   int(saved["num_routes"]), int(saved["shared_bank_queries"]), list(saved["image_ids"]))
        all_train, bundle = load_metadata(args.dataset, str(args.metadata_dir), "train")
        test_ids = set(manifest["image_ids"])
        rows = [row for row in all_train if row["image_id"] not in test_ids]
        rng = np.random.default_rng(args.bank_seed)
        choose = sorted(rng.choice(len(rows), size=args.bank_max, replace=False).tolist(), key=lambda i: rows[i]["image_id"])
        chosen = [rows[i] for i in choose]
        ds = PixelDataset(chosen, info["image_size"], args.dataset, bundle)
        images = torch.stack([ds[i]["image"] for i in range(len(ds))])

        def target(pixels: torch.Tensor):
            with torch.no_grad():
                c = model.forward_components(pixels)
            return {"router_logits": c["router_logits"], "fused_logits": c["fused_logits"]}

        acquired = acquire_surrogate_targets(target, images, batch_size=args.batch_size, device=device, image_ids=ds.image_ids)
        payload = {"dataset": args.dataset, "model": args.model, "checkpoint_sha256": info.get("checkpoint_sha256"),
                   "bank_seed": args.bank_seed, "images": acquired.images.to(torch.float16),
                   "router_labels": acquired.router_labels, "fused_probabilities": acquired.fused_probabilities,
                   "target_queries": acquired.target_queries, "num_routes": acquired.num_routes,
                   "shared_bank_queries": acquired.shared_bank_queries, "image_ids": acquired.image_ids}
        temporary = bank_path.with_name(f".{bank_path.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, bank_path)
        print(f"Acquired and cached {len(acquired.images)} surrogate labels: {bank_path}")
        return acquired


def cmd_transfer(args: argparse.Namespace) -> None:
    device = _torch_device(args.device)
    model, info = _model_and_info(args, device)
    manifest = _load_manifest(args.manifest)
    bank = _acquisition_bank(args, model, info, manifest, device)
    if args.acquisition_n > len(bank.images):
        raise ValueError("--acquisition-n exceeds cached acquisition bank")
    train_data = bank.subset(args.acquisition_n)
    surrogate, diagnostics = train_surrogate(train_data, target_kind=args.target, seed=args.seed, epochs=args.surrogate_epochs,
                                             batch_size=args.batch_size, lr=args.surrogate_lr, device=str(device), width=args.surrogate_width)
    ckpt_path = Path(args.output_dir) / "surrogates" / run_stem(args.dataset, args.model, "transfer_pgd", args.target, args.epsilon, args.seed, args.acquisition_n)
    ckpt_path = ckpt_path.with_suffix(".pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite trained surrogate: {ckpt_path}")
    torch.save({"model": surrogate.state_dict(), "config": surrogate.config, "diagnostics": diagnostics}, ckpt_path)
    arrays, errors, _ = _evaluate_attack(args, model, info, manifest, "transfer_pgd", args.target, args.epsilon, args.seed, 0, transfer_surrogate=surrogate,
                                          transfer_extra=diagnostics)
    # Per-image surrogate diagnostics use the persisted adversarial components
    # only for target assessment; no target evaluations are used in generation.
    metadata = _raw_metadata(args, info, manifest, attack="transfer_pgd", target=args.target, epsilon=args.epsilon, seed=args.seed,
                             query_budget=None, errors=errors, extra={"surrogate_acquisition_n": args.acquisition_n,
                             "surrogate_shared_acquisition_queries": bank.shared_bank_queries, "surrogate_nominal_acquisition_queries": args.acquisition_n,
                             "surrogate_checkpoint": str(ckpt_path), "surrogate_diagnostics": diagnostics,
                             "pgd_steps": args.pgd_steps, "pgd_restarts": args.pgd_restarts, "target_queries_during_generation": 0})
    stem = run_stem(args.dataset, args.model, "transfer_pgd", args.target, args.epsilon, args.seed, args.acquisition_n)
    raw = Path(args.output_dir) / "raw_runs" / stem
    npz, _ = save_npz_and_metadata(raw.with_suffix(".npz"), arrays, metadata, overwrite=args.overwrite)
    write_rows(raw.with_suffix(".csv"), _per_image_rows(arrays, info["class_names"], info["expert_names"]))
    print(f"Wrote {npz}")


def cmd_smoke(args: argparse.Namespace) -> None:
    """A one-image component/injection check. It does not create a result claim."""
    device = _torch_device(args.device)
    model, info = _model_and_info(args, device)
    rows, bundle = load_metadata(args.dataset, str(args.metadata_dir), "test_fixed")
    ds = PixelDataset(rows[:1], info["image_size"], args.dataset, bundle)
    item = ds[0]; x = item["image"].unsqueeze(0).to(device)
    c = _components(model, x)
    target = _attack_target(model, "router")
    r = square_attack(target, x, epsilon=min(args.epsilon, 8/255), query_budget=max(2, args.smoke_queries), seed=args.seed,
                      checkpoints=(max(2, args.smoke_queries),), target_kind="router")
    a = _components(model, r.x_adv.to(device))
    with torch.no_grad(): swaps = controlled_conditions(model, c, a)
    print(json.dumps({"smoke": True, "image_id": item["image_id"], "architecture": info["architecture"],
                      "clean_shape": {k: list(v.shape) for k, v in c.items()}, "queries": r.queries,
                      "linf": float((r.x_adv.cpu() - x.cpu()).abs().max()),
                      "CC_matches_native": bool(torch.allclose(swaps["CC"], c["fused_logits"], atol=2e-6, rtol=1e-5)),
                      "AA_matches_native": bool(torch.allclose(swaps["AA"], a["fused_logits"], atol=2e-6, rtol=1e-5))}, indent=2))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("bdd100k", "nuimages"), required=True)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--model", choices=("csa", "sparse"), default="csa")
    parser.add_argument("--run-dir", type=Path, default=None, help="CSA model directory containing config.json and checkpoints/.")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--sparse-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-splits"); add_common(audit); audit.set_defaults(func=cmd_audit_splits)
    manifest = sub.add_parser("make-manifest"); add_common(manifest)
    manifest.add_argument("--n-images", type=int, default=1000); manifest.add_argument("--seed", type=int, default=20261029)
    manifest.add_argument("--stratification", choices=("proportional", "balanced"), default="proportional")
    manifest.add_argument("--batch-size", type=int, default=32); manifest.add_argument("--minimum-subset", type=int, default=30)
    manifest.add_argument("--max-candidates", type=int, default=0, help="Smoke test only: scan only this many fixed-test images; never use its output for results.")
    manifest.add_argument("--manifest", type=Path, default=None); manifest.add_argument("--overwrite", action="store_true"); manifest.set_defaults(func=cmd_make_manifest)
    attack = sub.add_parser("attack"); add_common(attack)
    attack.add_argument("--manifest", required=True, type=Path); attack.add_argument("--attack", choices=("square", "hopskipjump"), required=True)
    attack.add_argument("--target", choices=("router", "fused"), required=True); attack.add_argument("--epsilon", type=float, default=8/255)
    attack.add_argument("--query-budget", type=int, default=1000); attack.add_argument("--query-checkpoints", type=int, nargs="+", default=(100, 250, 500, 1000))
    attack.add_argument("--seed", type=int, required=True); attack.add_argument("--square-p-init", type=float, default=.05)
    attack.add_argument("--hsj-steps", type=int, default=64); attack.add_argument("--hsj-initial-gradient-evals", type=int, default=100)
    attack.add_argument("--hsj-max-gradient-evals", type=int, default=10000); attack.add_argument("--hsj-gradient-batch-size", type=int, default=16)
    attack.add_argument("--hsj-max-initialization", type=int, default=1000); attack.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    attack.add_argument("--progress-every", type=int, default=10); attack.add_argument("--overwrite", action="store_true"); attack.add_argument("--smoke", action="store_true")
    attack.set_defaults(func=cmd_attack)
    transfer = sub.add_parser("transfer"); add_common(transfer)
    transfer.add_argument("--manifest", required=True, type=Path); transfer.add_argument("--target", choices=("router", "fused"), required=True)
    transfer.add_argument("--epsilon", type=float, default=8/255); transfer.add_argument("--seed", type=int, required=True)
    transfer.add_argument("--acquisition-n", type=int, required=True); transfer.add_argument("--bank-max", type=int, default=2000); transfer.add_argument("--bank-seed", type=int, default=20261029); transfer.add_argument("--bank-path", type=Path, default=None)
    transfer.add_argument("--batch-size", type=int, default=32); transfer.add_argument("--surrogate-epochs", type=int, default=20); transfer.add_argument("--surrogate-lr", type=float, default=1e-3); transfer.add_argument("--surrogate-width", type=int, default=32)
    transfer.add_argument("--pgd-steps", type=int, default=100); transfer.add_argument("--pgd-restarts", type=int, default=3); transfer.add_argument("--query-checkpoints", type=int, nargs="+", default=())
    transfer.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16"); transfer.add_argument("--progress-every", type=int, default=10); transfer.add_argument("--overwrite", action="store_true"); transfer.add_argument("--smoke", action="store_true")
    transfer.set_defaults(func=cmd_transfer)
    smoke = sub.add_parser("smoke"); add_common(smoke); smoke.add_argument("--epsilon", type=float, default=8/255); smoke.add_argument("--seed", type=int, default=7); smoke.add_argument("--smoke-queries", type=int, default=4); smoke.set_defaults(func=cmd_smoke)
    args = parser.parse_args()
    if args.metadata_dir is None:
        args.metadata_dir = DEFAULT_METADATA[args.dataset]
    args.metadata_dir = Path(args.metadata_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    if not args.metadata_dir.exists():
        parser.error(f"Missing metadata directory: {args.metadata_dir}")
    args.func(args)


if __name__ == "__main__":
    main()
