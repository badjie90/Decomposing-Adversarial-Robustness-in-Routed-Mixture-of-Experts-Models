"""Independent surrogate acquisition, training and zero-target-query generation.

Router imitation uses target hard routing labels; fused imitation uses target
sigmoid probabilities. These are separate threat models. No ground-truth labels
are used. Separate models must be trained for every target model, target kind,
N and training seed. A shared acquisition bank is allowed, but its actual target
cost and each N-prefix's nominal acquisition cost are both recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .attacks import AttackResult, CountedTarget, _decision, _objective, _success, _validate, project_linf


@dataclass
class AcquisitionData:
    images: torch.Tensor
    router_labels: torch.Tensor
    fused_probabilities: torch.Tensor
    target_queries: int
    num_routes: int
    shared_bank_queries: int | None = None
    image_ids: list[str] | None = None

    def subset(self, n):
        if not 1 <= n <= len(self.images):
            raise ValueError("N outside acquisition bank")
        return AcquisitionData(self.images[:n], self.router_labels[:n], self.fused_probabilities[:n],
                               n, self.num_routes, self.shared_bank_queries or self.target_queries,
                               self.image_ids[:n] if self.image_ids is not None else None)


@torch.no_grad()
def acquire_surrogate_targets(target, images, batch_size=32, device=None, image_ids=None):
    """Query a preselected, evaluation-disjoint clean training bank exactly once.

    Images must be unnormalized pixels. The caller fixes/saves image_ids and
    prevents train/evaluation overlap. One joint forward provides router labels
    and soft fused outputs; that is N image queries, not 2N. The two response
    channels are both requested explicitly and must be declared in the paper.
    """
    if images.ndim != 4 or len(images) < 1 or batch_size < 1:
        raise ValueError("nonempty [N,C,H,W] images and positive batch_size required")
    if image_ids is not None and (len(image_ids) != len(images) or len(set(image_ids)) != len(images)):
        raise ValueError("image_ids must uniquely identify each training image")
    if not bool(torch.isfinite(images).all()) or float(images.min()) < 0 or float(images.max()) > 1:
        raise ValueError("acquisition images must be finite [0,1] pixels")
    counter = CountedTarget(target, len(images))
    routes, probabilities, num_routes = [], [], None
    for start in range(0, len(images), batch_size):
        batch = images[start:start+batch_size]
        if device is not None:
            batch = batch.to(device)
        outputs = counter(batch, "surrogate_acquisition")
        routing = outputs.get("router_logits", outputs.get("alpha"))
        if routing is None or "fused_logits" not in outputs:
            raise KeyError("joint acquisition needs routing outputs and fused_logits")
        num_routes = routing.shape[1]
        routes.append(routing.argmax(1).detach().cpu())
        probabilities.append(outputs["fused_logits"].sigmoid().detach().cpu())
    return AcquisitionData(images.detach().cpu(), torch.cat(routes), torch.cat(probabilities),
                           counter.queries, num_routes, counter.queries,
                           list(image_ids) if image_ids is not None else None)


class IndependentSurrogate(nn.Module):
    """A small independently initialized CNN, sharing no target model parameters.

    Only one task head is optimized in each training run. Adaptive pooling makes
    image size independent of checkpoint shape. GroupNorm avoids batch-statistic
    dependence when attacking individual images.
    """
    def __init__(self, num_routes, num_classes, width=32, in_channels=3):
        super().__init__()
        if min(num_routes, num_classes, width, in_channels) < 1:
            raise ValueError("model dimensions must be positive")
        self.config = dict(num_routes=num_routes, num_classes=num_classes,
                           width=width, in_channels=in_channels)
        layers, channels = [], in_channels
        for out in (width, 2*width, 4*width):
            layers.extend([nn.Conv2d(channels, out, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(math.gcd(out, 8), out), nn.SiLU(),
                           nn.Conv2d(out, out, 3, padding=1, bias=False),
                           nn.GroupNorm(math.gcd(out, 8), out), nn.SiLU()])
            channels = out
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.router_head = nn.Linear(channels, num_routes)
        self.fused_head = nn.Linear(channels, num_classes)

    def forward(self, pixels):
        features = self.encoder((pixels-0.5)/0.5)
        logits = self.router_head(features)
        return {"router_logits": logits, "alpha": logits.softmax(-1),
                "fused_logits": self.fused_head(features)}


def train_surrogate(data: AcquisitionData, target_kind="router", seed=0, epochs=20,
                    batch_size=32, lr=1e-3, device="cpu", width=32, weight_decay=1e-4):
    """Train from cached targets. This function has no target callable.

    Training agreement is diagnostic only. Held-out agreement must be measured
    on the fixed evaluation bank with ``surrogate_diagnostics``. Each call makes
    a fresh independent model; use at least five seeds and all requested Ns.
    """
    if target_kind not in ("router", "fused") or min(epochs, batch_size) < 1 or lr <= 0:
        raise ValueError("invalid surrogate training configuration")
    if len(data.images) != len(data.router_labels) or len(data.images) != len(data.fused_probabilities):
        raise ValueError("inconsistent acquisition arrays")
    device = torch.device(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Restore caller's global RNG states on exit; all shuffling has a private RNG.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model = IndependentSurrogate(data.num_routes, data.fused_probabilities.shape[1], width,
                                     data.images.shape[1]).to(device)
        inactive = model.fused_head if target_kind == "router" else model.router_head
        inactive.requires_grad_(False)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=lr, weight_decay=weight_decay)
        rng = torch.Generator().manual_seed(seed)
        history = []
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(data.images), generator=rng)
            summed_loss = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start:start+batch_size]
                out = model(data.images[indices].to(device, dtype=torch.float32))
                if target_kind == "router":
                    loss = F.cross_entropy(out["router_logits"], data.router_labels[indices].to(device))
                else:
                    loss = F.binary_cross_entropy_with_logits(out["fused_logits"], data.fused_probabilities[indices].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                summed_loss += float(loss.detach())*len(indices)
            history.append({"epoch": epoch+1, "loss": summed_loss/len(order)})
        model.eval()
        model.target_kind = target_kind
        diagnostics = surrogate_diagnostics(model, data.images, data.router_labels,
                                             data.fused_probabilities, batch_size=batch_size)
    diagnostics = {f"training_{k}": v for k, v in diagnostics.items()}
    diagnostics.update({"target_kind": target_kind, "seed": seed, "N": len(data.images),
                        "epochs": epochs, "batch_size": batch_size, "lr": lr,
                        "weight_decay": weight_decay, "target_queries": data.target_queries,
                        "shared_bank_queries": data.shared_bank_queries, "history": history,
                        "model_config": model.config, "ground_truth_used": False,
                        "router_head_trained": target_kind == "router",
                        "fused_head_trained": target_kind == "fused"})
    return model, diagnostics


@torch.no_grad()
def surrogate_diagnostics(surrogate, images, target_routes=None, target_probabilities=None, batch_size=32):
    """Agreement against already cached target responses; zero new target calls.

    The inactive head is not reported. Router agreement for a fused-only surrogate
    would otherwise measure an untrained random head and invite misinterpretation.
    """
    if len(images) < 1:
        raise ValueError("diagnostics require images")
    device = next(surrogate.parameters()).device
    kind = getattr(surrogate, "target_kind", "router")
    route_correct, label_correct, vector_correct, soft_sum, count = 0, 0, 0, 0.0, 0
    surrogate.eval()
    for start in range(0, len(images), batch_size):
        out = surrogate(images[start:start+batch_size].to(device, dtype=torch.float32))
        n = len(out["router_logits"])
        if target_routes is not None and kind == "router":
            reference = target_routes[start:start+n].to(device)
            route_correct += int((out["router_logits"].argmax(1) == reference).sum())
        if target_probabilities is not None and kind == "fused":
            reference = target_probabilities[start:start+n].to(device)
            equal = (out["fused_logits"] >= 0) == (reference >= 0.5)
            label_correct += int(equal.sum())
            vector_correct += int(equal.all(1).sum())
            soft_sum += float((out["fused_logits"].sigmoid()-reference).abs().sum())
            count += reference.numel()
    result = {"N_evaluated": len(images)}
    if target_routes is not None and kind == "router":
        result["routing_agreement"] = route_correct/len(images)
    if target_probabilities is not None and kind == "fused":
        result.update(fused_label_agreement=label_correct/count,
                      fused_vector_agreement=vector_correct/len(images),
                      fused_soft_probability_mae=soft_sum/count)
    return result


def transfer_pgd(surrogate, x, epsilon=8/255, steps=40, restarts=5, step_size=None,
                 seed=0, target_kind="router"):
    """Generate using surrogate gradients and select restarts by surrogate loss.

    No target callable is accepted, so generation and selection cannot secretly
    consume target queries. Report target success only after separate evaluator
    calls. Router references are the *surrogate's* own clean top-1 predictions;
    fused references are its clean thresholded predictions. This distinction
    matters whenever clean surrogate/target routing agreement is imperfect.
    """
    # Cached acquisition images can be float16 to limit disk use, but attacks
    # always run in float32 so projection and gradient steps are stable.
    x = x.to(dtype=torch.float32)
    _validate(x, epsilon, target_kind)
    if steps < 1 or restarts < 1:
        raise ValueError("steps and restarts must be positive")
    if hasattr(surrogate, "target_kind") and surrogate.target_kind != target_kind:
        raise ValueError("surrogate was trained for a different attack target")
    step_size = epsilon/4 if step_size is None else step_size
    if step_size < 0:
        raise ValueError("step_size must be nonnegative")
    surrogate.eval()
    rng = torch.Generator(device=x.device).manual_seed(seed)
    with torch.no_grad():
        clean = surrogate(x)
        reference = _decision(clean, target_kind).detach()
        best_score = float(_objective(clean, target_kind, reference)[0])
        clean_route = int(clean["router_logits"].argmax(1)[0])
    best, trace, best_success = x.detach().clone(), [], False
    for restart in range(restarts):
        delta = (torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=rng)*2-1)*epsilon
        candidate = project_linf(x+delta, x, epsilon).detach()
        for step in range(steps+1):
            candidate.requires_grad_(True)
            outputs = surrogate(candidate)
            objective = _objective(outputs, target_kind, reference).mean()
            score = float(objective.detach())
            successful = bool(_success(_decision(outputs, target_kind), reference)[0])
            # Pure surrogate objective selection, including all intermediate PGD
            # iterates. Ground-truth or target-model outputs never choose a restart.
            if score > best_score:
                best_score, best, best_success = score, candidate.detach().clone(), successful
            if step == steps:
                trace.append({"restart": restart, "surrogate_objective": score,
                              "surrogate_success": successful})
                break
            gradient = torch.autograd.grad(objective, candidate, only_inputs=True)[0]
            candidate = project_linf(candidate.detach()+step_size*gradient.sign(), x, epsilon).detach()
    with torch.no_grad():
        final = surrogate(best)
        final_route = int(final["router_logits"].argmax(1)[0])
    return AttackResult(best, 0, None, None, trace, "target_not_queried",
                        {"algorithm": "transfer_pgd", "target_kind": target_kind, "seed": seed,
                         "epsilon": epsilon, "steps": steps, "restarts": restarts,
                         "step_size": step_size, "surrogate_success": best_success,
                         "surrogate_clean_route": clean_route, "surrogate_adv_route": final_route,
                         "surrogate_route_flip": clean_route != final_route,
                         "surrogate_objective": best_score, "ground_truth_used": False,
                         "target_queries_during_generation": 0,
                         "restart_selection": "maximum_surrogate_objective",
                         "reference": "surrogate_clean_prediction"})
