"""Auditable pixel-space black-box attacks.

Square follows Andriushchenko et al., https://arxiv.org/abs/1912.00049.
L_inf HopSkipJump follows Chen et al., https://arxiv.org/abs/1904.02144:
random adversarial initialization, boundary projection, binary-decision gradient
estimation, geometric step search, and repeated boundary projection. Its search
may query outside the epsilon ball, as in the original minimization algorithm;
only *queried*, epsilon-feasible candidates can be returned as successful.

The first clean decision is charged. Every candidate in every batch is charged,
including failed initialization, finite differences and line searches. Evaluator
diagnostics must run separately. Fused attacks use clean model pseudo-labels,
never dataset truth. A hard fused decision changes if ANY thresholded label does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F

TargetKind = Literal["router", "fused"]


@dataclass
class AttackResult:
    x_adv: torch.Tensor
    queries: int
    first_success_query: int | None
    success: bool | None
    trace: list[dict]
    status: str
    metadata: dict = field(default_factory=dict)
    snapshots: dict[int, dict] = field(default_factory=dict)


class QueryBudgetExhausted(RuntimeError):
    pass


class CountedTarget:
    """One evaluation per image, not per Python call. Oversized calls are rejected.

    Callers can slice a batch to ``remaining``. Rejecting instead of silently
    truncating avoids attributing outputs to inputs that were never evaluated.
    """
    def __init__(self, target: Callable, budget: int):
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("query budget must be a positive integer")
        self.target, self.budget, self.queries = target, budget, 0
        self.phase_counts: dict[str, int] = {}

    @property
    def remaining(self):
        return self.budget - self.queries

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, phase="search"):
        n = len(x)
        if n < 1 or n > self.remaining:
            raise QueryBudgetExhausted(f"Requested {n} evaluations; {self.remaining} remain")
        self.queries += n
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + n
        outputs = self.target(x)
        if not isinstance(outputs, dict):
            raise TypeError("target must return a dictionary")
        return outputs


def project_linf(candidate, original, epsilon):
    return torch.maximum(torch.minimum(candidate, original + epsilon), original - epsilon).clamp(0, 1)


def _validate(x, epsilon, target_kind):
    if x.ndim != 4 or x.shape[0] != 1 or not x.is_floating_point():
        raise ValueError("attack one floating-point image [1,C,H,W] at a time")
    if not bool(torch.isfinite(x).all()) or float(x.min()) < 0 or float(x.max()) > 1:
        raise ValueError("attack inputs must be finite pixels in [0,1]")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    if target_kind not in ("router", "fused"):
        raise ValueError("target_kind must be router or fused")


def _decision(outputs, kind):
    if kind == "router":
        values = outputs.get("router_logits", outputs.get("alpha"))
        if values is None:
            raise KeyError("router_logits or alpha is required")
        return values.argmax(-1)
    return outputs["fused_logits"] >= 0


def _success(decisions, reference):
    return (decisions != reference).reshape(len(decisions), -1).any(-1)


def _objective(outputs, kind, reference):
    """Always maximize: negative original-route margin, or pseudo-label BCE."""
    if kind == "fused":
        logits = outputs["fused_logits"]
        return F.binary_cross_entropy_with_logits(logits, reference.float().expand_as(logits), reduction="none").mean(-1)
    logits = outputs.get("router_logits")
    if logits is None:
        logits = outputs["alpha"].clamp_min(1e-30).log()
    index = reference.long().reshape(1, 1).expand(len(logits), 1)
    clean = logits.gather(1, index).squeeze(1)
    others = logits.scatter(1, index, -torch.inf).amax(1)
    return others - clean


class _Ledger:
    def __init__(self, counter, x, epsilon, kind, checkpoints, score_based):
        self.counter, self.x, self.epsilon, self.kind = counter, x.detach(), epsilon, kind
        self.score_based = score_based
        self.reference = None
        self.best = x.detach().clone()
        self.best_objective = -math.inf
        self.best_norm = math.inf
        self.best_success = False
        self.first_success = None
        self.trace = []
        self.checkpoints = set(int(q) for q in checkpoints if 0 < q <= counter.budget)
        self.snapshots = {}

    def snapshot(self, q):
        return {"x_adv": self.best.detach().cpu().clone(), "queries": q,
                "success": self.best_success, "first_success_query": self.first_success,
                "objective": self.best_objective if self.score_based else None}

    def evaluate(self, candidates, phase):
        before = self.counter.queries
        out = self.counter(candidates, phase)
        decisions = _decision(out, self.kind).detach()
        if self.reference is None:
            assert len(candidates) == 1 and phase == "clean_reference"
            self.reference = decisions.clone()
        successes = _success(decisions, self.reference)
        # HSJ never reads scores for an objective or candidate selection.
        scores = _objective(out, self.kind, self.reference).detach() if self.score_based else None
        distances = (candidates - self.x).flatten(1).abs().amax(1)
        for j in range(len(candidates)):
            q, norm = before + j + 1, float(distances[j])
            feasible = norm <= self.epsilon + 1e-7
            success = bool(successes[j]) and feasible
            if success and self.first_success is None:
                self.first_success = q
            score = float(scores[j]) if scores is not None else None
            if feasible:
                if self.score_based:
                    # Preserve an achieved hard-decision success (BCE alone need
                    # not monotonically preserve a multi-label vector change).
                    improve = (success and not self.best_success) or (
                        success == self.best_success and score > self.best_objective)
                else:
                    improve = success and (not self.best_success or norm < self.best_norm)
                if improve:
                    self.best = candidates[j:j+1].detach().clone()
                    self.best_success, self.best_norm = success, norm
                    if score is not None:
                        self.best_objective = score
            self.trace.append({"query": q, "phase": phase, "decision_changed": bool(successes[j]),
                               "epsilon_feasible": feasible, "success": success,
                               "best_success": self.best_success, "linf": norm,
                               "objective": score})
            if q in self.checkpoints:
                self.snapshots[q] = self.snapshot(q)
        return successes, scores

    def result(self, status, metadata):
        # Early stopping still supplies all requested larger-budget summaries,
        # recording the actual number of calls rather than the nominal budget.
        for q in sorted(self.checkpoints):
            if q not in self.snapshots:
                self.snapshots[q] = self.snapshot(self.counter.queries)
        if float((self.best-self.x).abs().max()) > self.epsilon + 1e-7:
            raise AssertionError("returned perturbation violates epsilon")
        metadata.update({"phase_queries": dict(self.counter.phase_counts),
                         "target_kind": self.kind, "epsilon": self.epsilon,
                         "clean_reference_included": True, "ground_truth_used": False})
        return AttackResult(self.best, self.counter.queries, self.first_success,
                            self.best_success, self.trace, status, metadata, self.snapshots)


def _square_probability(initial, iteration, schedule_queries):
    position = int(10000 * iteration / max(1, schedule_queries))
    boundaries = (10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000)
    divisor = 2 ** sum(position > boundary for boundary in boundaries)
    return initial / divisor


def square_attack(target, x, epsilon=8/255, query_budget=1000, target_kind="router",
                  seed=0, checkpoints=(100, 250, 500, 1000), p_init=0.05,
                  schedule_queries=1000, stop_on_success=False):
    """L_inf Square; checkpoints are prefixes of ONE fixed 1000-query schedule.

    Keeping schedule_queries fixed also makes separate Q runs exact prefixes.
    Router objective is the negative clean-route logit margin. Fused objective
    maximizes BCE against the clean thresholded prediction, without true labels.
    """
    _validate(x, epsilon, target_kind)
    if not 0 < p_init <= 1 or schedule_queries < 1:
        raise ValueError("invalid square schedule")
    rng = torch.Generator(device=x.device).manual_seed(seed)
    counter = CountedTarget(target, query_budget)
    ledger = _Ledger(counter, x, epsilon, target_kind, checkpoints, True)
    ledger.evaluate(x, "clean_reference")
    c, h, w = x.shape[1:]
    iteration = 0
    while counter.remaining and epsilon > 0:
        if iteration == 0:
            signs = torch.randint(2, (1, c, 1, w), device=x.device, generator=rng) * 2 - 1
            candidate = project_linf(x + epsilon * signs, x, epsilon)
            phase = "initialization"
        else:
            p = _square_probability(p_init, iteration - 1, schedule_queries)
            side = min(max(1, round(math.sqrt(p * h * w))), h, w)
            row = int(torch.randint(h-side+1, (), device=x.device, generator=rng))
            col = int(torch.randint(w-side+1, (), device=x.device, generator=rng))
            candidate = ledger.best.clone()
            signs = torch.randint(2, (1, c, 1, 1), device=x.device, generator=rng) * 2 - 1
            candidate[:, :, row:row+side, col:col+side] = (
                x[:, :, row:row+side, col:col+side] + epsilon * signs)
            candidate = project_linf(candidate, x, epsilon)
            phase = "square_search"
        ledger.evaluate(candidate, phase)
        iteration += 1
        if stop_on_success and ledger.best_success:
            break
    status = "success" if ledger.best_success else "budget_exhausted" if not counter.remaining else "zero_epsilon"
    return ledger.result(status, {"algorithm": "square_linf", "seed": seed,
                                 "schedule_queries": schedule_queries, "p_init": p_init,
                                 "objective": "negative_router_margin" if target_kind == "router" else "clean_pseudolabel_bce"})


def hopskipjump_attack(target, x, epsilon=8/255, query_budget=1000, target_kind="router",
                       seed=0, checkpoints=(100, 250, 500, 1000), steps=64,
                       initial_gradient_evals=100, max_gradient_evals=10000,
                       gradient_batch_size=16, max_initialization=1000, gamma=1.0,
                       starting_point=None, stop_on_success=False):
    """Hard-decision L_inf HopSkipJump with exact image query accounting.

    ``starting_point`` is optional and its hard decision is charged. A failed
    search returns clean x unless an epsilon-feasible queried success exists;
    failure is never repaired using score-based amplification. Search is allowed
    over [0,1] outside epsilon, as required by HSJ boundary initialization.
    Float32 precision floors prevent endless binary searches and zero probes.
    """
    _validate(x, epsilon, target_kind)
    if min(steps, initial_gradient_evals, max_gradient_evals, gradient_batch_size,
           max_initialization) < 1 or gamma <= 0:
        raise ValueError("HSJ counts and gamma must be positive")
    rng = torch.Generator(device=x.device).manual_seed(seed)
    counter = CountedTarget(target, query_budget)
    ledger = _Ledger(counter, x, epsilon, target_kind, checkpoints, False)
    ledger.evaluate(x, "clean_reference")
    d = x[0].numel()
    theta = gamma / (d*d)
    tolerance_floor = 2 * torch.finfo(x.dtype).eps
    stage, current = "initialization", None
    completed_steps = 0

    def decision(candidate, phase):
        return bool(ledger.evaluate(candidate, phase)[0][0])

    def boundary(point):
        nonlocal stage
        stage = "boundary_search"
        anchor = point.clone()
        low, high = 0.0, float((point-x).abs().max())
        tolerance = max(high * theta, tolerance_floor)
        best = point
        while high - low > tolerance:
            middle = (low+high)/2
            candidate = project_linf(anchor, x, middle)
            if torch.equal(candidate, best):
                break
            if decision(candidate, stage):
                high, best = middle, candidate
            else:
                low = middle
        return best

    try:
        if epsilon == 0:
            return ledger.result("zero_epsilon", {"algorithm": "hopskipjump_linf", "seed": seed})
        for attempt in range(max_initialization):
            candidate = (starting_point.detach().to(x).clamp(0, 1)
                         if attempt == 0 and starting_point is not None else
                         torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=rng))
            if candidate.shape != x.shape:
                raise ValueError("starting_point shape differs from x")
            if decision(candidate, stage):
                current = candidate
                break
        if current is None:
            return ledger.result("initialization_failed", {"algorithm": "hopskipjump_linf", "seed": seed})
        # Count the epsilon projection explicitly. An unqueried clipped point
        # cannot be considered successful even if the unconstrained point is.
        decision(project_linf(current, x, epsilon), "epsilon_projection")
        current = boundary(current)
        for iteration in range(1, steps+1):
            if stop_on_success and ledger.best_success:
                break
            stage = "gradient_estimation"
            distance = float((current-x).abs().max())
            delta = max(0.1 if iteration == 1 else d*theta*distance,
                        tolerance_floor * math.sqrt(d))
            evaluations = min(int(initial_gradient_evals*math.sqrt(iteration)),
                              max_gradient_evals, counter.remaining)
            if evaluations == 0:
                raise QueryBudgetExhausted()
            sum_noise = torch.zeros_like(x)
            sum_signed = torch.zeros_like(x)
            sign_sum, done = 0.0, 0
            while done < evaluations:
                n = min(gradient_batch_size, evaluations-done, counter.remaining)
                noise = torch.rand((n, *x.shape[1:]), device=x.device, dtype=x.dtype, generator=rng)*2-1
                noise /= noise.flatten(1).norm(dim=1).clamp_min(1e-12).reshape(-1, 1, 1, 1)
                probes = (current + delta*noise).clamp(0, 1)
                actual_noise = (probes-current)/delta
                changed, _ = ledger.evaluate(probes, stage)
                signs = changed.to(x.dtype)*2-1
                sum_noise += actual_noise.sum(0, keepdim=True)
                sum_signed += (actual_noise*signs.reshape(-1, 1, 1, 1)).sum(0, keepdim=True)
                sign_sum += float(signs.sum())
                done += n
            # Center binary responses only if both sides of the boundary occur.
            grad = sum_signed/evaluations
            if abs(sign_sum) < evaluations:
                grad -= (sign_sum/evaluations)*(sum_noise/evaluations)
            direction = grad.sign()
            stage = "step_search"
            step_size = distance/math.sqrt(iteration)
            while True:
                candidate = (current + step_size*direction).clamp(0, 1)
                if decision(candidate, stage):
                    current = candidate
                    break
                step_size *= 0.5
                if step_size < tolerance_floor:
                    break
            decision(project_linf(current, x, epsilon), "epsilon_projection")
            current = boundary(current)
            completed_steps = iteration
        status = "success" if ledger.best_success else "steps_exhausted"
    except QueryBudgetExhausted:
        status = "success" if ledger.best_success else f"budget_exhausted_{stage}"
    return ledger.result(status, {"algorithm": "hopskipjump_linf", "seed": seed,
                                 "decision_only": True, "initial_gradient_evals": initial_gradient_evals,
                                 "completed_steps": completed_steps, "last_stage": stage,
                                 "search_domain": "[0,1]; returned candidate must satisfy epsilon",
                                 "binary_tolerance_floor": tolerance_floor,
                                 "gradient_delta_floor": tolerance_floor*math.sqrt(d)})
