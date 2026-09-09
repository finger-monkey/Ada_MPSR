from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


METRIC_NAMES = ("derivative_order", "nonlinear_terms", "nesting_depth", "expression_length")
_NONLINEAR_OPERATORS = {"*", "^", "**", "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "pow"}


@dataclass
class Candidate:
    expression: Any
    derivative_order: float = 0.0
    nonlinear_terms: float = 0.0
    nesting_depth: float = 0.0
    expression_length: float | None = None
    operators: tuple[str, ...] = ()
    residual: float = math.inf
    parameters: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expression_length is None:
            self.expression_length = float(len(str(self.expression)))
        if not self.operators:
            self.operators = tuple(_operator_tokens(str(self.expression)))
        if self.nonlinear_terms == 0.0:
            self.nonlinear_terms = float(sum(token in _NONLINEAR_OPERATORS for token in self.operators))


def _operator_tokens(expression: str) -> list[str]:
    tokens = re.findall(r"(?:\*\*|[+\-/^*]|[A-Za-z_][A-Za-z_0-9]*)", expression)
    return [token for token in tokens if token in _NONLINEAR_OPERATORS or token in {"+", "-", "/"}]


def _parse_expression(expression: Any) -> Any:
    import sympy as sp

    if isinstance(expression, sp.Basic):
        return expression
    return sp.sympify(str(expression), evaluate=False)


def _expression_analysis(expression: Any) -> dict[str, Any]:
    try:
        import sympy as sp

        parsed = _parse_expression(expression)
        derivative_orders = [sum(order for _, order in item.variable_count) for item in parsed.atoms(sp.Derivative)]
        nonlinear_functions = (sp.Mul, sp.Pow, sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.Abs)

        def depth(node: Any) -> int:
            if not getattr(node, "args", ()):
                return 0
            return 1 + max(depth(child) for child in node.args)

        nonlinear_count = sum(isinstance(node, nonlinear_functions) for node in sp.preorder_traversal(parsed))
        return {
            "expression": parsed,
            "derivative_order": float(max(derivative_orders, default=0)),
            "nonlinear_terms": float(nonlinear_count),
            "nesting_depth": float(depth(parsed)),
            "expression_length": float(len(str(parsed))),
            "operators": tuple(_operator_tokens(str(parsed))),
        }
    except Exception:
        return {
            "expression": expression,
            "derivative_order": 0.0,
            "nonlinear_terms": float(sum(token in _NONLINEAR_OPERATORS for token in _operator_tokens(str(expression)))),
            "nesting_depth": 0.0,
            "expression_length": float(len(str(expression))),
            "operators": tuple(_operator_tokens(str(expression))),
        }


def candidate_from_expression(
    expression: Any,
    residual: float = math.inf,
    metadata: Mapping[str, Any] | None = None,
) -> Candidate:
    values = _expression_analysis(expression)
    return Candidate(
        expression=values["expression"],
        derivative_order=values["derivative_order"],
        nonlinear_terms=values["nonlinear_terms"],
        nesting_depth=values["nesting_depth"],
        expression_length=values["expression_length"],
        operators=values["operators"],
        residual=residual,
        metadata=dict(metadata or {}),
    )


def _tree_nodes(expression: Any) -> tuple[Any, list[Any]]:
    import sympy as sp

    parsed = _parse_expression(expression)
    nodes = [node for node in sp.preorder_traversal(parsed) if getattr(node, "args", ())]
    if not nodes:
        nodes = [parsed]
    return parsed, nodes


def _mutation_terminal(expression: Any, rng: np.random.Generator) -> Any:
    import sympy as sp

    parsed = _parse_expression(expression)
    symbols = sorted(parsed.free_symbols, key=str)
    if symbols and rng.random() < 0.75:
        return symbols[int(rng.integers(len(symbols)))]
    return sp.Integer(int(rng.integers(-2, 3)))


def mutate_candidate(
    candidate: Candidate,
    probability: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Candidate:
    generator = rng or np.random.default_rng()
    try:
        import sympy as sp

        parsed, nodes = _tree_nodes(candidate.expression)
        target = nodes[int(generator.integers(len(nodes)))]
        mutation_kind = int(generator.integers(4))
        if mutation_kind == 0:
            replacement = _mutation_terminal(parsed, generator)
        elif mutation_kind == 1:
            unary_functions = (sp.sin, sp.cos, sp.exp, sp.sqrt, sp.Abs)
            replacement = unary_functions[int(generator.integers(len(unary_functions)))](target)
        elif mutation_kind == 2:
            replacement = target + _mutation_terminal(parsed, generator)
        else:
            replacement = target * _mutation_terminal(parsed, generator)
        mutated = parsed.xreplace({target: replacement})
        values = candidate_from_expression(mutated, candidate.residual, candidate.metadata)
        values.metadata.update({"operation": "mutation", "mutation_probability": float(probability)})
        return values
    except Exception:
        expression = f"({candidate.expression}) + 0"
        return candidate_from_expression(expression, candidate.residual, {**candidate.metadata, "operation": "mutation"})


def crossover_candidates(
    first: Candidate,
    second: Candidate,
    rng: np.random.Generator | None = None,
) -> Candidate:
    generator = rng or np.random.default_rng()
    try:
        first_tree, first_nodes = _tree_nodes(first.expression)
        _, second_nodes = _tree_nodes(second.expression)
        first_target = first_nodes[int(generator.integers(len(first_nodes)))]
        second_target = second_nodes[int(generator.integers(len(second_nodes)))]
        child = first_tree.xreplace({first_target: second_target})
        metadata = {**first.metadata, "operation": "crossover", "parent": str(second.expression)}
        return candidate_from_expression(child, first.residual, metadata)
    except Exception:
        expression = f"(({first.expression}) + ({second.expression})) / 2"
        return candidate_from_expression(expression, first.residual, {**first.metadata, "operation": "crossover"})


def _candidate_metrics(candidate: Candidate) -> np.ndarray:
    return np.asarray(
        [
            candidate.derivative_order,
            candidate.nonlinear_terms,
            candidate.nesting_depth,
            candidate.expression_length or 0.0,
        ],
        dtype=float,
    )


def complexity_weights(candidate_sets: Mapping[str, Sequence[Candidate]]) -> dict[str, float]:
    totals = np.zeros(4, dtype=float)
    for candidates in candidate_sets.values():
        for candidate in candidates:
            totals += _candidate_metrics(candidate)
    total = float(totals.sum())
    values = np.full(4, 0.25) if total <= 0.0 else totals / total
    return dict(zip(METRIC_NAMES, values.tolist()))


def structural_complexity(candidate: Candidate, weights: Mapping[str, float]) -> float:
    return float(np.dot(_candidate_metrics(candidate), [weights.get(name, 0.0) for name in METRIC_NAMES]))


def constraint_degree(variable: str, equations: Iterable[Any]) -> int:
    pattern = re.compile(rf"(?<![A-Za-z_0-9]){re.escape(variable)}(?![A-Za-z_0-9])")
    equation_count = 0
    variable_count = 0
    for equation in equations:
        matches = pattern.findall(str(equation))
        if matches:
            equation_count += 1
            variable_count += len(matches)
    return equation_count + variable_count


def priority_scores(
    candidate_sets: Mapping[str, Sequence[Candidate]],
    equations: Iterable[Any],
    previous_scores: Mapping[str, float] | None = None,
) -> dict[str, float]:
    weights = complexity_weights(candidate_sets)
    scores = {}
    for variable, candidates in candidate_sets.items():
        complexity = min((structural_complexity(item, weights) for item in candidates), default=0.0)
        scores[variable] = complexity + constraint_degree(variable, equations)
    if previous_scores is not None:
        scores = {key: float(previous_scores.get(key, value)) for key, value in scores.items()}
    return scores


def selection_probabilities(scores: Mapping[str, float], epsilon: float = 1e-12) -> dict[str, float]:
    names = list(scores)
    inverse = np.asarray([1.0 / max(float(scores[name]), epsilon) for name in names], dtype=float)
    total = float(inverse.sum())
    probabilities = np.full(len(names), 1.0 / max(len(names), 1)) if total <= 0.0 else inverse / total
    return dict(zip(names, probabilities.tolist()))


def sample_decoupling_order(
    scores: Mapping[str, float],
    rng: np.random.Generator | None = None,
) -> list[str]:
    generator = rng or np.random.default_rng()
    remaining = list(scores)
    order = []
    while remaining:
        probabilities = selection_probabilities({name: scores[name] for name in remaining})
        names = list(probabilities)
        selected = names[int(generator.choice(len(names), p=[probabilities[name] for name in names]))]
        order.append(selected)
        remaining.remove(selected)
    return order


def update_priority_scores(
    previous_scores: Mapping[str, float],
    orders: Sequence[Sequence[str]],
    residuals: Sequence[Mapping[str, float] | Sequence[float] | float],
    alpha: float = 0.9,
) -> dict[str, float]:
    updates = {name: 0.0 for name in previous_scores}
    for order, residual in zip(orders, residuals):
        if isinstance(residual, Mapping):
            values = {name: float(residual.get(name, 0.0)) for name in order}
        elif np.isscalar(residual):
            values = {name: float(residual) for name in order}
        else:
            values = {name: float(value) for name, value in zip(order, residual)}
        for position, name in enumerate(order, start=1):
            updates[name] += values.get(name, 0.0) / position
    return {name: alpha * float(previous_scores[name]) + updates.get(name, 0.0) for name in previous_scores}


def structure_entropy(candidates: Sequence[Candidate]) -> float:
    counts: dict[str, int] = {}
    total = 0
    for candidate in candidates:
        for operator in candidate.operators or tuple(_operator_tokens(str(candidate.expression))):
            counts[operator] = counts.get(operator, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=float) / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def normalized_entropy(
    entropy: float,
    historical_min: float,
    historical_max: float,
    epsilon: float = 1e-12,
) -> float:
    return float(np.clip((entropy - historical_min) / (historical_max - historical_min + epsilon), 0.0, 1.0))


def mutation_probability(
    entropy: float,
    historical_min: float,
    historical_max: float,
    epsilon: float = 1e-12,
) -> float:
    return 1.0 - normalized_entropy(entropy, historical_min, historical_max, epsilon)


@dataclass
class StructureEntropyController:
    historical_min: float = math.inf
    historical_max: float = -math.inf
    epsilon: float = 1e-12

    def update(self, candidates: Sequence[Candidate]) -> float:
        entropy = structure_entropy(candidates)
        self.historical_min = min(self.historical_min, entropy)
        self.historical_max = max(self.historical_max, entropy)
        return entropy

    def probability(self, candidates: Sequence[Candidate]) -> float:
        entropy = self.update(candidates)
        if self.historical_max - self.historical_min <= self.epsilon:
            return 0.5
        return mutation_probability(entropy, self.historical_min, self.historical_max, self.epsilon)


def local_parameter_optimization(
    parameters: Sequence[float],
    residual_function: Callable[[np.ndarray], float],
    regularization: float = 5e-4,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    **options: Any,
) -> Any:
    from scipy.optimize import minimize

    initial = np.asarray(parameters, dtype=float)

    def objective(values: np.ndarray) -> float:
        residual = float(residual_function(values))
        return residual + regularization * float(np.dot(values, values))

    return minimize(objective, initial, method="L-BFGS-B", bounds=bounds, **options)


def validate_consistency(
    expressions: Mapping[str, Any],
    residual_function: Callable[[Mapping[str, Any]], Any],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    raw = residual_function(expressions)
    if isinstance(raw, Mapping):
        residuals = {str(key): float(value) for key, value in raw.items()}
        maximum = max(residuals.values(), default=0.0)
    else:
        residuals = {"system": float(raw)}
        maximum = float(raw)
    maximum = abs(maximum)
    return {
        "residuals": residuals,
        "max_residual": maximum,
        "tolerance": float(tolerance),
        "passed": bool(maximum <= tolerance),
    }


def decoupling_discriminability_index(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size <= 1 or np.mean(values) == 0.0:
        return 0.0
    return float(np.std(values) / np.mean(values) * (np.max(values) - np.min(values)) / (values.size - 1))


def decoupling_efficiency(residual_history: Sequence[float]) -> float:
    values = np.asarray(residual_history, dtype=float)
    if values.size <= 1 or values[0] == 0.0:
        return 0.0
    return float(np.sum(values[:-1] - values[1:]) / (values.size * values[0]))


class AdaptiveDecouplingScheduler:
    def __init__(
        self,
        candidate_sets: Mapping[str, Sequence[Candidate]],
        equations: Iterable[Any],
        alpha: float = 0.9,
    ) -> None:
        self.candidate_sets = {name: list(items) for name, items in candidate_sets.items()}
        self.equations = list(equations)
        self.alpha = alpha
        self.scores = priority_scores(self.candidate_sets, self.equations)
        self.entropy = {name: StructureEntropyController() for name in self.candidate_sets}

    def probabilities(self) -> dict[str, float]:
        return selection_probabilities(self.scores)

    def sample_order(self, rng: np.random.Generator | None = None) -> list[str]:
        return sample_decoupling_order(self.scores, rng)

    def mutation_probability(self, variable: str) -> float:
        return self.entropy[variable].probability(self.candidate_sets[variable])

    def update(
        self,
        order: Sequence[str],
        residuals: Mapping[str, float] | Sequence[float] | float,
    ) -> dict[str, float]:
        self.scores = update_priority_scores(self.scores, [order], [residuals], self.alpha)
        return dict(self.scores)


def evolve_candidate_pool(
    candidate_sets: Mapping[str, Sequence[Candidate]],
    equations: Iterable[Any],
    residual_function: Callable[[Mapping[str, Any]], Mapping[str, float] | Sequence[float] | float],
    iterations: int = 10,
    tolerance: float = 1e-6,
    alpha: float = 0.9,
    rng: np.random.Generator | None = None,
    mutation_function: Callable[[Candidate, float, np.random.Generator], Candidate] | None = None,
    crossover_function: Callable[[Candidate, Candidate, np.random.Generator], Candidate] | None = None,
) -> dict[str, Any]:
    pools = {name: list(items) for name, items in candidate_sets.items()}
    scheduler = AdaptiveDecouplingScheduler(pools, equations, alpha)
    generator = rng or np.random.default_rng()
    selected = {name: items[0] for name, items in pools.items() if items}
    history: list[float] = []
    orders: list[list[str]] = []

    for _ in range(iterations):
        order = scheduler.sample_order(generator)
        orders.append(order)
        current_expressions = {name: item.expression for name, item in selected.items()}
        round_residuals = residual_function(current_expressions)
        for variable in order:
            pool = pools[variable]
            if not pool:
                continue
            current = selected[variable]
            probability = scheduler.mutation_probability(variable)
            trial = current
            if generator.random() < probability:
                if mutation_function is not None:
                    trial = mutation_function(current, probability, generator)
                else:
                    trial = mutate_candidate(current, probability, generator)
            elif len(pool) > 1:
                partner = pool[int(generator.integers(len(pool)))]
                if crossover_function is not None:
                    trial = crossover_function(current, partner, generator)
                else:
                    trial = crossover_candidates(current, partner, generator)
            trial_expressions = {name: item.expression for name, item in selected.items()}
            trial_expressions[variable] = trial.expression
            trial_residual = residual_function(trial_expressions)
            current_value = _residual_for_variable(round_residuals, variable)
            trial_value = _residual_for_variable(trial_residual, variable)
            if trial_value <= current_value:
                selected[variable] = replace(trial, residual=trial_value)
                pool.append(selected[variable])
                round_residuals = trial_residual
        scheduler.update(order, round_residuals)
        full_residual = residual_function({name: item.expression for name, item in selected.items()})
        history.append(_residual_norm(full_residual))
        if history[-1] <= tolerance:
            break

    expressions = {name: item.expression for name, item in selected.items()}
    return {
        "expressions": expressions,
        "candidates": selected,
        "scores": scheduler.scores,
        "probabilities": scheduler.probabilities(),
        "orders": orders,
        "residual_history": history,
        "ddi": decoupling_discriminability_index(list(scheduler.scores.values())),
        "de": decoupling_efficiency(history),
        "validation": validate_consistency(expressions, residual_function, tolerance),
    }


def _residual_for_variable(residual: Mapping[str, float] | Sequence[float] | float, variable: str) -> float:
    if isinstance(residual, Mapping):
        return float(residual.get(variable, _residual_norm(residual)))
    if np.isscalar(residual):
        return float(residual)
    return _residual_norm(residual)


def _residual_norm(residual: Mapping[str, float] | Sequence[float] | float) -> float:
    if isinstance(residual, Mapping):
        values = list(residual.values())
    elif np.isscalar(residual):
        return abs(float(residual))
    else:
        values = residual
    values_array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values_array ** 2))) if values_array.size else 0.0
