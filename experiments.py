from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from run_context import RunContext, create_run_context


ENTRYPOINTS = {
    "adaptive_coupled_symbolic_regression": ("adaptive_coupled_symbolic_regression", "main"),
    "compare_coupled_temperature_models": ("compare_coupled_temperature_models", "main"),
    "fit_independent_symbolic_models": ("fit_independent_symbolic_models", "main"),
    "fit_sequential_coupled_models": ("fit_sequential_coupled_models", "main"),
}


def _load_entrypoint(module_name: str, function_name: str) -> Callable:
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def main(
    output_dir: str | Path = "reproduction_runs",
    run_id: str | None = None,
    selected: list[str] | None = None,
    show_plot: bool = False,
) -> dict:
    names = selected or list(ENTRYPOINTS)
    unknown = sorted(set(names) - set(ENTRYPOINTS))
    if unknown:
        raise ValueError(f"Unknown entrypoint: {', '.join(unknown)}")
    context = create_run_context(
        output_dir,
        run_id,
        {"launcher": "experiments", "entrypoints": names},
    )
    results = {}
    failures = {}
    for name in names:
        module_name, function_name = ENTRYPOINTS[name]
        try:
            function = _load_entrypoint(module_name, function_name)
            kwargs = {"run_context": context}
            if name != "adaptive_coupled_symbolic_regression":
                kwargs["show_plot"] = show_plot
            results[name] = function(**kwargs)
        except Exception as exc:
            failures[name] = str(exc)
    status = "failed" if failures else "completed"
    context.finish(status, {"completed": list(results), "failed": failures})
    return {"run_id": context.run_id, "status": status, "results": results, "failures": failures}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reproduction_runs")
    parser.add_argument("--run-id")
    parser.add_argument("--entrypoint", action="append", dest="selected")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(arguments.output_dir, arguments.run_id, arguments.selected, arguments.show_plot)

