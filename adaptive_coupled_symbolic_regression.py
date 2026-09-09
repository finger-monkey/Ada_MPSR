from pathlib import Path
import warnings

import numpy as np
from pysr import PySRRegressor
from sympy import pretty

from run_context import get_default_context


warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent


def build_model(iters=100):
    return PySRRegressor(
        niterations=iters,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "sqrt"],
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        maxsize=20,
        verbosity=0,
        deterministic=True,
        parallelism="serial",
        random_state=42,
    )


def compute_priority(loss, expr):
    if expr is None or expr.empty:
        return float("inf")
    expression = str(expr["sympy_format"].values[0]) if hasattr(expr["sympy_format"], "values") else str(expr["sympy_format"])
    return loss * len(expression)


def main(run_context=None):
    context = run_context or get_default_context()
    entrypoint = "adaptive_coupled_symbolic_regression"
    context.on_start(entrypoint, {"data": "stable_*_data_fixed.npy"})
    try:
        T = np.load(DATA_DIR / "stable_T_data_fixed.npy")[-1]
        u = np.load(DATA_DIR / "stable_u_data_fixed.npy")[-1]
        nx = len(T)
        x = np.linspace(0, 1.0, nx).reshape(-1, 1)
        T_expr_best, u_expr_best = None, None
        T_best_loss, u_best_loss = float("inf"), float("inf")
        T_pred_best, u_pred_best = None, None
        min_rounds = 5

        for round_index in range(1, 20):
            print(f"Adaptive optimization round {round_index}")
            if T_expr_best is None and u_expr_best is None:
                to_fit = "T"
            else:
                priority_T = compute_priority(T_best_loss, T_expr_best)
                priority_u = compute_priority(u_best_loss, u_expr_best)
                to_fit = "T" if priority_T > priority_u else "u"

            if to_fit == "T":
                X_T = np.hstack([x, u_pred_best.reshape(-1, 1)]) if u_pred_best is not None else np.hstack([x, u.reshape(-1, 1)])
                model_T = build_model(iters=200)
                model_T.fit(X_T, T)
                T_pred = model_T.predict(X_T)
                loss_T = np.mean((T_pred - T) ** 2)
                if loss_T < T_best_loss:
                    T_best_loss = loss_T
                    T_expr_best = model_T.get_best()
                    T_pred_best = T_pred
                print(f"Fitted variable: T(x, u) | MSE = {loss_T:.3e}")
                print("Expression:")
                print(pretty(T_expr_best["sympy_format"], use_unicode=False))
            else:
                X_u = np.hstack([x, T_pred_best.reshape(-1, 1)]) if T_pred_best is not None else np.hstack([x, T.reshape(-1, 1)])
                model_u = build_model(iters=100)
                model_u.fit(X_u, u)
                u_pred = model_u.predict(X_u)
                loss_u = np.mean((u_pred - u) ** 2)
                if loss_u < u_best_loss:
                    u_best_loss = loss_u
                    u_expr_best = model_u.get_best()
                    u_pred_best = u_pred
                print(f"Fitted variable: u(x, T) | MSE = {loss_u:.3e}")
                print("Expression:")
                print(pretty(u_expr_best["sympy_format"], use_unicode=False))

            if round_index >= min_rounds and T_best_loss < 1e-3 and u_best_loss < 1e-3:
                break

        print("\nFinal expressions and errors")
        print("------------------------------------------------")
        print("T(x, u):")
        if T_expr_best is not None:
            print("Expression:")
            print(pretty(T_expr_best["sympy_format"], use_unicode=False))
            print(f"MSE: {T_best_loss:.3e}")
        else:
            print("T(x, u) fitting failed")
        print("------------------------------------------------")
        print("u(x, T):")
        if u_expr_best is not None:
            print("Expression:")
            print(pretty(u_expr_best["sympy_format"], use_unicode=False))
            print(f"MSE: {u_best_loss:.3e}")
        else:
            print("u(x, T) fitting failed")
        print("------------------------------------------------")
        result = {
            "T_expression": None if T_expr_best is None else str(T_expr_best["sympy_format"].values[0]),
            "u_expression": None if u_expr_best is None else str(u_expr_best["sympy_format"].values[0]),
            "T_mse": float(T_best_loss),
            "u_mse": float(u_best_loss),
            "rounds": round_index,
        }
        context.on_end(entrypoint, "completed", result)
        return result
    except Exception as exc:
        context.on_end(entrypoint, "failed", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()

