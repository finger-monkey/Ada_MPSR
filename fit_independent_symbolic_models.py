from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pysr import PySRRegressor

from run_context import get_default_context


DATA_DIR = Path(__file__).resolve().parent


def build_model():
    return PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "sqrt"],
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        maxsize=20,
        verbosity=1,
    )


def main(run_context=None, show_plot=True):
    context = run_context or get_default_context()
    entrypoint = "fit_independent_symbolic_models"
    context.on_start(entrypoint, {"data": "stable_T_data.npy, stable_u_data.npy"})
    try:
        T = np.load(DATA_DIR / "stable_T_data.npy")[-1]
        u = np.load(DATA_DIR / "stable_u_data.npy")[-1]
        nx = len(T)
        x = np.linspace(0, 1.0, nx).reshape(-1, 1)

        print("Fitting T(x)...")
        model_T = build_model()
        model_T.fit(x, T)
        expr_T = model_T.get_best()
        T_pred = model_T.predict(x)
        T_mse = np.mean((T_pred - T) ** 2)
        print("\nRecovered expression T(x):\n", expr_T)

        print("\nFitting u(x)...")
        model_u = build_model()
        model_u.fit(x, u)
        expr_u = model_u.get_best()
        u_pred = model_u.predict(x)
        u_mse = np.mean((u_pred - u) ** 2)
        print("\nRecovered expression u(x):\n", expr_u)

        if show_plot:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(x, T, label="Reference T(x)")
            plt.plot(x, T_pred, "--", label="Fitted T(x)")
            plt.title("T(x) fitting result")
            plt.xlabel("x")
            plt.ylabel("Temperature")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(x, u, label="Reference u(x)")
            plt.plot(x, u_pred, "--", label="Fitted u(x)")
            plt.title("u(x) fitting result")
            plt.xlabel("x")
            plt.ylabel("Displacement")
            plt.legend()
            plt.tight_layout()
            plt.show()

        result = {
            "T_expression": str(expr_T["sympy_format"].values[0]),
            "u_expression": str(expr_u["sympy_format"].values[0]),
            "T_mse": float(T_mse),
            "u_mse": float(u_mse),
        }
        context.on_end(entrypoint, "completed", result)
        return result
    except Exception as exc:
        context.on_end(entrypoint, "failed", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()

