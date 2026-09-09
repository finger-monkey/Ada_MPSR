from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pysr import PySRRegressor

from run_context import get_default_context


DATA_DIR = Path(__file__).resolve().parent


def build_model(niterations):
    return PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "sqrt"],
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        maxsize=20,
        verbosity=0,
    )


def main(run_context=None, show_plot=True):
    context = run_context or get_default_context()
    entrypoint = "fit_sequential_coupled_models"
    context.on_start(entrypoint, {"data": "stable_*_data_fixed.npy"})
    try:
        T = np.load(DATA_DIR / "stable_T_data_fixed.npy")[-1]
        u = np.load(DATA_DIR / "stable_u_data_fixed.npy")[-1]
        nx = len(T)
        x = np.linspace(0, 1.0, nx).reshape(-1, 1)

        model_T1 = build_model(200)
        model_T1.fit(x, T)
        T_expr1 = model_T1.get_best()
        T_pred1 = model_T1.predict(x)
        T_loss1 = np.mean((T_pred1 - T) ** 2)

        X_u = np.hstack([x, T_pred1.reshape(-1, 1)])
        model_u = build_model(10)
        model_u.fit(X_u, u)
        u_expr = model_u.get_best()
        u_pred = model_u.predict(X_u)
        u_loss = np.mean((u_pred - u) ** 2)

        X_T2 = np.hstack([x, u_pred.reshape(-1, 1)])
        model_T2 = build_model(200)
        model_T2.fit(X_T2, T)
        T_expr2 = model_T2.get_best()
        T_pred2 = model_T2.predict(X_T2)
        T_loss2 = np.mean((T_pred2 - T) ** 2)

        if show_plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(x, T, label="T(x)")
            plt.plot(x, T_pred1, "--", label="Initial T(x)")
            plt.plot(x, T_pred2, ":", label="Feedback T(x)")
            plt.title("Temperature fitting")
            plt.xlabel("x")
            plt.ylabel("Temperature")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(x, u, label="u(x)")
            plt.plot(x, u_pred, "--", label="Fitted u(x)")
            plt.title("Displacement fitting")
            plt.xlabel("x")
            plt.ylabel("Displacement")
            plt.legend()
            plt.tight_layout()
            plt.show()

        print("Final expressions and metrics")
        print("T(x) initial:", T_expr1)
        print("T(x) initial MSE: {:.3e}".format(T_loss1))
        print("u(x):", u_expr)
        print("u(x) MSE: {:.3e}".format(u_loss))
        print("T(x) feedback:", T_expr2)
        print("T(x) feedback MSE: {:.3e}".format(T_loss2))
        result = {
            "T_initial_expression": str(T_expr1["sympy_format"].values[0]),
            "u_expression": str(u_expr["sympy_format"].values[0]),
            "T_feedback_expression": str(T_expr2["sympy_format"].values[0]),
            "T_initial_mse": float(T_loss1),
            "u_mse": float(u_loss),
            "T_feedback_mse": float(T_loss2),
        }
        context.on_end(entrypoint, "completed", result)
        return result
    except Exception as exc:
        context.on_end(entrypoint, "failed", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()

