from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pysr import PySRRegressor
from scipy.signal import savgol_filter

from run_context import get_default_context


DATA_DIR = Path(__file__).resolve().parent


def build_model():
    return PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "sqrt", "exp", "log"],
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        maxsize=30,
        verbosity=1,
        random_state=42,
        procs=1,
    )


def main(run_context=None, show_plot=True):
    context = run_context or get_default_context()
    entrypoint = "compare_coupled_temperature_models"
    context.on_start(entrypoint, {"data": "stable_*_data_fixed.npy"})
    try:
        T = savgol_filter(np.load(DATA_DIR / "stable_T_data_fixed.npy")[-1], window_length=21, polyorder=3)
        u = savgol_filter(np.load(DATA_DIR / "stable_u_data_fixed.npy")[-1], window_length=21, polyorder=3)
        nx = T.shape[0]
        x = np.linspace(0, 1.0, nx).reshape(-1, 1)
        mask = (np.abs(T) < 1e5) & (np.abs(u) < 1e2) & np.isfinite(T) & np.isfinite(u)
        x_valid = x[mask]
        u_valid = u[mask].reshape(-1, 1)
        T_valid = T[mask]
        print(f"Valid samples: {len(x_valid)}")

        model_T1 = build_model()
        model_T1.fit(x_valid, T_valid)
        expr_T1 = model_T1.get_best()
        T_pred1 = model_T1.predict(x_valid)
        loss_T1 = np.mean((T_pred1 - T_valid) ** 2)

        X_T2 = np.hstack([x_valid, u_valid])
        model_T2 = build_model()
        model_T2.fit(X_T2, T_valid)
        expr_T2 = model_T2.get_best()
        T_pred2 = model_T2.predict(X_T2)
        loss_T2 = np.mean((T_pred2 - T_valid) ** 2)

        if show_plot:
            plt.figure(figsize=(10, 4))
            plt.plot(x_valid, T_valid, label="Reference T(x)", linewidth=2)
            plt.plot(x_valid, T_pred1, "--", label="T(x) first fit")
            plt.plot(x_valid, T_pred2, ":", label="T(x) coupled fit")
            plt.title("Temperature fitting comparison")
            plt.xlabel("x")
            plt.ylabel("Temperature (K)")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

        print("\nSymbolic regression results")
        print("--------------------------------------------------")
        print("Independent fit T(x) = f(x)")
        print("Expression:", expr_T1["sympy_format"])
        print("MSE: {:.4e}".format(loss_T1))
        print("\nCoupled fit T(x) = f(x, u(x))")
        print("Expression:", expr_T2["sympy_format"])
        print("MSE: {:.4e}".format(loss_T2))
        print("--------------------------------------------------")
        result = {
            "independent_expression": str(expr_T1["sympy_format"].values[0]),
            "coupled_expression": str(expr_T2["sympy_format"].values[0]),
            "independent_mse": float(loss_T1),
            "coupled_mse": float(loss_T2),
            "valid_samples": int(len(x_valid)),
        }
        context.on_end(entrypoint, "completed", result)
        return result
    except Exception as exc:
        context.on_end(entrypoint, "failed", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()

