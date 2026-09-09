                          
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, exp, sin, lambdify

        
x, y = symbols('x y')

        
sigma_0 = 1.0          
pi = np.pi

                                      
                         
u_expr = sin(pi * x) * sin(pi * y)
T_expr = exp(-x**2 - y**2) + 0.5 * u_expr**2

       
u_func = lambdify((x, y), u_expr, modules='numpy')
T_func = lambdify((x, y), T_expr, modules='numpy')

      
x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_vals, y_vals)

               
U = u_func(X, Y)
T_vals = T_func(X, Y)

            
noise_level = 0.01
U_noisy = U + noise_level * np.random.randn(*U.shape)
T_noisy = T_vals + noise_level * np.random.randn(*T_vals.shape)

         
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, T_noisy, levels=50, cmap='inferno')
plt.colorbar(label="T(x, y)")
plt.title("Simulated Temperature Field: Joule Heating System")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

           
df_joule = pd.DataFrame({
    'x': X.flatten(),
    'y': Y.flatten(),
    'u': U_noisy.flatten(),
    'T': T_noisy.flatten()
})
file_path_joule = "./simulated_joule_heating_data.csv"
df_joule.to_csv(file_path_joule, index=False)

file_path_joule
