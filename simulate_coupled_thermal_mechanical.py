import numpy as np
import matplotlib.pyplot as plt

                       
nx, nt = 100, 300
L = 1.0
dx = L / (nx - 1)
dt = 1e-5                                   

x = np.linspace(0, L, nx)

                
T = np.ones(nx) * 300.0
u = np.zeros(nx)

         
T_data = []
u_data = []

                                        
def gradient(arr):
    grad = np.zeros_like(arr)
    grad[1:-1] = (arr[2:] - arr[:-2]) / (2 * dx)
    grad[0] = grad[1]
    grad[-1] = grad[-2]
    return grad

def divergence(arr):
    div = np.zeros_like(arr)
    div[1:-1] = (arr[2:] - 2 * arr[1:-1] + arr[:-2]) / dx**2
    div[0] = div[1]
    div[-1] = div[-2]
    return div

                      
for n in range(nt):
                                     
    strain_th = 1e-5 * (T - 300.0)
    u = np.cumsum(strain_th) * dx
    du_dx = gradient(u)

                                                      
    k_eff = 1.0 * (1 + 0.2 * np.clip(np.abs(du_dx), 0, 5))
    k_eff = np.clip(k_eff, 0.5, 2.0)

                                         
    dTdx = np.clip(gradient(T), -100, 100)
    flux = -k_eff * dTdx
    dflux_dx = gradient(flux)

                                                  
    T += dt * (dflux_dx + 0.001 * divergence(T))

                         
    T[0], T[-1] = 500.0, 300.0
    u[0], u[-1] = 0.0, u[-2]

                     
    if np.any(np.isnan(T)) or np.any(np.isinf(T)):
        print(f"Instability at step {n}")
        break

    T_data.append(T.copy())
    u_data.append(u.copy())

                   
T_data = np.array(T_data)
u_data = np.array(u_data)

              
np.save("stable_T_data.npy", T_data)
np.save("stable_u_data.npy", u_data)

                     
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x, T_data[-1])
plt.title("Final temperature distribution")
plt.xlabel("x")
plt.ylabel("T [K]")

plt.subplot(1, 2, 2)
plt.plot(x, u_data[-1])
plt.title("Final displacement distribution")
plt.xlabel("x")
plt.ylabel("u [m]")

plt.tight_layout()
plt.show()
