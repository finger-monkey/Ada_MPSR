
import numpy as np
import matplotlib.pyplot as plt

              
nx, nt = 100, 300
L, T_max = 1.0, 0.003
dx = L / (nx - 1)
dt = T_max / nt
x = np.linspace(0, L, nx)

              
alpha_T = 1e-4
E = 2e11
rho = 7800
alpha_th = 1e-5
T0 = 300
gamma = 5e3        

              
T = np.ones(nx) * T0
u = np.zeros(nx)
v = np.zeros(nx)
T[int(nx/2)] += 100          

               
cfl = alpha_T * dt / dx**2
print(f"CFL number for heat equation: {cfl:.4f} (should be < 0.5)")

                
T_record = [T.copy()]
u_record = [u.copy()]

              
for n in range(nt):
                  
    T_new = T.copy()
    for i in range(1, nx-1):
        d2T_dx2 = (T[i+1] - 2*T[i] + T[i-1]) / dx**2
        T_new[i] = T[i] + alpha_T * dt * d2T_dx2

                 
    sigma = np.zeros(nx)
    for i in range(1, nx-1):
        strain = (u[i+1] - u[i-1]) / (2*dx)
        sigma[i] = E * (strain - alpha_th * (T[i] - T0))

    a = np.zeros(nx)
    for i in range(1, nx-1):
        dsigma_dx = (sigma[i+1] - sigma[i-1]) / (2*dx)
        a[i] = dsigma_dx / rho

         
    v *= np.exp(-gamma * dt)
    v += a * dt
    u += v * dt

          
    T_new[0], T_new[-1] = T_new[1], T_new[-2]
    u[0], u[-1] = 0.0, 0.0

            
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(T_new)):
        print(f"Numerical instability at step {n}; stopping.")
        break

        
    T = T_new
    T_record.append(T.copy())
    u_record.append(u.copy())

              
T_record = np.array(T_record)
u_record = np.array(u_record)
np.save("stable_T_data2.npy", T_record)
np.save("stable_u_data2.npy", u_record)

             
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, T, label="T(x)")
plt.title("Final temperature distribution")
plt.xlabel("x")
plt.ylabel("Temperature (K)")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x, u, label="u(x)", color='orange')
plt.title("Final displacement distribution")
plt.xlabel("x")
plt.ylabel("Displacement (m)")
plt.grid()

plt.tight_layout()
plt.show()
