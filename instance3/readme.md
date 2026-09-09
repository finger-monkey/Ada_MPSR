# Instance1 of PDEs
This example shows how to formulate, compute, and plot the solution to a system of two partial differential equations.
Consider the system of PDEs
$$
\begin{aligned}
& \frac{\partial u_1}{\partial t}-0.024 \frac{\partial^2 u_1}{\partial x^2}+F\left(u_1-u_2\right)=0, \\
& \frac{\partial u_2}{\partial t}-0.170 \frac{\partial^2 u_2}{\partial x^2}-F\left(u_1-u_2\right)=0 .
\end{aligned}
$$
(The function $F(y)=e^{5.73 y}-e^{-11.46 y}$ is used as a shorthand.)
The equation holds on the interval $0 \leq x \leq 1$ for times $t \geq 0$. The initial conditions are
$$
\begin{aligned}
& u_1(x, 0)=1, \\
& u_2(x, 0)=0 .
\end{aligned}
$$

The boundary conditions are
$$
\begin{aligned}
\frac{\partial}{\partial x} u_1(0, t) & =0 \\
u_2(0, t) & =0 \\
\frac{\partial}{\partial x} u_2(1, t) & =0 \\
u_1(1, t) & =1
\end{aligned}
$$
