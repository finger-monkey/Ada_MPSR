
clear;
clc;
%% main
x = [0 0.005 0.01 0.05 0.1 0.2 0.5 0.7 0.9 0.95 0.99 0.995 1];
t = [0 0.005 0.01 0.05 0.1 0.5 1 1.5 2];
m = 0;
sol = pdepe(m,@pdefun,@pdeic,@pdebc,x,t);

u1 = sol(:,:,1);
u2 = sol(:,:,2);
surf(x,t,u1)
%title('u_1(x,t)')
xlabel('X')
ylabel('T')


% surf(x,t,u2)
% %title('u_2(x,t)')
% xlabel('Distance x')
% ylabel('Time t')



%%
function [c,f,s] = pdefun(x,t,u,dudx) % Equation to solve
c = [1; 1];
f = [0.024; 0.17] .* dudx;
y = u(1) - u(2);
F = exp(5.73*y)-exp(-11.47*y);
s = [-F; F];
end
% ---------------------------------------------
function u0 = pdeic(x) % Initial Conditions
u0 = [1; 0];
end
% ---------------------------------------------
function [pl,ql,pr,qr] = pdebc(xl,ul,xr,ur,t) % Boundary Conditions
pl = [0; ul(2)];
ql = [1; 0];
pr = [ur(1)-1; 0];
qr = [0; 1];
end
% ---------------------------------------------