clear;
clc;
load("datapredict.mat");



%%
% % 定义符号变量x
% syms x t
% 
% % 定义函数表达式
% y1 = x^2 + 3*x + t;
% y2 = x^2 + 3*x + t;
% 
% % 对函数表达式求导
% dy1_dx = diff(y1, x);
% dy1_dxdx = diff(dy1_dx, x);
% dy1_dt = diff(y1, t);
% 
% % 对函数表达式求导
% dy2_dx = diff(y2, x);
% dy2_dxdx = diff(dy2_dx, x);
% dy2_dt = diff(y2, t);
% 
% 
% 
% % 显示导数
% disp(dy1_dxdx -dy1_dt);
% disp(dy2_dxdx -dy2_dt);




x = [0 0.005 0.01 0.05 0.1 0.2 0.5 0.7 0.9 0.95 0.99 0.995 1];
t = [0 0.005 0.01 0.05 0.1 0.5 1 1.5 2];
ans = log(x + 1.0)
u = [ans;ans;ans;ans;ans;ans;ans;ans;ans]
surf(x,t,u)
