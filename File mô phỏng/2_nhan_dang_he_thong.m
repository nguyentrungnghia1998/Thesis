%% Simulation of Thesis
clear; close all; clc;
%% Time interval and simulation time
Step = 0.001;T_end = 40;
t = 0:Step:T_end;
%% Variables
GAMMA = cell(1,size(t,2));
W_c = cell(1,size(t,2));
W_a = cell(1,size(t,2));
x = cell(1,size(t,2));
x_hat = cell(1,size(t,2)); 
v = cell(1,size(t,2));
W_f = cell(1,size(t,2));
V_f = cell(1,size(t,2));
delta_hjb = cell(1,size(t,2));
F = cell(1,size(t,2));
F_hat = cell(1,size(t,2));
u = cell(1,size(t,2));
wi=rand(100,1)*1000-500;
d0=cell(1,size(t,2));
d1=cell(1,size(t,2));
u1 = cell(1,size(t,2));
u2 = cell(1,size(t,2));
d_=cell(1,size(t,2));
d2=cell(1,size(t,2));
d=cell(1,size(t,2));
nrmWf=cell(1,size(t,2));
nrmVf=cell(1,size(t,2));
%% Parameters
% important parameter: nuy, min(eig)
nuy = 0.1;
N=3; % number of actor-critic NN nodes
n=2; % number of states
L_f = 5; % identifier NN
eta_c = 20;
eta_a1 = 10;eta_a2 = 50;
k = 800;alpha = 300; gamma=5;beta_1 = 0.2;
GAMMA_wf=0.1*eye(L_f+1);GAMMA_vf = 0.1*eye(n);

Q = diag([4 3]); R = eye(1);
L1=14.2;
L2=217.10;
L3=716.3;
%% Initial conditions
W_f{1} = rand(L_f+1,n);
V_f{1} = rand(n,L_f);
x{1} = [-1;0.5];
x_hat{1} = [-1;0.5];
v{1} = [0;0];
p1{1}=[0;0];
p2{1}=[0;0];
p3{1}=[0;0];
%% System simulation
for i=1:size(t,2)
    %% Noise for PE condition
    d{i}=0.1*(sin(t(i))^2*cos(t(i))+sin(2*t(i))^2*cos(0.1*t(i))+sin(-1.2*t(i))^2*cos(0.5*t(i))+sin(t(i))^5);
           %% Estimated disturbance
    d2{i}=p3{i}+L3*x{i};
    d1{i}=p2{i}+L2*x{i};
    d0{i}=p1{i}+L1*x{i};
    d_{i}=pinv(g(x{i}))*d0{i};
    %% Compute control input
    u{i} = [0.6734 -0.7676]*x{i};    
    %% Identifier NN
    x_tilde = x{i} - x_hat{i};
    muy = k*x_tilde - k*(x{1} - x_hat{1}) + v{i};
    sigma_f = zeros(L_f+1,1);
    grad_sigma_f = zeros(L_f+1,L_f);
    Vfx = [1;V_f{i}'*x_hat{i}];
    for j = 1:L_f+1
        sigma_f(j) = 1/(1+exp(-Vfx(j)));
        if j>1
            grad_sigma_f(j,j-1) = sigma_f(j)*(1-sigma_f(j));
        end
    end
    
    %% Estimated model equation plus RISE feedback accounting for reconstruction error
    F{i}=f(x{i})+g(x{i})*u{i};
    F_hat{i}=W_f{i}'*sigma_f+g(x{i})*u{i}+muy;
    nrmWf{i}=norm(W_f{i});
    nrmVf{i}=norm(V_f{i});
    if i==size(t,2)
        break;
    end
    %% Update new states
    x{i+1} = x{i} + Step*(f(x{i}) + g(x{i})*(u{i}+d{i}));
    x_hat{i+1} = x_hat{i} + Step*(W_f{i}'*sigma_f + g(x{i})* (u{i}+d_{i})  + muy);  
    v{i+1} = v{i} + Step*((k*alpha + gamma)*x_tilde + beta_1*sign(x_tilde));
    W_f{i+1} = W_f{i} + Step*(GAMMA_wf*grad_sigma_f*V_f{i}'*F_hat{i}*x_tilde');
    V_f{i+1} = V_f{i} + Step*(GAMMA_vf*F_hat{i}*x_tilde'*W_f{i}'*grad_sigma_f);
    p3{i+1}=p3{i}+Step*(-L3*(f(x{i}) + g(x{i})*(u{i})+d0{i}));
    p2{i+1}=p2{i}+Step*(-L2*(f(x{i}) + g(x{i})*(u{i})+d0{i})+d2{i});
    p1{i+1}=p1{i}+Step*(-L1*(f(x{i}) + g(x{i})*(u{i})+d0{i})+d1{i});
end


figure(1);
nrmWf=cell2mat(nrmWf);
plot(t,nrmWf);
title('Norm of Wf');
legend('||Wf||');

figure(2);
nrmVf=cell2mat(nrmVf);
plot(t,nrmVf);
title('Norm of Vf');
legend('||Vf||');


figure(3)
F = cell2mat(F);
F_hat = cell2mat(F_hat);
subplot(2,1,1)
plot(t,F(1,:),'r',t,F_hat(1,:),'b--');
title('1');
legend('Real','Estimate')
subplot(2,1,2)
plot(t,F(2,:),'r',t,F_hat(2,:),'b--');
title('2');
legend('Real','Estimate')

figure(4)
plot(t,F(1,:)-F_hat(1,:),t,F(2,:)-F_hat(2,:));
title('Error identifier');
legend('Error dx1','Error dx2');
function a = f(x)
x1=x(1);
x2=x(2);
K1=0.03;
K2=0.5;
V=4;
S0=0.2187;
X0=4.8907;
F0=3.2029;
umax=1;
Y=0.5;
SF=10;
a=[umax*(x2+S0)/(K2*(x2+S0)^2+(x2+S0)+K1)*(x1+X0)-(x1+X0)*F0/V;
    -umax*(x2+S0)/(K2*(x2+S0)^2+(x2+S0)+K1)*(x1+X0)/Y+(SF-(x2+S0))*F0/V];
end
function a = g(x)
x1=x(1);
x2=x(2);
X0=4.8907;
S0=0.2187;
SF=10;
V=4;
a=[-(x1+X0)/V;
    (SF-(x2+S0))/V];
end