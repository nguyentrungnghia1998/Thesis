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
lamda=2;
L1=14.2;
L2=217.10;
L3=716.3;
%% Initial conditions
W_c{1} = 10*rand(N,1)-5;
% W_c{1} = [1.2974   -3.1006    0.6699   -0.1532   32.1481   -4.1638    6.7208   -0.3269   -0.3970    0.3590]';
W_a{1} = 10*rand(N,1)-5;
% W_a{1} = [0.5745   -2.8933   -0.6427   -0.6825   31.5326   -4.2684    8.9911    0.0278   -1.3186    0.9632]';
W_f{1} = rand(L_f+1,n);
V_f{1} = rand(n,L_f);

GAMMA{1} = 5000*eye(N);
x{1} = [-1;0.5];
x_hat{1} = [-1;0.5];
v{1} = [0;0];
p1{1}=[0;0];
p2{1}=[0;0];
p3{1}=[0;0];
%% System simulation
for i=1:size(t,2)
    %% Noise for PE condition
    if i<= 16001
        noise = nhieu(wi,t(i));
    else
        noise = 0;
    end
    d=0.1*(sin(t(i))^2*cos(t(i))+sin(2*t(i))^2*cos(0.1*t(i))+sin(-1.2*t(i))^2*cos(0.5*t(i))+sin(t(i))^5);
           %% Estimated disturbance
    d2{i}=p3{i}+L3*x{i};
    d1{i}=p2{i}+L2*x{i};
    d0{i}=p1{i}+L1*x{i};
    d_{i}= g(x{i})*d;
    %% Compute control input
    u{i} = -lamda*tanh(1/2/lamda*pinv(R)*g(x{i})'*gradPhi(x{i})'*W_a{i});
%     u1{i}= -lamda*tanh(1/2/lamda*pinv(R)*g(x{i})'*gradPhi(x{i})'*W_a{i}+noise)-pinv(g(x{i}))*d0{i};
    u1{i}= -lamda*tanh(1/2/lamda*pinv(R)*g(x{i})'*gradPhi(x{i})'*W_a{i}+noise)-pinv(g(x{i}))*d0{i};
%     if u1{i}>lamda
%         u1{i}=lamda;
%     end
%     if u1{i}<-lamda
%         u1{i}=-lamda;
%     end
    
    %% Real model equation
    F{i} = f(x{i}) + g(x{i})*u{i};
    
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
    F_hat{i} = W_f{i}'*sigma_f + g(x{i})*u{i} + muy;
    
    omega = gradPhi(x{i})*F_hat{i};
    %% Reset gain to prevent covariance wind-up in critic NN
    if min(eig(GAMMA{i})) <= 0.1
           GAMMA{i} = GAMMA{1};
    end
    %% Compute delta HJB and estimation error
    delta_hjb{i} = W_c{i}'*gradPhi(x{i})*F_hat{i}+x{i}'*Q*x{i}+funU(lamda,R,u{i});
    
    if i==size(t,2)
        break;
    end
    %% Update new states
    x{i+1} = x{i} + Step*(f(x{i}) + g(x{i})*(u1{i}+d));
    x_hat{i+1} = x_hat{i} + Step*(W_f{i}'*sigma_f + g(x{i})* u1{i} + d_{i}  + muy);
    
    W_c{i+1} = W_c{i} + Step*(-eta_c*(GAMMA{i}*omega/(1+nuy*omega'*GAMMA{i}*omega)*delta_hjb{i}));   
    GAMMA{i+1} = GAMMA{i} + Step*(-eta_c*GAMMA{i}*(omega*omega')/(1+nuy*omega'*GAMMA{i}*omega)*GAMMA{i});
    
    dWa=-eta_a1/sqrt(1+omega'*omega)*gradPhi(x{i})*(g(x{i})*pinv(R)*funB(lamda,u{i},1)*g(x{i})')*gradPhi(x{i})'*(W_a{i}-W_c{i})*delta_hjb{i}-eta_a2*(W_a{i}-W_c{i});
    W_a{i+1} = W_a{i} + Step*dWa;
    
    v{i+1} = v{i} + Step*((k*alpha + gamma)*x_tilde + beta_1*sign(x_tilde));
    W_f{i+1} = W_f{i} + Step*(GAMMA_wf*grad_sigma_f*V_f{i}'*F_hat{i}*x_tilde');
    V_f{i+1} = V_f{i} + Step*(GAMMA_vf*F_hat{i}*x_tilde'*W_f{i}'*grad_sigma_f);
    p3{i+1}=p3{i}+Step*(-L3*(f(x{i}) + g(x{i})*(u1{i})+d0{i}));
    p2{i+1}=p2{i}+Step*(-L2*(f(x{i}) + g(x{i})*(u1{i})+d0{i})+d2{i});
    p1{i+1}=p1{i}+Step*(-L1*(f(x{i}) + g(x{i})*(u1{i})+d0{i})+d1{i});
end


figure(1);
x = cell2mat(x);
plot(t,x);
title('States');
legend('x1','x2');
figure(2);
W_c=cell2mat(W_c);
plot(t,W_c);
title('W_c');
legend('Wc1','Wc2','Wc3');

figure(3);
W_a = cell2mat(W_a);
plot(t,W_a);
title('W_a');
legend('Wa1','Wa2','Wa3');

figure(4);
u1 = cell2mat(u1);
plot(t,u1);
title('u');

figure(5);
F = cell2mat(F);
F_hat = cell2mat(F_hat);
subplot(2,1,1)
plot(t,F(1,:),'r',t,F_hat(1,:),'b--');
title('1');
legend('Real','Estimate');
subplot(2,1,2)
plot(t,F(2,:),'r',t,F_hat(2,:),'b--');
title('2');
legend('Real','Estimate');

figure(6);
d0 = cell2mat(d0);
d_ = cell2mat(d_);
subplot(2,1,1);
plot(t,d0(1,:),'r',t,d_(1,:),'b--');
subplot(2,1,2)
plot(t,d0(2,:),'r',t,d_(2,:),'b--');
title('Estimate observer');
legend('Real','Estimate');

figure(7);
plot(t,4.8907+x(1,:),'r');
title('Biomass concentration');

figure(8);
plot(t,0.2187+x(2,:),'r');
title('Substrate concentration');

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
% a=[0 0 1 0;0 0 0 1;0 17.3564 -0.1172 -0.2411;0 67.377 -0.0989 -0.9359]*x;
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
% a=[0;0;14.8452;12.5278];
% A=[0 0 1 0;0 0 0 1;0 17.3564 -0.1172 -0.2411;0 67.377 -0.0989 -0.9359]
% B=[0;0;14.8452;12.5278];
end

% function a = Phi(x)
% %     a = [x(1)^2; x(1)*x(2); x(2)^2; x(1)^4 x(1)^3*x(2) x(1)^2*x(2)^2 x(1)*x(2)^3 x(2)^4];
%         a = [x(1)^2; x(1)*x(2); x(2)^2];
% end

function a = gradPhi(x)
%     a = [2*x(1) 0; x(2) x(1); 0 2*x(2);4*x(1)^3 0;3*x(1)^2*x(2) x(1)^3;2*x(1)*x(2)^2 2*x(1)^2*x(2);x(2)^3 3*x(1)*x(2)^2;0 4*x(2)^3];
      a=[2*x(1) 0;
          x(2) x(1);
          0 2*x(2)];
          
end

function a = funU(lamda,R,u)
R_=diag(R)';
a=2*lamda*atanh(u/lamda)'*R*u+lamda^2*R_*log(1-(u/lamda).^2);
end

function a = funB(lamda,u,m)
a=eye(m)-diag((u/lamda).^2);
end

function a=nhieu(wi,t)
a=0;
for i=1:100
    a=a+0.00001*sin(wi(i)*t);
end
end