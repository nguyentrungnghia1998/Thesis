%% Simulation of novel actor-critic-identifier - Bhashin 2013
clear; close all; clc;
%% Time interval and simulation time
Step = 0.001;T_end = 10;
t = 0:Step:T_end;
%% Variables
x = cell(1,size(t,2));
u = cell(1,size(t,2));
J = cell(1,size(t,2));
r = cell(1,size(t,2));

%% Parameters
% important parameter: nuy, min(eig)

Q = diag([4 3]); R = eye(1);
lamda=2;
%% Initial conditions
x{1} = [-1;0.5];
Wa0=[2.3642;2.2522;1.3138];
WaT=[2.936;1.7126;2.2839];
J{1}=0;
%% System simulation
for i=1:size(t,2)
    %% Compute control input
    u{i} = -lamda*tanh(1/2/lamda*pinv(R)*g(x{i})'*gradPhi(x{i})'*Wa0);
    %% Compute object fuction
    r{i}=x{i}'*Q*x{i}+2*funU(lamda,R,u{i});
    
    if i==size(t,2)
        break
    end
    %% Update new states
    x{i+1} = x{i} + Step*(f(x{i}) + g(x{i})*(u{i}));
    J{i+1} = J{i} + Step*r{i};
end
Jmax=J{10001};
J=cell2mat(J);
J=Jmax-J;

figure(1);
plot(t,J);
hold on;

%% Variables
x = cell(1,size(t,2));
u = cell(1,size(t,2));
J = cell(1,size(t,2));
r = cell(1,size(t,2));
%% Parameters
% important parameter: nuy, min(eig)

Q = diag([4 3]); R = eye(1);
lamda=2;
%% Initial conditions
x{1} = [-1;0.5];
Wa0=[2.3642;2.2522;1.3138];
WaT=[2.936;1.7126;2.2839];
J{1}=0;
%% System simulation
for i=1:size(t,2)
    %% Compute control input
    u{i} = -lamda*tanh(1/2/lamda*pinv(R)*g(x{i})'*gradPhi(x{i})'*WaT);
    %% Compute object fuction
    r{i}=x{i}'*Q*x{i}+2*funU(lamda,R,u{i});
    
    if i==size(t,2)
        break
    end
    %% Update new states
    x{i+1} = x{i} + Step*(f(x{i}) + g(x{i})*(u{i}));
    J{i+1} = J{i} + Step*r{i};
end
Jmax=J{10001};
J=cell2mat(J);
J=Jmax-J;
plot(t,J);
title('Object function before and after ADP');
legend('Before','After');

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

function a = funU(lamda,R,u)
R_=diag(R)';
a=2*lamda*atanh(u/lamda)'*R*u+lamda^2*R_*log(1-(u/lamda).^2);
end

function a = gradPhi(x)
      a=[2*x(1) 0;
          x(2) x(1);
          0 2*x(2)];        
end