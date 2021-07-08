%% Simulation of Thesis
clear; close all; clc;
%% Time interval and simulation time
Step = 0.001;T_end = 40;
t = 0:Step:T_end;
%% Boquansatbac1
x = cell(1,size(t,2));
u = cell(1,size(t,2));
d0=cell(1,size(t,2));
d_=cell(1,size(t,2));
d=cell(1,size(t,2));
%% Parameters
L1=1.57;
%% Initial conditions
x{1} = [-1;0.5];
p1{1}=[0;0];
%% System simulation
for i=1:size(t,2)
    %% Noise for PE condition
    d{i}=0.1*(sin(t(i))^2*cos(t(i))+sin(2*t(i))^2*cos(0.1*t(i))+sin(-1.2*t(i))^2*cos(0.5*t(i))+sin(t(i))^5);
    %% Estimated disturbance
    d0{i}=p1{i}+L1*x{i};
    d_{i}= pinv(g(x{i}))*d0{i};
    %% Compute control input
    u{i}=[0.6734 -0.7676]*x{i};
    if i==size(t,2)
        break
    end
    %% Update new states
    x{i+1} = x{i} + Step*(f(x{i}) + g(x{i})*(u{i}+d{i}));
    p1{i+1}=p1{i}+Step*(-L1*(f(x{i}) + g(x{i})*(u{i})+d0{i}));
end

figure(1);
d=cell2mat(d);
d_=cell2mat(d_);
plot(t,d,t,d_);
hold on;
title('Output disturbance observer');

figure(2);
plot(t,d_-d);
title('Error observer');
hold on;

%% Boquansatbac2
x = cell(1,size(t,2));
u = cell(1,size(t,2));
d0=cell(1,size(t,2));
d1=cell(1,size(t,2));
d_=cell(1,size(t,2));
d=cell(1,size(t,2));
%% Parameters
L1=18.76;
L2=54.93;
%% Initial conditions
x{1} = [-1;0.5];
p1{1}=[0;0];
p2{1}=[0;0];
%% System simulation
for i=1:size(t,2)
    %% Noise for PE condition
    d{i}=0.1*(sin(t(i))^2*cos(t(i))+sin(2*t(i))^2*cos(0.1*t(i))+sin(-1.2*t(i))^2*cos(0.5*t(i))+sin(t(i))^5);
    %% Estimated disturbance
    d0{i}=p1{i}+L1*x{i};
    d1{i}=p2{i}+L2*x{i};
    d_{i}= pinv(g(x{i}))*d0{i};
    %% Compute control input
    u{i}=[0.6734 -0.7676]*x{i};
    if i==size(t,2)
        break
    end
    %% Update new states
    x{i+1} = x{i} + Step*(f(x{i}) + g(x{i})*(u{i}+d{i}));
    p1{i+1}=p1{i}+Step*(-L1*(f(x{i}) + g(x{i})*(u{i})+d0{i})+d1{i});
    p2{i+1}=p2{i}+Step*(-L2*(f(x{i}) + g(x{i})*(u{i})+d0{i}));
end

figure(1);
d=cell2mat(d);
d_=cell2mat(d_);
plot(t,d_,'g');
figure(2);
plot(t,d_-d);

%% Boquansatbac2
x = cell(1,size(t,2));
u = cell(1,size(t,2));
d0=cell(1,size(t,2));
d1=cell(1,size(t,2));
d2=cell(1,size(t,2));
d_=cell(1,size(t,2));
d=cell(1,size(t,2));
%% Parameters
L1=14.2;
L2=217.1;
L3=716.3;
%% Initial conditions
x{1} = [-1;0.5];
p1{1}=[0;0];
p2{1}=[0;0];
p3{1}=[0;0];
%% System simulation
for i=1:size(t,2)
    %% Noise for PE condition
    d{i}=0.1*(sin(t(i))^2*cos(t(i))+sin(2*t(i))^2*cos(0.1*t(i))+sin(-1.2*t(i))^2*cos(0.5*t(i))+sin(t(i))^5);
    %% Estimated disturbance
    d0{i}=p1{i}+L1*x{i};
    d1{i}=p2{i}+L2*x{i};
    d2{i}=p3{i}+L3*x{i};
    d_{i}= pinv(g(x{i}))*d0{i};
    %% Compute control input
    u{i}=[0.6734 -0.7676]*x{i};
    if i==size(t,2)
        break
    end
    %% Update new states
    x{i+1} = x{i} + Step*(f(x{i}) + g(x{i})*(u{i}+d{i}));
    p1{i+1}=p1{i}+Step*(-L1*(f(x{i}) + g(x{i})*(u{i})+d0{i})+d1{i});
    p2{i+1}=p2{i}+Step*(-L2*(f(x{i}) + g(x{i})*(u{i})+d0{i})+d2{i});
    p3{i+1}=p3{i}+Step*(-L3*(f(x{i}) + g(x{i})*(u{i})+d0{i}));
end

figure(1);
d=cell2mat(d);
d_=cell2mat(d_);
plot(t,d_,'m');
legend('Actual disturbance','First Order','Second Order','Third Order');
figure(2);
plot(t,d_-d);
legend('First Order','Second Order','Third Order');

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
