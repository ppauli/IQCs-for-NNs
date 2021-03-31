%close all
clear all
%clc

%% Load NN
W{1} = importdata('Wb_s32_tanh/W1.csv');
W{2} = importdata('Wb_s32_tanh/W2.csv');
W{3} = importdata('Wb_s32_tanh/W3.csv');

%W{1} = ones(2,4);
%W{2} = ones(2,2);
%W{3} = ones(1,2);

b{1} = zeros(size(W{1},1),1);
b{2} = zeros(size(W{2},1),1);
b{3} = zeros(size(W{3},1),1);

%% parameters
% Nominal speed of the vehicle travels at.
U = 28; % m/s
% Model
% Front cornering stiffness for one wheel.
Ca1 = -61595; % unit: Newtons/rad
% Rear cornering stiffness for one wheel.
Ca3 = -52095; % unit: Newtons/rad

% Front cornering stiffness for two wheels.
Caf = Ca1*2; % unit: Newtons/rad
% Rear cornering stiffness for two wheels.
Car = Ca3*2; % unit: Newtons/rad

% Vehicle mass
m = 1670; % kg
% Moment of inertia
Iz = 2100; % kg/m^2

% Distance from vehicle CG to front axle
a = 0.99; % m
% Distance from vehicle CG to rear axle
bd = 1.7; % m

g = 9.81;

% sampling period
dt = 0.02;

% Continuous-time state space matrices
% States are lateral displacement(e) and heading angle error(deltaPsi) and
% their derivatives.
% Inputs are front wheel angle and curvature of the road.
Ac = [0 1 0 0; ...
    0, (Caf+Car)/(m*U), -(Caf+Car)/m, (a*Caf-bd*Car)/(m*U); ...
    0 0 0 1; ...
    0, (a*Caf-bd*Car)/(Iz*U), -(a*Caf-bd*Car)/Iz, (a^2*Caf+bd^2*Car)/(Iz*U)];
B1c = [0;
    -Caf/m;
    0; ...
    -a*Caf/Iz];
% discribes how the curvature enters the system.
B2c = [0;
    (a*Caf-bd*Car)/m-U^2;
    0; ...
    (a^2*Caf+bd^2*Car)/Iz];

%% x^+ = AG*x + BG1*q + BG2*usat
sys.A = Ac*dt + eye(4);

% describes how q enters the system
sys.B = B1c*dt;
% describes how usat enters the system
B2 = B1c*dt;
n_x = size(sys.A, 1);
nu = 1;
nq = 1;

% the input to Delta: p = usat
xs = zeros(n_x,1);
sys.C = eye(n_x);

%% Find local sector bounds alpha and beta and local slope bounds mu and nu
% mu<=alpha<beta<=nu
n_h = size(W{1},1);
l = length(W)-1; % number hidden layers
dv{1} = 0.7*ones(n_h,1); % Delta v^0 (given)
umax = 30/180*pi;


sys.delta_norm = 0.1;
sys.l_minus = 1;
sys.l_plus = 1;
sys.bias_free = 0;
sys.odd = 1;
sys.layer_wise = 0; % layer_wise=1 and bias-free=0 leads to one instead of n_h multipliers per layer
sys.split = 2;

d1_l=0.3;
d1_u=1.4;

[mu_phi,nu_phi,mu_sat]=find_slope_bounds(W,b,xs,dv,umax);
[alpha_phi,beta_phi,alpha_sat]=find_sector_bounds(W,b,xs,dv,umax);

bounds.mu_sat=mu_sat;
bounds.nu_sat=1;
bounds.alpha_sat=alpha_sat;
bounds.beta_sat=1;

bounds.alpha_phi=alpha_phi;
bounds.beta_phi=beta_phi;
bounds.mu_phi=mu_phi;
bounds.nu_phi=nu_phi;

%[d1,d1_vec,traceX]=bisection(sys,W,b,xs,'circle',d1_l,d1_u,umax);
%[d1_ZF,d1_vec_ZF,traceX_ZF]=bisection(sys,W,b,xs,'causal_ZF',d1_l,d1_u,umax);
%[d1_acZF,d1_vec_acZF,traceX_acZF]=bisection(sys,W,b,xs,'acausal_ZF',d1_l,d1_u,umax);

%[d1,traceX,traceX_lb]=convex_linesearch(sys,W,b,xs,'circle',d1_l,d1_u,umax)
%[d1_ZF,traceX_ZF,traceX_lb_ZF]=convex_linesearch(sys,W,b,xs,'causal_ZF',d1_l,d1_u,umax)
%[d1_aZF,traceX_aZF,traceX_lb_aZF]=convex_linesearch(sys,W,b,xs,'acausal_ZF',d1_l,d1_u,umax)

[X,res,dec_var]=solve_SDP(sys,W,bounds,dv,'circle');
[X_c,res_c,dec_var_ZF]=solve_SDP(sys,W,bounds,dv,'causal_ZF');
%[X_ac,res_ac,dec_var_acZF]=solve_SDP(sys,W,bounds,dv,'acausal_ZF');


%radii = 1./sqrt(eig(X(1:n_x,1:n_x)))
%traceP = trace(X(1:n_x,1:n_x))
%volume = det(inv(X(1:n_x,1:n_x)))