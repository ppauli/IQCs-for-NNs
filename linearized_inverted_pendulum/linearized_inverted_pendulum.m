close all
clear all
clc

%% Load system dynamics and NN controller

load('inv_pend_MPC_bias_free.mat') % this gives the linearized inverted pendulum and an NN with n_1=n_2=32
n_x=size(sys.A,2);

%% Load NN
%W{1} = importdata('Wb_s32_tanh/W1.csv');
%W{2} = importdata('Wb_s32_tanh/W2.csv');
%W{3} = importdata('Wb_s32_tanh/W3.csv');

%b{1} = zeros(size(W{1},1),1);
%b{2} = zeros(size(W{2},1),1);
%b{3} = zeros(size(W{3},1),1);


% Find equilibrium state xs
x0=randn(n_x,1);
sim_time=1000;
[star,x_vec]=simulate_system(W,b,sys,x0,sim_time); % Checking by simulation what the equilibrium point is
xs=zeros(n_x,1);


%% Find local sector bounds alpha and beta and local slope bounds mu and nu
% mu<=alpha<beta<=nu
n_h=size(W{1},1);
l=length(W)-1; % number hidden layers

%%
sys2.A=sys.A;
sys2.B = sys.B;
sys2.C = eye(n_x);
sys2.odd = 1;        % indicate if activation is odd
sys2.split = 1;      % Splits down Zames-Falb multipliers in <split> blocks per layer
sys2.full_block = 0; % Full block circle and CY multipliers

dv_vec=[0.2:0.1:1.4];

for ii=1:length(dv_vec)
    ii/length(dv_vec)
    
    dv{1}=dv_vec(ii)*ones(n_h,1);
    
    [mu_phi,nu_phi]=find_slope_bounds(W,b,xs,dv,'tanh');
    [alpha_phi,beta_phi]=find_sector_bounds(W,b,xs,dv,'tanh');
    
    bounds.alpha_phi=alpha_phi;
    bounds.beta_phi=beta_phi;
    bounds.mu_phi=mu_phi;
    bounds.nu_phi=nu_phi;
    
    sys2.layer_wise = 0;
    sys2.l_plus = 1;
    sys2.l_minus = 1;
    sys2.bias_free = 0;
    
    [X{1}{ii},res{1}{ii},time(1,ii)] = solve_SDP(sys2,W,bounds,dv,'circle');
    sys2.full_block = 1;
    [X{2}{ii},res{2}{ii},time(2,ii)] = solve_SDP(sys2,W,bounds,dv,'circle');
    
    sys2.full_block = 0;    
    sys2.bias_free = 1;
    sys2.layer_wise = 1;
    
    [X{3}{ii},res{3}{ii},time(3,ii)] = solve_SDP(sys2,W,bounds,dv,'causal_ZF');
    [X{4}{ii},res{4}{ii},time(4,ii)] = solve_SDP(sys2,W,bounds,dv,'acausal_ZF');
    
    sys2.layer_wise=0;
    
    [X{5}{ii},res{5}{ii},time(5,ii)] = solve_SDP(sys2,W,bounds,dv,'causal_ZF');
    [X{6}{ii},res{6}{ii},time(6,ii)] = solve_SDP(sys2,W,bounds,dv,'acausal_ZF');
    
    sys2.l_plus = 2;
    sys2.l_minus = 2;
    sys2.bias_free = 0;
    
    [X{7}{ii},res{7}{ii},time(7,ii)] = solve_SDP(sys2,W,bounds,dv,'causal_ZF');
    [X{8}{ii},res{8}{ii},time(8,ii)] = solve_SDP(sys2,W,bounds,dv,'acausal_ZF');

    sys2.bias_free = 1;
    sys2.layer_wise = 1;
    
    [X{9}{ii},res{9}{ii},time(9,ii)] = solve_SDP(sys2,W,bounds,dv,'causal_ZF');
    [X{10}{ii},res{10}{ii},time(10,ii)] = solve_SDP(sys2,W,bounds,dv,'acausal_ZF');
    
    sys2.layer_wise = 0;
    
    [X{11}{ii},res{11}{ii},time(11,ii)] = solve_SDP(sys2,W,bounds,dv,'causal_ZF');
    [X{12}{ii},res{12}{ii},time(12,ii)] = solve_SDP(sys2,W,bounds,dv,'acausal_ZF');
    
    sys2.layer_wise = 1;
    
    [X{13}{ii},res{13}{ii},time(13,ii)] = solve_SDP(sys2,W,bounds,dv,'causal_ZF');
    [X{14}{ii},res{14}{ii},time(14,ii)] = solve_SDP(sys2,W,bounds,dv,'acausal_ZF');
end

%%
trace_X=zeros(size(X,2),size(X{1},2));
for ii=1:size(X{1},2)
    for jj=1:size(X,2)
        if strcmp(res{jj}{ii},'Successfully solved (MOSEK)')
            trace_X(jj,ii)=trace(X{jj}{ii}(1:n_x,1:n_x));
        else
            trace_X(jj,ii)=NaN;
        end
    end
end

figure
plot(dv_vec,trace_X')
legend('diag-C','FB-C','cZF-1','acZF-1','cZF-1-RL','acZF-1-RL','cZF-1-R','acZF-1-R','cZF-2','acZF-2','cZF-2-RL','acZF-2-RL','cZF-2-R','acZF-2-R')

save('res.mat')