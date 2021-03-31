close all
clear all
clc

mu=0.1;
nu=1;

n_zf=1; % max(n_b,n_f) order of FIR filter
n_x=2; % dimension state
n_u=1; % dimension input to LTI system
n_y=1; % dimension output of LTI system
n_h=2; % dimension of hidden layers
l=1; % number hidden layers

n_vec=[n_y];
for ii=1:l
    n_vec=[n_vec,n_h];
end
n_vec=[n_vec, n_u];

n=sum(n_vec(2:end-1));

nu=ones(n,1)*nu;
mu=ones(n,1)*mu;

A_psi=[zeros(n,2*n*(n_zf+1));...
    eye(n_zf*n), zeros(n_zf*n, (n_zf+2)*n);...
    zeros(n,2*n*(n_zf+1));...
    zeros(n_zf*n, (n_zf+1)*n), eye(n_zf*n), zeros(n_zf*n, n)];

B_psi=[eye(n), zeros(n,n); ...
    zeros(n_zf*n,2*n); ...
    zeros(n,n), eye(n); ...
    zeros(n_zf*n,2*n)];

C_psi=[eye(n), zeros(n,n*(2*n_zf+1));...
    -eye(n), zeros(n,n*(2*n_zf+1));...
    zeros(n,(n_zf+1)*n), eye(n), zeros(n,n_zf*n);...
    zeros(n,(n_zf+1)*n), -eye(n), zeros(n,n_zf*n);...
    zeros(n,2*(n_zf+1)*n);...
    kron(eye(n_zf+1),diag(nu)), -eye((n_zf+1)*n);...
    zeros(n,2*(n_zf+1)*n);...
    -kron(eye(n_zf+1),diag(mu)), eye((1+n_zf)*n)];

D_psi=[zeros(n,2*n);...
    eye(n),zeros(n,n);...
    zeros(n,2*n);
    zeros(n,n), eye(n);...
    diag(nu), -eye(n);...
    zeros((n_zf+1)*n,2*n);...
    -diag(mu), eye(n);...
    zeros((n_zf+1)*n,2*n)];

n_xi=size(A_psi,1);

%%
A=randn(n_x,n_x);
B=randn(n_x,n_u);
C=randn(n_y,n_x);
%load('NN_inv_pendulum_bias_free_3.mat')
%A=sys.A;
%B=sys.B;
%C=sys.C;

[W,b]=generate_random_NN(n_vec);

N1l_1=[zeros(n_h,n); blkdiag(W{2:l}),zeros(n-n_h,n_h)];
R_w=[N1l_1;eye(n)];
R_y=[W{1}; zeros(2*n-n_h,n_y)];
R_v=[zeros(n_u,n-n_h), W{l+1}];

A_tot=[A, zeros(n_x,n_xi);...
    B_psi*R_y*C, A_psi];
B_tot=[B*R_v;...
    B_psi*R_w];
C_tot=[D_psi*R_y*C, C_psi];
D_tot=[D_psi*R_w];

