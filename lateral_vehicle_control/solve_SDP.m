function [X_ROA,res,dec_var]=solve_SDP(sys,W,bounds,dv,kind)
l_minus=sys.l_minus;
l_plus=sys.l_plus;
n_zf = max(l_plus,l_minus);
sys.n_zf = n_zf;
bias_free=sys.bias_free;
odd=sys.odd;
split=sys.split;
layer_wise=sys.layer_wise;
delta_norm=sys.delta_norm;

bounds=find_effective_bounds(sys,bounds,W);

sys_tot=state_space_realization_tot(sys,W,bounds,kind);
l=length(W)-1; % number hidden layers
[n_xtot,n_utot]=size(sys_tot.B);
B = sys.B;
C = sys.C;
n_u = size(B,2);
[n_y, n_x] = size(C);

n_vec=[n_y];
for ii=1:l
    n_vec=[n_vec,size(W{ii},1)];
end
n_vec=[n_vec, n_u];
n_h=size(W{ii},1);
n=sum(n_vec(2:end-1));

A_tot=sys_tot.A;
B_tot=sys_tot.B;
C_tot=sys_tot.C;
D_tot=sys_tot.D;


alpha = bounds.alpha;
beta = bounds.beta;
alpha_sat = bounds.alpha_sat;
beta_sat = bounds.beta_sat;

% M matrix for off-by-one IQC
M_off = [0, 1;...
    1, 0];

Psi_sec = [beta_sat, -1;...
    -alpha_sat,    1];
% M matrix for sector IQC
M_sec = [0, 1;...
    1, 0];

lambda_sec = sdpvar;
lambda_off = sdpvar;
dec_var = 2;

Pi_sec = lambda_sec*Psi_sec'...
    *M_sec*Psi_sec;

Pi_off = lambda_off*M_off;

M11 = sdpvar(2,2);
dec_var = dec_var + 4;

M_LTI = blkdiag(delta_norm*M11, -1/delta_norm*M11);


%%

lambda = diag(sdpvar(n,1));
dec_var = dec_var + n;
Pi_c = [diag(beta),-eye(n);-diag(alpha),eye(n)]'...
    *[zeros(n) lambda; lambda zeros(n)]*[diag(beta),-eye(n);-diag(alpha),eye(n)];
con = [lambda >= 0, lambda_off >= 0, lambda_sec >= 0, M11 >= 0];

switch kind
    case 'circle'
        Pi=blkdiag(M_LTI,Pi_sec,Pi_off,Pi_c);
    case 'causal_ZF'
        M = sys_tot.M;
        dec_var = dec_var + sys_tot.dec_var;
        Pi_zf = [zeros(n) eye(n);eye(n) zeros(n)];
        %con=[con, (sum(M,2))>=0];
        M_ji=M(:,1:n);
        for kk=1:n_zf
            M_ji=[M_ji; M(:,kk*n+(1:n))];
        end
        if bias_free ==1
            for ii=1:size(M,1)
                for jj=1:size(M,2)
                    if ii~=jj && odd==0
                        con=[con, M(ii,jj) <= 0];
                    elseif ii==jj
                        con=[con, M(ii,jj) >= 0];
                        con=[con, 2*M(ii,jj)>=sum(abs(M(ii,:)))];    % Diagonal dominance
                        con=[con, 2*M(ii,jj)>=sum(abs(M_ji(:,ii)))]; % Diagonal dominance
                    end
                end
            end
        else
            con=[con, sum(M,2)>=0];
            for ii=1:n_zf
                con=[con, M(1:n,ii*n+(1:n))<=0];
            end
        end
        Pi = blkdiag(M_LTI,Pi_sec,Pi_off,Pi_c,Pi_zf);
    case 'acausal_ZF'
        if bias_free == 1
            if layer_wise == 1
                M_var = sdpvar(n_h/split,(l_minus+l_plus+1)*n);
                dec_var = dec_var + n_h/split*n*(l_minus+l_plus+1);
                M_var2 = [];
                for ii = 1:(l_minus+l_plus+1)
                    M_tmp = [];
                    for jj=1:l*split
                        M_tmp = blkdiag(M_tmp,M_var(:,(l*(ii-1)+jj-1)*n_h/split+(1:n_h/split)));
                    end
                    M_var2 =[M_var2, M_tmp];
                end
            elseif layer_wise == 0
                M_var = sdpvar(n,(l_minus+l_plus+1)*n);
                dec_var = dec_var + n^2*(l_minus+l_plus+1);
                M_var2 = M_var;
            end
            M = zeros(n*(n_zf+1),'like',sdpvar);
            M(1:n,1:n) = M_var2(1:n,1:n);
            for ii = 1:l_minus
                M(1:n, ii*n+(1:n)) = M_var2(1:n,ii*n+(1:n));
            end
            for ii = 1:l_plus
                M(ii*n+(1:n), 1:n) = M_var2(1:n,(l_minus+ii)*n+(1:n));
            end
            M_ji = M_var2(:,1:n);
            for kk=1:(l_plus+l_minus)
                M_ji = [M_ji; M_var2(:,kk*n+(1:n))];
            end
            for ii=1:n
                for jj=1:(l_minus+l_plus+1)*n
                    if ii~=jj && odd==0
                        con = [con, M_var2(ii,jj) <= 0];
                    elseif ii==jj
                        con = [con, M_var2(ii,jj) >= 0];
                        con = [con, 2*M_var2(ii,jj) >= sum(abs(M_var2(ii,:)))]; % Diagonal dominance
                        con = [con, 2*M_var2(ii,jj) >= sum(abs(M_ji(:,ii)))];  % Diagonal dominance
                    end
                end
            end
        else
            M_var = sdpvar(n,l_minus+l_plus+1);
            dec_var = dec_var + n*(l_minus+l_plus+1);
            M = zeros(n*(n_zf+1),'like',sdpvar);
            M(1:n,1:n) = diag(M_var(1:n,l_plus+1));
            for ii = 1:l_minus
                M(1:n,ii*n + (1:n)) = diag(M_var(1:n,l_plus+1+ii));
            end
            for ii = 1:l_plus
                M(ii*n + (1:n),1:n) = diag(M_var(1:n,ii));
            end
            for ii=1:l_minus
                con=[con, diag(M_var(1:n,l_plus+1+ii)) <= 0];
            end
            for ii=1:l_plus
                con=[con, diag(M_var(1:n,ii)) <= 0];
            end
            con=[con, sum(M_var,2)>=0];
        end
        Pi_zf = [zeros(n*(n_zf + 1)), M'; M, zeros(n*(n_zf + 1))];
        Pi = blkdiag(M_LTI,Pi_sec,Pi_off,Pi_c,Pi_zf);
end

X = sdpvar(n_xtot,n_xtot,'symmetric');
dec_var = dec_var + n_xtot*(n_xtot+1)/2;
Q = [W{1}*C zeros(size(W{1},1),n_xtot-n_x)];

con2=[];
for ii=1:length(dv{1})
    con2=[con2,[dv{1}(ii)^2, Q(ii,:) ; Q(ii,:)', X]>=0];
end

con3=[eye(n_xtot),zeros(n_xtot,n_utot);A_tot,B_tot]'*[-X,zeros(n_xtot,n_xtot);zeros(n_xtot,n_xtot),X]...
    *[eye(n_xtot),zeros(n_xtot,n_utot);A_tot,B_tot]+[C_tot,D_tot]'*Pi*[C_tot,D_tot];

ops = sdpsettings('solver','mosek','verbose',1,'debug',1);
tic
res=optimize([[con3<=-1e-8,X>=1e-8],con,con2],trace(X(1:n_x,1:n_x)),ops);
toc
res=res.info;

X=value(X);
X_ROA=X(1:n_x,1:n_x);

end

function sys_tot=state_space_realization_tot(sys,W,bounds,kind)

A=sys.A;
B=sys.B;
C=sys.C;
n_zf = sys.n_zf;
bias_free = sys.bias_free;
layer_wise = sys.layer_wise;
split = sys.split;

n_x=size(A,2);% dimension state
n_u=size(B,2); % dimension input to LTI system
n_y=size(C,1); % dimension output of LTI system
l=length(W)-1; % number hidden layers

n_vec = [n_y];
for ii=1:l
    n_vec = [n_vec,size(W{ii},1)];
end
n_vec = [n_vec, n_u];
n_h = size(W{1},1);

n=sum(n_vec(2:end-1));

N1l_1=[zeros(size(W{1},1),n); blkdiag(W{2:l}),zeros(n-size(W{1},1),size(W{l+1},2))];
R_w=[N1l_1;eye(n)];
R_y=[W{1}; zeros(2*n-size(W{1},1),n_y)];
R_u=[zeros(n_u,n-size(W{l+1},2)), W{l+1}];

alpha=bounds.alpha;
beta=bounds.beta;
mu=bounds.mu;
nu=bounds.nu;
mu_sat=bounds.mu_sat;
nu_sat=bounds.nu_sat;

%% multiplier for LTI uncertainty

dt=0.02;

A_LTI = -eye(2);
n_LTI = size(A_LTI,1);
A_LTI = A_LTI*dt + eye(n_LTI);
B_LTI1 = [1; 0];
B_LTI1 = B_LTI1*dt;
B_LTI2 = [0; 1];
B_LTI2 = B_LTI2*dt;
C_LTI = [0,1,0,0;0,0,0,1]';
D_LTI1 = [1,0,0,0]';
D_LTI2 = [0,0,1,0]';
[n_r_LTI,n_LTI] = size(C_LTI);

%% multiplier for saturation

A_sat = 0;
B_sat1 = -nu_sat;
B_sat2 = 1;
C_sat = [0; 0; 1; 0];
D_sat1 = [1; 0; nu_sat; -mu_sat];
D_sat2 = [0; 1; -1; 1];
[n_r_sat,n_sat] = size(C_sat);


%%
switch kind
    case 'circle'
        A_tot = [A, zeros(n_x,n_LTI+n_sat);...
            zeros(n_LTI,n_x), A_LTI, zeros(n_LTI,n_sat);...
            zeros(n_sat,n_x+n_LTI), A_sat];
        B_tot = [zeros(n_x,n),B, B;...
            zeros(n_LTI,n), B_LTI1, B_LTI2;...
            B_sat1*R_u, B_sat2 zeros(n_sat,n_u)];
        C_tot = [zeros(n_r_LTI,n_x), C_LTI, zeros(n_r_LTI,n_sat);...
            zeros(n_r_sat,n_x+n_LTI), C_sat;...
            R_y*C, zeros(2*n,n_LTI+n_sat)];
        D_tot = [zeros(n_r_LTI,n), D_LTI1, D_LTI2;...
            D_sat1*R_u, D_sat2, zeros(n_r_sat,n_u);...
            R_w, zeros(2*n,2*n_u)];
    case 'causal_ZF'
        A_psi = zeros(n_zf*n);
        A_psi(1+n:n_zf*n,1:(n_zf-1)*(n)) = eye((n_zf-1)*n);
        B_psi = [-diag(nu), eye(n);...
            zeros((n_zf-1)*n,2*n)];
        if bias_free == 1 && layer_wise == 1
            M_var = sdpvar(n_h/split,n*(n_zf+1));
            dec_var = n_h/split*n*(n_zf+1);
            M = [];
            for ii = 1:n_zf+1
                M_tmp = [];
                for jj=1:l*split
                    M_tmp = blkdiag(M_tmp,M_var(:,(l*(ii-1)+jj-1)*n_h/split+(1:n_h/split)));
                end
                M =[M, M_tmp];
            end
        elseif bias_free == 1 && layer_wise == 0
            M_var = sdpvar(n,n*(n_zf+1));
            dec_var = n^2*(n_zf+1);
            M = M_var;
        elseif bias_free == 0 && layer_wise == 1
            M_var = sdpvar(l*split,n_zf+1);
            dec_var = l*split*(n_zf+1);
            M = [];
            for ii = 1:n_zf+1
                M_tmp = [];
                for jj=1:l*split
                    M_tmp = blkdiag(M_tmp,M_var(jj,ii)*eye(n_h/split));
                end
                M =[M, M_tmp];
            end
        else
            M_var = sdpvar(n,(n_zf+1));
            dec_var = n*(n_zf+1);
            for ii=1:n_zf+1
                M(1:n,n*(ii-1)+(1:n)) = diag(M_var(:,ii));
            end
        end
        C_psi=[];
        for ii=1:n_zf
            C_psi = [C_psi, [-M(:,ii*n+(1:n));zeros(n,n)]];
        end
        C_psi = [zeros(2*n,n_zf*n);... % this is for the circle criterion
            C_psi];
        D_psi = [eye(2*n);... % this is for the circle criterion
            M(:,1:n)*diag(nu) -M(:,1:n)*eye(n);...
            -diag(mu) eye(n)];
        sys_tot.M = M;
        sys_tot.dec_var = dec_var;
    case 'acausal_ZF'
        A_psi=[zeros(n,2*n*(n_zf));...
            eye((n_zf-1)*n), zeros((n_zf-1)*n, (n_zf+1)*n);...
            zeros(n,2*n*(n_zf));...
            zeros((n_zf-1)*n, n_zf*n), eye((n_zf-1)*n), zeros((n_zf-1)*n, n)];
        
        B_psi=[eye(n), zeros(n,n); ...
            zeros((n_zf-1)*n,2*n); ...
            zeros(n,n), eye(n); ...
            zeros((n_zf-1)*n,2*n)];
        
        C_psi=[zeros(2*n,2*n_zf*n);... % this is for the circle criterion
            zeros(n,2*n_zf*n);...
            kron(eye(n_zf),diag(nu)), -eye((n_zf)*n);...
            zeros(n,2*(n_zf)*n);...
            -kron(eye(n_zf),diag(mu)), eye((n_zf)*n)];
        
        D_psi=[eye(2*n);... % this is for the circle criterion
            diag(nu), -eye(n);...
            zeros((n_zf)*n,2*n);...
            -diag(mu), eye(n);...
            zeros((n_zf)*n,2*n)];
    otherwise
        error('Unexpected kind.')
end

switch kind
    case 'circle'
    otherwise
        [n_r_psi, n_psi] = size(C_psi);
        
        A_tot = [A, zeros(n_x,n_LTI+n_sat+n_psi);...
            zeros(n_LTI,n_x), A_LTI, zeros(n_LTI,n_sat+n_psi);...
            zeros(n_sat,n_x+n_LTI), A_sat, zeros(n_sat,n_psi);...
            B_psi*R_y*C, zeros(n_psi,n_LTI+n_sat), A_psi];
        B_tot = [zeros(n_x,n),B, B;...
            zeros(n_LTI,n), B_LTI1, B_LTI2;...
            B_sat1*R_u, B_sat2 zeros(n_sat,n_u);...
            B_psi*R_w,zeros(n_psi,2*n_u)];
        C_tot = [zeros(n_r_LTI,n_x), C_LTI, zeros(n_r_LTI,n_sat+n_psi);...
            zeros(n_r_sat,n_x+n_LTI), C_sat, zeros(n_r_sat,n_psi);...
            D_psi*R_y*C, zeros(n_r_psi,n_LTI+n_sat), C_psi];
        D_tot = [zeros(n_r_LTI,n), D_LTI1, D_LTI2;...
            D_sat1*R_u, D_sat2, zeros(n_r_sat,n_u);...
            D_psi*R_w, zeros(n_r_psi,2*n_u)];
end

sys_tot.A=A_tot;
sys_tot.B=B_tot;
sys_tot.C=C_tot;
sys_tot.D=D_tot;
end

function bounds = find_effective_bounds(sys,bounds,W)
bias_free = sys.bias_free;
layer_wise = sys.layer_wise;
split = sys.split;
n_h=size(W{1},1);
l=length(W)-1;

mu_phi=bounds.mu_phi;
nu_phi=bounds.nu_phi;
alpha_phi=bounds.alpha_phi;
beta_phi=bounds.beta_phi;

alpha=[];
beta=[];
for ii=1:l
    alpha=[alpha,alpha_phi{ii}];
    beta=[beta,beta_phi{ii}];
end

mu=[];
nu=[];
for ii=1:l
    mu=[mu,mu_phi{ii}];
    nu=[nu,nu_phi{ii}];
end


if bias_free == 1 && layer_wise == 1
    l_rep = n_h/split;
    n_split = length(mu)/l_rep;
    for ii = 1:n_split
        mu(l_rep*(ii-1)+(1:l_rep))=min(mu(l_rep*(ii-1)+(1:l_rep)))*ones(size(mu(l_rep*(ii-1)+(1:l_rep))));
        nu(l_rep*(ii-1)+(1:l_rep))=max(nu(l_rep*(ii-1)+(1:l_rep)))*ones(size(nu(l_rep*(ii-1)+(1:l_rep))));
    end
elseif bias_free == 1 && layer_wise == 0
    mu = min(mu)*ones(size(mu));
    nu = max(nu)*ones(size(nu));
end

bounds.alpha = alpha;
bounds.beta = beta;
bounds.mu = mu;
bounds.nu = nu;
end

