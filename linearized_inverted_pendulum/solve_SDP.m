function [X_ROA,res,time,sys_tot,dec_var]=solve_SDP(sys,W,bounds,dv,kind)
l_minus=sys.l_minus;
l_plus=sys.l_plus;
n_zf = max(l_plus,l_minus);
sys.n_zf = n_zf;
bias_free=sys.bias_free;
odd=sys.odd;
split=sys.split;
layer_wise=sys.layer_wise;
full_block=sys.full_block;

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
mu = bounds.mu;
nu = bounds.nu;

%%

lambda = diag(sdpvar(n,1));
dec_var = n;
if full_block == 0
    Pi_c = [diag(beta),-eye(n);-diag(alpha),eye(n)]'...
        *[zeros(n) lambda; lambda zeros(n)]*[diag(beta),-eye(n);-diag(alpha),eye(n)];
    con = [lambda >= 0];
elseif full_block == 1
    Pi_c = sdpvar(2*n,2*n,'symmetric');
    dec_var = 2*n*(2*n+1)/2;
    con = build_fullblock_constraints(alpha,beta,[],Pi_c);
    %con = [con,Pi_c(n+(1:n),n+(1:n))<=0];
    for ii=1:n
        con=[con, Pi_c(n+ii,n+ii)<=0];
    end
end


switch kind
    case 'circle'
        Pi=Pi_c;
    case 'circle_Yacubovich'
        if full_block == 0
            clear Pi_c
            lambda2 = diag(sdpvar(n,1)); 
            dec_var = dec_var + n;
            con = [con,lambda2 >= 0];
            Pi_cy = [diag([beta,nu]),-eye(2*n);-diag([alpha,mu]),eye(2*n)]'...
                *[zeros(2*n) blkdiag(lambda,lambda2); blkdiag(lambda,lambda2) zeros(2*n)]...
                *[diag([beta,nu]),-eye(2*n);-diag([alpha,mu]),eye(2*n)];
        elseif full_block == 1
            clear con Pi_c
            Pi_cy = sdpvar(4*n,4*n,'symmetric');
            dec_var = (4*n*(4*n+1))/2;
            con = build_fullblock_constraints([alpha,mu],[beta,nu],[],Pi_cy);
            %con = [con,Pi_cy(2*n+(1:2*n),2*n+(1:2*n))<=0];
            for ii=1:2*n
                con=[con, Pi_cy(2*n+ii,2*n+ii)<=0];
            end
        end
        Pi = Pi_cy;
    case 'causal_ZF'
        M = sys_tot.M;
        dec_var = dec_var + sys_tot.dec_var;
        Pi_zf = [zeros(n) eye(n);eye(n) zeros(n)];
        %con=[con, (sum(M,2))>=0];
        M_ji=M(:,1:n);
        for kk=1:n_zf
            M_ji=[M_ji; M(:,kk*n+(1:n))];
        end
        if bias_free==1
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
        Pi = blkdiag(Pi_c,Pi_zf);
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
        Pi = blkdiag(Pi_c,Pi_zf);
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
res=optimize([[con3<=-10e-10,X>=10e-8],con,con2],trace(X(1:n_x,1:n_x)),ops);
toc
res=res.info;
X=value(X);
X_ROA=X(1:n_x,1:n_x);
time=toc;

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

n_vec=[n_y];
for ii=1:l
    n_vec=[n_vec,size(W{ii},1)];
end
n_vec=[n_vec, n_u];
n_h=size(W{ii},1);

n=sum(n_vec(2:end-1));

N1l_1=[zeros(size(W{1},1),n); blkdiag(W{2:l}),zeros(n-size(W{1},1),size(W{l+1},2))];
R_w=[N1l_1;eye(n)];
R_y=[W{1}; zeros(2*n-size(W{1},1),n_y)];
R_u=[zeros(n_u,n-size(W{l+1},2)), W{l+1}];

alpha=bounds.alpha;
beta=bounds.beta;
mu=bounds.mu;
nu=bounds.nu;

%%
switch kind
    case 'circle'
        A_tot = A;
        B_tot = B*R_u;
        C_tot = R_y*C;
        D_tot = R_w;
    case 'circle_Yacubovich'
        A_psi=[zeros(2*n,2*n)];
        B_psi=[eye(n), zeros(n,n); ...
            zeros(n,n), eye(n)];
        C_psi=[zeros(n,2*n);...
            -eye(n), zeros(n,n);...
            zeros(n,2*n);...
            zeros(n,n),-eye(n)];
        D_psi=[eye(n), zeros(n,n);...
            eye(n),zeros(n,n);...
            zeros(n,n),eye(n);...
            zeros(n,n),eye(n)];
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
        n_xi=size(A_psi,1);
        
        A_tot=[A, zeros(n_x,n_xi);...
            B_psi*R_y*C, A_psi];
        B_tot=[B*R_u;...
            B_psi*R_w];
        C_tot=[D_psi*R_y*C, C_psi];
        D_tot=[D_psi*R_w];
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

function R = build_fullblock_constraints(alpha,beta,v,Pi)

lv = length(v);
if lv < length(alpha)
    R=[...
    build_fullblock_constraints(alpha,beta,[v,alpha(lv+1)],Pi),...
    build_fullblock_constraints(alpha,beta,[v,beta(lv+1)],Pi)];
else
    R = ([eye(lv);diag(v)]'*Pi*[eye(lv);diag(v)] >= 0);
end

end