function [d1,d1_vec,traceX]=bisection(sys,W,b,xs,kind,d1_l,d1_u,umax)

eps=0.01;
n_h=size(W{1},1);
d1=d1_l;
ii=1;

while abs(d1_u-d1_l)>eps
    d1=(d1_l+d1_u)/2;
    dv{1} = d1*ones(n_h,1);
    [mu_phi,nu_phi,mu_sat]=find_slope_bounds(W,b,xs,dv,umax);
    [alpha_phi,beta_phi,alpha_sat]=find_sector_bounds(W,b,xs,dv,umax);
    
    bounds.alpha_phi=alpha_phi;
    bounds.beta_phi=beta_phi;
    bounds.mu_phi=mu_phi;
    bounds.nu_phi=nu_phi;
    
    bounds.mu_sat=mu_sat;
    bounds.nu_sat=1;
    bounds.alpha_sat=alpha_sat;
    bounds.beta_sat=1;
    
    [X,res]=solve_SDP(sys,W,bounds,dv,kind);
    if strcmp(res,'Successfully solved (MOSEK)')
        d1_l=d1;
        traceX(ii)=trace(X);
        d1_vec(ii)=d1;
        ii=ii+1;    
    else
        d1_u=d1;
    end
end