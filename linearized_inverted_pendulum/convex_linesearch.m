function [d1,traceX,traceX_lb]=convex_linesearch(sys,W,b,xs,kind,d1_l,d1_u,act)

opt_fun = @(d1) d1_to_traceX(d1,sys,W,b,xs,kind,act);
itter = 25; %number of iterations

[traceX, d1, traceX_lb] = convexLineSearch(opt_fun,d1_l,d1_u,itter);
end

function traceX = d1_to_traceX(d1,sys,W,b,xs,kind,act)
n_h=size(W{1},1);
dv{1} = d1*ones(n_h,1);
[mu_phi,nu_phi]=find_slope_bounds(W,b,xs,dv,act);
[alpha_phi,beta_phi]=find_sector_bounds(W,b,xs,dv,act);

bounds.alpha_phi=alpha_phi;
bounds.beta_phi=beta_phi;
bounds.mu_phi=mu_phi;
bounds.nu_phi=nu_phi;

[X,res]=solve_SDP(sys,W,bounds,dv,kind);
if strcmp(res,'Successfully solved (MOSEK)')
    traceX=trace(X);
else
    traceX = inf;
end
end

function [f_min, x_min,lower_bound] = convexLineSearch(f,x_a,x_b,itter)
golden_section = (sqrt(5)-1)/2;
f_b = f(x_b);
f_a = f(x_a);

x_1 = x_b - golden_section*(x_b - x_a);
f_1 = f(x_1);

lower_bound = calc_convex_lowerbound_3(x_a,x_1,x_b,f_a,f_1,f_b);
for jj = 1:itter
    x_2 = select_golden(x_a,x_1,x_b,f_a,f_1,f_b);
    f_2 = f(x_2);
    
    lower_bound = max(lower_bound,calc_convex_lowerbound_4(x_a,x_1,x_2,x_b,f_a,f_1,f_2,f_b));
    if f_2 < f_1
        if x_1 < x_2
            x_a = x_1;
            f_a = f_1;
        else
            x_b = x_1;
            f_b = f_1;
        end
        x_1 = x_2;
        f_1 = f_2;
    else
        if x_1 < x_2
            x_b = x_2;
            f_b = f_2;
        else
            x_a = x_2;
            f_a = f_2;
        end
    end
end
f_min = f_1;
x_min = x_1;
end






function x_new = select_quadratic(x_1,x_2,x_3,f_1,f_2,f_3)
VM = [x_1^2 x_1 1; x_2^2 x_2 1; x_3^2 x_3 1];
c = VM\[f_1;f_2;f_3];
x_new = -c(2)/(2*c(1));
end

function x_new = select_golden(x_1,x_2,x_3,f_1,f_2,f_3)
golden_section = (sqrt(5)-1)/2;
if (x_2 - x_1) < (x_3 - x_2)
    x_new = x_3 - golden_section*(x_3 - x_2);
else
    x_new = x_1 + golden_section*(x_2 - x_1);
end
end

function lb = calc_convex_lowerbound_3(x_1,x_2,x_3,f_1,f_2,f_3)
lb = min(f_2 + (f_1 - f_2)/(x_1 - x_2)*(x_3 - x_1),f_2 + (f_3 - f_2)/(x_3 - x_2)*(x_1 - x_2));
end
function lb = calc_convex_lowerbound_4(x_1,x_2,x_3,x_4,f_1,f_2,f_3,f_4)
if x_2 < x_1
    tmp = x_1;
    x_1 = x_2;
    x_2 = tmp;
    tmp = f_1;
    f_1 = f_2;
    f_2 = tmp;
end
lb_1 = f_2 + (f_3 - f_2)/(x_3 - x_2)*(x_1 - x_2);
lb_2 = max(f_2 + (f_1 - f_2)/(x_1 - x_2)*(x_3 - x_1),f_3 + (f_4 - f_3)/(x_4 - x_3)*(x_2 - x_4));
lb_3 = f_3 + (f_3 - f_2)/(x_3 - x_2)*(x_4 - x_3);
lb = min([lb_1,lb_2,lb_3]);
end