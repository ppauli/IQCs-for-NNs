function [alpha_phi,beta_phi,alpha_sat]=find_slope_bounds(W,b,xs,dv,umax)

l=length(W)-1;

vs{1}=W{1}*xs+b{1};
v_max{1}=vs{1}+dv{1};
v_min{1}=vs{1}-dv{1};

for ii=2:l % Interval bound propagation
    vs{ii}=W{ii}*tanh(vs{ii-1})+b{ii};
    mu=(tanh(v_max{ii-1})+tanh(v_min{ii-1}))/2;
    r=(tanh(v_max{ii-1})-tanh(v_min{ii-1}))/2;
    mu2=W{ii}*mu+b{ii};
    r2=abs(W{ii})*r;
    v_min{ii}=mu2-r2;
    v_max{ii}=mu2+r2;
end
%%

for jj=1:l
    for kk=1:size(W{jj},1)
        if sign(v_min{jj}(kk)*v_max{jj}(kk))==-1
            beta_phi{jj}(kk)=1;
            if abs(v_min{jj}(kk))>v_max{jj}(kk)
                alpha_phi{jj}(kk)=1-tanh(v_min{jj}(kk))^2; % derivative of tanh
            else
                alpha_phi{jj}(kk)=1-tanh(v_max{jj}(kk))^2;
            end
        else
            if sign(v_min{jj}(kk))==-1
                alpha_phi{jj}(kk)=1-tanh(v_min{jj}(kk))^2;
                beta_phi{jj}(kk)=1-tanh(v_max{jj}(kk))^2;
            else
                alpha_phi{jj}(kk)=1-tanh(v_max{jj}(kk))^2;
                beta_phi{jj}(kk)=1-tanh(v_min{jj}(kk))^2;
            end            
        end
    end
end


%%
vs{l+1}=W{l+1}*tanh(vs{l})+b{l+1};
mu=(tanh(v_max{l})+tanh(v_min{l}))/2;
r=(tanh(v_max{l})-tanh(v_min{l}))/2;
mu2=W{l+1}*mu+b{l+1};
r2=abs(W{l+1})*r;
u_min=mu2-r2;
u_max=mu2+r2;

if u_max > umax || u_min < -umax
    alpha_sat = 0;
else
    alpha_sat = 1;
end
