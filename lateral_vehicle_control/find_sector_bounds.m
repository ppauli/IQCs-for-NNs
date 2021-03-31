function [alpha_phi,beta_phi,alpha_sat]=find_sector_bounds(W,b,xs,dv,umax)

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
        alpha_phi{jj}(kk)=min((tanh(v_max{jj}(kk))-tanh(vs{jj}(kk)))/...
            (v_max{jj}(kk)-vs{jj}(kk)),(tanh(vs{jj}(kk))-tanh(v_min{jj}(kk)))/(vs{jj}(kk)-v_min{jj}(kk)));
        if alpha_phi{jj}(kk)==(tanh(vs{jj}(kk))-tanh(v_min{jj}(kk)))/(vs{jj}(kk)-v_min{jj}(kk))
            vec=linspace(0.01,v_max{jj}(kk)-vs{jj}(kk),1000);
        else
            vec=linspace(v_min{jj}(kk)-vs{jj}(kk),-0.01,1000);
        end
        for ii=1:1000
            beta_vec(ii)=(tanh(vec(ii)+vs{jj}(kk))-tanh(vs{jj}(kk)))/(vec(ii));
        end
        beta_phi{jj}(kk)=max(beta_vec);
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

alpha_sat = min((sat(u_max,umax,-umax)-sat(vs{l+1},umax,-umax))./(u_max-vs{l+1}), (sat(vs{l+1},umax,-umax)-sat(u_min,umax,-umax))./(vs{l+1}-u_min));

