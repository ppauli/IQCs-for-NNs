function [star,x_vec]=simulate_system(W,b,sys,x0,sim_time)

A=sys.A;
B=sys.B;
C=sys.C;
%Ts=sys.Ts;
nx=size(sys.A,1);

l=length(W)-1;
x=x0;
x_max=x0;
x_min=x0;
x_vec=[x0];


for ii=1:sim_time
    w=x;
    for jj=1:l
        v=W{jj}*w+b{jj};
        w=tanh(v);
        if ii==1
            v_max{jj}=v;
            v_min{jj}=v;
        end
        for kk=1:length(v)
            if v(kk)>v_max{jj}(kk)
                v_max{jj}(kk)=v(kk);
            elseif v(kk)<v_min{jj}(kk)
                v_min{jj}(kk)=v(kk);
            end
        end
        if ii == sim_time
            v_s{jj}=v;
            w_s{jj}=w;
        end
    end
    u=W{l+1}*w+b{l+1};
    x=A*x+B*u;
    x_vec=[x_vec x];
    if ii==sim_time
        x_s=x;
        u_s=u;
    end
end

for jj=1:l
    d{jj}=min(v_max{jj}-v_s{jj},v_s{jj}-v_min{jj});
end

star.vs=v_s;
star.ws=w_s;
star.xs=x_s;
star.us=u_s;
star.v_max=v_max;
star.v_min=v_min;

