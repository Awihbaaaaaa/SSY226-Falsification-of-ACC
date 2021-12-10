clear
clc
v_lead=[0:0.1:66.6];
v_ego=[0:0.1:30];

L_lead=length(v_lead);
L_ego=length(v_ego);

Z=zeros(L_lead,L_ego);

for i=1:L_lead
    for j= 1:L_ego
        Z(i,j)=dmin(v_ego(j),v_lead(i));
    end
end

figure(1)
mesh(Z);
grid on
ylabel('v lead x 0.1 m/s')
xlabel('v ego x 0.1 m/s')
zlabel('dmin m')

max(max(Z));