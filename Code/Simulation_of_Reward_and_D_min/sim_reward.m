k=0.001;
d_min=[0:0.1:186];
d_rel=[0:0.1:200];

L_dmin=length(d_min);
L_rel=length(d_rel);

Z=zeros(L_dmin,L_rel);

for i=1:L_dmin
    for j= 1:L_rel
        Z(i,j)=reward(d_min(i),d_rel(j),k);
    end
end

figure(1)
mesh(Z);
grid on
ylabel('dmin x 0.1 m')
xlabel('drel x 0.1 m')
zlabel('reward')