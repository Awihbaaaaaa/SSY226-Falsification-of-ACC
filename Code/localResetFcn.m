function in = localResetFcn(in,v_set)
% Reset the initial position of the lead car.
in = setVariable(in,'x0_lead',40+randi(60,1,1));
in = setVariable(in,'v0_lead',randi(40,1,1)-1);
in = setVariable(in,'v0_ego',randi(v_set,1,1)-1);
end