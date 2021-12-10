function in = localResetFcn(in)
% Reset the initial position of the lead car.
in = setVariable(in,'x0_lead',40+randi(60,1,1));
in = setVariable(in,'v0_lead',randi(20,1,1)-1);
%in = setVariable(in,'v0_ego',randi(30,1,1)-1);
end