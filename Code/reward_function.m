function R = reward_function(phi,y_rel)
    R = exp(-phi*y_rel) -1;

end