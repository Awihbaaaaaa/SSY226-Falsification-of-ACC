clc, clear all, close all

%% Soft Actor Critic Agent

mdl = 'modified_DDPG_ACC';
open_system(mdl);

Ts = 0.1;
Tf = 200;
% T = 30;

G_ego = tf(1,[0.5,1,0]);

x0_lead = 50;   % initial position for lead car (m)
v0_lead = 25;   % initial velocity for lead car (m/s)

x0_ego = 10;   % initial position for ego car (m)
v0_ego = 20;   % initial velocity for ego car (m/s)

t_gap = 1.4;
D_default = 10;

v_set = 30;

amin_ego = -3;
amax_ego = 3;

%RL Agent definitions
agentblk = [mdl '/RL Agent'];

nrObs = 3;

% Create the observation and action info
obsInfo = rlNumericSpec([nrObs 1], 'LowerLimit', -inf*ones(nrObs,1),'UpperLimit', inf*ones(nrObs,1));
obsInfo.Name = 'observations';
obsInfo.Description = 'information on velocity error and ego velocity';

% Action info
actInfo = rlNumericSpec([1 1],'LowerLimit', -3, 'UpperLimit', 3);
actInfo.Name = 'acceleration';

% Define environment
env = rlSimulinkEnv(mdl, agentblk, obsInfo,actInfo);

