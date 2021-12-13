clc, clear all, close all

%%
mdl = 'modified_DDPG_ACC';
open_system(mdl);

Ts = 0.1;
Tf = 60;
% T = 30;

G_ego = tf(1,[0.5,1,0]);

x0_lead = 50;   % initial position for lead car (m)
v0_lead = 0;   % initial velocity for lead car (m/s)

x0_ego = 10;   % initial position for ego car (m)
v0_ego = 0;   % initial velocity for ego car (m/s)

t_gap = 1.4;
D_default = 10;

v_set = 30;

amin_ego = -3;
amax_ego = 3;

v_min_lead = 0;
v_max_lead = 66.6;

v_min_ego = 0;
v_max_ego = 66.6;
%RL Agent definitions
agentblk = [mdl '/RL Agent'];

nrObs = 3;
observationInfo = rlNumericSpec([nrObs 1],'LowerLimit',-inf*ones(nrObs,1),'UpperLimit',inf*ones(nrObs,1));
observationInfo.Name = 'observations';
observationInfo.Description = 'information on velocity error and ego velocity';

actionInfo = rlNumericSpec([1 1],'LowerLimit',-3,'UpperLimit',3);
actionInfo.Name = 'acceleration';

env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);
env.ResetFcn = @(in)localResetFcn(in,v_set);

rng('default');

% Change to a much smaller network
L = 16; % number of neurons

statePath = [
    featureInputLayer(nrObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')
    ];

actionPath = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(L, 'Name', 'fc5')];

% The critic network tells how good the action found from the actor action 
% and how it should adjust 
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
    
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');

plot(criticNetwork)

criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4);

critic = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);


% The actor network decides which action should be taken
actorNetwork = [
    featureInputLayer(nrObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')
    tanhLayer('Name','tanh1')
    scalingLayer('Name','ActorScaling1','Scale',2.5,'Bias',-0.5)];


actorOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'ActorScaling1'},actorOptions);

agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',64);
agentOptions.NoiseOptions.Variance = 0.6;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor,critic,agentOptions);

maxepisodes = 150;
maxsteps = ceil(Tf/Ts);
trainingOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',260);


doTraining = true;


if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainingOpts);
    save("test.mat","agent");
else
    % Load a pretrained agent for the example.
    load("test.mat","agent")       
end



%%
load("test.mat","agent")
x0_lead = 200;   % initial position for lead car (m)
v0_lead = 0;   % initial velocity for lead car (m/s)

x0_ego = 10;   % initial position for ego car (m)
v0_ego = 10;   % initial velocity for ego car (m/s)

t_gap = 1.4;
D_default = 10;

v_set = 30;

amin_ego = -3;
amax_ego = 3;
sim(mdl)
%rlACCplot(logsout,D_default,t_gap,v_set)