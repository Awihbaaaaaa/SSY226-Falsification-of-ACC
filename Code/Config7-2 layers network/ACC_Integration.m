clc, clear all, close all

%%
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

% Getting the observation info
observationInfo = rlNumericSpec([nrObs 1],'LowerLimit',-inf*ones(nrObs,1),'UpperLimit',inf*ones(nrObs,1));
observationInfo.Name = 'observations';
observationInfo.Description = 'information on velocity error and ego velocity';

% Changed the upperlimit from 2
actionInfo = rlNumericSpec([1 1],'LowerLimit',-3,'UpperLimit',3);
actionInfo.Name = 'acceleration';

% Creating the environment interface
env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);

% The reset function randomizes the initial position of the lead car.
env.ResetFcn = @(in)localResetFcn(in);

rng('default');

% Change to a much smaller network
L = 10; % number of neurons

statePath = [
    featureInputLayer(nrObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    %reluLayer('Name','relu1')
    %fullyConnectedLayer(L,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
%     fullyConnectedLayer(L,'Name','fc3')
%     reluLayer('Name','relu3')
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
%     reluLayer('Name','relu1')
%     fullyConnectedLayer(L,'Name','fc2')
%     reluLayer('Name','relu2')
%     fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')
    tanhLayer('Name','tanh1')
    scalingLayer('Name','ActorScaling1','Scale',2.5,'Bias',-0.5)];

actorOptions = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
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

maxepisodes = 200;
maxsteps = ceil(Tf/Ts);
trainingOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',500);

%trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 132;

doTraining = true;

if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainingOpts);
    save("agent7.mat","agent");
else
    % Load a pretrained agent for the example.
    load('SimulinkACCDDPG.mat','agent')       
end

% sim(mdl)
% rlACCplot(logsout,D_default,t_gap,v_set)