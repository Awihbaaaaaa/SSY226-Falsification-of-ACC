clc, clear all, close all

% Soft Actor Critic Agent

mdl = 'modified_DDPG_ACC';

Ts = 0.1;
Tf = 200;

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

%RL Agent definitions
agentblk = [mdl '/RL Agent'];

nrObs = 3;

% Create the observation and action info
obsInfo = rlNumericSpec([nrObs 1], 'LowerLimit', -inf*ones(nrObs,1),'UpperLimit', inf*ones(nrObs,1));
obsInfo.Name = 'myobs';
obsInfo.Description = 'information on velocity of the lead and ego cars';

% Action info
actInfo = rlNumericSpec([1 1],'LowerLimit', -3, 'UpperLimit', 3);
actInfo.Name = 'myact';

% Define environment
env = rlSimulinkEnv(mdl, agentblk, obsInfo,actInfo);

% The number of neurons
L = 10;

%% Create stochastic actor
% Actor Network
% input path layers with size number of observations by 1 input and 
% 1 output)
clc
inPath = [
    featureInputLayer(nrObs , 'Normalization','none','Name','myobs') 
    fullyConnectedLayer(16,'Name','commonFC1')
    reluLayer('Name','CommonRelu')]; % The output must be twice the number of the output

% path layers for mean value (2 by 1 input and 2 by 1 output)
% using scalingLayer to scale the range
meanPath = [ fullyConnectedLayer(L,'Name','MeanFC1')
             reluLayer('Name','MeanRelu')
             fullyConnectedLayer(1,'Name','Mean')];

% path layers for standard deviations (2 by 1 input and output)
% using softplus layer to make it non negative
sdevPath =  [
    fullyConnectedLayer(L,'Name','StdFc1')
    reluLayer('Name','StdRel')
    fullyConnectedLayer(1,'Name','StdFc2')
    softplusLayer('Name', 'StandardDeviation')];

% conctatenate two inputs (along dimension #3) to form a single (4 by 1) output layer
outLayer = concatenationLayer(1,2,'Name','mean&sdev');

% add layers to the actor network object
actNet = layerGraph(inPath);
actNet = addLayers(actNet,meanPath);
actNet = addLayers(actNet,sdevPath);
actNet = addLayers(actNet,outLayer);

% connect layers: the mean value path output MUST be connected to the FIRST input of the concatenationLayer
actNet = connectLayers(actNet,'CommonRelu','MeanFC1/in');              % connect output of inPath to meanPath input
actNet = connectLayers(actNet,'CommonRelu','StdFc1/in');             % connect output of inPath to sdevPath input
actNet = connectLayers(actNet,'Mean','mean&sdev/in1');       % connect output of meanPath to gaussPars input #1
actNet = connectLayers(actNet,'StandardDeviation','mean&sdev/in2');       % connect output of sdevPath to gaussPars input #2

plot(actNet)

% Actor training options
actorOpts = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,...
                'GradientThreshold',1,'L2RegularizationFactor',1e-5);

% Creating the actor
actor = rlStochasticActorRepresentation(actNet, obsInfo, actInfo,actorOpts,...
    'Observation',{'myobs'});

% Checking the actor
% getAction(actor,{ones(3,1)})

%%
% Critic Network 
% observation path layers
clc
obsPath = [
    featureInputLayer(nrObs, 'Normalization','none','Name','observation') 
    fullyConnectedLayer(L,'Name','CriticStateFC1')
    reluLayer('Name','CriticStateRelu1')
    fullyConnectedLayer(1,'Name','CriticStateFC2')
    ];

% action path layers
actPath = [
    featureInputLayer(1, 'Normalization','none','Name','action') 
    fullyConnectedLayer(1,'Name','CriticActionFC1')];

% common path to output layers
comPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')
    ];

% add layers to network object
criticNet = layerGraph(obsPath);
criticNet = addLayers(criticNet,actPath); 
criticNet = addLayers(criticNet,comPath);
criticNet = connectLayers(criticNet,'CriticStateFC2','add/in1');% connect layers
criticNet = connectLayers(criticNet,'CriticStateFC1','add/in2');

criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,... 
                                        'GradientThreshold',1,'L2RegularizationFactor',2e-4);

critic1 = rlQValueRepresentation(criticNet,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);

% Checking the critic network
% getValue(critic,{rand(3,1)},{rand(1,1)})

% Specifying options for SAC agent
opt = rlSACAgentOptions;

opt.SampleTime = 0.1;

% Higher entropy target encourage more exploration
opt.EntropyWeightOptions.TargetEntropy = -5;

opt.MiniBatchSize = 32;

opt.NumWarmStartSteps = 32;

opt.DiscountFactor = 0.99;

opt.TargetSmoothFactor = 1e-3;

opt.ExperienceBufferLength = 1e6;

agent = rlSACAgent(actor,critic1,opt);

%%

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
    save("agent_sac.mat","agent");
else
    % Load a pretrained agent for the example.
    load('agent_acc.mat','agent')       
end







