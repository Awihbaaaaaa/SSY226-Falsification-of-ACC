clc, clear all, close all

% Soft Actor Critic Agent

mdl = 'modified_DDPG_ACC';
% open_system(mdl);

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
obsInfo.Name = 'myobs';
obsInfo.Description = 'information on velocity of the lead and ego';

% Action info
actInfo = rlNumericSpec([1 1],'LowerLimit', -3, 'UpperLimit', 3);
actInfo.Name = 'myact';

% Define environment
env = rlSimulinkEnv(mdl, agentblk, obsInfo,actInfo);

% The number of neurons
L = 10;

% Actor Network
% input path layers with size number of observations by 1 input and 
% 1 output)
inPath = [ imageInputLayer([nrObs 1 1], 'Normalization','none','Name','myobs') 
           fullyConnectedLayer(1,'Name','infc')]; % The output must be twice the number of the output

% path layers for mean value (2 by 1 input and 2 by 1 output)
% using scalingLayer to scale the range
meanPath = [ tanhLayer('Name','tanh'); % output range: (-1,1)
             scalingLayer('Name','scale','Scale',actInfo.UpperLimit) ]; % output range: (-3,3)

% path layers for standard deviations (2 by 1 input and output)
% using softplus layer to make it non negative
sdevPath =  softplusLayer('Name', 'splus');

% conctatenate two inputs (along dimension #3) to form a single (4 by 1) output layer
outLayer = concatenationLayer(3,2,'Name','mean&sdev');

% add layers to the actor network object
actNet = layerGraph(inPath);
actNet = addLayers(actNet,meanPath);
actNet = addLayers(actNet,sdevPath);
actNet = addLayers(actNet,outLayer);

% connect layers: the mean value path output MUST be connected to the FIRST input of the concatenationLayer
actNet = connectLayers(actNet,'infc','tanh/in');              % connect output of inPath to meanPath input
actNet = connectLayers(actNet,'infc','splus/in');             % connect output of inPath to sdevPath input
actNet = connectLayers(actNet,'scale','mean&sdev/in1');       % connect output of meanPath to gaussPars input #1
actNet = connectLayers(actNet,'splus','mean&sdev/in2');       % connect output of sdevPath to gaussPars input #2

plot(actNet)

% Actor training options
actorOpts = rlRepresentationOptions('LearnRate',8e-3,'GradientThreshold',1);

% Creating the actor
actor = rlStochasticActorRepresentation(actNet, obsInfo, actInfo, 'Observation','myobs',actorOpts);

% Checking the actor
% getAction(actor,{ones(3,1)})

% Critic Network 
% observation path layers
obsPath = [featureInputLayer(nrObs, 'Normalization','none','Name','myobs') 
    fullyConnectedLayer(1,'Name','obsout')];

% action path layers
actPath = [featureInputLayer(1, 'Normalization','none','Name','myact') 
    fullyConnectedLayer(1,'Name','actout')];

% common path to output layers
comPath = [additionLayer(2,'Name', 'add')  fullyConnectedLayer(1, 'Name', 'output')];

% add layers to network object
criticNet = addLayers(layerGraph(obsPath),actPath); 
criticNet = addLayers(criticNet,comPath);

% connect layers
criticNet = connectLayers(criticNet,'obsout','add/in1');
criticNet = connectLayers(criticNet,'actout','add/in2');

critic = rlQValueRepresentation(criticNet,obsInfo,actInfo,'Observation',{'myobs'},'Action',{'myact'});

% Checking the critic network
% getValue(critic,{rand(3,1)},{rand(1,1)})

% Specifying options for SAC agent
opt = rlSACAgentOptions('DiscountFactor',0.95);

opt.SampleTime = 0.1;

% Higher entropy target encourage more exploration
opt.EntropyWeightOptions.TargetEntropy = -5;

opt.MiniBatchSize = 32;

agent = rlSACAgent(actor,critic,opt);



%% 




