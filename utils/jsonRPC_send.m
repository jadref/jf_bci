function event = jsonRPC_send(event, fname, varargin)
%first argument should be the function, the others are the arguments
if( isstruct(event) )
   socket = event.Experiment.StimulusPresentation.Socket;
else
   socket=event;
end

s.method = fname;
s.params = varargin;
s.version = 1;
rpcstr = serialize(s,[],'json');

out = java.io.BufferedWriter(java.io.OutputStreamWriter(socket.getOutputStream()));
out.write(rpcstr);
out.flush(); % send command now!
return;

%---------------------------
function testCase();

% open the socket
socket=java.net.Socket('127.0.0.1',6666);
event.Experiment.StimulusPresentation.Socket=socket; % simulate and events structure

jsonRPC_send(socket,'disp','hello world');


jsonRPC_send(socket,'init_display','windowPos',[1024 50 1025+512 50+512]);
title='Title'; 
options={'where' 'do' 'you' 'want' 'to' 'go' 'today'}';
jsonRPC_send(socket,'setRegion',[],'title',title,'TextColor',[0 255 0]);
jsonRPC_send(socket,'setRegion',[],'options',options,'TextColor',[255 255 255]);
jsonRPC_send(socket,'drawDisplay');

jsonRPC_send(socket,'highlightSymb',[],'options',2);


% play a stimulus sequence
T=100;
optStimSeq = randn(numel(options),T);      % random flashing sequence
syncStimSeq = ones(1,T); syncStimSeq(2:2:end)=0; % sync alternates every frame

jsonRPC_send(socket,'animateRegions',[],'options',optStimSeq,'sync',syncStimSeq,...
            'stimTime_ms',100:100:1000);

jsonRPC_send('sca');
jsonRPC_send('quit');
