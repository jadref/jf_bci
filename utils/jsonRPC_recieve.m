function jsonRPC_recieve(port,verb)
% Initializes the socket connection, the midi connection and the screen
if ( nargin < 1 ) port=6666; end;
if ( nargin < 2 ) verb=1; end;

% Open listener socket 
global socket; % persistent so can re-start listen if error
if ( isempty(socket) ) 
   server = java.net.ServerSocket(port);
   disp(sprintf('Waiting for incoming connection on port %.0d...', port));
   socket = server.accept();
   disp(sprintf('Connected to %s', char(socket.toString)));
   server.close(); % Close listener socket -- but leave connected client socket open
end

% Listen to the socket
in = java.io.BufferedReader(java.io.InputStreamReader(socket.getInputStream()));
% channel = java.nio.channels.Channels.newChannel(socket.getInputStream());
% buffer  = java.nio.ByteBuffer.allocateDirect(1024); % get in 1k lumps
% in.read(buffer,0,1024);
nesting = 0;
str = [];
buff = zeros(1,10000,'int8');
while(true) % loop-forever listening for commands to execute

   %chr = in.read(); 
   nChr=1; buff(nChr)=in.read(); 
   while(in.ready() && nChr<numel(buff) ) nChr=nChr+1; buff(nChr)=in.read(); end; % read until out of data
   
   % decode the data
   nestingLvl(buff(1:nChr)=='{')=1; nestingLvl(buff(1:nChr)=='}')=-1; 
   nestingLvl(1)=nestingLvl(1)+nesting; nestingLvl=cumsum(nestingLvl);
   endStruct = find(diff(nestingLvl==0)<0); 
   strtStruct= find(diff(nestingLvl==0)>0); 
   if ( nesting>0 ) strtStruct=[1 strtStruct]; end; % start ends of structs   
   for is=1:numel(endStruct);
      str = cat(1,str,buff(strtStruct(is):endStruct(is))); 
      s = unserialize(str, 'json');
      method=s.method;
      if ( isfield(s,'params') ) params=s.params; else params={}; end;
      if ( ~iscell(params) ) params={params}; end;
      if ( verb>0  ) disp({method params{:}}); end;
      feval(method, params{:}); 
      str = []; 
   end
   str=buff(strtStruct(is):nChr); % grab what's left
   
%    if ( chr == '{' )
%       nesting = nesting + 1;
%    end
%    if ( nesting > 0 )
%       if ( verb>1 ) fprintf('%c',chr); end;
%       str = [str chr];
%    end
%    if( chr =='}' )
%       nesting = nesting - 1;
%       if ( nesting == 0 )
%          s = unserialize(str, 'json');
%          method=s.method;
%          if ( isfield(s,'params') ) params=s.params; else params={}; end;
%          if ( ~iscell(params) ) params={params}; end;
%          if ( verb>0  ) disp({method params{:}}); end;
%          feval(method, params{:}); 
%          str = [];
%       end
%    end
end

socket.close();
return;
%---------------------------------------------------------------
function testCase()



   