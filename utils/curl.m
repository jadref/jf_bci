function [str]=curl(varargin)
opts=struct('machine','','port',80,'filepath','','proto','http','waitTime',.01,'maxWait',3);
if ( numel(varargin)>1 || ~ischar(varargin{1}) ) % parse the options
   opts=parseOpts(opts,varargin);
else % split the url into its different parts
   tmpuri=varargin{1};
   % Protocol
   idx=strfind(tmpuri,'://'); 
   if ( isempty(idx) ) error('must specify a protocol'); end;
   idx=idx(1);opts.proto=tmpuri(1:idx-1);
   tmpuri=tmpuri(idx+3:end); % remove parsed
   % Machine name (+ port)
   idx=strfind(tmpuri,'/');
   if ( isempty(idx) ) opts.machine=uri(strt:end); tmpuri=''; % default query
   else
      idx=idx(1); 
      opts.machine=tmpuri(1:idx-1); % extract machine name
      tmpuri=tmpuri(idx(1)+1:end);
      idx=strfind(opts.machine,':'); % extract the port if necessary
      if( ~isempty(idx) ) 
         opts.port=str2num(opts.machine(idx(1)+1:end)); opts.machine=opts.machine(1:idx(1)-1);
      end
   end
   % Filepath
   opts.filepath=tmpuri; if ( isempty(opts.filepath) ) opts.filepath='/'; end;
end

if ( ~strmatch(opts.proto,'http') ) error('Only http is supported!'); end;

% build the request string
httpreq=sprintf('GET %s HTTP/1.1',opts.filepath);
httpreq=sprintf('%s\nConnection: close\nAccept: text/html\n\n',httpreq);

% open a socket to the server
socket   = java.net.Socket(opts.machine,opts.port);
instream = java.io.InputStreamReader(socket.getInputStream()); % character input stream
outstream= java.io.OutputStreamWriter(socket.getOutputStream()); % character output stream

% send the request string
outstream.write(httpreq); outstream.flush();
%outstream.close();

% read the response string
% wait until some data is ready
for i=1:opts.waitTime:opts.maxWait; if ( instream.ready() ) break; end; pause(opts.waitTime); end;
% read the data
str=''; while ( instream.ready() ) str=[str instream.read()]; end

% close the ports
instream.close();
outstream.close();
socket.close();
return;
%------------------------------------------------------------------------------
function testCase()
curl('http://www.google.com/index.html');
