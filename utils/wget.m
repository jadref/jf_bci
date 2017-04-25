function [str]=wget(varargin)
% returns result of http request with specified url (first string argument)
%
%  [str]=wget(varargin)
%  OR
%  [str]=wget(url,varargin)
%
% Inputs:
%  url -- [str] fully qualify url
% Options:
%  machine -- [str] destination machine name/ip address
%  port    -- [int] destination port number                      (80)
%  filepath-- [str] destination path to get
%  proto   -- [str] protocol to use. *must* be 'http'            ('http')
%  pollInterval -- [float] time in seconds to sleep between checking for server response (.01)
%  timeOut -- [float] maximum time (in seconds) to wait for data before giving up.       (3)
%  verb    -- [bool] verbose output
% Output:
%  str     -- [str] the raw string reponse of the server
opts=struct('machine','','port',80,'filepath','','proto','http','pollInterval',.01,'timeOut',3, 'verb', 0);
[opts,varargin]=parseOpts(opts,varargin);

if ( isstr(varargin{1}) ) % parse the request-string
   % split the url into its different parts
   tmpuri=varargin{1};
   % Protocol
   idx=strfind(tmpuri,'://'); 
   if ( isempty(idx) ) error('must specify a protocol'); end;
   idx=idx(1);opts.proto=tmpuri(1:idx-1);
   tmpuri=tmpuri(idx+3:end); % remove parsed
   % Machine name (+ port)
   idx=strfind(tmpuri,'/');
   if ( isempty(idx) ) opts.machine=tmpuri; tmpuri=''; % default query
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
   opts.filepath=tmpuri; if ( isempty(opts.filepath) ) opts.filepath='index.html'; end;

end

host=[opts.machine ':' num2str(opts.port)];

if ( ~strmatch(opts.proto,'http') ) error('Only http is supported!'); end;

% build the request string
if isempty(strfind(opts.filepath,'?'))
   httpreq=sprintf('GET %s HTTP/1.1',opts.filepath);
else
   httpreq=sprintf('GET /%s HTTP/1.1',opts.filepath); % added a / after GET, needed to handle complex urls with arguments, don't know why PD
end 
httpreq=sprintf('%s\nHost: %s', httpreq, host); % add destination host to the request

% httpreq=sprintf('%s\nUser-Agent: Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.5; en-US; rv:1.9.0.6) Gecko/2009011912 Firefox/3.0.6',httpreq);
% httpreq=sprintf('%s\nAccept-Language: en-us,en;q=0.5',httpreq);
% httpreq=sprintf('%s\nAccept-Encoding: gzip,deflate\n',httpreq);
% httpreq=sprintf('%s\nAccept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7',httpreq);
% httpreq=sprintf('%s\nKeep-Alive: 300\n',httpreq);
% httpreq=sprintf('%s\nConnection: keep-alive\n',httpreq);
% httpreq=sprintf('%s\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',httpreq);

httpreq=sprintf('%s\nConnection: close', httpreq);
httpreq=sprintf('%s\nAccept: text/html\n\n',httpreq);

if opts.verb
    disp(['send httpreq: ' httpreq]);
end

% open a socket to the server
socket   = java.net.Socket(opts.machine,opts.port);
instream = java.io.InputStreamReader(socket.getInputStream()); % character input stream
outstream= java.io.OutputStreamWriter(socket.getOutputStream()); % character output stream

% send the request string
outstream.write(httpreq); outstream.flush();
%outstream.close();

% read the response string, with timeOuts
str=''; tic;
while ( toc()<opts.timeOut ) % timeOut if waiting for *new* data for too long
   if( instream.ready() ) tic; end; % reset timer if data to read
   cstr=''; while( instream.ready() ) cstr=[cstr instream.read()]; end; % read ALL data, if ready
   if ( regexp(cstr,['^[0-9a-f][0-9a-f][0-9a-f]' sprintf('\r\n')]) )
      cstr=cstr(6:end); % strip the chunk info
   end
   str=[str cstr];
   pause(opts.pollInterval);
end

% close the ports
instream.close();
outstream.close();
socket.close();
return;
%------------------------------------------------------------------------------
function testCase()
curl('http://www.google.com/index.html');
