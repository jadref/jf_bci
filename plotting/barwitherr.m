function barwitherr(X,Y,varargin)
%**************************************************************************
%
%   This is a simple extension of the bar plot to include error bars.  It
%   is called in exactly the same way as bar but with an extra input
%   parameter "errors" passed first.
%
%   Parameters:
%   errors - the errors to be plotted
%   varargin - parameters as passed to conventional bar plot
%   See bar and errorbar documentation for more details.
%
%   Example:
%   y = randn(3,4);         % random y values (3 groups of 4 parameters) 
%   errY = 0.1.*y;          % 10% error
%   barwitherr(errY, y);    % Plot with errorbars
%
%   set(gca,'XTickLabel',{'Group A','Group B','Group C'})
%   legend('Parameter 1','Parameter 2','Parameter 3','Parameter 4')
%   ylabel('Y Value')
%
%   Note: Ideally used for group plots with non-overlapping bars because it
%   will always plot in bar centre (so can look odd for over-lapping bars) 
%   and for stacked plots the errorbars will be at the original y value is 
%   not the stacked value so again odd appearance as is.
%
%   24/02/2011  Created     Martina F. Callaghan
%
%**************************************************************************

% Check how the function has been called based on requirements for "bar"
errors=[];
if( nargin > 2 && isnumeric(varargin{1}) )
  errors=varargin{1}; varargin=varargin(2:end);
end

% Check that the size of "errors" corresponsds to the size of the y-values:
if any(size(Y) ~= size(errors))
    error('The values and errors have to be the same length')
end

[nRows nCols] = size(Y);
if ( ~isempty(X) )
  hdls = bar(X,Y,varargin{:}); % standard implementation of bar fn
else
  hdls = bar(Y,varargin{:}); % standard implementation of bar fn
end

if ( ~isempty(errors) )
isheld=ishold; hold on
if nRows > 1
    for col = 1:nCols
        % Extract the x location data needed for the errorbar plots:
        x = get(get(hdls(col),'children'),'xdata');
        % Use the mean x Y to call the standard errorbar fn; the
        % errorbars will now be centred on each bar:
        errorbar('v6',mean(x,1),Y(:,col),errors(:,col), '.k')
    end
else
    x = get(get(hdls,'children'),'xdata');
    errorbar('v6',mean(x,1),Y,errors,'.k')
end
if( ~isheld ) hold off; end; % reset hold status
end
return;