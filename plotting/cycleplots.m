function []=cycleplots(range)
fnum=1;
while true;
   set(0,'CurrentFigure',i);
   figure(fnum);shg;drawnow;
   k=waitkey(fnum)
   switch k;
    case {'space','n','N','uparrow','rightarrow','pageup'}; fnum=fnum+1;
    case {'backspace','b','B','downarrow','leftarrow','pagedown'}; fnum=fnum-1;
    case {'home'}; fnum=1;
    case {'end'};  fnum=numel(range);
    case {'escape','q','Q'}; break;
  end
  fnum=max(1,min(fnum,numel(range)));
end
  