function [hdl]=subplott(h,w,is)
% like sub-plot but numbers in normal matlab col-first order
ist=false(h,w); ist(is)=true; ist=find(ist');
hdl=subplot(h,w,ist);
return;