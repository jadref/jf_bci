

% Path with oxy3read functions  
addpath('')


%%% Read oxy3 file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% The oxy3read_function opens a screen where you can browse for your      %
% oxy3 file. The optical densities of all channels is given as output     %
% and xmlInfo contains all the measurement settings and properties        %
%                                                                         %    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example:
[OD,xmlInfo,ADvalues]=oxy3read_function();


%%% Change measurement settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          
%                                                                         %
% Here you can change the measurement settings by redefining xmlInfo.X    %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example:
% xmlInfo.abs_K=1.1;


%%% Single channel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %   
% Needs input in the form of:                                             %      
% [t,O2Hb,HHb]=single_ch(OD,xmlInfo,subtemplate,Rx,[L1,L2])               %
%                                                                         %
%   L1 and L2 form one transmitter                                        %
%   Output: oxy- and deoxyhemoglobin concentration changes                % 
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example:
[t,O2Hb,HHb]=single_ch(OD,xmlInfo,2,1,[3,4]);


%%% TSI channel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% Needs input in the form of:                                             %                                     
% [t,relO2Hb,relHHb,ua,absO2Hb,absHHb,TSI]=tsi_ch(OD,xmlInfo,...          %
%                                                subtemplate,Rx,Txs)      %
%                                                                         %
%   For 3-channel TSI, Txs is of the form: [L1,L2,L3,L4,L5,L6]            %
%   where (L1,L2) is the transmitter at the smallest distance from        %
%   the receiver and (L5,L6) is the transmitter at the largest distance   %
%   from the receiver                                                     %
%                                                                         %  
%   For 2-channel TSI, Txs is of the form: [L1,L2,L3,L4]                  %
%   where (L1,L2) is the transmitter at the smallest distance from        %
%   the receiver and (L3,L4) is the transmitter at the largest distance   %
%   from the receiver                                                     %
%                                                                         %  
%   L1, L3, L5 have the same wavelength as well as L2, L4, L6             %
%                                                                         %  
%   Output: both relative and absolute concentrations, absorption         %
%   coefficient and TSI                                                   %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Example:
% [t,relO2Hb,relHHb,ua,absO2Hb,absHHb,TSI]=...
%                        tsi_ch(OD,xmlInfo,1,1,[1,2,3,4,5,6]);
   
                   
%%% Additional info %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% The extinction coefficients found by Cope are used in the calculations. %
% No correction for DPF is applied.                                       %
% Set the subtemplate number at 1 when only one template is used (for     %
% example in Portamon measurements).                                      %
%                                                                         %    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                    



