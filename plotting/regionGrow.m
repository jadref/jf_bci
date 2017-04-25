function [Im]=regionGrow(Im)
echo on
delay=0;
% This function grows a region about the input region.
[h,w]=size(Im);
chg=1;
imagesc(Im);
while chg ~= 0;
  chg=0;

  % Fill North
  Frng=find(Im==1);Frng=Frng(Frng>w); % get the seed set, ignore top edge.
  while ~ isempty(Frng);
    Frng=Frng-w;             % look north
    nFrng=Frng(Im(Frng)==0); %get the set of pixels with 0 north pt.
    Im( nFrng ) = 1;         %set them
    if ( nFrng ) chg=1; end
    Frng=nFrng(nFrng>w);   %kill off any at top edge
  end
  if ( delay > 0 ) disp('North');imagesc(Im); pause(delay); end

  
  % Fill South
  Frng=find(Im==1);Frng=Frng(Frng<w*(h-1)); %get the seed set, ignore bot edge.
  while ~ isempty(Frng);
    Frng=Frng+w;             % look south
    nFrng=Frng(Im(Frng)==0); %get the set of pixels with 0 north pt.
    Im( nFrng ) = 1;     %set them
    if ( nFrng ) chg=2; end
    Frng=nFrng(nFrng < w*(h-1));   %kill off any at bottom edge
  end
  if ( delay > 0 ) disp('South');imagesc(Im); pause(delay); end

  % Fill East
  Frng=find(Im==1);Frng=Frng(mod(Frng,w)<w);%get the seed set, ignore east edge
  while ~ isempty(Frng)
    Frng=Frng+1;             % look north
    nFrng=Frng(Im(Frng)==0); %get the set of pixels with 0 north pt.
    Im( nFrng ) = 1;     %set them
    if ( nFrng ) chg=3; end
    Frng = nFrng(floor(nFrng/w)<w);  %kill of any at the east edge  
  end
  if ( delay > 0 )  disp('East');imagesc(Im); pause(delay); end

  % Fill West
  Frng=find(Im==1);Frng=Frng(mod(Frng,w)>1);%get the seed set, ignore west edge
  while ~ isempty(Frng)
    Frng=Frng-1;             % look north
    nFrng=Frng(Im(Frng)==0); %get the set of pixels with 0 north pt.
    Im( nFrng ) = 1;     %set them
    if ( nFrng ) chg=4; end
    Frng = nFrng(floor(nFrng/w)>1);  %kill of any at the west edge
  end
  if ( delay > 0 ) disp('West');imagesc(Im); pause(delay); end

end
imagesc(Im);
