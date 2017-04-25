% Use MCMC to compute the average and variance of a probability
% distribution

x = -10:0.1:10;
f = exp(5-(x-1).*(x-1).*(sin(x)+1.2));

clf
hold on
plot(x,f)

x0 = -5;
p0 = exp(5-(x0-1).*(x0-1).*(sin(x0)+1.2));
xt = [x0];
accept = 0;
av = 0;
var = 0;
cnt = 0;
avtext = text('Position', [-8,130], 'String', '');
vartext = text('Position', [-8,120], 'String', '');
accepttext = text('Position', [-8,110], 'String', '');
for t=1:1000
  x1 = x0 + 3*randn;
  p1 = exp(5-(x1-1).*(x1-1).*(sin(x1)+1.2));
  if (p1/p0>rand)
    x0 = x1;
    p0 = p1;
    accept = accept + 1; 
  end
  xt = [xt x0];
  if (t>100) % through away transience
    cnt = cnt + 1;
    delta = (x0-av)/cnt;
    av = av + delta;
    var = var + (cnt-1)*delta*(x0-av);
    str = ['Av= ', num2str(av)];
    delete(avtext)
    avtext = text('Position', [-8,130], 'String', str);
    if (cnt>1)
      str = ['Var= ', num2str(var/(cnt-1))];
      delete(vartext)
      vartext = text('Position', [-8,120], 'String', str);
    end
    str = ['Accept= ', num2str(accept/t)];
    delete(accepttext)
    accepttext = text('Position', [-8,110], 'String', str);
  end
  hist(xt,30)
  figure(1)
end
