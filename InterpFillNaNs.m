function x = InterpFillNaNs(x)

ismissing = isnan(x);
if all(ismissing) || ~any(ismissing),
  return;
end
[t0s,t1s] = get_interval_ends(ismissing);
t1s = t1s-1;
for imissing = 1:numel(t0s),
  t0 = t0s(imissing);
  t1 = t1s(imissing);
  if t0 == 1,
    x(1:t1) = x(t1+1);
  elseif t1 == numel(x),
    x(t0:t1) = x(t0-1);
  else
    x(t0:t1) = x(t0-1)+(x(t1+1)-x(t0-1))/(t1-t0+2).*(1:t1-t0+1);
  end
  
end
