function perf_profile(H,solvers,Title)

[np,ns] = size(H); % Grab the dimensions


% For each problem and solver, determine the number of evaluations
% required to reach the cutoff value
T = H;

% Other colors, lines, and markers are easily possible:
colors  = ['b' 'r' 'k' 'm' 'c' 'g' 'y'];   lines   = {'-' '-.' '--'};
markers = [ 's' 'o' '^' 'v' 'p' '<' 'x' 'h' '+' 'd' '*' '<' ];

% Compute ratios and divide by smallest element in each row.
r = (T-repmat(min(T,[],2),1,ns))./(1.e-4+abs(repmat(min(T,[],2),1,ns)));

% Replace all NaN's with twice the max_ratio and sort.
max_ratio = max(max(r));
max_ratio = 10;
%disp('perf_profile: max_ratio impostato a mano')
r(isnan(r)) = 2*max_ratio;
r = sort(r);

%keyboard
max_ratio;

%ax(1) = axes('Position',[0.1 0.1 0.5 0.8]);
% Plot stair graphs with markers.
hl = zeros(ns,1);
for s = 1:ns
    [xs,ys] = stairs(r(:,s),(1:np)/np);

    % Only plot one marker at the intercept
    %if (xs(1)==1)
    %    vv = find(xs==1,1,'last');
    %    xs = xs(vv:end);   ys = ys(vv:end);
    %end

    sl = mod(s-1,3) + 1; sc = mod(s-1,7) + 1; sm = mod(s-1,12) + 1;
    option1 = [char(lines(sl)) colors(sc) markers(sm)];
    hl(s) = semilogx(xs,ys,option1);
    %hl(s) = plot(xs,ys,option1);
    hold on;
end

% Axis properties are set so that failures are not shown, but with the
% max_ratio data points shown. This highlights the "flatline" effect.
%axis([1 5 0 1]);
%axis([1 100 0 1]);
axis([1.e-3 max_ratio 0 1]);
legend(solvers);
title(Title);
xlabel('ratio, \alpha');
ylabel('percentage of problems');

hold off;


end