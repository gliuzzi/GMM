clear all
close all

%T = readtable("risultati.txt","delimiter",'|');
%T = T(1:978,:);
%T = readtable("risultati_completi_e-6.txt","delimiter",'|');
%T = T(1:1304,:);
%T = readtable("risultati.txt","delimiter",'|');
%T = T(1:1176,:);
T = readtable("risultati163.txt","delimiter",'|');
T = T(1:1141,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WARNING VERY IMPORTANT: Depending on how the
%   results have been obtained (which tolerance)
%   the same tolerance MUST be used in this script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gtol = 1.e-6;

%%%
% le colonne di T sono le seguenti:
% Var1 : NOT USED
% Var2 : Solver name
% Var3 : Problem name
% Var4 : Problem dim.
% Var5 : time
% Var6 : n.it
% Var7 : f.opt
% Var8 : gradient norm
% Var9 : function evaluations
% Var10: gradient evaluations
% Var11: DON'T KNOW, DON'T MIND
% Var12: DON'T KNOW, DON'T MIND
% Var13: DON'T KNOW, DON'T MIND
%%%

table_size = size(T) ; 
rows = table_size(1);

% get solver names
S = {};
for row = 1:rows 
    name = string(T.Var2(row));
    
    if name == "--"
        break
    end
    S = [S name];
    disp(name)
end   

% get problem names
P = {};
lastp = "";
for row = 1:rows
   name = string(T.Var3(row));
   if name == lastp
       continue
   end
   if name == "--"
       continue
   end
   disp(name);
   P = [P name];
   lastp = name;
end

[~, ns] = size(S);
[~, np] = size(P);

Htime = zeros(np,ns);
Hiter = zeros(np,ns);
Hfval = zeros(np,ns);
Hgrad = zeros(np,ns);
Hnewi = zeros(np,ns);

ip = 1;
is = 1;
for row = 1:rows
    solver = string(T.Var2(row));
    problem = string(T.Var3(row));
    if solver == "--"
        continue
    end
    ip = find(ismember(P,problem));
    is = find(ismember(S,solver));
    Hfval(ip,is) = T.Var7(row);
    Hgrad(ip,is) = T.Var8(row);
    nf = T.Var9(row);
    n  = T.Var4(row);
    ng = T.Var10(row);
    if T.Var8(row) <= gtol
        Htime(ip,is) = T.Var5(row);
        Hiter(ip,is) = T.Var6(row);
        Hnewi(ip,is) = nf+5*ng; 
    else 
        Htime(ip,is) = nan;
        Hiter(ip,is) = nan;
        Nnewi(ip,is) = nan;
    end
end

Istaz = [];
I = [];
for ip = 1:np

    if max(Hfval(ip,:)) - min(Hfval(ip,:)) < 1.e-3
        I = [I ip];
    end

end

LS = {
    '--k^', %'-bs', %GMM1
    '-ko', %'--k^', %GMM3
    '-.ks', %'-.ro', %GMM2
    '-rx', %'-g*', %L-BFGS
    '-bv', %'-bv' %CG    
    '-mv', %'-mv' %Hager    
    '-gv' %'-gv' %Hager (no hess)    
    };

CS = {
    [0 0.5 1], %GMM1
    [0 0.5 0.5], %GMM3
    [0 0 1], %GMM2
    [1 0 0], %L-BFGS
    [0.5 1 0], %CG    
    [0 1 0.5], %Hager    
    [0.5 1 0.5] %Hager (no hess)    
    };

SS = {
'GMM$_1$',
'GMM$_3$',
'GMM$_2$',
'L-BFGS$_{scipy}$',
'CG$_{scipy}$',
%'PASA',
'PASA (no hess)'
};

confronti = {[1,2,3,5]};
confronti = {[1,2,3,4]};
confronti = {[3,4]};

compare = {[1,2,3,5],[1,2,3,4],[3,4]};
compare = {[1,2,3,4,6],[3,6]};
compare = {[1,2,3,4]};

for confronti = compare
    nc = size(confronti,2);

    figure('Position',[0,0,2000,600])
    i = 1;
    for pp = confronti
        pair = pp{1};
        subplot(nc,3,i);
        perf_profile(Htime(:,pair),SS(pair),'Time',LS(pair),CS(pair))
        subplot(nc,3,i+1)
        perf_profile(Hiter(:,pair),SS(pair),'Iter',LS(pair),CS(pair))
        subplot(nc,3,i+2)
        perf_profile(Hnewi(:,pair),SS(pair),'f.evals eq.',LS(pair),CS(pair))
        i = i+2;
    end

    figure('Position',[0,0,2000,600])
    i = 1;
    for pp = confronti
        pair = pp{1};
        I = [];
        nbest = zeros(1,size(pair,2));
        for ip = 1:np
            bestf = min(Hfval(ip,pair));
            worsf = max(Hfval(ip,pair));

            if worsf - bestf < 1.e-3
                I = [I ip];
            else
                [v,ind] = min(Hfval(ip,pair));
                for ii = 1:size(pair,2)
                    if abs(v-Hfval(ip,pair(ii))) < 1.e-3
                        nbest(1,ii) = nbest(1,ii)+1;
                    end
                end
            end

        end
        subplot(nc,3,i);
        perf_profile(Htime(I,pair),SS(pair),'Time',LS(pair),CS(pair))
        subplot(nc,3,i+1)
        perf_profile(Hiter(I,pair),SS(pair),'Iter',LS(pair),CS(pair))
        subplot(nc,3,i+2)
        perf_profile(Hnewi(I,pair),SS(pair),'f.evals eq.',LS(pair),CS(pair))
        i = i+2;
        nu = size(I,2);
        for p = 1:size(pair,2)
            fprintf("%20s wins on %3d/%3d\n",SS{pair(p)},nbest(1,p),np)
        end
    end
end