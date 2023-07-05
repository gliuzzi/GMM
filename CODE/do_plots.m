clear all
close all

%T = readtable("risultati_2023-02-17.txt","delimiter",'|');
%T = readtable("risultati_2023-02-20.txt","delimiter",'|');
%T = readtable("risultati_2023-02-20_fg2.txt","delimiter",'|');
%T = T(1:1364,:);
%T = readtable("risultati_2023-02-28.txt","delimiter",'|');
%T = T(1:364,:);
%T = readtable("risultati_2023-03-07.txt","delimiter",'|');
%T = T(1:364,:);
%T = readtable("risultati.txt","delimiter",'|');
%T = T(1:218,:);
%T = readtable("risultati_2023-03-11.txt","delimiter",'|');
%T = T(1:291,:);
%T = readtable("risultati-tutti-new.txt","delimiter",'|');
%T = T(1:915,:);
T = readtable("risultati-tot-7-new.txt","delimiter",'|');
T = T(1:1464,:);
%T = readtable("risultati.txt","delimiter",'|');
%T = T(1:1464,:);

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

fprintf('\n');
fprintf('Found %3d problems\n',np);
fprintf('Found %3d  solvers. They are:\n',ns);
for i = 1:ns
    fprintf('\t %3d : %s\n',i,S{i});
end
fprintf('\n');

reply = input('Is this ok? Y/N [Y]','s');
if isempty(reply)
    reply = 'Y';
end
if reply == 'Y'

    Htime = zeros(np,ns);
    Hiter = zeros(np,ns);
    Hfval = zeros(np,ns);
    Hgrad = zeros(np,ns);

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
        if T.Var8(row) <= 1.e-3
            Htime(ip,is) = T.Var5(row);
            Hiter(ip,is) = T.Var6(row);
        else 
            Htime(ip,is) = nan;
            Hiter(ip,is) = nan;
        end
    end

    Istaz = [];
    I = [];
    QPS = 0;
    LBFGS = 0;
    for ip = 1:np
        bestf = min(Hfval(ip,:));
        worsf = max(Hfval(ip,:));
        %for is = 1:ns
        %    if Hgrad(ip,is) > 1.e-3 && Hfval(ip,is) > bestf + 1.e-3(worsf-bestf)
        %        Htime(ip,is) = nan;
        %        Hiter(ip,is) = nan;
        %    end
        %end

        if not(Hgrad(ip,6) <= 1.e-3)
            LBFGS = LBFGS +1;
        end
        if not(Hgrad(ip,4) <= 1.e-3)
            disp(P(ip));
            QPS = QPS +1;
        end

        if max(Hfval(ip,:)) - min(Hfval(ip,:)) < 1.e-3
            I = [I ip];
        else
            %if (Hfval(ip,2) < Hfval(ip,3))
            %    fprintf("%15s %15.6e %15.6e %15.6e %15.6e ** \n",P(ip),Hfval(ip,1),Hfval(ip,2),Hfval(ip,3),Hfval(ip,4));        
            %else
            %    fprintf("%15s %15.6e %15.6e %15.6e %15.6e \n",P(ip),Hfval(ip,1),Hfval(ip,2),Hfval(ip,3),Hfval(ip,4));        
            %end
        end

    end

    disp(QPS)
    disp(LBFGS)

    for i = 1:ns
        S{i} = strrep(S{i},'_','\_');
    end
    
    figure()
    subplot(2,3,1);
    perf_profile(Htime(:,[2,6]),S([2,6]),'Time')
    subplot(2,3,2);
    perf_profile(Htime(:,[4,6]),S([4,6]),'Time')
    subplot(2,3,3);
    perf_profile(Htime(:,[2,4]),S([2,4]),'Time')
    subplot(2,3,4);
    perf_profile(Hiter(:,[2,6]),S([2,6]),'Time')
    subplot(2,3,5);
    perf_profile(Hiter(:,[4,6]),S([4,6]),'Time')
    subplot(2,3,6);
    perf_profile(Hiter(:,[2,4]),S([2,4]),'Time')

    figure()
    subplot(1,2,1);
    perf_profile(Htime(:,[5,7]),S([5,7]),'Time')
    subplot(1,2,2);
    perf_profile(Hiter(:,[5,7]),S([5,7]),'Iter')


    figure()
    subplot(2,3,1);
    perf_profile(Htime(I,[2,1]),S([2,1]),'Time')
    subplot(2,3,2);
    perf_profile(Htime(I,[2,3]),S([2,3]),'Time')
    subplot(2,3,3);
    perf_profile(Htime(I,[2,4]),S([2,4]),'Time')

    subplot(2,3,4);
    perf_profile(Hiter(I,[2,1]),S([2,1]),'Iter')
    subplot(2,3,5);
    perf_profile(Hiter(I,[2,3]),S([2,3]),'Iter')
    subplot(2,3,6);
    perf_profile(Hiter(I,[2,4]),S([2,4]),'Iter')

    figure()
    subplot(1,2,1);
    perf_profile(Htime,S,'Time')

    subplot(1,2,2);
    perf_profile(Hiter,S,'Iter')

    figure()
    subplot(1,2,1);
    perf_profile(Htime(I,:),S,'Time')

    subplot(1,2,2);
    perf_profile(Hiter(I,:),S,'Iter')

end