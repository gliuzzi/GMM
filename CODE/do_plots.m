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
%T = readtable("risultati-tot-7.txt","delimiter",'|');
%T = T(1:1464,:);
T = readtable("ris_20231206.txt","delimiter",'|');
T = T(1:2282,:);

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

SS = {
'$QPS_1$',
'QPS-Diagonale1',
'QPS-Diagonale2',
'$QPS_3$',
'QPS-Diagonale4',
'QPS-Diagonale5',
'QPS-Diagonale6',
'QPS-Diagonale7',
'QPS-Diagonale8',
'QPS-Newton',
'$QPS_2$',
'$lBFGS_{scipy}$',
'$CG_{scipy}$'
};

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

%keyboard

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

%keyboard
disp(QPS)
disp(LBFGS)

confronti = {[4,12], [5,12], [11,12], [1,12]};
confronti = {[4,13], [5,13], [11,13], [1,13], [11,1]};
confronti = {[4,13], [11,1]};
confronti = {[4,13,11,1,12],[11,12]};
%confronti = {[2,3,4,5,6,7,8,9]};
confronti = {[9,13]};
confronti = {[1,11,4,13]};
confronti = {[1,11,4,12]};
confronti = {[11,12]};

nc = size(confronti,2);

figure('Position',[0,0,1000,1000])
i = 1;
for pp = confronti
    pair = pp{1};
    subplot(2,nc,i);
    perf_profile(Htime(:,pair),SS(pair),'Time')
    subplot(2,nc,nc+i)
    perf_profile(Hiter(:,pair),SS(pair),'Iter')
    i = i+1;
end

figure('Position',[0,0,1000,1000])
i = 1;
for pp = confronti
    pair = pp{1};
    I = [];
    %nbest = 0;
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
            %if(Hfval(ip,pair(1)) < Hfval(ip,pair(2)))
            %    nbest = nbest+1;
            %end
        end
    
    end
    subplot(2,nc,i);
    perf_profile(Htime(I,pair),SS(pair),'Time')
    subplot(2,nc,nc+i)
    perf_profile(Hiter(I,pair),SS(pair),'Iter')
    i = i+1;
    nu = size(I,2);
    for p = 1:size(pair,2)
        fprintf("%20s wins on %3d/%3d\n",S(pair(p)),nbest(1,p),np)
    end
    %fprintf("%20s wins on %3d/%3d\n",S(pair(1)),nbest,np)
    %fprintf("%20s wins on %3d/%3d\n",S(pair(2)),np-nu-nbest,np)
    %disp(S(pair))
    %disp(size(I))
    %disp([S(pair(1))," is best on ", num2str(nbest)])
    %disp([S(pair(2))," is best on ", num2str(np-nu-nbest)])
end
