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
T = readtable("risultati.txt","delimiter",'|');
T = T(1:218,:);
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
    if T.Var8(row) <= 1.e-3
        Htime(ip,is) = T.Var5(row);
        Hiter(ip,is) = T.Var6(row);
    else
        Htime(ip,is) = nan;
        Hiter(ip,is) = nan;
    end
end

I = [];
for ip = 1:np
    if max(Hfval(ip,:)) - min(Hfval(ip,:)) < 1.e-3
        I = [I ip];
    end
end


