clear all, close all, clc
addpath('./utils');
addpath('./DATA');
addpath('./npy-matlab/npy-matlab/');

% 读取数据
xdat = readNPY('S01_clin_45hz.npy');
xdat = xdat';
% size(xdat): T, dim

% 截取感兴趣数据
% xdat = xdat(4240000:4600000, :);
% xdat = xdat(4480000:4600000, :);
xdat = xdat(4240000:4480000, :);
N = length(xdat);
sample_frep = 2048;
dt = 1/sample_frep;
tspan = [dt:dt:N];

%% 构建hankel矩阵
stackmax = 300;  % the number of shift-stacked rows
rmax = 20;  % maximum number of singular values

clear V, clear H
H = zeros(stackmax,size(xdat,1)-stackmax);
for k=1:stackmax
    H(k,:) = xdat(k:end-stackmax-1+k,1);
end
[U,S,V] = svd(H,'econ');
sigs = diag(S);
beta = size(H,1)/size(H,2);
thresh = optimal_SVHT_coef(beta,0) * median(sigs);
r = length(sigs(sigs>thresh))
r=min(rmax,r)
sigs_e = sigs(1:12);

%% 计算系统矩阵
x = V(1:end-1,1:r);
xprime = V(2:end,1:r);
Xi = pinv(xprime)*x;
B = Xi(1:r-1,r)
A = Xi(1:r-1,1:r-1)
sys = ss(A,B,eye(r-1),0*B,dt);

% 系统仿真
L = 1:size(x,1);
% [y,t] = lsim(sys,x(L,r),dt*(L-1),x(1,1:r-1));
[y,t] = lsim(sys,zeros(size(x,1),1),dt*(L-1),x(1,1:r-1));

%% #可视化
%% 图1：原始吸引子
L = 1:N;
plot3(xdat(L,1),xdat(L,2),xdat(L,3),'Color',[.1 .1 .1],'LineWidth',1.5)
axis on
view(-5,12)
axis tight
xlabel('x'), ylabel('y'), zlabel('z')
set(gca,'FontSize',14)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
set(gca, 'LooseInset', [0,0,0,0]);

%% 图2：时间序列可视化
% 相当重要
figure
plot([1:length(xdat)]',xdat(:,1),'k','LineWidth',2)
axis on

%% 图3： 时间嵌入吸引子
figure
L = 1:N-stackmax;
plot3(V(L,1),V(L,2),V(L,3),'Color',[.1 .1 .1],'LineWidth',1.5)
axis tight
xlabel('v_1'), ylabel('v_2'), zlabel('v_3')
set(gca,'FontSize',14)
view(34,22)
set(gcf,'Position',[100 100 600 400])
set(gcf,'PaperPositionMode','auto')
set(gca, 'LooseInset', [0,0,0,0]);

%% 图4： 模型时间序列
n = 7; % num_modes_disp

start_ind = 1;
end_ind = N - stackmax-1;
L = start_ind:end_ind;   % 连续绘图
L2 = start_ind:50:end_ind;   % 散点绘图

type='compact';

figure
for i=1:n-1
    tsubplot(n,1,i,type);
    plot(L,x(L,i),'Color',[.4 .4 .4],'LineWidth',1.5)
    hold on
    s_ylabel = sprintf('v_%d',i);
    xlabel('t'), ylabel(s_ylabel);
    plot(L2,y(L2,i),'.','Color',[0 0 .5],'LineWidth',5,'MarkerSize',10)
    xlim([start_ind end_ind])
    box on, axis on
    axis tight
end

tsubplot(n,1,n,type);
% plot(L,x(L,r),'Color',[.5 0 0],'LineWidth',1.5)
plot(L,zeros(size(x(L,r),1),1),'Color',[.5 0 0],'LineWidth',1.5)
xlabel('t'), ylabel('v_r')
xlim([start_ind end_ind])
box on, axis on
axis tight
% set(gcf,'Position',[100 100 2*250 2*175])
set(gcf,'PaperPositionMode','auto')

%% 图5： 重建吸引子
figure
plot3(y(L,1),y(L,2),y(L,3),'Color',[0 0 .5],'LineWidth',1.5)
axis tight
axis on
view(102,13)
set(gcf,'Position',[100 100 3*250 3*175])
set(gcf,'PaperPositionMode','auto')

%% 图6： 强迫项分析
figure
Vtest = std(V(:,r))*randn(200000,1);
[h,hc] = hist(V(:,r)-mean(V(:,r)),[-.02:.0025:.02]);%[-.03  -.02 -.015  -.0125 -.01:.0025:.01 .0125  .015 .02  .03]);
[hnormal,hnormalc] = hist(Vtest-mean(Vtest),[-.01:.0025:.01])
semilogy(hnormalc,hnormal/sum(hnormal),'--','Color',[.2 .2 .2],'LineWidth',4)
hold on
semilogy(hc,h/sum(h),'Color',[0.5 0 0],'LineWidth',4)
ylim([.0001 1])
xlim([-.02 .02])
axis on
set(gcf,'Position',[100 100 2*250 2*175])
set(gcf,'PaperPositionMode','auto')

%% 图7： U模态
figure
CC = [2 15 32;
    2 35 92;
    22 62 149;
    41 85 180;
    83 124 213;
    112 148 223;
    114 155 215];
plot(U(:,2*r+1 : 3*r),'Color',[.95 .8 .8],'LineWidth',0.4)
hold on
plot(U(:,r+1 : 2*r),'Color',[.8 .95 .8],'LineWidth',1)
plot(U(:,1 : r),'Color',[.5 .5 .5],'LineWidth',1.5)
for k=7:-1:1
    plot(U(:,k),'linewidth',1.5+2*k/10,'Color',CC(k,:)/255)
end

axis on
set(gcf,'Position',[100 100 2*250 2*175])
set(gcf,'PaperPositionMode','auto')

%% 爆发预测
L = 1:length(V);
inds = V(L,r).^2>5*1.e-6;
L = L(inds);
startvals = [];
endvals = [];
start = 55540;
clear interval hits endval newhit
numhits = 85;
for k=1:numhits;
    k
    startvals = [startvals; start];
    endmax = start+900;
    interval = start:endmax;
    hits = find(inds(interval));
    endval = start+hits(end);
    endvals = [endvals; endval];
    newhit = find(inds(endval+1:end));
    if length(newhit) >= 1
        start = endval+newhit(1);
    end
end
%
figure
for k=1:numhits
    plot3(V(startvals(k):endvals(k),1),V(startvals(k):endvals(k),2),V(startvals(k):endvals(k),3),'r','LineWidth',1.5), hold on
end
for k=1:numhits-1
    plot3(V(endvals(k):startvals(k+1),1),V(endvals(k):startvals(k+1),2),V(endvals(k):startvals(k+1),3),'Color',[.25 .25 .25],'LineWidth',1.5), hold on
end
axis tight
axis on
view(102,13)
set(gcf,'Position',[100 100 3*250 3*175])
set(gcf,'PaperPositionMode','auto')
