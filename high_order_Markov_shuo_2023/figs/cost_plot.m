

memory_cost_ag1 = load("memoryCost_apbm_1st.mat");
memory_cost_ag1 = memory_cost_ag1.memoryCost;
memory_cost_ag1 = [memory_cost_ag1(1)*ones(25,1);memory_cost_ag1];
memory_cost_ag3 = load("memoryCost_agapbm_3rd.mat");
memory_cost_ag3 = memory_cost_ag3.memoryCost;
memory_cost_ag3 = [memory_cost_ag3(1)*ones(25,1);memory_cost_ag3];
memory_cost_ap3 = load("memoryCost_apapbm_3rd.mat");
memory_cost_ap3 = memory_cost_ap3.memoryCost;
memory_cost_ap3 = [memory_cost_ap3(1)*ones(25,1);memory_cost_ap3];

time_cost_ag1 = load("timeCost_apbm_1st.mat");
time_cost_ag1 = time_cost_ag1.timeData;
time_cost_ag3 = load("timeCost_agapbm_3rd.mat");
time_cost_ag3 = time_cost_ag3.timeData;
time_cost_ap3 = load("timeCost_apapbm_3rd.mat");
time_cost_ap3 = time_cost_ap3.timeData;

mc = 1:1:200;
h_time_cost = figure;
plot(mc, time_cost_ag1, '-o', "Color","#f0662b");
hold on
plot(mc, time_cost_ag3, '-o', "Color", "#0072bd");
plot(mc, time_cost_ap3, '-o', "Color","#068a0e");
xlabel('Number of Monte Carlo Simulations', "FontSize",16);
ylabel('Time Cost (s)', "FontSize",16);
% title('Time Cost Over 200 Monte Carlo Simulations', "FontSize",16);
legend("1st-order APBM", "3rd-order AG-APBM", "3rd-order AP-APBM", "FontSize",14)

mc = 1:1:200;
h_memory_cost = figure;
plot(mc, memory_cost_ag1, '-o', "Color","#f0662b");
hold on
plot(mc, memory_cost_ag3, '-o', "Color", "#0072bd");
plot(mc, memory_cost_ap3, '-o',"Color","#068a0e");
xlabel('Number of Monte Carlo Simulations', "FontSize",16);
ylabel('Memory Usage (MB)', "FontSize",16);
% title('Memory Usage Over 200 Monte Carlo Simulations', "FontSize",16);
legend("1st-order APBM", "3rd-order AG-APBM", "3rd-order AP-APBM", "FontSize",14)