"""Code to fit cell population growth data using Modified Gompertz, Gompertz, and Logistic functions.
Authored by: Swarnavo Sarkar (swarnavo.sarkar@nist.gov)
"""

import sys,os
import numpy as np
import pandas as pd
import scipy.optimize as optim
import functions
import glob
from scipy.stats import gamma
import math

casename = 'growth_'

t_factor_table = {}
t_factor_table[1] = 12.706
t_factor_table[2] = 4.303
t_factor_table[3] = 3.182
t_factor_table[3] = 2.776

def get_CI(T2,sT2):
    theta = (sT2**2)/T2
    k = T2/theta

    low = gamma.ppf(0.025,k,0,theta)
    up = gamma.ppf(0.975,k,0,theta)

    #return up, low
    return (up - T2), (T2 - low)

def get_CI_t(T2,sT2,n):
    interval = sT2/math.sqrt(n)

    up = t_factor_table[n]*interval
    low = up

    #return up, low
    return up, low

data_directory = '/Users/sns9/Research/GrowthCurves/PublicationData/Study3'
#data_directory = '/Users/sns9/Research/COVID-19'
os.chdir(data_directory)

data_files = glob.glob(casename+'*.csv')#'growth_60.csv'


all_files = {}
times = []

for ff in data_files:
    t = float(ff.lstrip(casename).rstrip('.csv'))
    times.append(t)
    all_files[int(t)] = ff

times.sort()

p_array = np.zeros(shape=(len(data_files),10))
column_names = ['T','k','kerr1','kerr2','a','aerr1','aerr2','b','berr1','berr2']
parameters = pd.DataFrame(p_array,index=[i for i in range(0,len(data_files))],columns=column_names)

#column_names = ['CT','T1','T2']
column_names = ['CT','T1','T1err1','T1err2','T2','T2err1','T2err2']
inflection_times = pd.DataFrame(np.zeros(shape=(len(data_files),7)),index=[i for i in range(0,len(data_files))],columns=column_names)
report_inflection_times = pd.DataFrame(np.zeros(shape=(len(data_files),7)),index=[i for i in range(0,len(data_files))],columns=column_names)

for idx in range(0,len(times)):
    t = times[idx]
    filename = all_files[int(t)]

    pop_series = pd.read_csv(filename)

    time_array = pop_series['Time (h)'].to_numpy()
    time_array = time_array.astype(float)
    time_size = time_array.shape[0]

    stack_size = time_size*(len(pop_series.columns.values)-1)

    time_stack = np.zeros(stack_size,)
    data_stack = np.zeros(stack_size,)

    time_l, data_l = [], []

    #mean_data = np.zeros(time_array.shape)

    ik = 0

    n = len(pop_series.columns) - 2

    for col in pop_series.columns.values[1:]:
        this_data = pop_series[col].to_numpy()

        for p in range(0,this_data.shape[0]):
            if this_data[p]==this_data[p]:
                time_l.append(time_array[p])
                data_l.append(this_data[p])

    time_stack = np.array(time_l)
    data_stack = np.array(data_l)

    max_d = np.max(data_stack)

    max_inf_time = time_l[data_l.index(max_d)]

    # modified gompertz fit
    m_d = max(data_l)
    t_inf = time_l[data_l.index(m_d)]

    check = 1
    iter = 0

    k_up, k_low = 2.0*m_d, 0.0
    b_up, b_low = 1.0/(time_array[1]-time_array[0]), 1.0/(time_array[-1]-time_array[0])
    a_up, a_low = max_inf_time*b_up, 0.0


    while check==1:
        all_bounds = (np.array([k_low,a_low,b_low]),np.array([k_up,a_up,b_up]))

        #print(all_bounds)

        popt, pcov = optim.curve_fit(functions.gompertz,time_stack,data_stack,bounds=all_bounds)
        perr = np.sqrt(np.diag(pcov))

        check = 0
        #
        # for i in range(0,4):
        #     if perr[i]/popt[i]>1.0:
        #         check = 1
        #         print('Error in ',t, k_up,iter)
        #         break
        #
        # if check==1:
        #     k_up *= 2.0

        iter += 1

    """Print parameters
    """

    parameters.loc[idx,'T'] = t
    parameters.loc[idx,'k'] = popt[0]
    parameters.loc[idx,'a'] = popt[1]
    parameters.loc[idx,'b'] = popt[2]
    #parameters.loc[idx,'d'] = popt[3]
    parameters.loc[idx,'kerr1'] = perr[0]
    parameters.loc[idx,'aerr1'] = perr[1]
    parameters.loc[idx,'berr1'] = perr[2]
    #parameters.loc[idx,'derr1'] = perr[3]
    parameters.loc[idx,'kerr2'] = perr[0]
    parameters.loc[idx,'aerr2'] = perr[1]
    parameters.loc[idx,'berr2'] = perr[2]
    #parameters.loc[idx,'derr2'] = perr[3]

    """Print modified Gompertz value
    """

    #mg_fit = functions.modified_gompertz(time_array,popt[0],popt[1],popt[2],popt[3])

    #mg_sol = 'MGsol:,'+str(popt[0])+','+str(popt[1])+','+str(popt[2])+','+str(popt[3])
    #mg_cov = 'MGcov:,'+str(perr[0])+','+str(perr[1])+','+str(perr[2])+','+str(perr[3])

    #pop_series.insert(len(pop_series.columns.values),"mg_fit",list(mg_fit),True)

    #pop_series.to_csv('gompertzD_'+filename,index=None)

    """Inflection times
    """

    times_set = np.linspace(1,1000,10000)

    t1, t2 = functions.compute_simple_gompertz_inflections(times_set,popt[0],popt[1],popt[2])
    #max_t, min_t = functions.compute_secondary_inflections(times_set,popt[0],popt[1],popt[2])

    inflection_times.loc[idx,'CT'] = t
    inflection_times.loc[idx,'T1'] = t1
    inflection_times.loc[idx,'T2'] = t2
    #inflection_times.loc[idx,'T3'] = min_t

    report_inflection_times.loc[idx,'CT'] = t
    report_inflection_times.loc[idx,'T1'] = t1
    report_inflection_times.loc[idx,'T2'] = t2

    T1_error = t1*(abs(perr[1]/popt[1]) + abs(perr[2]/popt[2]))

    up, low = get_CI(t1,T1_error)
    #up, low = get_CI_t(t1,T1_error,n)

    inflection_times.loc[idx,'T1err1'] = up #up#T2_error
    inflection_times.loc[idx,'T1err2'] = min(low,t1) #low#T2_error

    report_inflection_times.loc[idx,'T1err1'] = max(t1 - low,0) #up#T2_error
    report_inflection_times.loc[idx,'T1err2'] = t1 + up #low#T2_error

    T2_error = t2*(abs(perr[1]/popt[1]) + abs(perr[2]/popt[2]))

    up, low = get_CI(t2,T2_error)
    #up, low = get_CI_t(t2,T2_error,n)

    inflection_times.loc[idx,'T2err1'] = up #up#T2_error
    inflection_times.loc[idx,'T2err2'] = min(low,t2) #low#T2_error

    report_inflection_times.loc[idx,'T2err1'] = max(t2 - low,0) #up#T2_error
    report_inflection_times.loc[idx,'T2err2'] = t2 + up #low#T2_error

    # Compute modified gompertz growth properties
    #out_string_mg = functions.compute_mg_properties(popt[0],popt[1],popt[2],popt[3])

    rate_times = np.linspace(0.0,np.max(time_array),200)
    fit_N = functions.gompertz(rate_times,popt[0],popt[1],popt[2])

    total_data = np.zeros(shape=(rate_times.shape[0],2))
    total_data[:,0] = rate_times
    total_data[:,1] = fit_N

    np.savetxt('fit_'+filename,total_data,delimiter=',')

    #mg_rate, fine_Ns, g_Ns = functions.compute_mg_rates(popt[0],popt[1],popt[2],popt[3],rate_times)

    # of = open('mg_rate_'+filename,'w')
    #
    # print('T,N,dN,gN',file=of)
    #
    # for p in range(0,len(list(rate_times))):
    #     print(str(rate_times[p])+','+str(fine_Ns[p])+','+str(mg_rate[p])+','+str(g_Ns[p]),file=of)
    #
    # of.close()

    # properties_file = open('gD_properties_'+filename,'w')
    #
    # print('t_inf, N_inf, max growth rate',file=properties_file)
    # print(out_string_mg,file=properties_file)
    # properties_file.close()

    # error_file = open('errors_'+filename,'w')
    # print(',k,a,b,d',file=error_file)
    # print(mg_sol,file=error_file)
    # print(mg_cov,file=error_file)
    # error_file.close()


    # time_set = np.linspace(1,1000,1000)
    #
    # of = open('long_mg_projection.csv','w')
    #
    # for i in range(0,time_set.shape[0]):
    #     mg_pop = functions.modified_gompertz(time_set[i],popt[0],popt[1],popt[2],popt[3])+1
    #     print(str(time_set[i])+','+str(mg_pop),file=of)
    #
    # of.close()

    # times_set = np.linspace(1,1000,10000)
    #
    # p_t = functions.compute_primary_inflections(times_set,popt[0],popt[1],popt[2],popt[3])
    # max_t, min_t = functions.compute_secondary_inflections(times_set,popt[0],popt[1],popt[2],popt[3])
    #
    # of = open('mg_time_'+filename,'w')
    # print(str(p_t)+','+str(max_t)+','+str(min_t),file=of)
    # of.close()

parameters.rename(columns={'kerr1': '+','aerr1': '+', 'berr1': '+'}, inplace=True)
parameters.rename(columns={'kerr2': '-','aerr2': '-', 'berr2': '-'}, inplace=True)

inflection_times.rename(columns={'T1err1': '+','T1err2': '-'}, inplace=True)
inflection_times.rename(columns={'T2err1': '+','T2err2': '-'}, inplace=True)

report_inflection_times.rename(columns={'T1err1': 'Lower CI','T1err2': 'Upper CI'}, inplace=True)
report_inflection_times.rename(columns={'T2err1': 'Lower CI','T2err2': 'Upper CI'}, inplace=True)

#print(list(parameters))

parameters.to_csv('gompertz_parameter_summary.csv',index=None)
inflection_times.to_csv('gompertz_inflection_points_summary.csv',index=None)

report_inflection_times.to_csv('gompertz_inflection_points_report.csv',index=None)
