import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d, corrcoef
from math import ceil
import os

def auto_seperate_discharge_relaxation(data:pd.DataFrame, dir:str, col_state:int, col_time:int, lb_discharge:str):
    start_index_discharge = data[(
        data.iloc[:, col_state] == lb_discharge) & (data.iloc[:, col_time] == 0)]
    start_index_discharge = np.array(start_index_discharge.index)
    k = 1
    try:
        os.mkdir(dir + '/discharge')
        print(f'>>> Discharge data will be saved to {dir}/discharge/')
    except:
        print(f'>>> Discharge data will be saved to {dir}/discharge/')
    for i in range(len(start_index_discharge)-1):
        try:
            start = start_index_discharge[i] - 1
            end = start_index_discharge[i+1]
            if end - start < 3:
                continue
            temp = data.iloc[start:end, :]
            outfile = dir + '/discharge/'+str(k)+'.xlsx'
            temp.to_excel(outfile, index=False)
            k += 1
            print(f'\t{outfile}. {start}:{end}, len={end-start}')
        except Exception as e:
            print(e)
            continue
    return start_index_discharge


def auto_seperate_charge_relaxation(data: pd.DataFrame, dir: str, col_state: int, col_time: int, lb_charge: str):
    start_index_charge = data[(data.iloc[:, col_state] == lb_charge) & (data.iloc[:, col_time] == 0)]
    start_index_charge = np.array(start_index_charge.index)
    k = 1
    try:
        os.mkdir(dir + '/charge')
        print(f'>>> Charge data will be saved to {dir}/charge/')
    except:
        print(f'>>> Charge data will be saved to {dir}/charge/')
    for i in range(len(start_index_charge)-1):
        try:
            start = start_index_charge[i] - 1
            end = start_index_charge[i+1]
            if end - start < 3:
                continue
            temp = data.iloc[start:end, :]
            outfile = dir + '/charge/'+str(k)+'.xlsx'
            temp.to_excel(outfile, index=False)
            k += 1
            print(f'\t{outfile}. {start}:{end}, len={end-start}')
        except Exception as e:
            print(e)
            continue
    return start_index_charge


def divide_data(data, dir, col_state, col_time, lb_charge, lb_discharge):    
    # * seperate massive data into many smaller files
    # * so that GITT fitting can be handeled more easily
    print("\n>>> Auto seperate discharge relaxation loops...")
    idx_discharge = auto_seperate_discharge_relaxation(data, dir, col_state, col_time, lb_discharge)
    print("\n>>> Auto seperate charge relaxation loops...")
    idx_charge = auto_seperate_charge_relaxation(data, dir, col_state, col_time, lb_charge)
    num_charge_files = max(len(idx_charge)-1, 0)
    num_discharge_files = max(len(idx_discharge)-1, 0)
    print(
        f"\nThere are {num_charge_files} of charge files and {num_discharge_files} of discharge files.")
    return num_charge_files, num_discharge_files


def load_data(filename: str):
    """
    Data example:
    工步类型 | 时间(h) | 总时间(h) | 电压(V) | 电流(A) | 比容量(mAh/g)
    搁置     | 0.000  |  0.000   |0.787   | 0.102   | 0.00
    Make sure headers contain at least as the Data example.
    Also see example file 'eg.xlsx' for more information.
    """
    data = pd.read_excel(filename, sheet_name=0)
    return data


def pick_discharge_data(data, min_time: float, max_time: float, col_state, col_time, lb_discharge):
    # pick up the discharge data that time in range (mintime, maxtime)
    # '工步类型' means 'step type' and '恒流放电' means 'gavalnostatic discharge'
    pick_data = data[data.iloc[:, col_state] == lb_discharge]
    # '时间' means 'step time'
    pick_data = pick_data[pick_data.iloc[:, col_time] < max_time]
    pick_data = pick_data[pick_data.iloc[:, col_time] > min_time]
    return pick_data


def pick_charge_data(data, min_time: float, max_time: float, col_state, col_time, lb_charge):
    # pick up the charge data that time in range (mintime, maxtime)
    # '工步类型' means 'step type' and '恒流充电' means 'gavalnostatic charge'
    pick_data = data[data.iloc[:, col_state] == lb_charge]
    # '时间' means 'time'
    pick_data = pick_data[pick_data.iloc[:, col_time] < max_time]
    pick_data = pick_data[pick_data.iloc[:, col_time] > min_time]
    return pick_data


def get_dEs(data, col_voltage):
    # static potential difference after and before this pulse/relaxation loop.
    # ! The first data point should be the last one of previous relaxation.
    # ! And the data only contain one loop.
    # get the static potential of the previous relaxation.
    Es1 = data.iloc[0, col_voltage]
    # get the static potential of the present relaxation.
    Es2 = data.iloc[-1, col_voltage]
    return Es2 - Es1


def fit(dis_time, disp_volt, h1):
    # fit single pulse
    # V(t) = E_I + k * \sqrt{t}
    sqrt_dis_time = np.sqrt(dis_time)
    slope, intercept = polyfit(sqrt_dis_time, disp_volt, 1)
    r2 = corrcoef(sqrt_dis_time, disp_volt,)[0, 1]
    r2 = r2 ** 2
    y_ = poly1d((slope, intercept))
    # print(f"{slope=:.4e}, {intercept=:.4e}, {r2=:.4f}")
    h1.scatter(sqrt_dis_time, disp_volt)
    h1.plot(sqrt_dis_time, y_(sqrt_dis_time), 'r--')
    h1.set_title(f"{slope=:.4e}, {r2=:.4f}")
    return slope, intercept, r2


def auto_fit_discharge(filename, min_time, max_time, ax1, ax2, cols, lb_discharge, unit_factor):
    col_time_tot, col_time, col_current, col_voltage, col_capacity, col_state = cols
    uf_time_set, uf_time_fit, uf_current, uf_voltage, uf_capacity, uf_length = unit_factor
    data = load_data(filename)
    data.iloc[:, col_time_tot] *= uf_time_set
    data.iloc[:, col_time]  *= uf_time_set
    data.iloc[:, col_current] *= uf_current
    data.iloc[:, col_voltage] *= uf_voltage
    data.iloc[:, col_capacity] *= uf_capacity
    
    min_time = min_time * uf_time_fit
    max_time = max_time * uf_time_fit

    dis_data = pick_discharge_data(data, min_time, max_time, col_state, col_time, lb_discharge)
    dis_time = dis_data.iloc[:, col_time]
    # pulse time in seconds
    tau = data[data.iloc[:, col_state] == lb_discharge].iloc[:, col_time].max()
    voltage = dis_data.iloc[:, col_voltage]
    capacity = dis_data.iloc[0, col_capacity]
    iR_drop = data.iloc[1, col_voltage] - data.iloc[0, col_voltage]
    current = dis_data.iloc[2, col_current]
    # static potential before pulse
    Es0 = data.iloc[0, col_voltage]
    # Ohmic resistance
    Rohm = iR_drop / current
    slope, inter, r2 = fit(dis_time, voltage, ax1)
    # charge transfer resistance
    Rct = (inter - Es0) / current - Rohm
    dEs = get_dEs(data, col_voltage)
    ax2.plot((data.iloc[:, col_time_tot] - data.iloc[0, col_time_tot]), data.iloc[:, col_voltage], 'o--')
    if r2 < 0.8:
        info = f"{capacity:.2f}\t{slope:.6e}\t{dEs:.6f}\t{inter:.6e}\t{r2:.6f}\t!!!"
    elif r2 < 0.9:
        info = f"{capacity:.2f}\t{slope:.6e}\t{dEs:.6f}\t{inter:.6e}\t{r2:.6f}\t!!"
    elif r2 < 0.95:
        info = f"{capacity:.2f}\t{slope:.6e}\t{dEs:.6f}\t{inter:.6e}\t{r2:.6f}\t!"
    else:
        info = f"{capacity:.2f}\t{slope:.6e}\t{dEs:.6f}\t{inter:.6e}\t{r2:.6f}"
    print(info)
    return info, capacity, slope, dEs, tau, Rohm, Rct, inter, r2


def auto_fit_charge(filename, min_time, max_time, ax1, ax2, cols, lb_charge, unit_factor):
    col_time_tot, col_time, col_current, col_voltage, col_capacity, col_state = cols
    uf_time_set, uf_time_fit, uf_current, uf_voltage, uf_capacity, uf_length = unit_factor
    data = load_data(filename)
    data.iloc[:, col_time_tot] *= uf_time_set
    data.iloc[:, col_time] *= uf_time_set
    data.iloc[:, col_current] *= uf_current
    data.iloc[:, col_voltage] *= uf_voltage
    data.iloc[:, col_capacity] *= uf_capacity

    min_time = min_time * uf_time_fit
    max_time = max_time * uf_time_fit

    dis_data = pick_charge_data(data, min_time, max_time, col_state, col_time, lb_charge)
    dis_time = dis_data.iloc[:, col_time]  # convert to seconds
    # pulse time in seconds
    tau = data[data.iloc[:, col_state] == lb_charge].iloc[:,col_time].max()
    iR_drop = data.iloc[1, col_voltage] - data.iloc[0, col_voltage]
    voltage = dis_data.iloc[:, col_voltage]
    Es0 = data.iloc[0, col_voltage]
    current = dis_data.iloc[2, col_current]
    # Ohmic resistance
    Rohm = iR_drop / current
    # print(dis_data.head)
    capacity = dis_data.iloc[0, col_capacity]/3600
    slope, inter, r2 = fit(dis_time, voltage, ax1)
    # charge transfer resistance
    Rct = (inter - Es0) / current - Rohm
    dEs = get_dEs(data, col_voltage)
    ax2.plot((data.iloc[:, col_time_tot] - data.iloc[0, col_time_tot]), data.iloc[:, col_voltage], 'o--')
    if r2 < 0.8:
        info = f"{capacity:.2f}\t{slope:.6e}\t{dEs:.6f}\t{inter:.6e}\t{r2:.6f}\t!!!"
    elif r2 < 0.9:
        info = f"{capacity:.2f}\t{slope:.6e}\t{dEs:.6f}\t{inter:.6e}\t{r2:.6f}\t!!"
    elif r2 < 0.95:
        info = f"{capacity:.2f}\t{slope:.6e}\t{dEs:.6f}\t{inter:.6e}\t{r2:.6f}\t!"
    else:
        info = f"{capacity:.2f}\t{slope:.6e}\t{dEs:.6f}\t{inter:.6e}\t{r2:.6f}"
    print(info)
    return info, capacity, slope, dEs, tau, Rohm, Rct, inter, r2



def func_D(capacity, slope, Es, tau, R):
    # this function should return Diffusity in cm^2/s
    # OR other any function: func(capacity, slope, Es, tau, *args)
    return 4/np.pi * (R/(3*tau) * Es / slope)**2


def func_D_tube(capacity, slope, Es, tau, L):
    # diffusivity of tubic transport alone the axial direction
    return 4/np.pi * (L/(tau) * Es / slope)**2


def discharge_process(dir, fignum, s1, args, func_D, savefig, min_time, max_time, cols, lb_discharge, unit_factor):
    # process discharge data in dir, and return GITT results to files.
    s2 = ceil(fignum/s1)
    fig1 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    fig2 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    print("\n====================\n>>> Discharge " + dir)
    print("---------------------\ncapacity\tslope\tdEs\tinter\tR2")
    with open(dir+'/_discharge.txt', 'w') as f:
        f.write(
            "D(m^2/s)\tCapacity(mAh/g)\tSlope(V/s^0.5)\tdEs(V)\ttau(s)\tRohm(ohm)\tRct(ohm)\tInter\tR2\n")
    with open(dir+'/_discharge.txt', 'a+') as f:
        for i in range(1, fignum+1):
            try:
                filename = dir + '/discharge/' + str(i) + '.xlsx'
                ax1 = fig1.add_subplot(s2, s1, i)
                ax2 = fig2.add_subplot(s2, s1, i)
                info, capacity, slope, Es, tau, Rohm, Rct, inter, r2 = auto_fit_discharge(
                    filename, min_time, max_time, ax1, ax2, cols, lb_discharge, unit_factor)
                capacity /= 3600
                D = func_D(capacity, slope, Es, tau, *args) * unit_factor[-1]**2
                info = f"{D:.6e}\t{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{tau:.3f}\t{Rohm:.4f}\t{Rct:.4f}\t{inter:.6f}\t{r2:.6f}\n"
                f.write(info)
                ax1.set_xlabel(r"$\sqrt{t}\rm\ (s)$")
                ax1.set_ylabel(r"$E\rm(t)\ (V)$")
                ax2.set_xlabel(r"t (s)")
                ax2.set_ylabel(r"E(t) (V)")
            except Exception as e:
                print(f"{i} is failed!")
                print(e)
    if savefig:
        fig1.savefig(dir+'/_discharge.png')
    plt.close("all")


def charge_process(dir, fignum, s1, args, func_D, savefig, min_time, max_time, cols, lb_charge, unit_factor):
    # process charge data in dir, and return GITT results to files.
    s2 = ceil(fignum/s1)
    fig1 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    fig2 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    print("\n====================\n>>> Charge " + dir)
    print("---------------------\ncapacity\tslope\tdEs\tinter\tR2")
    with open(dir+'/_charge.txt', 'w') as f:
        f.write(
            "D(m^2/s)\tCapacity(mAh/g)\tSlope(V/s^0.5)\tdEs(V)\ttau(s)\tRohm(ohm)\tRct(ohm)\tInter\tR2\n")
    with open(dir+'/_charge.txt', 'a+') as f:
        for i in range(1, fignum+1):
            try:
                filename = dir + '/charge/' + str(i) + '.xlsx'
                ax1 = fig1.add_subplot(s2, s1, i)
                ax2 = fig2.add_subplot(s2, s1, i)
                info, capacity, slope, Es, tau, Rohm, Rct, inter, r2 = auto_fit_charge(
                    filename, min_time, max_time, ax1, ax2, cols, lb_charge, unit_factor)
                capacity /= 3600
                D = func_D(capacity, slope, Es, tau, *args)
                info = f"{D:.6e}\t{capacity:.2f}\t{slope:.6e}\t{Es:.6f}\t{tau:.3f}\t{Rohm:.4f}\t{Rct:.4f}\t{inter:.6f}\t{r2:.6f}\n"
                f.write(info)
                ax1.set_xlabel(r"$\sqrt{t}\rm\ (s)$")
                ax1.set_ylabel(r"$E\rm(t)\ (V)$")
                ax2.set_xlabel(r"t (s)")
                ax2.set_ylabel(r"E(t) (V)")
            except Exception as e:
                print(f"{i} is failed!")
                print(e)
    if savefig:
        fig1.savefig(dir+'/_charge.png')
    plt.close("all")