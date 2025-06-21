# python3.11
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d, corrcoef
from math import ceil
import sys
import tkinter as tk
from tkinter import filedialog


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


def pick_discharge_data(data, min_time: float, max_time: float):
    # pick up the discharge data that time in range (mintime, maxtime)
    # '工步类型' means 'step type' and '恒流放电' means 'gavalnostatic discharge'
    pick_data = data[data['工步类型'] == '恒流放电']
    # '时间' means 'step time'
    pick_data = pick_data[pick_data['时间(h)'] < max_time]
    pick_data = pick_data[pick_data['时间(h)'] > min_time]
    return pick_data


def pick_charge_data(data, min_time: float, max_time: float):
    # pick up the charge data that time in range (mintime, maxtime)
    # '工步类型' means 'step type' and '恒流充电' means 'gavalnostatic charge'
    pick_data = data[data['工步类型'] == '恒流充电']
    # '时间' means 'time'
    pick_data = pick_data[pick_data['时间(h)'] < max_time]
    pick_data = pick_data[pick_data['时间(h)'] > min_time]
    return pick_data


def get_dEs(data):
    # static potential difference after and before this pulse/relaxation loop.
    # ! The first data point should be the last one of previous relaxation.
    # ! And the data only contain one loop.
    # get the static potential of the previous relaxation.
    Es1 = data['电压(V)'].iloc[0]
    # get the static potential of the present relaxation.
    Es2 = data['电压(V)'].iloc[-1]
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


def auto_fit_discharge(filename, min_time, max_time, ax1, ax2):
    data = load_data(filename)
    dis_data = pick_discharge_data(data, min_time, max_time)
    dis_time = dis_data['时间(h)']*3600  # convert to seconds
    # pulse time in seconds
    tau = data[data['工步类型'] == '恒流放电']['时间(h)'].max()*3600
    voltage = dis_data['电压(V)']
    capacity = dis_data['比容量(mAh/g)'].iloc[0]
    iR_drop = data['电压(V)'].iloc[1] - data['电压(V)'].iloc[0]
    current = dis_data['电流(A)'].iloc[2]
    # static potential before pulse
    Es0 = data['电压(V)'].iloc[0]
    # Ohmic resistance
    Rohm = iR_drop / current
    slope, inter, r2 = fit(dis_time, voltage, ax1)
    # charge transfer resistance
    Rct = (inter - Es0) / current - Rohm
    dEs = get_dEs(data)
    ax2.plot((data['总时间(h)'] - data['总时间(h)'].iloc[0])
             * 3600, data['电压(V)'], 'o--')
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


def auto_fit_charge(filename, min_time, max_time, ax1, ax2):
    data = load_data(filename)
    dis_data = pick_charge_data(data, min_time, max_time)
    dis_time = dis_data['时间(h)']*3600  # convert to seconds
    # pulse time in seconds
    tau = data[data['工步类型'] == '恒流充电']['时间(h)'].max()*3600
    iR_drop = data['电压(V)'].iloc[1] - data['电压(V)'].iloc[0]
    voltage = dis_data['电压(V)']
    Es0 = data['电压(V)'].iloc[0]
    current = dis_data['电流(A)'].iloc[2]
    # Ohmic resistance
    Rohm = iR_drop / current
    # print(dis_data.head)
    capacity = dis_data['比容量(mAh/g)'].iloc[0]
    slope, inter, r2 = fit(dis_time, voltage, ax1)
    # charge transfer resistance
    Rct = (inter - Es0) / current - Rohm
    dEs = get_dEs(data)
    ax2.plot((data['总时间(h)'] - data['总时间(h)'].iloc[0])
             * 3600, data['电压(V)'], 'o--')
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


def auto_seperate_discharge_relaxation(data, dir):
    start_index_discharge = data[(
        data['工步类型'] == '恒流放电') & (data['时间(h)'] == 0)]
    start_index_discharge = np.array(start_index_discharge.index)
    k = 1
    for i in range(len(start_index_discharge)):
        try:
            start = start_index_discharge[i] - 1
            end = start_index_discharge[i+1]
            if end - start < 3:
                continue
            temp = data.iloc[start:end, :]
            outfile = dir + '/discharge'+str(k)+'.xlsx'
            temp.to_excel(outfile)
            k += 1
            print(f'\t{outfile}. {start}:{end}, len={end-start}')
        except:
            continue
    return start_index_discharge


def auto_seperate_charge_relaxation(data, dir):
    start_index_charge = data[(data['工步类型'] == '恒流充电') & (data['时间(h)'] == 0)]
    start_index_charge = np.array(start_index_charge.index)
    k = 1
    for i in range(len(start_index_charge)):
        try:
            start = start_index_charge[i] - 1
            end = start_index_charge[i+1]
            if end - start < 3:
                continue
            temp = data.iloc[start:end, :]
            outfile = dir + '/charge'+str(k)+'.xlsx'
            temp.to_excel(outfile)
            k += 1
            print(f'\t{outfile}. {start}:{end}, len={end-start}')
        except:
            continue
    return start_index_charge


def divide_data(filename, dir, sheet_name):
    try:
        import os
        os.mkdir(dir)
        print(f"Creating directory {dir}")
    except:
        pass
    # * read gitt data
    print(f">>> Reading data from file: {filename}\nThis may take minutes...")
    data_list = pd.read_excel(filename, sheet_name=sheet_name)
    print(f">>> Concat dataframe of sheet_name {sheet_name}...")
    data = pd.concat(data_list.values(), ignore_index=True)
    # * seperate massive data into many smaller files
    # * so that GITT fitting can be handeled more easily
    print("\n>>> Auto seperate discharge relaxation loops...")
    idx_discharge = auto_seperate_discharge_relaxation(data, dir)
    print("\n>>> Auto seperate charge relaxation loops...")
    idx_charge = auto_seperate_charge_relaxation(data, dir)
    num_charge_files = len(idx_charge)-1
    num_discharge_files = len(idx_discharge)-1
    print(
        f"\nThere are {num_charge_files} of charge files and {num_discharge_files} of discharge files.")
    return num_charge_files, num_discharge_files


def func_D(capacity, slope, Es, tau, R):
    # this function should return Diffusity in cm^2/s
    # OR other any function: func(capacity, slope, Es, tau, *args)
    return 4/np.pi * (R/(3*tau) * Es / slope)**2


def func_D_tube(capacity, slope, Es, tau, L):
    # diffusivity of tubic transport alone the axial direction
    return 4/np.pi * (L/(tau) * Es / slope)**2


def discharge_process(dir, fignum, s1, args, func_D=func_D, savefig=True, min_time=4/3600, max_time=60/3600):
    # process discharge data in dir, and return GITT results to files.
    s2 = ceil(fignum/s1)
    fig1 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    fig2 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    print("\n====================\n>>> Discharge " + dir)
    print("---------------------\ncapacity\tslope\tdEs\tinter\tR2")
    with open(dir+'/_discharge.txt', 'w') as f:
        f.write(
            "D(cm^2/s)\tCapacity(mAh/g)\tSlope(V/s^0.5)\tdEs(V)\ttau(s)\tRohm(ohm)\tRct(ohm)\tInter\tR2\n")
    with open(dir+'/_discharge.txt', 'a+') as f:
        for i in range(1, fignum+1):
            try:
                filename = dir + '/discharge' + str(i) + '.xlsx'
                ax1 = fig1.add_subplot(s2, s1, i)
                ax2 = fig2.add_subplot(s2, s1, i)
                info, capacity, slope, Es, tau, Rohm, Rct, inter, r2 = auto_fit_discharge(
                    filename, min_time, max_time, ax1, ax2)
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
        fig1.savefig(dir+'/_discharge.png')
    plt.close("all")


def charge_process(dir, fignum, s1, args, func_D=func_D, savefig=True, min_time=4/3600, max_time=60/3600):
    # process charge data in dir, and return GITT results to files.
    s2 = ceil(fignum/s1)
    fig1 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    fig2 = plt.figure(figsize=(s1*4, s2*5), dpi=300)
    print("\n====================\n>>> Charge " + dir)
    print("---------------------\ncapacity\tslope\tdEs\tinter\tR2")
    with open(dir+'/_charge.txt', 'w') as f:
        f.write(
            "D(cm^2/s)\tCapacity(mAh/g)\tSlope(V/s^0.5)\tdEs(V)\ttau(s)\tRohm(ohm)\tRct(ohm)\tInter\tR2\n")
    with open(dir+'/_charge.txt', 'a+') as f:
        for i in range(1, fignum+1):
            try:
                filename = dir + '/charge' + str(i) + '.xlsx'
                ax1 = fig1.add_subplot(s2, s1, i)
                ax2 = fig2.add_subplot(s2, s1, i)
                info, capacity, slope, Es, tau, Rohm, Rct, inter, r2 = auto_fit_charge(
                    filename, min_time, max_time, ax1, ax2)
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


class GITT_GUI:
    def __init__(self, init_window_name):
        self.init_window_name = init_window_name
        self._import_filename = tk.StringVar()
        self._dirpath = tk.StringVar()
        self._record_sheetname_list = ['record']
        self._min_time = tk.DoubleVar(value=2)
        self._max_time = tk.DoubleVar(value=20)
        self._num_discharge = tk.IntVar()
        self._num_charge = tk.IntVar()
        self._radius = tk.DoubleVar(value=1e-4)
        self._preview_num = tk.IntVar(value=1)

    def set_init_window(self):
        self.init_window_name.title("GITT_v0.1")  # 窗口名
        self.init_window_name.geometry('800x400+10+10')
        # 标签

        tk.Label(self.init_window_name, textvariable=self._dirpath, wraplength=600).grid(
            row=1, column=1, columnspan=4, padx=10, pady=5, sticky=tk.W)
        tk.Button(self.init_window_name, text='Export data dirpath', command=self.set_export_dirpath).grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W+tk.E)

        # tk.Label(self.init_window_name, text='Data filename').grid(row=1, column=0, padx=5, pady=5)
        tk.Label(self.init_window_name, textvariable=self._import_filename, wraplength=600).grid(
            row=2, column=1, columnspan=4, padx=10, pady=5, sticky=tk.W)
        tk.Button(self.init_window_name, text='Import datafile', command=self.set_import_filename).grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W+tk.E)

        self.entry_sheetname = tk.Entry(
            self.init_window_name, text='record', width=45)
        self.entry_sheetname.grid(
            row=3, column=1, columnspan=4, padx=10, pady=5, sticky=tk.W+tk.E)
        tk.Button(self.init_window_name, text='Get sheetname', command=self.set_sheetname).grid(
            row=3, column=0, padx=5, pady=5, sticky=tk.W+tk.E)

        self.button_divide_data = tk.Button(
            self.init_window_name, text='Divide data', command=self.divide_data, state=tk.DISABLED)
        self.button_divide_data.grid(
            row=4, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        tk.Label(self.init_window_name, text="num_charge: ").grid(
            row=4, column=1, padx=5, pady=5, sticky=tk.E)
        self.entry_num_charge = tk.Entry(
            self.init_window_name, textvariable=self._num_charge)
        self.entry_num_charge.grid(
            row=4, column=2, padx=5, pady=5, sticky=tk.W+tk.E)
        tk.Label(self.init_window_name, text="num_discharge: ").grid(
            row=4, column=3, padx=5, pady=5, sticky=tk.E)
        self.entry_num_discharge = tk.Entry(
            self.init_window_name, textvariable=self._num_discharge)
        self.entry_num_discharge.grid(
            row=4, column=4, padx=5, pady=5, sticky=tk.W+tk.E)

        # tk.Button(self.init_window_name, text='Set time region', command=self.set_time_region).grid(row=4, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        tk.Label(self.init_window_name, text="min_time (s): ").grid(
            row=5, column=1, padx=5, pady=5, sticky=tk.E)
        self.entry_min_time = tk.Entry(
            self.init_window_name, textvariable=self._min_time)
        self.entry_min_time.grid(
            row=5, column=2, padx=5, pady=5, sticky=tk.W+tk.E)
        tk.Label(self.init_window_name, text="max_time (s): ").grid(
            row=5, column=3, padx=5, pady=5, sticky=tk.E)
        self.entry_max_time = tk.Entry(
            self.init_window_name, textvariable=self._max_time)
        self.entry_max_time.grid(
            row=5, column=4, padx=5, pady=5, sticky=tk.W+tk.E)

        tk.Entry(self.init_window_name, textvariable=self._radius).grid(
            row=6, column=2, padx=5, pady=5, sticky=tk.W+tk.E)
        tk.Label(self.init_window_name, text="Particle radius (cm): ").grid(
            row=6, column=1, padx=5, pady=5, sticky=tk.E)
        tk.Entry(self.init_window_name, textvariable=self._preview_num).grid(
            row=6, column=4, padx=5, pady=5, sticky=tk.W+tk.E)
        tk.Label(self.init_window_name, text="Preview No.").grid(
            row=6, column=3, padx=5, pady=5, sticky=tk.E)

        self.button_pre_charge = tk.Button(
            self.init_window_name, text='Preview fit charge', command=self.preview_fit_charge, state=tk.DISABLED)
        self.button_pre_charge.grid(
            row=7, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        self.button_pre_discharge = tk.Button(
            self.init_window_name, text='Preview fit discharge', command=self.preview_fit_discharge, state=tk.DISABLED)
        self.button_pre_discharge.grid(
            row=7, column=3, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)

        self.button_fit_all = tk.Button(
            self.init_window_name, text='Fit All', command=self.fit_all, state=tk.DISABLED)
        self.button_fit_all.grid(
            row=8, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        self.button_fit_charge = tk.Button(
            self.init_window_name, text='Fit charge', command=self.fit_charge, state=tk.DISABLED)
        self.button_fit_charge.grid(
            row=8, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        self.button_fit_discharge = tk.Button(
            self.init_window_name, text='Fit discharge', command=self.fit_discharge, state=tk.DISABLED)
        self.button_fit_discharge.grid(
            row=8, column=3, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        # tk.Button(self.init_window_name, text='Quit', command=self.init_window_name.).grid(row=8, column=3, padx=5, pady=5)

        self.button_show_result = tk.Button(
            self.init_window_name, text='Show result', command=self.show_result)
        self.button_show_result.grid(
            row=9, column=0, padx=5, pady=5, sticky=tk.E+tk.W)

        self.button_export_fit_parameter = tk.Button(
            self.init_window_name, text="Export fit parameter", command=self.export_fit_parameter)
        self.button_export_fit_parameter.grid(
            row=9, column=1, padx=5, pady=5, sticky=tk.E+tk.W)

        self.button_import_fit_parameter = tk.Button(
            self.init_window_name, text="Import fit parameter", command=self.import_fit_parameter)
        self.button_import_fit_parameter.grid(
            row=9, column=2, padx=5, pady=5, sticky=tk.E+tk.W)

    def set_import_filename(self):
        filename = filedialog.askopenfilename(
            title="select file to be processed", filetypes=[("xlsx", '.xlsx')])
        if filename.strip() != '':
            self._import_filename.set(filename)  # 设置变量filename的值
            if self._dirpath.get() != '' and len(self._record_sheetname_list) > 0:
                self.button_divide_data.config(state=tk.ACTIVE)
        else:
            print("do not choose file")

    def set_export_dirpath(self):
        fileDir = filedialog.askdirectory(
            title="select dirpath to export data")  # 选择目录，返回目录名
        if fileDir.strip() != '':
            self._dirpath.set(fileDir)  # 设置变量outputpath的值
            import os
            files = os.listdir(fileDir)
            num_charge = len(
                [f for f in files if 'charge' in f and '.xlsx' in f and 'dis' not in f])
            num_discharge = len(
                [f for f in files if 'discharge' in f and '.xlsx' in f])
            self._num_charge.set(num_charge)
            self._num_discharge.set(num_discharge)

            file_fit_para = fileDir + '/_fit_parameter.fit'
            if os.path.isfile(file_fit_para):
                para = np.loadtxt(file_fit_para, delimiter=',')
                self._min_time.set(para[0])
                self._max_time.set(para[1])
                self._radius.set(para[2])

            if self._num_charge.get() > 0:
                self.button_pre_charge.config(state=tk.NORMAL)
                self.button_fit_charge.config(state=tk.NORMAL)
            if self._num_discharge.get() > 0:
                self.button_pre_discharge.config(state=tk.NORMAL)
                self.button_fit_discharge.config(state=tk.NORMAL)
            if self._import_filename.get() != '' and len(self._record_sheetname_list) > 0:
                self.button_divide_data.config(state=tk.ACTIVE)
            if self._num_charge.get() > 0 and self._num_discharge.get() > 0:
                self.button_fit_all.config(state=tk.NORMAL)
        else:
            print("do not choose Dir")

    def set_sheetname(self):
        result = self.entry_sheetname.get()
        self._record_sheetname_list = result.split(',')
        if self._import_filename.get() != '' and self._dirpath.get() != '' and len(self._record_sheetname_list) > 0:
            self.button_divide_data.config(state=tk.ACTIVE)
        print(self._record_sheetname_list)

    def divide_data(self):
        num_charge, num_discharge = divide_data(self._import_filename.get(),
                                                self._dirpath.get(),
                                                sheet_name=self._record_sheetname_list)
        self._num_charge.set(num_charge)
        self._num_discharge.set(num_discharge)
        print(self._num_charge.get(), self._num_discharge.get())
        if self._num_charge.get() > 0:
            self.button_pre_charge.config(state=tk.NORMAL)
            self.button_fit_charge.config(state=tk.NORMAL)
        if self._num_discharge.get() > 0:
            self.button_pre_discharge.config(state=tk.NORMAL)
            self.button_fit_discharge.config(state=tk.NORMAL)
        if self._num_charge.get() > 0 and self._num_discharge.get() > 0:
            self.button_fit_all.config(state=tk.NORMAL)
        

    def preview_fit_discharge(self):
        i = self._preview_num.get()
        try:
            filename = self._dirpath.get() + '/discharge' + str(i) + '.xlsx'
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
            info, capacity, slope, Es, tau, Rohm, Rct, inter, r2 = auto_fit_discharge(
                filename, self._min_time.get()/3600, self._max_time.get()/3600, ax1, ax2)
            D = func_D(capacity, slope, Es, tau, self._radius.get())
            ax1.set_xlabel(r"$\sqrt{t}\rm\ (s)$")
            ax1.set_ylabel(r"$E\rm(t)\ (V)$")
            ax2.set_xlabel(r"t (s)")
            ax2.set_ylabel(r"E(t) (V)")
            plt.show()
        except Exception as e:
            print(f"{i} is failed!")
            print(e)

    def preview_fit_charge(self):
        i = self._preview_num.get()
        try:
            filename = self._dirpath.get() + '/charge' + str(i) + '.xlsx'
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
            info, capacity, slope, Es, tau, Rohm, Rct, inter, r2 = auto_fit_charge(
                filename, self._min_time.get()/3600, self._max_time.get()/3600, ax1, ax2)
            D = func_D(capacity, slope, Es, tau, self._radius.get())
            ax1.set_xlabel(r"$\sqrt{t}\rm\ (s)$")
            ax1.set_ylabel(r"$E\rm(t)\ (V)$")
            ax2.set_xlabel(r"t (s)")
            ax2.set_ylabel(r"E(t) (V)")
            plt.show()
        except Exception as e:
            print(f"{i} is failed!")
            print(e)

    def fit_all(self):
        self.fit_charge()
        self.fit_discharge()

    def fit_charge(self):
        charge_process(self._dirpath.get(), fignum=self._num_charge.get(), s1=6, args=(
            self._radius.get(),), func_D=func_D, savefig=True, min_time=self._min_time.get()/3600, max_time=self._max_time.get()/3600)
        print(">>> Charge process finished.")

    def fit_discharge(self):
        discharge_process(self._dirpath.get(), fignum=self._num_discharge.get(), s1=6, args=(
            self._radius.get(),), func_D=func_D, savefig=True, min_time=self._min_time.get()/3600, max_time=self._max_time.get()/3600)
        print(">>> Discharge process finished.")

    def show_result(self):
        file_charge = self._dirpath.get() + '/_charge.txt'
        file_discharge = self._dirpath.get() + '/_discharge.txt'
        try:
            data_charge = pd.read_csv(file_charge, delimiter='\t')
            capacity = data_charge['Capacity(mAh/g)']
            De = data_charge['D(cm^2/s)']
            fig = plt.figure()
            plt.subplot(311)
            plt.semilogy(capacity, De, 'o--', mfc='white', label='charge')
            plt.xlabel('Capacity(mAh/g)')
            plt.ylabel(r'$D\rm (cm^2/s)$')
            data_discharge = pd.read_csv(file_discharge, delimiter='\t')
            capacity = data_discharge['Capacity(mAh/g)']
            De = data_discharge['D(cm^2/s)']
            plt.semilogy(capacity, De, 'o--', mfc='white', label='discharge')
            plt.legend()

            plt.subplot(312)
            capacity = data_charge['Capacity(mAh/g)']
            Rct = data_charge['Rct(ohm)']
            plt.plot(capacity, Rct, 'r*--', label='charge')

            capacity = data_discharge['Capacity(mAh/g)']
            Rct = data_discharge['Rct(ohm)']
            plt.plot(capacity, Rct, 'b*--', label='discharge')
            plt.xlabel('Capacity(mAh/g)')
            plt.ylabel('Rct(ohm)')
            plt.legend()

            plt.subplot(313)
            capacity = data_charge['Capacity(mAh/g)']
            Rohm = data_charge['Rohm(ohm)']
            plt.plot(capacity, Rohm, 'r^--', label='charge')

            capacity = data_discharge['Capacity(mAh/g)']
            Rohm = data_discharge['Rohm(ohm)']
            plt.plot(capacity, Rohm, 'b^--', label='discharge')
            plt.xlabel('Capacity(mAh/g)')
            plt.ylabel('Rohm(ohm)')
            plt.legend()
            plt.show()
        except Exception as e:
            print(e)

    def export_fit_parameter(self):
        filename = self._dirpath.get() + '/_fit_parameter.fit'
        with open(filename, 'w') as f:
            info = f"{self._min_time.get()},{self._max_time.get()},{self._radius.get()}"
            f.write(info)
            print(f"min_time={self._min_time.get()}\t max_time={self._max_time.get()}\t radius={self._radius.get()}")
            print(f">>> Export fitting parameters to file: {filename}")

    def import_fit_parameter(self):
        filename = filedialog.askopenfilename(
            title="select fit parameter", filetypes=[("fit", '.fit')])
        if filename.strip() != '':
            para = np.loadtxt(filename, delimiter=',')
            self._min_time.set(para[0])
            self._max_time.set(para[1])
            self._radius.set(para[2])
            print(f">>> Import fitting parameters from file: {filename}")
            print(f"min_time={self._min_time.get()}\t max_time={self._max_time.get()}\t radius={self._radius.get()}")
        else:
            print("do not choose file")


def gui_start():
    init_window = tk.Tk()  # 实例化出一个父窗口
    GITT_PORTAL = GITT_GUI(init_window)
    # 设置根窗口默认属性
    GITT_PORTAL.set_init_window()
    init_window.mainloop()


if __name__ == '__main__':
    gui_start()
    """
    导出数据时，务必导出以下几列，格式及单位务必严格遵守，列名务必保持一致。建议以BTSv8.0导出。
    导出前设置时间单位格式为“小时”或者“h”，设置容量累计模式。
    
    Data example:
    工步类型 | 时间(h) | 总时间(h) | 电压(V) | 电流(A) | 比容量(mAh/g)
    搁置     | 0.000  |  0.000   |0.787   | 0.102   | 0.00
    Make sure headers contain at least as the Data example.
    Also see example file 'eg.xlsx' for more information.
    """

    """
    # ! Step 1: divide data into smaller xlsx files
    ## * 直接设置数据文件名和分割数据存放文件夹
    # filename = '20240108/NCM-CNT-GITT-cycle-144.73mg_10th.xlsx'
    # dir = '20240108/NCM-CNT-GITT-cycle-144.73mg_10th'
    ## * 交互模式读取文件名和文件夹
    filename = filedialog.askopenfilename()
    dir = filedialog.askdirectory()
    ## * 所有记录数据sheet的名称，填入下方列表，程序可自动合并
    sheet_name = ['record']
    ## * 如果已分割数据，则注释第1行，将charge和discharge数量填入第2行，并取消注释
    num_charge, num_discharge = divide_data(filename, dir, sheet_name=sheet_name)
    # num_charge, num_discharge = 75, 58

    # ! Fitting discharge data
    ## * 如线性度差，调整min_time和max_time使拟合区间处于扩散控制区，单位h
    ## * args中为颗粒半径，单位为cm，其单位与扩散系数单位一致
    ## !!! 其余参数请勿修改
    discharge_process(dir, fignum=num_discharge, s1=6, args=(
        1e-6*100,), func_D=func_D, savefig=True, min_time=20/3600, max_time=200/3600)

    # ! Fitting charge data
    ## * 如线性度差，调整min_time和max_time使拟合区间处于扩散控制区，单位h
    ## * args中为颗粒半径，单位为cm，其单位与扩散系数单位一致
    ## !!! 其余参数请勿修改
    charge_process(dir, fignum=num_charge, s1=6, args=(
        1e-6*100,), func_D=func_D, savefig=True, min_time=20/3600, max_time=200/3600)
    """


# D = 4/pi * (n_M*V_M/(A*tau) * dEs / slope)**2
# D = 4/pi * (R/(3*tau) * dEs / slope)**2
