import math
import cmath
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fractions import Fraction
from ConvTest import ConvTest
from comparesignal2 import SignalSamplesAreEqual
from DerivativeSignal import DerivativeSignal
from Shift_Fold_Signal import *
from signalcompare import *


listofamp = []

def window():
    signals_page = tkinter.Toplevel()
    signals_page.minsize(500, 500)
    ampvar = StringVar()
    analogfrequencyvar = StringVar()
    samplingfrequencyvar = StringVar()
    phaseshiftvar = StringVar()

    # functions
    def draw_fun():
        if (float(samplingfrequencyvar.get()) == 0):
            time = np.arange(0, 2 * np.pi, 0.01)
            if (mycombo.get() == "sin"):
                fun = float(ampvar.get()) * np.sin(
                    (2 * np.pi * float(analogfrequencyvar.get())) * time + float(phaseshiftinput.get()))
            if (mycombo.get() == "cos"):
                fun = float(ampvar.get()) * np.cos(
                    (2 * np.pi * float(analogfrequencyvar.get())) * time + float(phaseshiftinput.get()))
            plt.plot(time, fun)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title("Continuous Signal")
            plt.show()
        else:
            time = np.linspace(0, 10, 10 * int(samplingfrequencyvar.get()))
            if (mycombo.get() == "sin"):
                fun = float(ampvar.get()) * np.sin(
                    ((2 * np.pi * float(analogfrequencyvar.get())) / float(samplingfrequencyvar.get())) * time + float(
                        phaseshiftinput.get()))
            if (mycombo.get() == "cos"):
                fun = float(ampvar.get()) * np.cos(
                    (2 * np.pi * float(analogfrequencyvar.get())) / float(samplingfrequencyvar.get()) * time + float(
                        phaseshiftinput.get()))
            plt.stem(time, fun)
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.title("Discrete Signal")
            plt.show()

    # Creating Widgets
    draw = Button(signals_page, text="Draw", command=draw_fun)
    mylabel = Label(signals_page, text="Ploter Function", font=("arial", 15))

    mycombo = ttk.Combobox(signals_page, values=("sin", "cos"), font='arial', width=20, state='readonly',
                           justify="center")

    amp = ttk.Label(signals_page, text="Amplitude : ")
    ampinput = Entry(signals_page, textvariable=ampvar)

    analogfrequency = ttk.Label(signals_page, text="AnalogFrequency : ")
    analogfrequencyinput = Entry(signals_page, textvariable=analogfrequencyvar)

    samplingfrequency = ttk.Label(signals_page, text="SamplingFrequency : ")
    samplingfrequencyinput = Entry(signals_page, textvariable=samplingfrequencyvar)

    phaseshift = ttk.Label(signals_page, text="PhaseShift : ")
    phaseshiftinput = Entry(signals_page, textvariable=phaseshiftvar)

    # Put Widgets On Screen
    mylabel.pack()

    mycombo.set("Select Function")
    mycombo.place(x=200, y=50)

    amp.place(x=100, y=90)
    ampinput.place(x=220, y=90)

    analogfrequency.place(x=100, y=120)
    analogfrequencyinput.place(x=220, y=120)

    samplingfrequency.place(x=100, y=150)
    samplingfrequencyinput.place(x=220, y=150)

    phaseshift.place(x=100, y=180)
    phaseshiftinput.place(x=220, y=180)

    draw.place(x=200, y=220)


def addition_window():
    Addition_page = tkinter.Toplevel()
    Addition_page.minsize(500, 500)
    # create variables
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    # functions

    def load1():
        load(x1, y1)

    def load2():
        load(x2, y2)

    def addition():
        i = 0
        while i < len(y1):
            y1[i] = y2[i] + y1[i]
            i += 1
        # SignalSamplesAreEqual("signal1-signal2.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()
        # Creating Widgets

    Label1 = ttk.Label(Addition_page, text="Addition page")
    loadfile1 = Button(Addition_page, text="LOAD FILE 1", command=load1)
    loadfile2 = Button(Addition_page, text="LOAD FILE 2", command=load2)
    draw = Button(Addition_page, text="Draw", command=addition)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=200, y=50)
    loadfile2.place(x=200, y=100)
    draw.place(x=200, y=150)


def subtraction_window():
    Subtraction_page = tkinter.Toplevel()
    Subtraction_page.minsize(500, 500)
    # create variables
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    # functions

    def load1():
        load(x1,y1)
    def load2():
        load(x2,y2)
    def subtraction():
        i = 0
        while i < len(y1):
            y1[i] = y2[i] - y1[i]
            i += 1
        #SignalSamplesAreEqual("signal1-signal2.txt", x1, y1)
        plt.plot(x1,y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Subtraction_page, text="Subtraction page")
    draw = Button(Subtraction_page, text="Draw", command=subtraction)
    loadfile1 = Button(Subtraction_page, text="LOAD FILE 1", command=load1)
    loadfile2 = Button(Subtraction_page, text="LOAD FILE 2", command=load2)
    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=200, y=50)
    loadfile2.place(x=200, y=100)
    draw.place(x=200, y=150)


def multiplication_window():
    Multiplication_page = tkinter.Toplevel()
    Multiplication_page.minsize(500, 500)

    # create variables
    const = StringVar()
    x1 = []
    y1 = []

    # functions

    def load1():
        load(x1,y1)

    def multiplication():
        i = 0
        if (const != -1):
            while i < len(y1):
                y1[i] = int(const.get()) * y1[i]
                i += 1
        #SignalSamplesAreEqual("MultiplySignalByConstant-Signal1 - by 5.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()
        # Creating Widgets
    Label1 = ttk.Label(Multiplication_page, text="Multiplication page")
    con = ttk.Label(Multiplication_page, text="constant : ")
    cons = Entry(Multiplication_page, textvariable=const)
    loadfile1 = Button(Multiplication_page, text="LOAD FILE", command=load1)
    draw = Button(Multiplication_page, text="Draw", command=multiplication)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    con.place(x=100, y=90)
    cons.place(x=220, y=90)
    loadfile1.place(x=200, y=150)
    draw.place(x=200, y=200)
def squaring_window(listofamp):
    print("square list : ")
    print(listofamp)
    Squaring_page = tkinter.Toplevel()
    Squaring_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []

    # functions

    def load1():
        load(x1,y1)

    def squaring():
        i = 0
        while i < len(y1):
            y1[i] = x1[i] * x1[i]
            i += 1
        #SignalSamplesAreEqual("Output squaring signal 1.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Squaring_page, text="Squaring page")
    loadfile1 = Button(Squaring_page, text="LOAD FILE", command=load1)
    draw = Button(Squaring_page, text="Draw", command=squaring)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)

def shifting_window():
    Shifting_page = tkinter.Toplevel()
    Shifting_page.minsize(500, 500)

    # create variables
    const = StringVar()
    x1 = []
    y1 = []

    # functions

    def load1():
        load(x1,y1)

    def shifting():
        i = 0
        if (const != -1):
            while i < len(y1):
                x1[i] = x1[i] - int(const.get())
                i += 1
        #SignalSamplesAreEqual("output shifting by minus 500.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Shifting_page, text="Shifting page")
    con = ttk.Label(Shifting_page, text="constant : ")
    cons = Entry(Shifting_page, textvariable=const)
    loadfile1 = Button(Shifting_page, text="LOAD FILE", command=load1)
    draw = Button(Shifting_page, text="Draw", command=shifting)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    con.place(x=100, y=90)
    cons.place(x=220, y=90)
    loadfile1.place(x=200, y=150)
    draw.place(x=200, y=200)

def normalization_window():
    Normalization_page = tkinter.Toplevel()
    Normalization_page.minsize(500, 500)

    # create variables
    const = StringVar()
    x1 = []
    y1 = []

    # functions

    def load1():
        load(x1,y1)
    def normalization():
        if (mycombo.get() == "0 to 1"):
            i = 0
            while i < len(y1):
                y1[i] = (y1[i] - np.min(y1)) / (np.max(y1) - np.min(y1))
                i += 1
        else:
            i = 0
            while i < len(y1):
                y1[i] = 2 * (y1[i] - np.min(y1)) / (np.max(y1) - np.min(y1)) - 1
                i += 1
        #SignalSamplesAreEqual("normalize of signal 1 -- output.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Normalization_page, text="Normalization page")
    mycombo = ttk.Combobox(Normalization_page, values=("0 to 1", "-1 to 1"), font='arial', width=20, state='readonly',
                           justify="center")
    loadfile1 = Button(Normalization_page, text="LOAD FILE", command=load1)
    draw = Button(Normalization_page, text="Draw", command=normalization)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    mycombo.set("Select skale")
    mycombo.place(x=200, y=50)
    loadfile1.place(x=200, y=100)
    draw.place(x=200, y=150)


def accumulation_window():
    Accumulation_page = tkinter.Toplevel()
    Accumulation_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []

    # functions

    def load1():
        load(x1,x2)
    def accumulation():
        i = 1
        while i < len(y1):
            y1[i] = x1[i] + y1[i - 1]
            i += 1
        #SignalSamplesAreEqual("output accumulation for signal1.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()
        # Creating Widgets
    Label1 = ttk.Label(Accumulation_page, text="Accumulation page")
    loadfile1 = Button(Accumulation_page, text="LOAD FILE", command=load1)
    draw = Button(Accumulation_page, text="Draw", command=accumulation)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
def quantization_window():
    quantization_page = tkinter.Toplevel()
    quantization_page.minsize(500, 500)

    # create variables
    const = StringVar()
    x1 = []
    y1 = []
    lists = []
    listofmidpoint = []
    listoferror = []
    listofintrval = []
    bits = 0
    levels = 0
    telda = 0
    temp = 0
    x2 = []
    y2 = []

    # functions

    def load1():
        load(x1,y1)
    def quantization():
        if (mycombo.get() == "bits"):
            bits = const
            levels = math.pow(2, int(const.get()))
        else:
            bits = math.log(int(const.get()), 2)
            levels = int(const.get())
        telda = (max(y1) - min(y1)) / levels
        lists.append([min(y1), round(min(y1) + telda, 3)])
        listofmidpoint.append(round((lists[0][0] + lists[0][1]) / 2, 3))
        for i in range(1, int(levels)):
            lists.append([lists[i - 1][1], round(lists[i - 1][1] + telda, 3)])
            listofmidpoint.append(round((lists[i][0] + lists[i][1]) / 2, 3))
        for i in range(len(y1)):
            for j in range(len(lists)):
                if ((y1[i] > lists[j][0]) & (y1[i] <= lists[j][1])):
                    x = j
                    temp = ""
                    temp2 = ""
                    if (x == 0):
                        temp2 = "0"
                        x2.append(int(temp2))
                        y2.append(listofmidpoint[j])
                        listofintrval.append(j + 1)

                    else:
                        while (x != 0):
                            temp = str(x % 2)
                            x = int(x / 2)
                            temp2 = temp + temp2
                        x2.append(int(temp2))
                        y2.append(listofmidpoint[j])
                        listofintrval.append(j + 1)
                    listoferror.append(round(listofmidpoint[j] - y1[i], 3))
            if ((y1[i] == lists[0][0])):
                x2.append(int(temp2))
                y2.append(listofmidpoint[0])
                listofintrval.append(1)
                listoferror.append(round(listofmidpoint[0] - y1[i], 3))

        #SignalSamplesAreEqual("Quan1_Out.txt", x2, y2)
    Label1 = ttk.Label(quantization_page, text="Quantization page")
    con = ttk.Label(quantization_page, text="constant : ")
    cons = Entry(quantization_page, textvariable=const)
    mycombo = ttk.Combobox(quantization_page, values=("bits", "levels"), font='arial', width=20, state='readonly',
                           justify="center")
    loadfile1 = Button(quantization_page, text="LOAD FILE", command=load1)
    draw = Button(quantization_page, text="Draw", command=quantization)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    mycombo.set("Select ")
    mycombo.place(x=200, y=50)
    loadfile1.place(x=200, y=120)
    draw.place(x=200, y=150)
    con.place(x=100, y=90)
    cons.place(x=220, y=90)

def calc_dft(y1,xk):
    for k in range(len(y1)):
        for n in range(len(y1)):
            xk[k] += y1[n] * np.exp((-1j * 2 * np.pi * k * n) / len(y1))
    return xk
def dft_window():
    DFT_page = tkinter.Toplevel()
    DFT_page.minsize(500, 500)

    # create variables
    frq = StringVar()
    amp = StringVar()
    phs = StringVar()
    fund = 0
    x1 = []
    y1 = []
    xi = []
    yi = []
    xk = []
    xn = []
    listofph = []
    listofimag = []
    listofreal = []

    b = "0\n0\n8\n"

    # functions

    def load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=str, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (a[i][j][-1] == 'f'):
                    a[i][j] = a[i][j].replace("f", " ")
                if (j == 0):
                    if (mycombo.get() == "DFT"):
                        x1.append(float(a[i][j]))
                    else:
                        xi.append(float(a[i][j]))
                else:
                    if (mycombo.get() == "DFT"):
                        y1.append(float(a[i][j]))
                    else:
                        yi.append(float(a[i][j]))

    def dft(listofamp, listofph):
        xk = np.zeros(len(y1), dtype=complex)
        if (mycombo.get() == "DFT"):
            listofamp = np.zeros(len(y1))
            listofph = np.zeros(len(y1))
            calc_dft(y1,xk)

            for i in range(len(y1)):
                listofamp[i] = math.sqrt(math.pow(xk[i].real, 2) + math.pow(xk[i].imag, 2))
                listofph[i] = (math.atan2(xk[i].imag, xk[i].real))
            print("list of amp")
            print(listofamp)
            print("list of phs")
            print(listofph)
            texfile = open("philo.txt", "w")
            texfile.write(b)
            for i in range(len(listofamp)):
                if (listofamp[i] - int(listofamp[i]) > 0.0):
                    if (abs(listofph[i]) - abs(int(listofph[i])) > 0.0):
                        texfile.write(str(listofamp[i]) + "f " + str(listofph[i]) + "f\n")
                    else:
                        texfile.write(str(listofamp[i]) + "f " + str(listofph[i]) + "\n")
                else:
                    if (abs(listofph[i]) - abs(int(listofph[i])) > 0.0):
                        texfile.write(str(listofamp[i]) + " " + str(listofph[i]) + "f\n")
                    else:
                        texfile.write(str(listofamp[i]) + " " + str(listofph[i]) + "\n")
        else:
            # print(listofamp)
            listofimag = np.zeros(len(yi))
            listofreal = np.zeros(len(yi))
            xn = np.zeros(len(yi))
            xk = np.zeros(len(yi), dtype=complex)
            for i in range(len(yi)):
                # listofimag[i]=math.sin(math.degrees(y1[i]))*x1[i]
                # listofreal[i]=math.cos(math.degrees(y1[i]))*x1[i]
                xk[i] = xi[i] * np.exp(1j * yi[i])
            for n in range(len(yi)):
                for k in range(len(yi)):
                    xn[n] += xk[k] * np.exp((2j * np.pi * n * k) / len(yi))
                xn[n] += 1 / len(y1)
            print("list of xn")
            print(xn)
            print("list of xk")
            print(xk)
            print(SignalComapreAmplitude(listofamp, xi))
            print(SignalComaprePhaseShift(listofph, yi))

        listofamp[:] = listofamp
        listofph[:] = listofph

        mycombo1.place(x=300, y=300)
        con.place(x=100, y=220)
        cons.place(x=120, y=220)
        con1.place(x=100, y=250)
        cons1.place(x=120, y=250)

        if (mycombo1.get() != "Select point"):

            listofamp[int(mycombo1.get()) - 1] = float(amp.get())
            listofph[int(mycombo1.get()) - 1] = float(phs.get())
            print(listofamp)
            print(listofph)
            texfile = open("philo.txt", "w")
            texfile.write(b)
            for i in range(len(listofamp)):
                if (listofamp[i] - int(listofamp[i]) > 0.0):
                    if (abs(listofph[i]) - abs(int(listofph[i])) > 0.0):
                        texfile.write(str(listofamp[i]) + "f " + str(listofph[i]) + "f\n")
                    else:
                        texfile.write(str(listofamp[i]) + "f " + str(listofph[i]) + "\n")
                else:
                    if (abs(listofph[i]) - abs(int(listofph[i])) > 0.0):
                        texfile.write(str(listofamp[i]) + " " + str(listofph[i]) + "f\n")
                    else:
                        texfile.write(str(listofamp[i]) + " " + str(listofph[i]) + "\n")

        if (int(frq.get() != "")):
            listoffund = []
            listoffund = np.zeros(len(y1))
            fund = (2 * np.pi * int(frq.get())) / len(y1)
            print(fund)
            listoffund[0] = (2 * np.pi * int(frq.get())) / len(y1)
            for i in range(1, len(y1)):
                listoffund[i] = listoffund[i - 1] + fund
            print(listoffund)
            plt.bar(listoffund, listofamp)
            plt.show()
            plt.bar(listoffund, listofph)
            plt.show()

    # Creating Widgets
    Label1 = ttk.Label(DFT_page, text="DFT page")
    mycombo = ttk.Combobox(DFT_page, values=("DFT", "IDFT"), font='arial', width=20, state='readonly', justify="center")
    # mycombo2=ttk.Combobox(DFT_page,values=("update","None"),font='arial',width=20,state='readonly',justify="center")
    loadfile1 = Button(DFT_page, text="LOAD FILE", command=load1)
    draw = Button(DFT_page, text="Draw", command=lambda: dft(listofamp, listofph))
    # MOD = Button(DFT_page,text="mod",command=mod)
    mycombo1 = ttk.Combobox(DFT_page, values=("1", "2", "3", "4", "5", "6", "7", "8"), font='arial', width=20,
                            state='readonly', justify="center")
    con = ttk.Label(DFT_page, text="amp : ")
    cons = Entry(DFT_page, textvariable=amp)
    con1 = ttk.Label(DFT_page, text="phs : ")
    cons1 = Entry(DFT_page, textvariable=phs)
    con2 = ttk.Label(DFT_page, text="frq : ")
    cons2 = Entry(DFT_page, textvariable=frq)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    mycombo.set("Select skale")
    mycombo.place(x=200, y=50)
    # mycombo2.set("Select update")
    # mycombo2.place(x=200,y=200)
    loadfile1.place(x=200, y=90)
    draw.place(x=200, y=150)
    mycombo1.set("Select point")
    con2.place(x=350, y=220)
    cons2.place(x=300, y=350)


# MOD.place(x=200,y=170)
def dct_window():
    DCT_page = tkinter.Toplevel()
    DCT_page.minsize(500, 500)

    # create variables
    coefficients = StringVar()
    x1 = []
    y1 = []
    yk = []

    # functions

    def load1():
        load(x1,y1)
    def dct():
        yk = np.zeros(len(y1))
        for k in range(len(y1)):
            for n in range(len(y1)):
                yk[k] += math.sqrt(2 / len(y1)) * y1[n] * math.cos(
                    (math.pi / (4 * len(y1))) * (2 * n - 1) * (2 * k - 1))
            yk[k] = np.round(yk[k], 5)
        print(yk)

        #SignalSamplesAreEqual("DCT_output.txt", yk)
        b = "0\n1\n" + str(coefficients.get()) + "\n"
        texfile = open("philo.txt", "w")
        texfile.write(b)
        for i in range(int(coefficients.get())):
            texfile.write(str('0 ') + str(yk[i]) + "\n")
            print(yk[i])
        plt.stem(x1, yk)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(DCT_page, text="DCT page")
    loadfile1 = Button(DCT_page, text="LOAD FILE", command=load1)
    draw = Button(DCT_page, text="Draw", command=dct)
    con = ttk.Label(DCT_page, text="constant : ")
    cons = Entry(DCT_page, textvariable=coefficients)
    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
    con.place(x=100, y=200)
    cons.place(x=220, y=200)


def folding_window():
    Remove_page = tkinter.Toplevel()
    Remove_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []
    yk = []

    # functions

    def load1():
        load(x1.y1)
    def remove():
        yk = np.zeros(len(y1))
        for n in range(len(y1)):
            yk[n] = np.round(y1[n] - np.mean(y1), 3)
        print(yk)
        plt.plot(x1, yk)
        plt.show()
        #SignalSamplesAreEqual("DC_component_output.txt", yk)
        # Creating Widgets

    Label1 = ttk.Label(Remove_page, text="Remove_page")
    loadfile1 = Button(Remove_page, text="LOAD FILE", command=load1)
    draw = Button(Remove_page, text="Draw", command=remove)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)


def smoothing_window():
    Smoothing_page = tkinter.Toplevel()
    Smoothing_page.minsize(500, 500)

    # create variables
    Window_size = StringVar()
    x1 = []
    y1 = []
    ys = []

    # functions

    def load1():
        LOAd(x1,y1)
    def smoothing():
        ys = np.zeros(len(y1) - int(Window_size.get()) + 1)
        counter1 = 0
        counter2 = 0
        sum = 0
        for i in range(len(y1)):
            counter1 += 1
            sum += y1[i]
            if (counter1 == int(Window_size.get())):
                print(counter2)
                ys[counter2] = sum / int(Window_size.get())
                counter2 += 1
                counter1 -= 1
                sum -= y1[i - int(Window_size.get()) + 1]
        print(ys)
        #SignalSamplesAreEqual("OutMovAvgTest2.txt", ys)

    # Creating Widgets
    Label1 = ttk.Label(Smoothing_page, text="DCT page")
    loadfile1 = Button(Smoothing_page, text="LOAD FILE", command=load1)
    draw = Button(Smoothing_page, text="Draw", command=smoothing)
    con = ttk.Label(Smoothing_page, text="Window_size : ")
    cons = Entry(Smoothing_page, textvariable=Window_size)
    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
    con.place(x=100, y=200)
    cons.place(x=220, y=200)


def remove2_window():
    Remove2_page = tkinter.Toplevel()
    Remove2_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []
    xk = []
    xn = []
    listofimag = []
    listofreal = []

    # functions

    def load1():
        load(x1,y1)

    def remove2():
        xk = np.zeros(len(y1), dtype=complex)
        listofamp = np.zeros(len(y1))
        listofph = np.zeros(len(y1))
        calc_dft(y1,xk)
        xk[0] = 0 + 0j
        print(xk)
        xn = np.zeros(len(y1))
        for n in range(len(y1)):
            for k in range(len(y1)):
                xn[n] += 1 / len(y1) * xk[k] * np.exp((2j * np.pi * n * k) / len(y1))
            xn[n] = round(xn[n],3)
        print(xn)
        plt.plot(x1, xn)
        # plt.show()
        #SignalSamplesAreEqual("DC_component_output.txt", xn)

        # Creating Widgets

    Label1 = ttk.Label(Remove2_page, text="Remove_page")
    loadfile1 = Button(Remove2_page, text="LOAD FILE", command=load1)
    draw = Button(Remove2_page, text="Draw", command=remove2)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)


def convolution_window():
    convolution_page = tkinter.Toplevel()
    convolution_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    yc = []
    xc = []

    # functions

    def load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                    if (len(xc) != len(a) - 1):
                        xc.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def load2():
        load(x2,y2)
        for i in range(len(x2)):
            xc.append(x2[i])
    def convolution():
        yc = np.zeros(len(x1) + len(x2) - 1)
        for n in range(len(yc)):
            for k in range(len(y1)):
                for m in range(len(y2)):
                    if (k + m == n):
                        yc[n] += y1[k] * y2[m]
        min = xc[0]
        for i in range(1, len(xc)):
            if (xc[i] < min):
                min = xc[i]
        for b in range(1, len(xc)):
            xc[b] = xc[b - 1] + 1
            print(xc[b])
        #ConvTest(xc, yc)

        # Creating Widgets

    Label1 = ttk.Label(convolution_page, text="Remove_page")
    loadfile1 = Button(convolution_page, text="LOAD FILE", command=load1)
    loadfile2 = Button(convolution_page, text="LOAD FILE 2", command=load2)
    draw = Button(convolution_page, text="Draw", command=convolution)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    loadfile2.place(x=150, y=150)
    draw.place(x=250, y=150)

def calc_folding(x,y,x1,y1):
    length = len(y) - 1
    for i in range(len(y)):
        x1[i] = x[i]
        y1[i] = y[length]
        length = length - 1
    return x1,y1
def folding_window():
    Folding_page = tkinter.Toplevel()
    Folding_page.minsize(500, 500)

    # create variables
    x = []
    y = []
    x1 = []
    y1 = []

    def load1():
        load(x,y)

    def folding():
        x1 = np.zeros(len(x))
        y1 = np.zeros(len(y))
        calc_folding(x, y, x1, y1)
        print(x1)
        print(y1)

       # SignalSamplesAreEqual("Output_fold.txt", y1)

    # Creating Widgets
    Label1 = ttk.Label(Folding_page, text="Folding")
    loadfile1 = Button(Folding_page, text="LOAD FILE", command=load1)
    draw = Button(Folding_page, text="Draw", command=folding)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)

def calc_shift(x,x1):
    shift_val_pos = int(shiftpos.get())
    for i in range(len(x)):
        x1[i] = x[i] + shift_val_pos
    print(x1)
    return x1
def folding_shift_window():
    Folding_Shift_page = tkinter.Toplevel()
    Folding_Shift_page.minsize(500, 500)
    x = []
    y = []
    x1 = []
    y1 = []
    shiftpos = StringVar()

    def load1():
       load(x,y)

    def folding_shift():
        # Folding
        x1 = np.zeros(len(x))
        y1 = np.zeros(len(y))
        calc_FOLD(x, y, x1, y1)
        # shifting
        calc_shift(x,x1)
       # Shift_Fold_Signal("Output_ShiftFoldedby-500.txt", x1, y1)

        # Creating Widgets

    Label1 = ttk.Label(Folding_Shift_page, text="Remove_page")
    loadfile1 = Button(Folding_Shift_page, text="LOAD FILE", command=load1)
    draw = Button(Folding_Shift_page, text="Draw", command=folding_shift)
    con = ttk.Label(Folding_Shift_page, text="shift : ")
    cons = Entry(Folding_Shift_page, textvariable=shiftpos)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
    con.place(x=100, y=200)
    cons.place(x=220, y=200)
def delay_window():
    Delay_page = tkinter.Toplevel()
    Delay_page.minsize(500, 500)
    x = []
    y = []
    x1 = []
    shiftpos = StringVar()

    def load1():
        load(x,y)

    def delay():
        # shifting
        calc_shift(x,x1)

        # Creating Widgets

    Label1 = ttk.Label(Delay_page, text="Remove_page")
    loadfile1 = Button(Delay_page, text="LOAD FILE", command=load1)
    draw = Button(Delay_page, text="Draw", command=delay)
    con = ttk.Label(Delay_page, text="shift : ")
    cons = Entry(Delay_page, textvariable=shiftpos)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
    con.place(x=100, y=200)
    cons.place(x=220, y=200)


def correlation_window():
    correlation_page = tkinter.Toplevel()
    correlation_page.minsize(500, 500)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    yres = []

    def load1():
        load(x1,y1)

    def load2():
        load(x2,y2)

    def nor_correlation():
        yres = np.zeros(len(x1))
        for i in range(len(x1)):
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for j in range(len(x1)):
                sum1 += y1[j] * y1[j]
                sum2 += y2[j] * y2[j]
                sum3 += y1[j] * y2[j]
            yres[i] = sum3 / len(x1) / (1 / len(x1) * (math.sqrt(sum1 * sum2)))
            y1.insert(0, y1[len(y1) - 1])
            y1.pop()
        print(yres)
        #SignalSamplesAreEqual("CorrOutput.txt", yres)

    Label1 = ttk.Label(correlation_page, text="correlation_page")
    loadfile1 = Button(correlation_page, text="LOAD FILE #1", command=load1)
    loadfile2 = Button(correlation_page, text="LOAD FILE #2", command=load2)
    draw = Button(correlation_page, text="Draw", command=nor_correlation)

    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    loadfile2.place(x=150, y=150)
    draw.place(x=250, y=150)


def time_delay_window():
    time_delay_page = tkinter.Toplevel()
    time_delay_page.minsize(500, 500)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    yres = []

    def load1():
        load(x1,y1)

    def load2():
        load(x2,y2)

    def time_delay():
        yres = np.zeros(len(x1))
        maxc = 0.0
        maxindex = 0
        for i in range(len(x1)):
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for j in range(len(x1)):
                sum1 += y1[j] * y1[j]
                sum2 += y2[j] * y2[j]
                sum3 += y1[j] * y2[j]
            yres[i] = sum3 / len(x1) / (1 / len(x1) * (math.sqrt(sum1 * sum2)))
            if (yres[i] > maxc):
                maxc = yres[i]
                maxindex = i
            y1.insert(0, y1[len(y1) - 1])
            y1.pop()
        res = maxindex / 100
        print(res)

    Label1 = ttk.Label(time_delay_page, text="time_delay_page")
    loadfile1 = Button(time_delay_page, text="LOAD FILE #1", command=load1)
    loadfile2 = Button(time_delay_page, text="LOAD FILE #2", command=load2)
    draw = Button(time_delay_page, text="Draw", command=time_delay)

    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    loadfile2.place(x=150, y=150)
    draw.place(x=250, y=150)


"""""another implementation
def template_matching_window():
    template_matching_page = tkinter.Toplevel()
    template_matching_page.minsize(500, 500)
    ylist = []
    ylist2 = []
    y1 = []
    join=[]

    res1 = 0.0
    res2 = 0.0
    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float)
        y1 = []
        for i in range(len(a)):
            y1.append(a[i])
        ylist.append(y1)
        print("Load1")

    def Load2():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float)
        y1 = []
        for i in range(len(a)):
            y1.append(a[i])
        ylist2.append(y1)
        print("Load2")

    def Load_test():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float)
        for i in range(len(a)):
            y1.append(a[i])

    def template_matching():
        avg = np.zeros(len(ylist[0]))
        avg2 = np.zeros(len(ylist[0]))
        res1 = np.zeros(len(ylist[0]))
        res2 = np.zeros(len(ylist[0]))
        for i in range (len(ylist[0])):
            for j in range (len(ylist)):
                avg[i] += ylist[j][i]
            avg[i] /=len(ylist)

        for i in range (len(ylist2[0])):
            for j in range (len(ylist2)):
                avg2[i] += ylist2[j][i]
            avg2[i] /=len(ylist2)

        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        maxc1 = 0.0
        maxc2 = 0.0
        for i in range(len(y1)):
            for j in range(len(y1)):
                sum1 += y1[j] * y1[j]
                sum2 += avg[j] * avg[j]
                sum3 += y1[j] * avg[j]
                sum4 += avg2[j] * avg2[j]
                sum5 += y1[j] * avg2[j]
            res1[i] = sum3 / len(y1) / (1 / len(y1) * (math.sqrt(sum1 * sum2)))
            res2[i] = sum5 / len(y1) / (1 / len(y1) * (math.sqrt(sum1 * sum4)))
            maxc1 = max(maxc1,res1[i])
            maxc2 = max(maxc2,res2[i])
            y1.insert(0, y1[len(y1) - 1])
            y1.pop()
        if(maxc1 > maxc2):
            print("res = ",maxc1)
            print("Class 1")
        else:
            print("res = ",maxc2)
            print("CLASS 2")
"""

def template_matching_window():
    template_matching_page = tkinter.Toplevel()
    template_matching_page.minsize(500, 500)
    ylist = []
    ylist2 = []
    y1 = []

    res1 = 0.0
    res2 = 0.0

    def load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float)
        y1 = []
        for i in range(len(a)):
            ylist.append(a[i])
        print("Load1")

    def load2():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float)
        y1 = []
        for i in range(len(a)):
            ylist2.append(a[i])
        print("Load2")

    def load_test():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float)
        for i in range(len(a)):
            y1.append(a[i])

    def template_matching():
        res1 = np.zeros(len(ylist))
        res2 = np.zeros(len(ylist2))
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        maxc1 = 0.0
        maxc2 = 0.0
        for i in range(len(y1)):
            for j in range(len(y1)):
                sum1 += y1[j] * y1[j]
                sum2 += ylist[j] * ylist[j]
                sum3 += y1[j] * ylist[j]
                sum4 += ylist2[j] * ylist2[j]
                sum5 += y1[j] * ylist2[j]
            res1[i] = sum3 / len(y1) / (1 / len(y1) * (math.sqrt(sum1 * sum2)))
            res2[i] = sum5 / len(y1) / (1 / len(y1) * (math.sqrt(sum1 * sum4)))
            maxc1 = max(maxc1, res1[i])
            maxc2 = max(maxc2, res2[i])
            ylist.insert(len(ylist), ylist[0])
            ylist.pop(0)
            ylist2.insert(len(ylist2), ylist2[0])
            ylist2.pop(0)

        if (maxc1 > maxc2):
            print("res = ", maxc1)
            print("Class 1")
        else:
            print("res = ", maxc2)
            print("CLASS 2")

    Label1 = ttk.Label(template_matching_page, text="template_matching_page")
    loadfile1 = Button(template_matching_page, text="LOAD CLASS 1", command=load1)
    loadfile2 = Button(template_matching_page, text="LOAD CLASS 2", command=load2)
    loadfile3 = Button(template_matching_page, text="LOAD TEST", command=load_test)
    draw = Button(template_matching_page, text="Draw", command=template_matching)

    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    loadfile2.place(x=150, y=200)
    loadfile3.place(x=150, y=300)
    draw.place(x=250, y=400)

def calc_idft(ykc,yc):
    for n in range(len(yc)):
        for k in range(len(yc)):
            yc[n] += ykc[k] * np.exp((2j * np.pi * n * k) / len(yc))
        yc[n] *= 1 / len(yc)
        yc[n] = np.round(yc[n], 3)
    print(yc)
    return yc
def fast_convolution_window():
    Fast_convolution_page = tkinter.Toplevel()
    Fast_convolution_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    yc = []
    xc = []
    yk = []
    yk2 = []
    ykc = []

    # functions

    def load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                    if (len(xc) != len(a) - 1):
                        xc.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def load2():
        load(x2,y2)
        for i in range(len(y2)):
            xc.append(x2[i])
    def fast_convolution():
        yc = np.zeros(len(x1) + len(x2) - 1)
        ykc = np.zeros(len(yc), dtype=complex)
        yk = np.zeros(len(yc), dtype=complex)
        yk2 = np.zeros(len(yc), dtype=complex)

        for i in range(len(y1), len(yc)):
            y1.append(0)
        for i in range(len(y2), len(yc)):
            y2.append(0)
        print(y1)
        print(y2)

        # DFT
        calc_dft(y1,yk)

        calc_dft(y2, yk2)
        # FAST
        for n in range(len(yc)):
            ykc[n] = yk[n] * yk2[n]

        # IDFT
        calc_idft(ykc, yc)
        # Your_indices
        min = xc[0]
        for i in range(1, len(xc)):
            if (xc[i] < min):
                min = xc[i]
        for b in range(1, len(xc)):
            xc[b] = xc[b - 1] + 1
            print(xc[b])
      #  ConvTest(xc, yc)
        # Creating Widgets

    Label1 = ttk.Label(Fast_convolution_page, text="Fast_convolution")
    loadfile1 = Button(Fast_convolution_page, text="LOAD FILE", command=load1)
    loadfile2 = Button(Fast_convolution_page, text="LOAD FILE 2", command=load2)
    draw = Button(Fast_convolution_page, text="Draw", command=fast_convolution)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    loadfile2.place(x=150, y=150)
    draw.place(x=250, y=150)


def fast_correlation_window():
    Fast_correlation_page = tkinter.Toplevel()
    Fast_correlation_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    yc = []
    xc = []
    yk = []
    yk2 = []
    ykc = []

    # functions

    def load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                    if (len(xc) != len(a) - 1):
                        xc.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def load2():
        load(x2, y2)
        for i in range(len(y2)):
            xc.append(x2[i])

    def fast_correlation():
        yc = np.zeros(len(x1))
        ykc = np.zeros(len(yc), dtype=complex)
        yk = np.zeros(len(yc), dtype=complex)
        yk2 = np.zeros(len(yc), dtype=complex)
        # DFT
        calc_dft(y1,yk)

        calc_dft(y2,yk2)
        #

        print(yk)
        for i in range(len(yc)):
            yk[i] = yk[i].conjugate()
        print(yk)
        for n in range(len(yc)):
            ykc[n] = yk[n] * yk2[n]
            ykc[n] /= len(yc)

        # IDFT
        calc_idft(ykc,yc)
        #SignalSamplesAreEqual("Corr_Output.txt", yc)

        # Creating Widgets

    Label1 = ttk.Label(Fast_correlation_page, text="Fast_correlation")
    loadfile1 = Button(Fast_correlation_page, text="LOAD FILE", command=load1)
    loadfile2 = Button(Fast_correlation_page, text="LOAD FILE 2", command=load2)
    draw = Button(Fast_correlation_page, text="Draw", command=fast_correlation)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    loadfile2.place(x=150, y=150)
    draw.place(x=250, y=150)


def load(x,y):
    filepath = filedialog.askopenfile()
    a = np.loadtxt(filepath, dtype=float, skiprows=3)
    for i in range(len(a)):
        for j in range(2):
            if (j == 0):
                x.append(a[i][j])
            else:
                y.append(a[i][j])
    return x,y

top = Tk()
top.title("Signal")
top.minsize(700, 700)
but1 = Button(text="LOAD FILE", command=load)
but1.pack()
but2 = Button(text="Open Signals Page", command=window)
but2.pack()
but3 = Button(text="Addition", command=addition_window)
but3.pack()
but4 = Button(text="Subtraction", command=subtraction_window)
but4.pack()
but5 = Button(text="Multiplication", command=multiplication_window)
but5.pack()
but6 = Button(text="Squaring", command=lambda: squaring_window(listofamp))
but6.pack()
but7 = Button(text="Shifting", command=shifting_window)
but7.pack()
but7 = Button(text="Normalization", command=normalization_window)
but7.pack()
but8 = Button(text="Accumulation ", command=accumulation_window)
but8.pack()
but9 = Button(text="quantization", command=quantization_window)
but9.pack()
but10 = Button(text="DFT", command=dft_window)
but10.pack()
but11 = Button(text="DCT", command=dct_window)
but11.pack()
but12 = Button(text="Remove DC component", command=folding_window)
but12.pack()
but13 = Button(text="Smoothing", command=smoothing_window)
but13.pack()
but14 = Button(text="Remove DC component 2", command=remove2_window)
but14.pack()
but15 = Button(text="convolution", command=convolution_window)
but15.pack()
but16 = Button(text="Sharpening", command=DerivativeSignal)
but16.pack()
but17 = Button(text="DELAYING AND ADVANCING ", command=delay_window)
but17.pack()
but18 = Button(text="Folding ", command=folding_window)
but18.pack()
but19 = Button(text="Folding and shift ", command=folding_shift_window)
but19.pack()
but20 = Button(text="Normalized Cross Correlation", command=correlation_window)
but20.pack()
but21 = Button(text="calculate time_delay", command=time_delay_window)
but21.pack()
but22 = Button(text="template_matching", command=template_matching_window)
but22.pack()
but23 = Button(text="Fast convolution.", command=fast_convolution_window)
but23.pack()
but24 = Button(text="Fast Correlation .", command=fast_correlation_window)
but24.pack()
top.mainloop()