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


def Window():
    signals_page = tkinter.Toplevel()
    signals_page.minsize(500, 500)

    ampvar = StringVar()
    analogfrequencyvar = StringVar()
    samplingfrequencyvar = StringVar()
    phaseshiftvar = StringVar()

    # functions
    def drawfun():
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
    draw = Button(signals_page, text="Draw", command=drawfun)
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


def Addition_window():
    Addition_page = tkinter.Toplevel()
    Addition_page.minsize(500, 500)
    # create variables
    const = StringVar()
    x1 = []
    y1 = []

    # functions

    def Load1():
        if (len(y1) == 0):
            filepath = filedialog.askopenfile()
            a = np.loadtxt(filepath, dtype=float, skiprows=3)
            for i in range(len(a)):
                for j in range(2):
                    if (j == 0):
                        x1.append(a[i][j])
                    else:
                        y1.append(a[i][j])
        else:
            filepath = filedialog.askopenfile()
            a = np.loadtxt(filepath, dtype=float, skiprows=3)
            for i in range(len(a)):
                for j in range(2):
                    if (j != 0):
                        y1[i] = y1[i] + a[i][j]

    def Addition():
        SignalSamplesAreEqual("Signal1+signal2.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Addition_page, text="Addition page")
    loadfile1 = Button(Addition_page, text="LOAD FILE", command=Load1)
    draw = Button(Addition_page, text="Draw", command=Addition)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=200, y=50)
    draw.place(x=200, y=100)


def Subtraction_window():
    Subtraction_page = tkinter.Toplevel()
    Subtraction_page.minsize(500, 500)

    # create variables

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Load2():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x2.append(a[i][j])
                else:
                    y2.append(a[i][j])

    def Subtraction():
        i = 0
        while i <= 1000:
            y1[i] = y2[i] - y1[i]
            i += 1
        SignalSamplesAreEqual("signal1-signal2.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Subtraction_page, text="Subtraction page")
    draw = Button(Subtraction_page, text="Draw", command=Subtraction)
    loadfile1 = Button(Subtraction_page, text="LOAD FILE 1", command=Load1)
    loadfile2 = Button(Subtraction_page, text="LOAD FILE 2", command=Load2)
    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=200, y=50)
    loadfile2.place(x=200, y=100)
    draw.place(x=200, y=150)


def Multiplication_window():
    Multiplication_page = tkinter.Toplevel()
    Multiplication_page.minsize(500, 500)

    # create variables
    const = StringVar()
    x1 = []
    y1 = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Multiplication():
        i = 0
        if (const != -1):
            while i <= 1000:
                y1[i] = int(const.get()) * y1[i]
                i += 1
        SignalSamplesAreEqual("MultiplySignalByConstant-Signal1 - by 5.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Multiplication_page, text="Multiplication page")
    con = ttk.Label(Multiplication_page, text="constant : ")
    cons = Entry(Multiplication_page, textvariable=const)
    loadfile1 = Button(Multiplication_page, text="LOAD FILE", command=Load1)
    draw = Button(Multiplication_page, text="Draw", command=Multiplication)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    con.place(x=100, y=90)
    cons.place(x=220, y=90)
    loadfile1.place(x=200, y=150)
    draw.place(x=200, y=200)


def Squaring_window(listofamp):
    print("square list : ")
    print(listofamp)
    Squaring_page = tkinter.Toplevel()
    Squaring_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Squaring():
        i = 0
        while i <= 1000:
            y1[i] = x1[i] * x1[i]
            i += 1
        SignalSamplesAreEqual("Output squaring signal 1.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Squaring_page, text="Squaring page")
    loadfile1 = Button(Squaring_page, text="LOAD FILE", command=Load1)
    draw = Button(Squaring_page, text="Draw", command=Squaring)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)


def Shifting_window():
    Shifting_page = tkinter.Toplevel()
    Shifting_page.minsize(500, 500)

    # create variables
    const = StringVar()
    x1 = []
    y1 = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Shifting():
        i = 0
        if (const != -1):
            while i <= 1000:
                x1[i] = x1[i] - int(const.get())
                i += 1
        SignalSamplesAreEqual("output shifting by minus 500.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Shifting_page, text="Shifting page")
    con = ttk.Label(Shifting_page, text="constant : ")
    cons = Entry(Shifting_page, textvariable=const)
    loadfile1 = Button(Shifting_page, text="LOAD FILE", command=Load1)
    draw = Button(Shifting_page, text="Draw", command=Shifting)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    con.place(x=100, y=90)
    cons.place(x=220, y=90)
    loadfile1.place(x=200, y=150)
    draw.place(x=200, y=200)


def Normalization_window():
    Normalization_page = tkinter.Toplevel()
    Normalization_page.minsize(500, 500)

    # create variables
    const = StringVar()
    x1 = []
    y1 = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Normalization():
        if (mycombo.get() == "0 to 1"):
            i = 0
            while i <= 1000:
                y1[i] = (y1[i] - np.min(y1)) / (np.max(y1) - np.min(y1))
                i += 1
        else:
            i = 0
            while i <= 1000:
                y1[i] = 2 * (y1[i] - np.min(y1)) / (np.max(y1) - np.min(y1)) - 1
                i += 1
        SignalSamplesAreEqual("normalize of signal 1 -- output.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Normalization_page, text="Normalization page")
    mycombo = ttk.Combobox(Normalization_page, values=("0 to 1", "-1 to 1"), font='arial', width=20, state='readonly',
                           justify="center")
    loadfile1 = Button(Normalization_page, text="LOAD FILE", command=Load1)
    draw = Button(Normalization_page, text="Draw", command=Normalization)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    mycombo.set("Select skale")
    mycombo.place(x=200, y=50)
    loadfile1.place(x=200, y=100)
    draw.place(x=200, y=150)


def Accumulation_window():
    Accumulation_page = tkinter.Toplevel()
    Accumulation_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Accumulation():
        i = 1
        while i <= 1000:
            y1[i] = x1[i] + y1[i - 1]
            i += 1
        SignalSamplesAreEqual("output accumulation for signal1.txt", x1, y1)
        plt.plot(x1, y1)
        plt.show()

        # Creating Widgets

    Label1 = ttk.Label(Accumulation_page, text="Accumulation page")
    loadfile1 = Button(Accumulation_page, text="LOAD FILE", command=Load1)
    draw = Button(Accumulation_page, text="Draw", command=Accumulation)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)


def Quantization_window():
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

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

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

        SignalSamplesAreEqual("Quan1_Out.txt", x2, y2)

    Label1 = ttk.Label(quantization_page, text="Quantization page")
    con = ttk.Label(quantization_page, text="constant : ")
    cons = Entry(quantization_page, textvariable=const)
    mycombo = ttk.Combobox(quantization_page, values=("bits", "levels"), font='arial', width=20, state='readonly',
                           justify="center")
    loadfile1 = Button(quantization_page, text="LOAD FILE", command=Load1)
    draw = Button(quantization_page, text="Draw", command=quantization)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    mycombo.set("Select ")
    mycombo.place(x=200, y=50)
    loadfile1.place(x=200, y=120)
    draw.place(x=200, y=150)
    con.place(x=100, y=90)
    cons.place(x=220, y=90)


def DFT_window():
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

    def Load1():
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

    def DFT(listofamp, listofph):
        xk = np.zeros(len(y1), dtype=complex)
        if (mycombo.get() == "DFT"):
            listofamp = np.zeros(len(y1))
            listofph = np.zeros(len(y1))
            for k in range(len(y1)):
                for n in range(len(y1)):
                    xk[k] += y1[n] * np.exp((-1j * 2 * np.pi * k * n) / len(y1))

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
    loadfile1 = Button(DFT_page, text="LOAD FILE", command=Load1)
    draw = Button(DFT_page, text="Draw", command=lambda: DFT(listofamp, listofph))
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
def DCT_window():
    DCT_page = tkinter.Toplevel()
    DCT_page.minsize(500, 500)

    # create variables
    coefficients = StringVar()
    x1 = []
    y1 = []
    yk = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def DCT():
        yk = np.zeros(len(y1))
        for k in range(len(y1)):
            for n in range(len(y1)):
                yk[k] += math.sqrt(2 / len(y1)) * y1[n] * math.cos(
                    (math.pi / (4 * len(y1))) * (2 * n - 1) * (2 * k - 1))
            yk[k] = np.round(yk[k], 5)
        print(yk)
        SignalSamplesAreEqual("DCT_output.txt", yk)
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
    loadfile1 = Button(DCT_page, text="LOAD FILE", command=Load1)
    draw = Button(DCT_page, text="Draw", command=DCT)
    con = ttk.Label(DCT_page, text="constant : ")
    cons = Entry(DCT_page, textvariable=coefficients)
    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
    con.place(x=100, y=200)
    cons.place(x=220, y=200)


def Folding_window():
    Remove_page = tkinter.Toplevel()
    Remove_page.minsize(500, 500)

    # create variables
    x1 = []
    y1 = []
    yk = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Remove():
        yk = np.zeros(len(y1))
        for n in range(len(y1)):
            yk[n] = np.round(y1[n] - np.mean(y1), 3)
        print(yk)
        plt.plot(x1, yk)
        plt.show()
        SignalSamplesAreEqual("DC_component_output.txt", yk)

        # Creating Widgets

    Label1 = ttk.Label(Remove_page, text="Remove_page")
    loadfile1 = Button(Remove_page, text="LOAD FILE", command=Load1)
    draw = Button(Remove_page, text="Draw", command=Remove)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)


def Smoothing_window():
    Smoothing_page = tkinter.Toplevel()
    Smoothing_page.minsize(500, 500)

    # create variables
    Window_size = StringVar()
    x1 = []
    y1 = []
    ys = []

    # functions

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Smoothing():
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
        SignalSamplesAreEqual("OutMovAvgTest2.txt", ys)

    # Creating Widgets
    Label1 = ttk.Label(Smoothing_page, text="DCT page")
    loadfile1 = Button(Smoothing_page, text="LOAD FILE", command=Load1)
    draw = Button(Smoothing_page, text="Draw", command=Smoothing)
    con = ttk.Label(Smoothing_page, text="Window_size : ")
    cons = Entry(Smoothing_page, textvariable=Window_size)
    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
    con.place(x=100, y=200)
    cons.place(x=220, y=200)


def Remove2_window():
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

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x1.append(a[i][j])
                else:
                    y1.append(a[i][j])

    def Remove2():
        xk = np.zeros(len(y1), dtype=complex)
        listofamp = np.zeros(len(y1))
        listofph = np.zeros(len(y1))
        for k in range(len(y1)):
            for n in range(len(y1)):
                xk[k] += y1[n] * np.exp((-1j * 2 * np.pi * k * n) / len(y1))
        xk[0] = 0 + 0j
        print(xk)
        xn = np.zeros(len(y1))
        for n in range(len(y1)):
            for k in range(len(y1)):
                xn[n] += 1 / len(y1) * xk[k] * np.exp((2j * np.pi * n * k) / len(y1))
            xn[n] = round(xn[n], 3)
        print(xn)
        plt.plot(x1, xn)
        # plt.show()
        SignalSamplesAreEqual("DC_component_output.txt", xn)

        # Creating Widgets

    Label1 = ttk.Label(Remove2_page, text="Remove_page")
    loadfile1 = Button(Remove2_page, text="LOAD FILE", command=Load1)
    draw = Button(Remove2_page, text="Draw", command=Remove2)

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

    def Load1():
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

    def Load2():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x2.append(a[i][j])
                    xc.append(a[i][j])
                else:
                    y2.append(a[i][j])

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
        ConvTest(xc, yc)

        # Creating Widgets

    Label1 = ttk.Label(convolution_page, text="Remove_page")
    loadfile1 = Button(convolution_page, text="LOAD FILE", command=Load1)
    loadfile2 = Button(convolution_page, text="LOAD FILE 2", command=Load2)
    draw = Button(convolution_page, text="Draw", command=convolution)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    loadfile2.place(x=150, y=150)
    draw.place(x=250, y=150)


def Folding_window():
    Folding_page = tkinter.Toplevel()
    Folding_page.minsize(500, 500)

    # create variables
    x = []
    y = []
    x1 = []
    y1 = []

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x.append(a[i][j])
                else:
                    y.append(a[i][j])

    def Folding():
        x1 = np.zeros(len(x))
        y1 = np.zeros(len(y))
        length = len(y) - 1
        for i in range(len(y)):
            x1[i] = x[i]
            y1[i] = y[length]
            length = length - 1
        print(x1)
        print(y1)
        SignalSamplesAreEqual("Output_fold.txt", y1)

    # Creating Widgets
    Label1 = ttk.Label(Folding_page, text="Folding")
    loadfile1 = Button(Folding_page, text="LOAD FILE", command=Load1)
    draw = Button(Folding_page, text="Draw", command=Folding)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)


def Folding_Shift_window():
    Folding_Shift_page = tkinter.Toplevel()
    Folding_Shift_page.minsize(500, 500)
    x = []
    y = []
    x1 = []
    y1 = []
    shiftpos = StringVar()

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x.append(a[i][j])
                else:
                    y.append(a[i][j])

    length = len(y)

    def Folding_Shift():
        # Folding
        x1 = np.zeros(len(x))
        y1 = np.zeros(len(y))
        length = len(y) - 1
        for i in range(len(y)):
            x1[i] = x[i]
            y1[i] = y[length]
            length = length - 1
        # shifting
        shift_val_pos = int(shiftpos.get())
        for i in range(len(x)):
            x1[i] = x[i] + shift_val_pos
        print(x1)
        Shift_Fold_Signal("Output_ShiftFoldedby-500.txt", x1, y1)

        # Creating Widgets

    Label1 = ttk.Label(Folding_Shift_page, text="Remove_page")
    loadfile1 = Button(Folding_Shift_page, text="LOAD FILE", command=Load1)
    draw = Button(Folding_Shift_page, text="Draw", command=Folding_Shift)
    con = ttk.Label(Folding_Shift_page, text="shift : ")
    cons = Entry(Folding_Shift_page, textvariable=shiftpos)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
    con.place(x=100, y=200)
    cons.place(x=220, y=200)


def Delay_window():
    Delay_page = tkinter.Toplevel()
    Delay_page.minsize(500, 500)
    x = []
    y = []
    x1 = []
    y1 = []
    shiftpos = StringVar()

    def Load1():
        filepath = filedialog.askopenfile()
        a = np.loadtxt(filepath, dtype=float, skiprows=3)
        for i in range(len(a)):
            for j in range(2):
                if (j == 0):
                    x.append(a[i][j])
                else:
                    y.append(a[i][j])

    def Delay():
        # shifting
        shift_val_pos = int(shiftpos.get())
        for i in range(len(x)):
            x1[i] = x[i] + shift_val_pos
        print(x1)

        # Creating Widgets

    Label1 = ttk.Label(Delay_page, text="Remove_page")
    loadfile1 = Button(Delay_page, text="LOAD FILE", command=Load1)
    draw = Button(Delay_page, text="Draw", command=Delay)
    con = ttk.Label(Delay_page, text="shift : ")
    cons = Entry(Delay_page, textvariable=shiftpos)

    # Put Widgets On Screen
    Label1.place(x=200, y=10)
    loadfile1.place(x=150, y=100)
    draw.place(x=150, y=150)
    con.place(x=100, y=200)
    cons.place(x=220, y=200)


def LOAd():
    filepath = filedialog.askopenfile()
    a = np.loadtxt(filepath, dtype=float, skiprows=3)
    x = []
    y = []
    for i in range(len(a)):
        for j in range(2):
            if (j == 0):
                x.append(a[i][j])
            else:
                y.append(a[i][j])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(x, y)
    ax2.stem(x, y)
    ax1.legend()
    ax1.set_title('Continuous')
    ax2.legend()
    ax2.set_title('Discrete')
    plt.show()


top = Tk()
top.title("Signal")
top.minsize(700, 700)
but1 = Button(text="LOAD FILE", command=LOAd)
but1.pack()
but2 = Button(text="Open Signals Page", command=Window)
but2.pack()
but3 = Button(text="Addition", command=Addition_window)
but3.pack()
but4 = Button(text="Subtraction", command=Subtraction_window)
but4.pack()
but5 = Button(text="Multiplication", command=Multiplication_window)
but5.pack()
but6 = Button(text="Squaring", command=lambda: Squaring_window(listofamp))
but6.pack()
but7 = Button(text="Shifting", command=Shifting_window)
but7.pack()
but7 = Button(text="Normalization", command=Normalization_window)
but7.pack()
but8 = Button(text="Accumulation ", command=Accumulation_window)
but8.pack()
but9 = Button(text="quantization", command=Quantization_window)
but9.pack()
but10 = Button(text="DFT", command=DFT_window)
but10.pack()
but11 = Button(text="DCT", command=DCT_window)
but11.pack()
but12 = Button(text="Remove DC component", command=Folding_window)
but12.pack()
but13 = Button(text="Smoothing", command=Smoothing_window)
but13.pack()
but14 = Button(text="Remove DC component 2", command=Remove2_window)
but14.pack()
but15 = Button(text="convolution", command=convolution_window)
but15.pack()
but16 = Button(text="Sharpening", command=DerivativeSignal)
but16.pack()
but17 = Button(text="DELAYING AND ADVANCING ", command=Delay_window)
but17.pack()
but18 = Button(text="Folding ", command=Folding_window)
but18.pack()
but19 = Button(text="Folding and shift ", command=Folding_Shift_window)
but19.pack()
top.mainloop()