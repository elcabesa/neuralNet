#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters  
from numpy.polynomial.polynomial import polyfit

def read_data():
    f = "./err.csv"
    data = pd.read_csv(f)
    return data

def print_graph():
    plt.ion()
    fig = plt.figure()
    fig.suptitle('training report')
    
    plt.ylabel('avg error')
    plt.xlabel('epoch')
    plt.show()
    while True:
        data = read_data()
        
        plt.clf()
        plt.grid()
        plt.ylim([0,3.25])
        plt.xlim([0,50000])
        #plt.xlim([0,1000000])
        
        index = data.index * 10
        
        plt.plot(index, data.TrainsetError, label = "trainset")
        plt.plot(index, data.ValidationError, label = "validation")
        
        b, m = polyfit(index[-500:-1], data.TrainsetError[-500:-1], 1)
        b1, m1 = polyfit(index[-500:-1], data.ValidationError[-500:-1], 1)
        
        plt.plot(index, b + m * index, label= "m= {0:.3g}/1000".format(m*1000))
        plt.plot(index, b1 + m1 * index, label= "m= {0:.3g}/1000 ".format(m1*1000))
        
        plt.legend(loc= 'upper right')
        plt.gcf().canvas.draw()
        
        plt.pause(20)
    
if __name__ == "__main__":
    print("PRINT LEARNING REPORT")
    
    register_matplotlib_converters()
    print_graph()
   
