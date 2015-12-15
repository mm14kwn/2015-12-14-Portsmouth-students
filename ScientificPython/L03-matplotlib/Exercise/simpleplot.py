#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
plt.style.use('./matplotlibrc.custom')
import numpy as np

def figdims(width,factor):
    widthpt=width*factor
    inperpt=1.0/72.27
    golden_ratio=(np.sqrt(5)-1.0)/2.0
    widthin=widthpt*inperpt
    heightin=widthin*golden_ratio
    return [widthin,heightin]

def kwnplot(filepath,savepath):
    "plots x-y data from file given by filepath"
    data=np.genfromtxt(filepath)
#    print data
    fig = plt.figure()
    plt.plot(data[:,0],data[:,1]);
    plt.xlabel('points')
    plt.ylabel('values')
    plt.suptitle('randplot')
    fig.savefig(savepath)
    plt.cla()

def kwnplot2(filepath1,filepath2,savepath,savepath2):
    data1=np.genfromtxt(filepath1)
    data2=np.genfromtxt(filepath2)
    fig=plt.figure(figsize=figdims(500,0.75))
    plt.plot(data1[:,0],data1[:,1],'kx--',label='data1');
    plt.plot(data2[:,0],data2[:,1],'rx-',label='data2');
    plt.legend()
    fig.tight_layout(pad=0.1)
    fig.savefig(savepath,dpi=600)
    plt.cla()
    plt.clf()
    fig=plt.figure(figsize=figdims(500,0.75))
    plt.subplot(2,1,1)
    plt.plot(data1[:,0],data1[:,1],'kx--',label='data1');
    plt.subplot(2,1,2)
    plt.plot(data2[:,0],data2[:,1],'rx-',label='data2');
    fig.tight_layout(pad=0.1)
    fig.savefig(savepath2,dpi=600)
    plt.cla()
    plt.clf()

kwnplot('./data/random1.dat','./rand1.pdf')

kwnplot('./data/random2.dat','./rand2.pdf')

kwnplot2('./data/random1.dat','./data/random2.dat','./rand3.pdf','./rand4.pdf')


