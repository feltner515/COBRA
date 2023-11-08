import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import random
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

tfd=tfp.distributions
tfpl=tfp.layers

#rotates an MxN matrix 180 degrees
def rotate_180(array, M, N, out):
    for i in range(M):
        for j in range(N):
            out[i, N-1-j] = array[M-1-i, j]
 

#creates a map (250x250 martrix) with the amount of energy imparted on each element surface of the FEA model using sphereical cap model

def vlocal(x_star,y_star,a,radius,areaeqdiameter,velocity):
    x_grid=(np.linspace(0,249,250)*0.02)+0.01
    y_grid=(np.linspace(0,249,250)*0.02)+0.01
    h=-np.sqrt((radius**2)-(a**2))+radius
    if np.isnan(h)==1:
        h=0
    if np.isinf(h) == 1:
        h=0
    

    e_total=((0.5*(4/3)*np.pi*(7.98*10**-9)*((areaeqdiameter/2)**3)*(velocity**2)))
    v_total=(1/3)*np.pi*(h**2)*((3*radius)-h)
    sf=e_total/v_total
    v=np.zeros((250,250))
    for n in range (0,250,1):
        for p in range (0,250,1):
            z=(np.sqrt((radius**2)-(x_grid[n]**2)+(2*x_grid[n]*x_star)-(x_star**2)-(y_grid[p]**2)+(2*y_grid[p]*y_star)-(y_star**2)))-(radius-h)
            if z < 0:
                z=0
            if np.isnan(z)==1:
                z=0
            if np.isinf(z) == 1:
                x=0
            v[n,p]=z*0.02*0.02*sf


    isthevaluenan=np.isnan(v)
    v[isthevaluenan]=0

    return(v)

def impactenergyfile(name,velocity):
    data=pd.read_csv("{}.txt".format(name))
    energyplot=np.zeros((250,250))
    energytemp=np.zeros((250,250,data.shape[0]))
    for n in range (0, data.shape[0],1):
        a=0.11442655*(((0.5*(4/3)*np.pi*(7.98*10**-9)*((data.AreaEqDiameter[n]/2)**3)*(velocity**2)))**0.28999685)*((data.impactdiameter[n])**0.33790162)
        energytemp[:,:,n]=vlocal(x_star=data.x[n],y_star=data.y[n],a=a,radius=data.impactdiameter[n]/2, areaeqdiameter=data.AreaEqDiameter[n], velocity=velocity)
        
    energyplot=np.sum(energytemp, axis=2)
        
    energyplot=energyplot.T

    np.savetxt('{}_impactenergy.csv'.format(name), energyplot, delimiter=',')
        
#Takes in the name, file number (for sets of multiple training and test data files), and the desired width of the training dataset matrix
#Opens the surface stress field from the FEA model (251x251 matrix- stress at each node on the surface)
#Iterates accross the surface of the part and isolates every region of size (numcells x numcells), then reshapes it to size (numcells**2)
#Each row of the table is a set of inputs
#The stress at the center of each (numcells x numcells) region is saved in a seperate table, each row of the stress cooresponds to the training target
#The average stresses has the same format, but instead of the stress at the center of the matrix, it is the average of the stress values of all nodes that make up the region

def gentestdata(name, numcells):
    densitydata=np.genfromtxt('{}_impactenergy.csv'.format(name), delimiter=',')
    stressdata=np.genfromtxt('{}_surfstress.csv'.format(name), delimiter=',')

    flatstress=np.array([])
    averagestress=np.empty(((numcells+1)**2,(int(len(densitydata)-numcells)+1)**2))
    flatdensity=np.empty((numcells, numcells,(int(len(densitydata)-numcells)+1)**2))
    tempdensity=np.empty((numcells,numcells))

    for n in range(0,int(len(densitydata)-numcells)+1):
        for p in range (0,int(len(densitydata)-numcells)+1):
            tempdensity[:,:]=densitydata[n:n+numcells, p:p+numcells]
            tempstress=stressdata[int(n+numcells/2), int(p+numcells/2)]
            flatstress=np.append(flatstress,tempstress)
            flatdensity[:, : ,p+(n*(len(densitydata)-numcells+1))]=tempdensity#.flatten()
            averagestress[:,p+(n*(len(densitydata)-numcells+1))]=stressdata[n:n+numcells+1,p:p+numcells+1].reshape(((numcells+1)**2,1)).flatten()


    averagestresses=np.average(averagestress,axis=0)

    return(flatdensity,flatstress,averagestresses)

filelist=glob.glob('./*_impactenergy.csv')
for n in range (0,len(filelist),1):
    temp=filelist[n]
    filelist[n]=temp[:-17]

#Generates the test and training data for all 10 datasets


trainingdensity, trainingstress, trainavgstress = gentestdata('CW32_Train1',18)
trainingdensity=trainingdensity.T
fulltrainingdensity=np.empty((len(filelist)*trainingdensity.shape[0],trainingdensity.shape[1], trainingdensity.shape[2]))
fulltrainingstress=np.empty((len(filelist)*trainingstress.shape[0]))
fulltrainavgstress=np.empty((len(filelist)*trainavgstress.shape[0]))
for n in range (1,len(filelist)+1,1):
    trainingdensity1, trainingstress1, trainavgstress1 = gentestdata(filelist[n-1],18)
    trainingdensity1=trainingdensity1.T
    fulltrainingdensity[(n-1)*trainingdensity.shape[0]:n*trainingdensity.shape[0],:,:]=trainingdensity1
    fulltrainingstress[(n-1)*trainingstress1.shape[0]:n*trainingstress1.shape[0]]=trainingstress1
    fulltrainavgstress[(n-1)*trainavgstress1.shape[0]:n*trainavgstress1.shape[0]]=trainavgstress1

valsetsize=(int(len(fulltrainingstress)*0.25))
dataset=range(len(fulltrainingstress))
validationdatapts=random.sample(dataset,valsetsize)
validationdatastress=fulltrainingstress[validationdatapts]
validationdatadensity=fulltrainingdensity[validationdatapts,:]
fulltrainingstress=np.delete(fulltrainingstress, validationdatapts)
fulltrainingdensity=np.delete(fulltrainingdensity, validationdatapts, axis=0)
fulltrainingdensity = fulltrainingdensity.reshape(-1, 1, 18, 18)
validationdatadensity = validationdatadensity.reshape(-1, 1, 18, 18)


def get_deterministic_model(input_shape, loss, optimizer, metrics):
    model=Sequential([#Conv2D(kernel_size=(1,1), filters=8, activation='relu', padding='VALID', input_shape=input_shape),
                      
                      #Flatten(),
                      Dense(units=512, activation='selu'), 
                      Dense(units=256, activation='selu'),
                      Dense(units=128, activation='selu'),
                      Dense(units=64, activation='selu'),
                      Dense(units=32, activation='selu'),
                      Dense(units=1, activation='selu')
                      ])
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return (model)


tf.random.set_seed(0)

deterministic_model=get_deterministic_model(input_shape=(1,18,18),
                                            loss=MeanSquaredError(),
                                            optimizer=Adam(),
                                            metrics=['mae']
                                            )

deterministic_model.fit(fulltrainingdensity,fulltrainingstress,epochs=5)

