import numpy as np
import pandas as pd

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

    return(energyplot)
        
