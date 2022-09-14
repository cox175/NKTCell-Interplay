#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:31:35 2022

@author: nxc045
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pyswarm import pso


#defining step functions: All step functions are the same as in NKTCellInterplay.py.
def stepPD(t):
    if t>=9:
        return 1
    else:
        return 0
    

def naivecellboost(C):
    if C>=500:
        return 1
    else:
        return 0
    
def NKdecay(t):
    if t>=9:
        return 1
    else:
        return 0
    
def naivestop(t):
    if t<=14:
        return 1
    else:
        return 0

def progenitorstop(t):
    if t<=16:
        return 1
    else:
        return 0

#All the cytokine depletion experiments were performed in the NK Depleted with Anti-PD1 setting,
#so we need only a single set of ODEs.
def combomodel(X, t, rhoC, rhoT1, rhoT2, rhoT3, alphaT3, rhoT4, alphaT4, rhoT5, thetaT, alphaT5, alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, beta12, beta23, beta24, beta35, beta45, alphaPD3, alphaPD4, alphaPD5, K):
    [C,N,T1,T2,T3,T4,T5]=X
    dCdt=rhoC*C-(alphaT3+stepPD(t)*alphaPD3)*C*T3-(alphaT4+stepPD(t)*alphaPD4)*C*T4-(alphaT5+stepPD(t)*alphaPD5)*C*T5-alphaN*N*C
    dNdt=rN-10*lambdaN*N+rhoN*N*C*(K-T1-T2-T3-T4-T5-N)-NKdecay(t)*thetaN*N
    dT1dt=rT+rhoT1*naivecellboost(C)-naivestop(t)*beta12*T1
    dT2dt=rhoT2*T2*C*(K-T1-T2-T3-T4-T5-N)+naivestop(t)*beta12*T1-progenitorstop(t)*(beta23*T2+beta24*T2)
    dT3dt=rhoT3*T3*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta23*T2-beta35*T3
    dT4dt=rhoT4*T4*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta24*T2-beta45*T4
    dT5dt=rhoT5*T5*C*(K-T1-T2-T3-T4-T5-N)+beta35*T3+beta45*T4-thetaT*T5
    dXdt=[dCdt, dNdt, dT1dt, dT2dt, dT3dt, dT4dt, dT5dt]
    return dXdt
    

def model(t, params, info, status):
    #Here we fix all the parameters except for K, which we fit.
    [K]=params
    C0=10
    rhoC=.4
    rT=.5
    rN=1.5
    lambdaN=.069
    lambdaT=.09
    thetaN=.049
    thetaT=.1
    rhoT5=0
    
    [cancercount, negative]=info

    rhoT1=10**.88
    rhoT2=10**-4.56
    rhoT3=10**-5.60
    rhoT4=10**-6.73
    alphaT3=np.log(2)/(cancercount*7)
    alphaT4=10**-2.54
    alphaT5=10**-4.25
    rhoN=10**-5.81
    alphaN=10**-4.4
    beta12=np.log(2)/(.02)
    beta23=np.log(2)/.33
    beta24=np.log(2)/.14
    beta35=np.log(2)/4.5
    beta45=np.log(2)/65

    if status=='responder':
        alphaPD3=0
        alphaPD4=0
        alphaPD5=10**-.9
    elif status=='nonresponder':
        alphaPD3=2*np.log(2)/(cancercount*2.5)
        alphaPD4=2*np.log(2)/(cancercount*.7)
        alphaPD5=2*np.log(2)/(cancercount*1.8)

    B=odeint(combomodel, [C0, .3*rN/lambdaN, rT/lambdaT,0,0,0,0], t, args=(rhoC, rhoT1, rhoT2, rhoT3, alphaT3, rhoT4, alphaT4, rhoT5, thetaT, alphaT5, alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, beta12, beta23, beta24, beta35, beta45, alphaPD3, alphaPD4, alphaPD5, K))                                                      
    return B

def residual(params, t, dataset, info, status):
     Cost=pd.Series(dtype='float64')
     C=pd.DataFrame(model(t, params, info, status))
     Cost=Cost.append(dataset.subtract(C.iloc[:,0].values, axis=0))
     norm=np.square(Cost)
     norm=np.sqrt(np.sum(norm))
     return norm

#import the data. Location will need to be changed to match
#where file is placed locally
data=pd.read_csv('../cytokinedepletion.csv', header=None)

t=data.iloc[46:64, 0].astype(float)

status='responder'        #comment to toggle between
#status= 'nonresponder    #responder and nonresponder

if status=='responder':
    IL2data=data.iloc[46:64,1:4].astype(float)
    avgIL2data=IL2data.T.mean()
  

elif status=='nonresponder':
    IL2data=data.iloc[46:64,4:11].astype(float)
    avgIL2data=IL2data.T.mean()


case_type="IL-2 Depletion"


cancercount=[avgIL2data[52]]

negative=-47.2

info=[cancercount, negative]

lb=[0]
ub=[500]

param_opt, resid=pso(residual, lb=lb, ub=ub, args=(t, avgIL2data, info, status), maxiter=50, swarmsize=100)

print(param_opt, resid)

timerange=np.linspace(0,23, 200)


plt.figure(1)
plt.title('IL-2 Depletion NK Depleted Anti-PD1 Cancer Cell Prediction')
plt.plot(t, avgIL2data, 'o', color='black',label='Data (Cancer)')
plt.plot(t, model(t, param_opt, info, status)[:,0], color='red', label='Model (Cancer)')
plt.ylabel("Cancer Cells/100000")
plt.legend()









