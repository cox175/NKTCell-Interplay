#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 16:53:06 2022

@author: nxc045
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pyswarm import pso



#defining step functions:
def stepPD(t):     #For the two cases with anti-PD1, 
    if t>=9:       #this function represents when the
        return 1   #inhibitor is active (after day 9)
    else:
        return 0


def naivecellboost(C): #This function represents the body's increased production
    if C>=500:         #of Naive T cells in the presence of antigen
        return 1       #in this case , C=500.
    else:
        return 0
    
def NKdecay(t):        #This function represents a time delay before NK cells 
    if t>=9:           # begin to decay, after 9 days.
        return 1
    else:
        return 0
    
def naivestop(t):     #As the tumor grows, it becomes harder for the naive cells to
    if t<=14:         #have access to antigen reciptors to receive a signal to differentiate
        return 1      #into progenitor T cells. This function models this effect by stopping naive
    else:             #differentiation after 14 days.
        return 0

def progenitorstop(t):  #Similar to the prior function, progenitor cell are made to stop
    if t<=16:           #differentiating at day 16. 
        return 1
    else:
        return 0

#The following function contains the ODEs to model the isotype experiment:
def isotypemodel(X,t,rhoC, rhoT1, rhoT2, rhoT3, alphaT3, rhoT4, alphaT4,rhoT5, \
                 thetaT, alphaT5,alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, \
                 beta12, beta23, beta24, beta35, beta45, K):
    [C,N,T1,T2,T3,T4,T5]=X
    dCdt=rhoC*C-alphaT3*C*T3-alphaT4*C*T4-alphaT5*C*T5-alphaN*N*C
    dNdt=rN-lambdaN*N+rhoN*N*C*(K-T1-T2-T3-T4-T5-N)-NKdecay(t)*thetaN*N
    dT1dt=rT+rhoT1*naivecellboost(C)-naivestop(t)*beta12*T1
    dT2dt=rhoT2*T2*C*(K-T1-T2-T3-T4-T5-N)+naivestop(t)*beta12*T1-progenitorstop(t)*(beta23*T2+beta24*T2)
    dT3dt=rhoT3*T3*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta23*T2-beta35*T3
    dT4dt=rhoT4*T4*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta24*T2-beta45*T4
    dT5dt=rhoT5*T5*C*(K-T1-T2-T3-T4-T5-N)+beta35*T3+beta45*T4-thetaT*T5
    dXdt=[dCdt, dNdt, dT1dt, dT2dt, dT3dt, dT4dt, dT5dt]
    return dXdt

#The following function contains the ODEs to model the isotype experiment with anti-PD1:
def PDmodel(X,t,rhoC, rhoT1, rhoT2, rhoT3, alphaT3, rhoT4, alphaT4,rhoT5,\
            thetaT, alphaT5,alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, \
            beta12, beta23, beta24, beta35, beta45, alphaPD3, alphaPD4, alphaPD5, K):
    [C,N,T1,T2,T3,T4,T5]=X
    dCdt=rhoC*C-(alphaT3+stepPD(t)*alphaPD3)*C*T3 \
            -(alphaT4+stepPD(t)*alphaPD4)*C*T4-(alphaT5+stepPD(t)*alphaPD5)*C*T5-alphaN*N*C
    dNdt=rN-lambdaN*N+rhoN*N*C*(K-T1-T2-T3-T4-T5-N)-NKdecay(t)*thetaN*N
    dT1dt=rT+rhoT1*naivecellboost(C)-naivestop(t)*beta12*T1
    dT2dt=rhoT2*T2*C*(K-T1-T2-T3-T4-T5-N)+naivestop(t)*beta12*T1-progenitorstop(t)*(beta23*T2+beta24*T2)
    dT3dt=rhoT3*T3*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta23*T2-beta35*T3
    dT4dt=rhoT4*T4*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta24*T2-beta45*T4
    dT5dt=rhoT5*T5*C*(K-T1-T2-T3-T4-T5-N)+beta35*T3+beta45*T4-thetaT*T5
    dXdt=[dCdt, dNdt, dT1dt, dT2dt, dT3dt, dT4dt, dT5dt]
    return dXdt


#The following function contains the ODEs to model the NK Depleted experiment:
def NKmodel(X,t,rhoC, rhoT1, rhoT2, rhoT3, alphaT3, rhoT4, alphaT4,rhoT5,\
            thetaT, alphaT5,alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, \
            beta12, beta23, beta24, beta35, beta45, K):
    [C,N,T1,T2,T3,T4,T5]=X
    dCdt=rhoC*C-alphaT3*C*T3-alphaT4*C*T4-alphaT5*C*T5-alphaN*N*C
    dNdt=rN-10*lambdaN*N+rhoN*N*C*(K-T1-T2-T3-T4-T5-N)-NKdecay(t)*thetaN*N
    dT1dt=rT+rhoT1*naivecellboost(C)-naivestop(t)*beta12*T1
    dT2dt=rhoT2*T2*C*(K-T1-T2-T3-T4-T5-N)+naivestop(t)*beta12*T1-progenitorstop(t)*(beta23*T2+beta24*T2)
    dT3dt=rhoT3*T3*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta23*T2-beta35*T3
    dT4dt=rhoT4*T4*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta24*T2-beta45*T4
    dT5dt=rhoT5*T5*C*(K-T1-T2-T3-T4-T5-N)+beta35*T3+beta45*T4-thetaT*T5
    dXdt=[dCdt, dNdt, dT1dt, dT2dt, dT3dt, dT4dt, dT5dt]
    return dXdt

#The following function contains the ODEs to model the NK Depleted experiment with anti-PD1:
def combomodel(X,t,rhoC, rhoT1, rhoT2, rhoT3, alphaT3, rhoT4, alphaT4,rhoT5,\
            thetaT, alphaT5,alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, \
            beta12, beta23, beta24, beta35, beta45, alphaPD3, alphaPD4, alphaPD5, K):
    [C,N,T1,T2,T3,T4,T5]=X
    dCdt=rhoC*C-(alphaT3+stepPD(t)*alphaPD3)*C*T3 \
            -(alphaT4+stepPD(t)*alphaPD4)*C*T4-(alphaT5+stepPD(t)*alphaPD5)*C*T5-alphaN*N*C
    dNdt=rN-10*lambdaN*N+rhoN*N*C*(K-T1-T2-T3-T4-T5-N)-NKdecay(t)*thetaN*N
    dT1dt=rT-lambdaT*T1+rhoT1*naivecellboost(C)-naivestop(t)*beta12*T1
    dT2dt=rhoT2*T2*C*(K-T1-T2-T3-T4-T5-N)+naivestop(t)*beta12*T1-progenitorstop(t)*(beta23*T2+beta24*T2)
    dT3dt=rhoT3*T3*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta23*T2-beta35*T3
    dT4dt=rhoT4*T4*C*(K-T1-T2-T3-T4-T5-N)+progenitorstop(t)*beta24*T2-beta45*T4
    dT5dt=rhoT5*T5*C*(K-T1-T2-T3-T4-T5-N)+beta35*T3+beta45*T4-thetaT*T5
    dXdt=[dCdt, dNdt, dT1dt, dT2dt, dT3dt, dT4dt, dT5dt]
    return dXdt


def model( t, params, i):
    
    #fixed paramters
    C0=10
    rhoC=.35
    rT=.5
    rN=1.5
    lambdaN=.069
    lambdaT=.09
    
    
    #isotype fitted parameters: uncomment these parameters and make sure all other fitted parameters are 
    #commented out
    
    rhoT1= 10**0.7495213
    rhoT2= 10**-5.70862656
    rhoT3= 10**-5.72481715
    alphaT3=10**-4.17123089
    rhoT4= 10**-6.18280523
    alphaT4= 10**-2.25177164
    rhoT5=0
    thetaT= 10**-0.61862392 
    alphaT5=10**-4.884995
    alphaN=10**-3.61922192
    rhoN= 10**-5.91201749
    thetaN=.049
    beta12= 10**-0.56744273
    beta23= 10**-0.37089137
    beta24= 10**-0.87702188
    beta35= 10**-0.32025071
    beta45= 10**-1.4351011
    K= 509.51288741

    
    #NK depleted fitted parameters: uncomment these and comment out the isotype parameters
    #(use alphaN, rhoN, and thetaN from isotype case; we did not fit NK cell parameters in 
    #this case since the cell counts were so low)
    

    #rhoT1= 10**1.00101164 
    #rhoT2= 10**-5.56038859
    #rhoT3= 10**-6.80159815
    #alphaT3= 10**-2.53611311
    #rhoT4= 10**-5.66847186
    #alphaT4= 10**-4.08105097
    #rhoT5=0 
    #thetaT= 10**-1.53083651
    #alphaT5= 10**-4.73411455
    #beta12= 10**1.14329813
    #beta23= 10**0.48979402
    #beta24= 10**-1.12197851
    #beta35= 10**-2.50602229
    #beta45= 10**-0.43371652
    #K= 10**2.70764175


    
    #Anti-PD1 Responder fitted values: uncomment these and uncomment the isotype values, comment out
    #everything else
    
    #alphaPD3=10**-5.44795034
    #alphaPD4=10**-0.98739226 
    #alphaPD5=10**-2.23766122
    
    #PD1 Non responder fitted values: uncomment these and uncomment the isotype values, comment out
    #everything else
    
    #alphaPD3=10**-2.04657226
    #alphaPD4=10**-4.84908771
    #alphaPD5=10**-5.1549298
    
    #Combo Responder fitted values: uncomment these and uncomment the NK Depleted values, comment out
    #everything else
    
    #alphaPD3=10**-4.8771434  
    #alphaPD4=10**-3.42591918 
    #alphaPD5=10**-0.21112293
    
    #Combo Nonresponder fitted values: uncomment these and uncomment the NK Depleted values, comment out
    #everything else
    
    #alphaPD3=10**-1.59254741 
    #alphaPD4=10**-5.49124765 
    #alphaPD5=10**-1.2126297 
    
    #For the code to work we need to define variables for alphaPD3, 4, and 5 in the isotype and NK depleted
    #treatment types
    
    alphaPD3=0
    alphaPD4=0 
    alphaPD5=0
    
    
    #This if block determines which set of ODEs the solver will run on based on treatment type
    #The value of i is chosen in the next function "residual" because it is used there first
    if i==0: 
        B=odeint(isotypemodel, [C0, rN/lambdaN, rT/lambdaT, 0,0,0,0], t, args=(rhoC, rhoT1, rhoT2, \
            rhoT3, alphaT3, rhoT4, alphaT4,rhoT5, thetaT, alphaT5,alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, \
            beta12, beta23, beta24, beta35, beta45, K))
    
    elif i==2:
        B=odeint(PDmodel, [C0, rN/lambdaN, rT/lambdaT,0,0,0,0], t, args=(rhoC, rhoT1, rhoT2, \
                rhoT3, alphaT3, rhoT4, alphaT4,rhoT5, thetaT, alphaT5,alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, \
           beta12, beta23, beta24, beta35, beta45, alphaPD3, alphaPD4, alphaPD5, K))
    elif i==1:
        B=odeint(NKmodel, [C0, (1/2)*rN/lambdaN, rT/lambdaT,0,0,0,0], t, args=(rhoC, rhoT1, rhoT2, \
                rhoT3, alphaT3, rhoT4, alphaT4,rhoT5, thetaT, alphaT5,alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, \
            beta12, beta23, beta24, beta35, beta45, K))
    elif i==3:
        B=odeint(combomodel, [C0, (1/2)*rN/lambdaN, rT/lambdaT,0,0,0,0], t, args=(rhoC, rhoT1, rhoT2, \
                rhoT3, alphaT3, rhoT4, alphaT4,rhoT5, thetaT, alphaT5,alphaN, rhoN, thetaN, rN, rT, lambdaN, lambdaT, \
            beta12, beta23, beta24, beta35, beta45, alphaPD3, alphaPD4, alphaPD5, K))
    
    return B

def residual(params, t, dataset):
    #This function calls the prior function "model" to solve the ODEs, and the calculates
    #the residual/cost function (i.e. how different the model is from the data) 

    Ccost=pd.Series(dtype='float64')
    NKcost=pd.Series(dtype='float64')
    T1cost=pd.Series(dtype='float64')
    T2cost=pd.Series(dtype='float64')
    T3cost=pd.Series(dtype='float64')
    T4cost=pd.Series(dtype='float64')
    T5cost=pd.Series(dtype='float64')
    i=0 #This variable determines which set of ODEs is solved. i=0 is isotype, i=1 is NK Depleted,
        #i=2 is isotype with anti-PD1, and i=3 is NK Depleted with anti-PD1.

    C=pd.DataFrame(model(t, params, i)) #runs the ODE and determines the model value for each cell type
    #at each time point
    
    Ccost=Ccost.append(dataset.iloc[0:15,i]-C.iloc[:,0]) #how different the cancer cell counts are between
    #the model and the data
    
    #this if block calculates how different the model and the data are for the immune cells
    if i==0 or i==1:   
        NKcost=NKcost.append(dataset.iloc[15:19,i]-(C.iloc[[3,5,9,14],1].values))
        T1cost=T1cost.append(dataset.iloc[19:23,i]-(C.iloc[[3,5,9,14],2].values))
        T2cost=T2cost.append(dataset.iloc[23:27,i]-(C.iloc[[3,5,9,14],3].values))
        T3cost=T3cost.append(dataset.iloc[27:31,i]-(C.iloc[[3,5,9,14],4].values))
        T4cost=T4cost.append(dataset.iloc[31:35,i]-(C.iloc[[3,5,9,14],5].values))
        T5cost=T5cost.append(dataset.iloc[35:39,i]-(C.iloc[[3,5,9,14],6].values))
    elif i==2 or i==3:
        NKcost=NKcost.append(dataset.iloc[16:18,i]-C.iloc[[3,5],1].values)
        T1cost=T1cost.append(dataset.iloc[19:21,i]-C.iloc[[3,5],2].values)
        T2cost=T2cost.append(dataset.iloc[22:24,i]-C.iloc[[3,5],3].values)
        T3cost=T3cost.append(dataset.iloc[25:27,i]-C.iloc[[3,5],4].values)
        T4cost=T4cost.append(dataset.iloc[28:30,i]-C.iloc[[3,5],5].values)
        T5cost=T5cost.append(dataset.iloc[31:33,i]-C.iloc[[3,5],6].values)
        
    #Here we convert the cost to a standard residual sum of squares     
    Cost=pd.concat([Ccost, 10*NKcost, 10*T1cost, 10*T2cost, 10*T3cost, 10*T4cost, 10*T5cost])
    norm=np.square(Cost)
    norm=np.sqrt(np.sum(norm))
    return norm
  

def main():  
    #import the data. File path will need to be updated to users file location
    data=pd.read_csv('../cellcounts.csv')
    immunecelldata=pd.read_csv('../subtype_proportions.csv')
    

    time=data['Day']
    status='Responder'     #switch between responder and non responder for anti-PD1 cases
    #status='NonResponder  #has no effect on the isotype and NK depleted cases.
    
    
    #Below we assign the data to various dataframes/series
    
    #immune data
    NKtime=immunecelldata.iloc[4:8,0]
    Ttime=immunecelldata.iloc[8:12,0]
    isoNKarea=immunecelldata.iloc[4:8,2]
    isoT1area=immunecelldata.iloc[8:12,2]
    isoT2area=immunecelldata.iloc[12:16,2] 
    isoT3area=immunecelldata.iloc[16:20,2] 
    isoT4area=immunecelldata.iloc[20:24,2] 
    isoT5area=immunecelldata.iloc[24:28,2] 
    nkdepNKarea=immunecelldata.iloc[4:8,4]
    nkdepT1area=immunecelldata.iloc[8:12,4]
    nkdepT2area=immunecelldata.iloc[12:16,4]
    nkdepT3area=immunecelldata.iloc[16:20,4]
    nkdepT4area=immunecelldata.iloc[20:24,4]
    nkdepT5area=immunecelldata.iloc[24:28,4]
    
    
    if status=='Responder': 
        allavgdata=data[['Control', 'NK Dep', 'PD Resp', 'Combo Resp']]
        column_dict={'Control':0,  'NK Dep':1, 'PD Resp':2,'Combo Resp':3}
        pdNKarea=immunecelldata.iloc[4:6,6]
        pdT1area=immunecelldata.iloc[8:10,6]
        pdT2area=immunecelldata.iloc[12:14,6]
        pdT3area=immunecelldata.iloc[16:18,6]
        pdT4area=immunecelldata.iloc[20:22,6]
        pdT5area=immunecelldata.iloc[24:26,6]
        comboNKarea=immunecelldata.iloc[4:6,8]
        comboT1area=immunecelldata.iloc[8:10,8]
        comboT2area=immunecelldata.iloc[12:14,8] 
        comboT3area=immunecelldata.iloc[16:18,8] 
        comboT4area=immunecelldata.iloc[20:22,8] 
        comboT5area=immunecelldata.iloc[24:26,8]
        

              
    elif status=='NonResponder':
        allavgdata=data[['Control',  'NK Dep', 'PD All Non Resp', 'Combo Non Resp']]
        pdNKarea=immunecelldata.iloc[4:6,7]
        pdT1area=immunecelldata.iloc[8:10,7]
        pdT2area=immunecelldata.iloc[12:14,7]
        pdT3area=immunecelldata.iloc[16:18,7]
        pdT4area=immunecelldata.iloc[20:22,7]
        pdT5area=immunecelldata.iloc[24:26,7]
        comboNKarea=immunecelldata.iloc[4:6,9]
        comboT1area=immunecelldata.iloc[8:10,9]
        comboT2area=immunecelldata.iloc[12:14,9] 
        comboT3area=immunecelldata.iloc[16:18,9] 
        comboT4area=immunecelldata.iloc[20:22,9] 
        comboT5area=immunecelldata.iloc[24:26,9]
        column_dict={'Control':0,  'NK Dep':1, 'PD All Non Resp':2,'Combo Non Resp':3}
    
    
    #Here we combine data together to get it into a more manageable form

    allNKdata=pd.concat([isoNKarea, nkdepNKarea, pdNKarea, comboNKarea], axis=1)
    allT1data=pd.concat([isoT1area,  nkdepT1area, pdT1area, comboT1area], axis=1)
    allT2data=pd.concat([isoT2area,  nkdepT2area, pdT2area, comboT2area], axis=1)
    allT3data=pd.concat([isoT3area,  nkdepT3area, pdT3area, comboT3area], axis=1)
    allT4data=pd.concat([isoT4area,  nkdepT4area, pdT4area, comboT4area], axis=1)
    allT5data=pd.concat([isoT5area,  nkdepT5area, pdT5area, comboT5area], axis=1)
    

    NK_index_dict={3:15, 4:16, 5:17}
    T_index_dict={6:18, 7:19, 8:20}

    allavgdata.rename(columns=column_dict, inplace=True)                
    allNKdata.rename(index=NK_index_dict, columns=column_dict, inplace=True)
    allT1data.rename(index=T_index_dict, columns=column_dict, inplace=True)
    allT2data.rename(index=T_index_dict, columns=column_dict, inplace=True)
    allT3data.rename(index=T_index_dict, columns=column_dict, inplace=True)
    allT4data.rename(index=T_index_dict, columns=column_dict, inplace=True)
    allT5data.rename(index=T_index_dict, columns=column_dict, inplace=True)
    


    alldata=pd.concat([allavgdata, allNKdata, allT1data, allT2data, allT3data, allT4data, allT5data], axis=0)
    
    #define the parameter space for each scenari:

    if status=='Responder': # set lower and upper bounds for any parameters you are fitting
                            #with optimization. Parameters need to be listed in the same order as in
                            #the function "model    
        lb=[ -6, -6, -5]
        ub=[ -1, -2, 0]

    elif status=='NonResponder':
              
        lb=[   -6,   -6,  -6] 
        ub=[   -2.5,  -2.5, -2]
       
    #uncomment this to fit the parameters with a particle swarm optimization algorithm
    #param_opt, resid=pso(residual, lb=lb, ub=ub, args=(time, alldata), maxiter=100, swarmsize=100, minfunc=1e-5, minstep=1e-6)

    #print(param_opt)
    #print(resid)
    
    param_opt=[] #We define an empty param_opt variable to make it simpler to switch between
                #fitting and just plotting results. Comment out if fitting data
                
   
    timerange=np.linspace(0,20,200)
    
    #what follows is the code to plot the fits. Each treatment type has 3 associated figures. They need to be
    #uncommented as necessary to display the images.
    
    #Isotype plots
    plt.figure(1)
    plt.title(f'Isotype Cancer Cell Fit')
    plt.ylabel('Cancer Cells/100,000')
    plt.plot(time, allavgdata.iloc[:,0], 'o', color='black', label='Data (Cancer)')   
    plt.plot(timerange, model(timerange, param_opt, i=0)[:,0], color='red', label='Model (Cancer)')
    plt.legend()
    
    plt.figure(2)
    plt.title("Isotype NK Cell Fit")
    plt.plot(timerange, model(timerange, param_opt, i=0)[:,1], color='green', label='Model (NK Cells)')
    plt.plot(NKtime, isoNKarea, 'o',color='green', label='Data (NK Cells)')
    plt.ylabel('NK Cells/100,000')
    plt.xlabel('Time (Days)')
    plt.legend()       

    plt.figure(3)
    plt.title("Isotype T Cell Fit")    
    plt.plot(timerange, model(timerange, param_opt, i=0)[:,2], color='blue', label='Model (Naive)')
    plt.plot(Ttime, isoT1area, 'o',color='blue', label='Data (Naive)')
    
    plt.plot(timerange, model(timerange, param_opt, i=0)[:,3], color='cyan', label='Model (Progenitor)')
    plt.plot(Ttime, isoT2area, 'o',color='cyan', label='Data (Progenitor)')
    
    plt.plot(timerange, model(timerange, param_opt, i=0)[:,4], color='purple', label='Model (Effector)')
    plt.plot(Ttime, isoT3area, 'o',color='purple', label='Data (Effector)')
    
    plt.plot(timerange, model(timerange, param_opt, i=0)[:,5], color='black', label='Model (New)')
    plt.plot(Ttime, isoT4area, 'o',color='black', label='Data (New)')
    
    plt.plot(timerange, model(timerange, param_opt, i=0)[:,6], color='orange', label='Model (Exhausted)')
    plt.plot(Ttime, isoT5area, 'o',color='orange', label='Data (Exhausted)')
        
    plt.ylabel('T cell Subtypes/100,000')  
    plt.xlabel('Time (Days)')
    plt.legend()

    
    '''
    #Isotype with Anti-PD1 Plots
    plt.figure(4)
    plt.title("Anti-PD1 Cancer Cell Fit")
    plt.plot(time, allavgdata.iloc[:,2], 'o', color='black', label='Data (Cancer)')
    
    plt.plot(timerange, model(timerange, param_opt, i=2)[:,0], color='red', label='Model (Cancer)')
    plt.ylabel('Cancer Cells/100,000')
    plt.legend()
    
    plt.figure(5)
    plt.title("Anti-PD1 NK Cell Fit")
    plt.plot(timerange, model(timerange, param_opt, i=2)[:,1], color='green', label='Model (NK Cells)')
    plt.plot([9,11], pdNKarea,'o', color='green', label='Data (NK Cells)')
    plt.ylabel('NK Cells/100,000')
    plt.xlabel('Time (Days)')    
    plt.legend()
    
    plt.figure(6)  
    plt.title("Anti-PD1 T Cell Fit")
    plt.plot(timerange2, model(timerange2, param_opt, i=2)[:,2], color='blue', label='Model (Naive)')
    plt.plot([9,11], pdT1area, 'o',color='blue', label='Data (Naive)')
    
    plt.plot(timerange2, model(timerange2, param_opt, i=2)[:,3], color='cyan', label='Model (Progenitor)')
    plt.plot([9,11], pdT2area, 'o',color='cyan', label='Data (Progenitor)')
    
    plt.plot(timerange2, model(timerange2, param_opt, i=2)[:,4], color='purple', label='Model (Effector)')
    plt.plot([9,11], pdT3area, 'o',color='purple', label='Data (Effector)')
    
    plt.plot(timerange2, model(timerange2, param_opt, i=2)[:,5], color='black', label='Model (New)')
    plt.plot([9,11], pdT4area, 'o',color='black', label='Data (New)')
    
    plt.plot(timerange2, model(timerange2, param_opt, i=2)[:,6], color='orange', label='Model (Exhausted)')
    plt.plot([9,11], pdT5area, 'o',color='orange', label='Data (Exhausted)')
    plt.ylabel('T cell Subtypes/100,000')  
    plt.xlabel('Time (Days)')
    plt.legend()


    #NK Depleted Plots
    plt.figure(7)
    plt.title(f'NK Depleted Cancer Cell Fit')
    plt.ylabel('Cancer Cells/100,000')
    plt.plot(time, allavgdata.iloc[:,1], 'o', color='black', label='Data (Cancer)')
    
    plt.plot(timerange, model(timerange, param_opt, i=1)[:,0], color='red', label='Model (Cancer)')
    plt.legend()
    
    plt.figure(8)
    plt.title("NK Depleted NK Cell Fit")
    plt.plot(timerange, model(timerange, param_opt, i=1)[:,1], color='green', label='Model (NK Cells)')
    plt.plot(NKtime, nkdepNKarea, 'o', color='green', label='Data (NK Cells)')
    plt.ylabel('NK Cells/100,000')
    plt.xlabel('Time (Days)')
    plt.legend()
    
    plt.figure(9) 
    plt.title("NK Depleted T Cell Fit")
    plt.plot(timerange, model(timerange, param_opt, i=1)[:,2], color='blue', label='Model (Naive)')
    plt.plot(Ttime, nkdepT1area, 'o',color='blue', label='Data (Naive)')
    
    plt.plot(timerange, model(timerange, param_opt, i=1)[:,3], color='cyan', label='Model (Progenitor)')
    plt.plot(Ttime, nkdepT2area, 'o',color='cyan', label='Data (Progenitor)')
    
    plt.plot(timerange, model(timerange, param_opt, i=1)[:,4], color='purple', label='Model (Effector)')
    plt.plot(Ttime, nkdepT3area, 'o',color='purple', label='Data (Effector)')
    
    plt.plot(timerange, model(timerange, param_opt, i=1)[:,5], color='black', label='Model (New)')
    plt.plot(Ttime, nkdepT4area, 'o',color='black', label='Data (New)')
    
    plt.plot(timerange, model(timerange, param_opt, i=1)[:,6], color='orange', label='Model (Exhausted)')
    plt.plot(Ttime, nkdepT5area, 'o',color='orange', label='Data (Exhausted)')
    plt.ylabel('T cell Subtypes/100,000')  
    plt.xlabel('Time (Days)')
    plt.legend()
    
    
    #NK Depleted with Anti-PD1 Plots
    plt.figure(10)
    plt.title("NK Depleted Anti-PD1 Cancer Cell Fit")
    plt.plot(time, allavgdata.iloc[:,3], 'o', color='black', label='Data (Cancer)')
    
    plt.plot(timerange2, model(timerange2, param_opt, i=3)[:,0], color='red', label='Model (Cancer)')
    plt.ylabel('Cancer Cells/100,000')
    plt.legend()
    
    plt.figure(11)
    plt.title("NK Depleted Anti-PD1 NK Cell Fit")
    plt.plot(timerange, model(timerange, param_opt, i=3)[:,1], color='green', label='Model (NK Cells)')
    plt.plot([9,11], comboNKarea, 'o', color='green', label='Data (NK Cells)')
    plt.ylabel('NK Cells/100,000')
    plt.xlabel('Time (Days)')    
    plt.legend()
    
    plt.figure(12)
    plt.title("NK Depleted Anti-PD1 T Cell Fit")    
    plt.plot(timerange, model(timerange, param_opt, i=3)[:,2], color='blue', label='Model (Naive)')
    plt.plot([9,11], comboT1area, 'o',color='blue', label='Data (Naive)')
    
    plt.plot(timerange, model(timerange, param_opt, i=3)[:,3], color='cyan', label='Model (Progenitor)')
    plt.plot([9,11], comboT2area, 'o',color='cyan', label='Data (Progenitor)')
    
    plt.plot(timerange, model(timerange, param_opt, i=3)[:,4], color='purple', label='Model (Effector)')
    plt.plot([9,11], comboT3area, 'o',color='purple', label='Data (Effector)')
    
    plt.plot(timerange, model(timerange, param_opt, i=3)[:,5], color='black', label='Model (New)')
    plt.plot([9,11], comboT4area, 'o',color='black', label='Data (New)')
    
    plt.plot(timerange, model(timerange, param_opt, i=3)[:,6], color='orange', label='Model (Exhausted)')
    plt.plot([9,11], comboT5area, 'o',color='orange', label='Data (Exhausted)')
    plt.ylabel('T cell Subtypes/100,000')  
    plt.xlabel('Time (Days)')
    plt.legend()
    '''
    
if __name__=="__main__":
    main()    
    
  
    
    
    
    
    
    
    
    
    
    
    
        
