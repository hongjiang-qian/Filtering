#------------------
# library loading
#------------------
import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#---------------------------------------
# Initialization Math Model
#---------------------------------------
start=time.perf_counter()
dimX=1; dimY=1; eta=0.04
N=100; n0=10; N_sample=256

sigma0_train=0.2

#Deterministic Matrix
x0=0

def f(x,a):
    return np.sin(5*a*x)
def s(x,a):
    return 0.7
def g(x,a):
    return 2*x
def s1(x,a):
    return sigma0_train
#--------------------------
# Simulating Markov Chain
#-------------------------
P= np.array([[0.98,0.02],[0.02,0.98]])
def mChain(N,P,ru=None):
    """
    simulate Markov chain based on the transition probability
    state: 1 and 2 """
    alpha=np.zeros((N+1,1)); alpha[0]=1
    for i in range(0,N):
        if ru is None:
            u=random.uniform(0,1)
        else:
            u=ru[i]
        k=int(alpha[i])
        if (u>P[k-1][k-1]):
            alpha[i+1]=3-alpha[i]
        else:
            alpha[i+1]=alpha[i]
    return alpha
#---------------------------------
# Monte Carlo Simulation for once
#---------------------------------
def mc_simulation(f,s,g,s1,alpha,N):
    """Monte Carlo Simulation
       N: time step horizon.
       sigma0_train: observation noise."""
    rng=np.random.default_rng()
    u=[rng.multivariate_normal(np.zeros(dimX),np.eye(dimX),1).reshape(dimX,1) for i in range(N)]
    v=[rng.multivariate_normal(np.zeros(dimY),np.eye(dimY),1).reshape(dimY,1) for i in range(N)]
    x_raw=np.zeros((N+1,dimX,1)); x_raw[0]=x0                            
    y_raw=np.zeros((N+1,dimY,1)); y_raw[0]=x0
    for k in range(N):
        x_raw[k+1]=x_raw[k]+eta*f(x_raw[k],alpha[k])+np.sqrt(eta)*s(x_raw[k],alpha[k])*u[k]
        y_raw[k+1]=y_raw[k]+eta*g(x_raw[k],alpha[k])+np.sqrt(eta)*s1(x_raw[k],alpha[k])*v[k]
    return x_raw, y_raw

def sample_generator():
    """sigma0_train: observation noise."""
    datas=np.zeros(((N-n0+2)*N_sample,n0,dimY)) #for each sample path, we have N-n0+2 
    labels=np.zeros(((N-n0+2)*N_sample,dimX))
    x_raws=np.zeros((N_sample, N+1, dimX, 1))
    y_raws=np.zeros((N_sample, N+1, dimY, 1))
    for i in range(N_sample):
        data=np.zeros((N-n0+2,n0,dimY)) #store data for each sample
        label=np.zeros((N-n0+2,dimX))
        # call mc_simulation function to generate sample
        alpha=mChain(N,P)
        x_raw,y_raw=mc_simulation(f,s,g,s1,alpha,N)
        x_raws[i]=x_raw; y_raws[i]=y_raw
        # convert x_raw...into row vector
        x_raw=x_raw.reshape(N+1,dimX) 
        y_raw=y_raw.reshape(N+1,dimY)
        # make data and label for each sample
        for k in range(N-n0+2):
            data[k]=y_raw[k:k+n0]
            label[k]=x_raw[k+n0-1]   
        datas[i*(N-n0+2):(i+1)*(N-n0+2)]=data
        labels[i*(N-n0+2):(i+1)*(N-n0+2)]=label
    datas=datas.reshape(((N-n0+2)*N_sample,dimY*n0))
    return datas,labels,x_raws,y_raws

def generate_data_wrapper():
    datas,labels,x_raws,y_raws=sample_generator()
    train_data, test_data, train_label, test_label=train_test_split(datas, labels, test_size=0.2, random_state=12)

    data_mean=train_data.mean()
    data_std=train_data.std()
    train_data=(train_data-data_mean)/data_std
    test_data=(test_data-data_mean)/data_std
    label_mean=train_label.mean()
    label_std=train_label.std()
    train_label=(train_label-label_mean)/label_std

    train_data, valid_data, train_label, valid_label=train_test_split(train_data, train_label, test_size=0.1, random_state=21)

    tr_d=[np.reshape(train_data[i], (n0*dimY,1)) for i in range(len(train_data))]
    tr_l=[np.reshape(train_label[j], (dimX, 1)) for j in range(len(train_label))]

    val_d=[np.reshape(valid_data[k], (n0*dimY, 1)) for k in range(len(valid_data))]
    val_l=[np.reshape(valid_label[l], (dimX, 1)) for l in range(len(valid_label))]
    te_d=[np.reshape(test_data[s], (n0*dimY, 1)) for s in range(len(test_data))]
    te_l=[np.reshape(test_label[t], (dimX, 1)) for t in range(len(test_label))]

    training_data=list(zip(tr_d, tr_l))
    validation_data=list(zip(val_d, val_l))
    test_data=list(zip(te_d, te_l))
    
    return (training_data, validation_data, test_data, data_mean, data_std, label_mean, label_std)

def generate_new_data(data_mean, data_std, alpha):
    x_new, y_new=mc_simulation(f,s,g,s1,alpha,N)
    x_new=x_new.reshape(N+1, dimX)
    y_new=y_new.reshape(N+1, dimY)
    data_new=np.zeros((N-n0+2, n0 ,dimY))
    for k in range(N-n0+2):
        data_new[k]=y_new[k:k+n0]
    data_new=data_new.reshape(N-n0+2,n0*dimY)
    data_new=(data_new-data_mean)/data_std
    datas=[np.reshape(data_new[i], (n0*dimY,1)) for i in range(len(data_new))]
    labels=[np.reshape(x_new[j], (dimX, 1)) for j in range(len(x_new))]
    return datas, labels, x_new

if __name__=='__main__':
    # Generate data
    training_data, validation_data, test_data, data_mean, data_std, label_mean, label_std=generate_data_wrapper()
    # Store all of data with Pickle
    import gzip
    import pickle
    obj1=(training_data, validation_data, test_data, data_mean, data_std, label_mean, label_std)
    f1=gzip.open('data/ex1Train.pklz','wb')
    pickle.dump(obj1, f1)
    f1.close()

