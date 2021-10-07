# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:46:12 2021

@author: Hongjiang Qian
"""
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

@jit(nopython=True)
def state_sim(N, x0, y0, dt, sig, sig1, w1, w2):
    f=lambda x: np.sin(5*x)
    h=lambda x: H*x
    x = np.zeros(N+1); x[0] = x0
    y = np.zeros(N+1); y[0] = y0
    for i in range(0, N):
        x[i+1] = x[i]+f(x[i])*dt+sig*np.sqrt(dt)*w1[i]
        y[i+1] = y[i]+h(x[i])*dt+sig1*np.sqrt(dt)*w2[i]
    return x, y

@jit(nopython=True)
def xhat(x, y, N, dt, xh0, R):
    """
    Calcualte the estiamte state xhat with parameter R.
    """
    f=lambda x: np.sin(5*x)
    h=lambda x: H*x
    xh = np.zeros(N+1); xh[0] = xh0
    dy = np.diff(y)
    for i in range(0, N):
        xh[i+1] = xh[i]+f(xh[i])*dt+R*(dy[i]-h(xh[i])*dt)
    return xh

#+===== extended kalman bucy =======
@jit(nopython=True)
def ExtendedKalBucy(x,y,N,dt,sig,sig1,xh0,R0):
    
    A=lambda x: 5*np.cos(5*x)
    f=lambda x: np.sin(5*x)
    h=lambda x: H*x
    
    a=sig*sig; a1=sig1*sig1
    a1_inv=a1**(-1)
    R=np.zeros((N+1)); R[0]=R0
    F=np.zeros((N+1)); F[0]=R0*H*a1_inv
    xh=np.zeros((N+1)); xh[0]=xh0
    dy=np.diff(y)
    
    for i in range(N):
        xh[i+1]=xh[i]+f(xh[i])*dt+F[i]*(dy[i]-h(xh[i])*dt)
        R[i+1]=R[i]+(A(xh[i])*R[i]+R[i]*A(xh[i])-R[i]*H*a1_inv*H*R[i]+a)*dt
        F[i+1]=R[i+1]*H*(a1_inv)
        
    return xh,R,F

#======= SA method =========
@jit(nopython=True)
def SA_update(M, dlt, epi, R0, x, y, maxitr=10000,verbose=False):
    Tc0 = J(M, N, x, y, dt, xh0, R0)
    Rold = R0
    R = []; R.append(R0)
    c = []; c.append(Tc0)
    for n in range(0, maxitr):
        # compute the gradient
        grad = (J(M,N,x,y,dt,xh0,Rold+dlt)-J(M,N,x,y,dt,xh0,Rold-dlt))/(2*dlt)
        Rnew = Rold-epi*grad
        R.append(Rnew)
        # compute the new cost
        Tc = J(M, N, x, y, dt, xh0, Rnew)
        c.append(Tc)
        if verbose:
            print(n)
            print(Tc)
        Rold = Rnew
    return Rnew, R, c

# ====== Deep Filter ========
def format_data(x, y, n0):
    """
    x and y are one sample path."""
    data = np.zeros((N-n0+2, n0))
    label = np.zeros((N-n0+2, 1))
    for k in range(N-n0+2):
        data[k] = y[k:k+n0]
        label[k] = x[k+n0-1]
    return data, label

def tr_sample(M, N, n0, x, y):
    """
    x and y are all Monte Carlo samples."""
    # 1. generate Monte Carlo samples
    datas = np.zeros(((N-n0+2)*M, n0))
    labels = np.zeros(((N-n0+2)*M, 1))
    for i in range(M):
        data, label = format_data(x[i], y[i], n0)
        datas[i*(N-n0+2):(i+1)*(N-n0+2)] = data
        labels[i*(N-n0+2):(i+1)*(N-n0+2)] = label
    # 2. prepare training_data and test_data
    train_data, test_data, train_label, test_label= train_test_split(datas,labels, test_size=0.1, random_state=1, shuffle=True)
    # normalize train_data
    tr_d_mean = train_data.mean()
    tr_d_std = train_data.std()
    train_data = (train_data-tr_d_mean)/tr_d_std
    test_data = (test_data-tr_d_mean)/tr_d_std

    return train_data, test_data, train_label, test_label,tr_d_mean, tr_d_std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='sigmoid', input_shape=(n0,)))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(16, activation='sigmoid'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizers.SGD(learning_rate=0.01),
                  loss="mean_squared_error")
    return model
  
# ===== cost function ========
@jit(nopython=True)
def cost(N, x, xh):
    """
    compute the cost for one sample.
    x: 1x(N+1) vector
    """
    res = np.mean((x[0:N]-xh[0:N])**2)
    return res

@jit(nopython=True)
def J(M, N, x, y, dt, xh0, R):
    """
    The average cost for all Monte Carlo samples
    x: Mx(N+1); M represents the number of Monte Carlo samples
    """
    res = 0
    for i in range(0, M):
        xh = xhat(x[i], y[i], N, dt, xh0, R)
        res += cost(N, x[i], xh)
    res = res/M
    return res

@jit(nopython=True)
def rel_err(N,n0,x,xh):
    """Compute relative error for test purpose"""
    num=np.sum(np.abs(x[:,n0-1:N+1]-xh[:,n0-1:N+1]))
    den=np.sum(np.abs(x[:,n0-1:N+1])+np.abs(xh[:,n0-1:N+1]))
    return num/den

#==== cost on a separate Monte Carlo samples =====
def cost_test(M, N, dt, xh0, R, R0, sig, sig1, sig2, KB_flag=True, DF_flag=True):
    """
    sig1 is used to genetate samples, sig2 is used in Kalman filter."""
    x = np.zeros((M, N+1))
    y = np.zeros((M, N+1))

    xh_sa=np.zeros((M,N+1))
    if DF_flag:
        xh_df = np.zeros((M,N+1))
    if KB_flag:
        xh_kb = np.zeros((M, N+1))

    for i in range(0, M):
        wx = np.random.normal(0, 1, N)
        wy = np.random.normal(0, 1, N)
        x[i], y[i] = state_sim(N, x0, y0, dt, sig, sig1, wx, wy)
        # SA Filter
        xh_sa[i]=xhat(x[i],y[i],N,dt,xh0,R)
        # Deep Filter
        if DF_flag:
            new_data, new_label = format_data(x[i], y[i], n0)
            new_data=(new_data-tr_d_mean)/tr_d_std
            xh_df_tmp=model.predict(new_data).reshape(N-n0+2) #output-dim:N-n0+2x1
            xh_df0_tmp=np.array([x0 for i in range(n0-1)])
            xh_df[i]=np.hstack((xh_df0_tmp,xh_df_tmp))
        # Kalman Bucy Filter
        if KB_flag:
            xh_kb[i], Rkb, Fkb= ExtendedKalBucy(x[i], y[i], N, dt, sig, sig2, xh0, R0)
    
    if KB_flag:
        if DF_flag:
            return (rel_err(N, n0, x, xh_sa), rel_err(N, n0, x, xh_kb),
                    rel_err(N, n0, x, xh_df))
        else:
            return (rel_err(N, n0, x, xh_sa), rel_err(N, n0, x, xh_kb))
    else:
        if DF_flag:
            return (rel_err(N, n0, x, xh_sa), rel_err(N, n0, x, xh_df))
        else:
            return (rel_err(N, n0, x, xh_sa))

#====== Plotting functions =======
def itr_costPlot(R,c,sig1,saveFig=True):
    itr=[i for i in range(0,maxitr+1)]
    # Iteration-cost plot
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    ax1.plot(itr,c,'C0',label=r"cost-$\sigma_1$={}".format(sig1))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cost")
    plt.legend()
    
    ax2=ax1.twinx()
    ax2.plot(itr,R,'C1',label=r"R value-$\sigma_1$={}".format(sig1))
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("R value")
    plt.legend()

    fig2=plt.figure()
    plt.plot(R,c,label=r"$\sigma_1$={}".format(sig1))
    plt.xlabel("R value")
    plt.ylabel("Cost J")
    plt.legend()
    if saveFig:
        fig1.savefig("Nonlind1-itr-c-r"+str(sig1)+".pdf",dpi=600,bbox_inches='tight')
        fig2.savefig("Nonlind1-r-c"+str(sig1)+".pdf",dpi=600,bbox_inches='tight')
             
def path_plot(x,xh_sa,xh_kb,xh_df,sig1,KB_flag,DF_flag): 
    
    xgrid=[i*dt for i in range(0,N+1)]
    fig3=plt.figure()
    if KB_flag:
        if DF_flag:
            plt.plot(xgrid,x,xgrid,xh_sa,xgrid,xh_kb,xgrid,xh_df)
            plt.legend(["True State","SA Filter","EKBF","Deep Filter"])
        else:
            plt.plot(xgrid,x,xgrid,xh_sa,xgrid,xh_kb)
            plt.legend(["True State","SA Filter","EKBF"])
    else:
        if DF_flag:
            plt.plot(xgrid,x,xgrid,xh_sa,xgrid,xh_df)
            plt.legend(["True State","SA Filter","Deep Filter"])
        else:
            plt.plot(xgrid,x,xgrid,xh_sa)
            plt.legend(["True State","SA Filter"])
    fig3.savefig("Nonlind1Path-"+str(sig1)+".pdf",dpi=600,bbox_inches='tight')

def RFPlot(R,Fkb,sig1):
    # Fkb-R(maxitr) graph
    # R(length=maxitr), Fkb(length=N+1)
    fig4=plt.figure()
    time=[i for i in range(N+1)]
    Len=int(maxitr/(N+1))-1
    Rclip=[R[i*Len] for i in range(N+1)]
    plt.plot(time,Rclip,time, Fkb)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend(["R value in SA","F value in EKBF"])
    fig4.savefig("Nonlind1-R-Fkb"+str(sig1)+".pdf",dpi=600, bbox_inches='tight')
    
    
if __name__=="__main__":
    T = 1; N = 500 ; dt = T/N
    n0 = 50  # window_sizes
    H = 1
    sig = 1.0; sig1=0.5
    if(sig1!=0):
        KB_flag=True 
    else:
        KB_flag=False
    DF_flag=True
    x0 = 0; y0 = 0; xh0 = 0; R0 = 1
    M = 1000; maxitr =5000
    dlt = 0.5; epi = 0.1

    # For SA and DF, generate Monte Carlo samples
    x = np.zeros((M, N+1))
    y = np.zeros((M, N+1))
    for i in range(0, M):
        wx = np.random.normal(0, 1, N)
        wy = np.random.normal(0, 1, N)
        x[i], y[i] = state_sim(N, x0, y0, dt, sig, sig1, wx, wy)
    
    # Deep Filtering
    if DF_flag:
        train_data, test_data, train_label, test_label,tr_d_mean, tr_d_std= tr_sample(M, N, n0, x, y)
        model = build_model(); epoch=50
        mymodel = model.fit(train_data, train_label, validation_split=0.2,epochs=epoch, batch_size=64,verbose=0)

        hist_dict = mymodel.history
        loss_val = hist_dict['loss']
        valid_loss = hist_dict['val_loss']
        epochs = range(1, epoch+1)

        fig5 = plt.figure()
        plt.plot(epochs, loss_val, epochs, valid_loss)
        plt.legend(["Train Loss", "Valid Loss"])
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        # plt.savefig("trloss"+str(sig1)+".pdf",dpi=600,bbox_inches='tight')
    
    # SA update
    Ropt,R,c = SA_update(M, dlt, epi, R0, x, y, maxitr,verbose=False)
    print("The optimal R is {}".format(Ropt))
    itr_costPlot(R,c,sig1,saveFig=True)
    
    # Remember in Kalman Bucy Filter, use the parameter sig2, here sig2=sig1.
    cost_tuple=cost_test(M, N, dt, xh0, Ropt, R0, sig, sig1, sig1, KB_flag, DF_flag)
    print(cost_tuple)

    # sample path plotting
    w1=np.random.normal(0,1,N)
    w2=np.random.normal(0,1,N)
    x_plot,y_plot=state_sim(N,x0,y0,dt,sig,sig1,w1,w2)

    xh_kb=np.zeros(N+1)
    xh_df=np.zeros(N+1)
    xh_sa=xhat(x_plot,y_plot,N,dt,xh0,Ropt)
    
    if KB_flag:
        xh_kb, Rkb, Fkb=ExtendedKalBucy(x_plot,y_plot,N,dt,sig,sig1,xh0,R0)
        RFPlot(R,Fkb,sig1)
    if DF_flag:
        new_data,new_label=format_data(x_plot,y_plot,n0)
        new_data=(new_data-tr_d_mean)/tr_d_std
        xh_df=model.predict(new_data).reshape(N-n0+2)
        xh_df0=np.array([x0 for i in range(n0-1)])
        xh_df=np.hstack((xh_df0,xh_df))       
    path_plot(x_plot,xh_sa,xh_kb,xh_df, sig1, KB_flag,DF_flag)

# #====================================
# #  Robustness: Fixing NM
# #  NM: training; AM: testing
# #====================================

# if __name__=="__main__":
#     T = 1; N = 500 ; dt = T/N
#     n0 = 50  # window_sizes
#     H = 1
#     sig = 1.0 
#     sigNM=0.5
#     sigAMs=[0.0,0.1,0.5,1.0,1.5,2.0]

#     DF_flag=True; KB_flag=True
#     x0 = 0; y0 = 0; xh0 = 0; R0 = 1
#     M = 1000; maxitr =5000
#     dlt = 0.5; epi = 0.1

#     # Monte Carlo samples
#     x = np.zeros((M, N+1))
#     y = np.zeros((M, N+1))
#     for i in range(0, M):
#         wx = np.random.normal(0, 1, N)
#         wy = np.random.normal(0, 1, N)
#         x[i], y[i] = state_sim(N, x0, y0, dt, sig, sigNM, wx, wy)
    
#     # Deep Filtering
#     if DF_flag:
#         train_data, test_data, train_label, test_label,tr_d_mean, tr_d_std = tr_sample(M, N, n0, x, y)
#         model = build_model(); epoch=50
#         mymodel = model.fit(train_data, train_label, validation_split=0.2,epochs=epoch, batch_size=64,verbose=0)

#         hist_dict = mymodel.history
#         loss_val = hist_dict['loss']
#         valid_loss = hist_dict['val_loss']
#         epochs = range(1, epoch+1)

#         fig5 = plt.figure()
#         plt.plot(epochs, loss_val, epochs, valid_loss)
#         plt.legend(["Train Loss", "Valid Loss"])
#         plt.xlabel("Epochs")
#         plt.ylabel("MSE")
#         # plt.savefig("trloss"+str(sig1)+".pdf",dpi=600,bbox_inches='tight')
    
#     # SA update
#     Ropt,R,c = SA_update(M, dlt, epi, R0, x, y, maxitr,verbose=False)
#     # print("The optimal R is {}".format(Ropt))
    
#     for sigAM in sigAMs:
#         cost_tuple=cost_test(M, N, dt, xh0, Ropt, R0, sig, sigAM, sigNM, KB_flag, DF_flag)
#         print("The cost when sigAM {} is {}".format(sigAM, cost_tuple))

#========================================
#  Robustness: Fixing AM (unkown) testing
#========================================

# if __name__=="__main__":
#     T = 1; N = 500 ; dt = T/N
#     n0 = 50  # window_sizes
#     H = 1
#     sig = 1.0
#     sigAM=0.5
#     sigNMs=[0.0,0.1,0.5,1.0,1.5,2.0]
    
#     DF_flag=True
#     x0 = 0; y0 = 0; xh0 = 0; R0 = 1
#     M = 1000; maxitr =5000
#     dlt = 0.5; epi = 0.1
    
#     for sigNM in sigNMs:
#         # Monte Carlo samples with sigNM
#         x = np.zeros((M, N+1))
#         y = np.zeros((M, N+1))
#         for i in range(0, M):
#             wx = np.random.normal(0, 1, N)
#             wy = np.random.normal(0, 1, N)
#             x[i], y[i] = state_sim(N, x0, y0, dt, sig, sigNM, wx, wy)

#         # Deep Filtering
#         if DF_flag:
#             train_data, test_data, train_label, test_label,tr_d_mean, tr_d_std \
#                 = tr_sample(M, N, n0, x, y)
#             model = build_model(); epoch=50
#             mymodel = model.fit(train_data, train_label, validation_split=0.2,
#                                 epochs=epoch, batch_size=64,verbose=0)

#         hist_dict = mymodel.history
#         loss_val = hist_dict['loss']
#         valid_loss = hist_dict['val_loss']
#         epochs = range(1, epoch+1)

#         fig5 = plt.figure()
#         plt.plot(epochs, loss_val, epochs, valid_loss)
#         plt.legend(["Train Loss", "Valid Loss"])
#         plt.xlabel("Epochs")
#         plt.ylabel("MSE")
#         #plt.savefig("trloss"+str(sig1)+".pdf",dpi=600,bbox_inches='tight')
#         plt.show()
    
#         # SA update
#         RoptNM,R,c = SA_update(M, dlt, epi, R0, x, y, maxitr,verbose=False)
#         print("The optimal R is {}".format(RoptNM))

#         if (sigNM==0):
#             KB_flag=False
#         else:
#             KB_flag=True
#         cost_tuple=cost_test(M, N, dt, xh0, RoptNM, R0, sig, sigAM, sigNM, KB_flag, DF_flag)
#         print(cost_tuple)