# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 19:18:32 2021

@author: Hongjiang Qian
"""
import numpy as np
from math import sqrt
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def state_sim(A,H,x0,y0,sig,sig1,N,dt,dimx,dimy,wx,wy):
    f=lambda x: A.dot(x)
    h=lambda x: H.dot(x)
    x=np.zeros((N+1,dimx)); x[0]=x0 
    y=np.zeros((N+1,dimy)); y[0]=y0
    for i in range(N):
        x[i+1]=x[i]+f(x[i])*dt+sig.dot(wx[i])*sqrt(dt)
        y[i+1]=y[i]+h(x[i])*dt+sig1.dot(wy[i])*sqrt(dt)
    return x,y

#======= SA algorithm ========
def xhat(x,y,N,dt,xh0,R,dimx):
    """ R_dim=(dimx,dimx)"""
    f=lambda x: A.dot(x)
    h=lambda x: H.dot(x)
    xh=np.zeros((N+1,dimx)); xh[0]=xh0
    dy=np.diff(y,axis=0) #!!! dim(N+1,dimx)->dim(N,dimx)
    for i in range(N):
        xh[i+1]=xh[i]+f(xh[i])*dt+R.dot(dy[i]-h(xh[i])*dt)
    return xh

def SA_update(M,dlt,epi,R0,x,y,maxitr=1000,verbose=True):
    """x, y are monte Carlo samples"""
    Tc0=J_SA(M,N,x,y,dt,xh0,R0)
    R=[]; R.append(R0)
    c=[]; c.append(Tc0)
    Rold=R0
    
    for n in range(maxitr):
        #!!! here the gradient should be a matrix.
        R11_right=Rold+np.array([[dlt,0],[0,0]])
        R11_left=Rold+np.array([[-dlt,0],[0,0]])
        R12_right=Rold+np.array([[0,dlt],[0,0]])
        R12_left=Rold+np.array([[0,-dlt],[0,0]])
        R21_right=Rold+np.array([[0,0],[dlt,0]])
        R21_left=Rold+np.array([[0,0],[-dlt,0]])
        R22_right=Rold+np.array([[0,0],[0,dlt]])
        R22_left=Rold+np.array([[0,0],[0,-dlt]])
        grad11=(J_SA(M,N,x,y,dt,xh0,R11_right)-J_SA(M,N,x,y,dt,xh0,R11_left))/(2*dlt)
        grad12=(J_SA(M,N,x,y,dt,xh0,R12_right)-J_SA(M,N,x,y,dt,xh0,R12_left))/(2*dlt)
        grad21=(J_SA(M,N,x,y,dt,xh0,R21_right)-J_SA(M,N,x,y,dt,xh0,R21_left))/(2*dlt)
        grad22=(J_SA(M,N,x,y,dt,xh0,R22_right)-J_SA(M,N,x,y,dt,xh0,R22_left))/(2*dlt)
        grad=np.array([[grad11,grad12],[grad21,grad22]])
        Rnew=Rold-epi*grad; R.append(Rnew)
        Tc=J_SA(M,N,x,y,dt,xh0,Rnew); c.append(Tc)
        if verbose:
            print("Iteration {} with cost {}".format(n,Tc))
        Rold=Rnew
    return Rnew, R, c

#====== Kalman Bucy =====
def kal_bucy(x,y,N,dt,sig,sig1,xh0,R0,dimx,dimy):
    f=lambda x: A.dot(x)
    h=lambda x: H.dot(x)
    # compute the Riccati equation
    a=sig.dot(sig.T); a1=sig1.dot(sig1.T)
    a1_inv=LA.inv(a1)
    R=np.zeros((N+1,dimx,dimx)); R[0]=R0
    for i in range(N):
        R[i+1]=R[i]+(A.dot(R[i])+R[i].dot(A.T)-R[i].dot(H.T).dot(a1_inv).dot(H).dot(R[i])+a)*dt
        
    F=np.zeros((N+1,dimx,dimy))
    for i in range(N+1):
        F[i]=R[i].dot(H.T).dot(a1_inv)
    # compute the estimate state
    xh=np.zeros((N+1,dimx)); xh[0]=xh0
    dy=np.diff(y,axis=0)
    for i in range(N):
        xh[i+1]=xh[i]+f(xh[i])*dt+F[i].dot(dy[i]-h(xh[i])*dt)
    return xh,R,F


#====== Deep Filtering =====
def format_data(x,y,n0):
    """x and y are one sample path."""
    data=np.zeros((N-n0+2,n0*dimy))
    label=np.zeros((N-n0+2,dimx))
    for k in range(N-n0+2):
        data[k]=y[k:k+n0].reshape(n0*dimy)
        label[k]=x[k+n0-1]
    return data,label
def tr_sample(M,N,n0,x,y):
    """x and y are all Monte Carlo sample paths"""
    datas=np.zeros(((N-n0+2)*M,n0*dimy))
    labels=np.zeros(((N-n0+2)*M,dimx))
    for i in range(M):
        data,label=format_data(x[i], y[i], n0)
        datas[i*(N-n0+2):(i+1)*(N-n0+2)]=data
        labels[i*(N-n0+2):(i+1)*(N-n0+2)]=label
        
    tr_data,te_data,tr_label,te_label=train_test_split(datas,labels, test_size=0.1,random_state=1,shuffle=True)
    tr_d_mean=tr_data.mean()
    tr_d_std=tr_data.std()
    tr_data=(tr_data-tr_d_mean)/tr_d_std
    te_data=(te_data-tr_d_mean)/tr_d_std
    
    return tr_data,te_data,tr_label,te_label,tr_d_mean, tr_d_std

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(128,activation='sigmoid',input_shape=(n0*dimy,)))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(16, activation='sigmoid'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.add(layers.Dense(dimx))
    
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01),
                  loss="mean_squared_error")
    return model

#======== cost functions ===========
# @jit(nopython=True)
def J_SA(M,N,x,y,dt,xh0,R):
    """
    compute the cost function for SA Filter
    x and y are Monte Carlo samples"""
    res=0
    for i in range(M):
        xh=xhat(x[i],y[i],N,dt,xh0,R,dimx)[0:N,:] # do not consider the last element.
        xt=x[i,0:N,:]
        res+=(LA.norm(xt-xh,'fro')**2)/N #!!! Frobenius norm
    res=res/M
    return res

def rel_err(N,n0,x,xh):
    """Compute relative error for test purpose
    x denotes the Monte Carlo samples and xh are the corresponding estimate states.
    """
    num=np.sum(np.abs(x[:,n0-1:N+1,:]-xh[:,n0-1:N+1,:]))
    den=np.sum(np.abs(x[:,n0-1:N+1,:])+np.abs(xh[:,n0-1:N+1,:]))
    return num/den
    
#======= cost function on new data =======
def cost_new(x, xh_sa, xh_kb, xh_df, KB_flag=True,DF_flag=True):

    if KB_flag:
        if DF_flag:
            return (rel_err(N,n0,x,xh_sa),rel_err(N,n0,x,xh_kb),rel_err(N,n0,x,xh_df))
        else:
            return (rel_err(N,n0,x,xh_sa),rel_err(N,n0,x,xh_kb))
    else:
        if DF_flag:
            return (rel_err(N,n0,x,xh_sa), rel_err(N,n0,x,xh_df))
        else:
            return rel_err(N,n0,x,xh_sa)

#============ Plotting==========
def path_plot(x,xh_sa,xh_kb,xh_df,s1,KB_flag,DF_flag):
    """Plot the sample path with s1"""
    xgrid=[i*dt for i in range(N+1)]
    if KB_flag:
        if DF_flag:
            fig1=plt.figure()
            plt.plot(xgrid,x[:,0],xgrid,xh_sa[:,0],xgrid,xh_kb[:,0],xgrid,xh_df[:,0])
            plt.legend(["True State","SA Filter","Kalman Bucy Filter","Deep Filter"])
            fig1.savefig("Lind2Path"+str(s1)+"_0.pdf",dpi=600,bbox_inches='tight')
        
            fig2=plt.figure()
            plt.plot(xgrid,x[:,1],xgrid,xh_sa[:,1],xgrid,xh_kb[:,1],xgrid,xh_df[:,1])
            plt.legend(["True State","SA Filter","Kalman Bucy Filter","Deep Filter"])
            fig2.savefig("Lind2Path"+str(s1)+"_1.pdf",dpi=600,bbox_inches='tight')
        else:
            fig1=plt.figure()
            plt.plot(xgrid,x[:,0],xgrid,xh_sa[:,0],xgrid,xh_kb[:,0])
            plt.legend(["True State","SA Filter","Kalman Bucy Filter"])
            fig1.savefig("Lind2Path"+str(s1)+"_0.pdf",dpi=600,bbox_inches='tight')
        
            fig2=plt.figure()
            plt.plot(xgrid,x[:,1],xgrid,xh_sa[:,1],xgrid,xh_kb[:,1])
            plt.legend(["True State","SA Filter","Kalman Bucy Filter"])
            fig2.savefig("Lind2Path"+str(s1)+"_1.pdf",dpi=600,bbox_inches='tight')
            
    else:
        if DF_flag:
            fig1=plt.figure()
            plt.plot(xgrid,x[:,0],xgrid,xh_sa[:,0],xgrid,xh_df[:,0])
            plt.legend(["True State","SA Filter", "Deep Filter"])
            fig1.savefig("Lind2Path"+str(s1)+"_0.pdf",dpi=600,bbox_inches='tight')
        
            fig2=plt.figure()
            plt.plot(xgrid,x[:,1],xgrid,xh_sa[:,1],xgrid,xh_df[:,1])
            plt.legend(["True State","SA Filter","Deep Filter"])
            fig2.savefig("Lind2Path"+str(s1)+"_1.pdf",dpi=600,bbox_inches='tight')
        else:
            fig1=plt.figure()
            plt.plot(xgrid,x[:,0],xgrid,xh_sa[:,0])
            plt.legend(["True State","SA Filter"])
            fig1.savefig("Lind2Path"+str(s1)+"_0.pdf",dpi=600,bbox_inches='tight')
        
            fig2=plt.figure()
            plt.plot(xgrid,x[:,1],xgrid,xh_sa[:,1])
            plt.legend(["True State","SA Filter"])
            fig2.savefig("Lind2Path"+str(s1)+"_1.pdf",dpi=600,bbox_inches='tight')
    
def R_costPlot(R,c,s1):
    # convert list to array
    R=np.array(R)
    fig3,ax=plt.subplots(nrows=2,ncols=2,constrained_layout=True)
    ax[0][0].plot(R[:,0,0],c,label="$s_1={}$".format(s1))
    ax[0][0].set_xlabel("$R_{11}$")
    ax[0][0].set_ylabel("Cost J")
    ax[0][0].legend()
    
    ax[0][1].plot(R[:,0,1],c,label="$s_1={}$".format(s1))
    ax[0][1].set_xlabel("$R_{12}$")
    ax[0][1].set_ylabel("Cost J")
    ax[0][1].legend()
    
    ax[1][0].plot(R[:,1,0],c,label="$s_1={}$".format(s1))
    ax[1][0].set_xlabel("$R_{21}$")
    ax[1][0].set_ylabel("Cost J")
    ax[1][0].legend()
    
    ax[1][1].plot(R[:,1,1],c,label="$s_1={}$".format(s1))
    ax[1][1].set_xlabel("$R_{22}$")
    ax[1][1].set_ylabel("Cost J")
    ax[1][1].legend()
    
    fig3.savefig("Lind2-r-c"+str(s1)+".pdf",dpi=600, bbox_inches="tight")

def RFPlot(R,Fkb,s1):
    fig4=plt.figure()
    time=[i for i in range(N+1)]
    Len=int(maxitr/(N+1))-1
    Rclip=[R[i*Len] for i in range(N+1)]
    Rclip=np.array(Rclip)
    
    fig4,ax4=plt.subplots(nrows=2,ncols=2,constrained_layout=True)
    ax4[0][0].plot(time,Rclip[:,0,0],time,Fkb[:,0,0])
    ax4[0][0].set_xlabel("Time Step")
    ax4[0][0].set_ylabel("$R_{11}$")
    
    ax4[0][1].plot(time,Rclip[:,0,1],time,Fkb[:,0,1])
    ax4[0][1].set_xlabel("Time Step")
    ax4[0][1].set_ylabel("$R_{12}$")
    
    ax4[1][0].plot(time,Rclip[:,1,0],time,Fkb[:,1,0])
    ax4[1][0].set_xlabel("Time Step")
    ax4[1][0].set_ylabel("$R_{21}$")
    
    ax4[1][1].plot(time,Rclip[:,1,1],time,Fkb[:,1,1])
    ax4[1][1].set_xlabel("Time Step")
    ax4[1][1].set_ylabel("$R_{22}$")

    ax4[0][0].legend(["R value in SA","F value in KB"])
    fig4.savefig("Lind2-R-Fkb"+str(s1)+".pdf",dpi=600, bbox_inches='tight')
    
if __name__=="__main__":
    
    dimx=2; dimy=2; dim_wx=2; dim_wy=2    
    s1=2.0
    if(s1==0):
        KB_flag=False
    else:
        KB_flag=True
    DF_flag=False
    A=np.array([[1,-1],[1,1]])
    H=np.array([[1,0],[0,1]])
    sig=np.array([[1,0],[0,1]])
    sig1=np.array([[s1,0],[0,1]])
    T=1; N=500; dt=T/N
    x0=y0=xh0=np.array([0,0])
    R0=np.ones((dimx,dimx))
    M=500; dlt=0.5; epi=0.1; maxitr=500
    level=10**(-8); n0=50

    # For SA_update and DF, generate random samples for Monte Carlo.
    x=np.zeros((M,N+1,dimx))
    y=np.zeros((M,N+1,dimy))
    mu_wx=np.zeros(dim_wx); cov_wx=np.eye(dim_wx)
    mu_wy=np.zeros(dim_wy); cov_wy=np.eye(dim_wy)
    for i in range(M):
        wx=np.random.multivariate_normal(mu_wx,cov_wx,N)
        wy=np.random.multivariate_normal(mu_wy,cov_wy,N)
        x[i],y[i]=state_sim(A,H,x0,y0,sig,sig1,N,dt,dimx,dimy,wx,wy)
    # DF
    if DF_flag:
        tr_data, te_data, tr_label, te_label,tr_d_mean, tr_d_std= tr_sample(M, N, n0, x, y)
        model = build_model(); epoch=200
        mymodel = model.fit(tr_data, tr_label, validation_split=0.2, epochs=epoch, batch_size=64, verbose=0)
        hist_dict = mymodel.history
        loss_val = hist_dict['loss']
        valid_loss = hist_dict['val_loss']
        epochs = range(1, epoch+1)

        fig5 = plt.figure()
        plt.plot(epochs, loss_val, epochs, valid_loss)
        plt.legend(["Train Loss", "Valid Loss"])
        plt.xlabel("Epochs")
        plt.ylabel("MSE")

    # SA
    Ropt,R,c=SA_update(M,dlt,epi,R0,x,y,maxitr,verbose=True)
    print("Ropt: {}".format(Ropt))
    R_costPlot(R,c,s1)

    # To compute cost on new data
    x_new=np.zeros((M,N+1,dimx))
    y_new=np.zeros((M,N+1,dimy))
    xh_kb_new=np.zeros((M,N+1,dimx))
    xh_df_new=np.zeros((M,N+1,dimx))
    xh_sa_new=np.zeros((M,N+1,dimx))

    for i in range(M):
        wx_new=np.random.multivariate_normal(mu_wx,cov_wx,N)
        wy_new=np.random.multivariate_normal(mu_wy,cov_wy,N)
        x_new[i],y_new[i]=state_sim(A,H,x0,y0,sig,sig1,N,dt,dimx,dimy,wx_new,wy_new)

        xh_sa_new[i]=xhat(x_new[i],y_new[i],N,dt,xh0,Ropt,dimx)

        if KB_flag:
            xh_kb_new[i],R_new,F_new=kal_bucy(x[i],y[i],N,dt,sig,sig1,xh0,R0,dimx,dimy)
        if DF_flag:
            new_data,new_label=format_data(x_new[i],y_new[i],n0)
            new_data=(new_data-tr_d_mean)/tr_d_std #!!! normalized the data
            xh_df_tmp=model.predict(new_data).reshape((N-n0+2,dimx))
            xh_df0_tmp=np.array([x0 for i in range(n0-1)])
            xh_df_new[i]=np.vstack((xh_df0_tmp,xh_df_tmp))
            
    cost_tuple=cost_new(x_new, xh_sa_new, xh_kb_new, xh_df_new, KB_flag, DF_flag)
    print(cost_tuple)

    # Sample path plotting
    w1=np.random.multivariate_normal(mu_wx,cov_wx,N)
    w2=np.random.multivariate_normal(mu_wy,cov_wy,N)
    x_plot,y_plot=state_sim(A,H,x0,y0,sig,sig1,N,dt,dimx,dimy,w1,w2)

    xh_sa_plot=np.zeros((N+1,dimx))
    xh_kb_plot=np.zeros((N+1,dimx))
    xh_df_plot=np.zeros((N+1,dimx))
    
    xh_sa_plot=xhat(x_plot,y_plot,N,dt,xh0,Ropt,dimx)
    if DF_flag:
        new_data,new_label=format_data(x_plot,y_plot,n0)
        new_data=(new_data-tr_d_mean)/tr_d_std
        xh_df_tmp=model.predict(new_data).reshape((N-n0+2,dimx))
        xh_df0_tmp=np.array([x0 for i in range(n0-1)])
        xh_df_plot=np.vstack((xh_df0_tmp,xh_df_tmp))  
    if KB_flag:
        xh_kb_plot,Rkb_plot,Fkb_plot=kal_bucy(x_plot,y_plot,N,dt,sig,sig1,xh0,R0,dimx,dimy)
        RFPlot(R,Fkb_plot,s1)
    path_plot(x_plot, xh_sa_plot, xh_kb_plot, xh_df_plot, s1, KB_flag, DF_flag)

#====================================
# Robustness: Fixing NM (training)
#====================================
# if __name__=="__main__":
    
#     dimx=2; dimy=2; dim_wx=2; dim_wy=2
#     sNM=0.5
#     sAMs=[0.0,0.1,0.5,1.0,1.5,2.0]
    
#     DF_flag=True; KB_flag=True
#     A=np.array([[1,-1],[1,1]])
#     H=np.array([[1,0],[0,1]])
#     sig=np.array([[1,0],[0,1]])
#     sigNM=np.array([[sNM,0],[0,1]])
    
#     T=1; N=500; dt=T/N
#     x0=y0=xh0=np.array([0,0])
#     R0=np.ones((dimx,dimx))
#     M=500; dlt=0.5; epi=0.1; maxitr=500
#     level=10**(-8); n0=50
    
#     # For SA_update and DF, generate random samples for Monte Carlo.
#     x=np.zeros((M,N+1,dimx))
#     y=np.zeros((M,N+1,dimy))
#     mu_wx=np.zeros(dim_wx); cov_wx=np.eye(dim_wx)
#     mu_wy=np.zeros(dim_wy); cov_wy=np.eye(dim_wy)
#     for i in range(M):
#         wx=np.random.multivariate_normal(mu_wx,cov_wx,N)
#         wy=np.random.multivariate_normal(mu_wy,cov_wy,N)
#         x[i],y[i]=state_sim(A,H,x0,y0,sig,sigNM,N,dt,dimx,dimy,wx,wy)

#     # DF
#     if DF_flag:
#         tr_data, te_data, tr_label, te_label,tr_d_mean, tr_d_std= tr_sample(M, N, n0, x, y)
#         model = build_model(); epoch=200
#         mymodel = model.fit(tr_data, tr_label, validation_split=0.2,epochs=epoch,
#                             batch_size=64,verbose=0)
#         hist_dict = mymodel.history
#         loss_val = hist_dict['loss']
#         valid_loss = hist_dict['val_loss']
#         epochs = range(1, epoch+1)

#         fig5 = plt.figure()
#         plt.plot(epochs, loss_val, epochs, valid_loss)
#         plt.legend(["Training Loss", "Validation Loss"])
#         plt.xlabel("Epochs")
#         plt.ylabel("MSE")
#         plt.show()

#     # SA
#     RoptNM,R,c=SA_update(M,dlt,epi,R0,x,y,maxitr,verbose=False)
#     print("RoptNM: {}".format(RoptNM))
#     # R_costPlot(R,c)

#     # To compute cost on new data (using AM model)
#     for sAM in sAMs:
#         sigAM=np.array([[sAM,0],[0,1]])
        
#         x_new=np.zeros((M,N+1,dimx))
#         y_new=np.zeros((M,N+1,dimy))

#         xh_sa_new=np.zeros((M,N+1,dimx))
#         xh_kb_new=np.zeros((M,N+1,dimx))
#         xh_df_new=np.zeros((M,N+1,dimx))

#         for i in range(M):
#             wx_new=np.random.multivariate_normal(mu_wx,cov_wx,N)
#             wy_new=np.random.multivariate_normal(mu_wy,cov_wy,N)
#             x_new[i],y_new[i]=state_sim(A,H,x0,y0,sig,sigAM,N,dt,dimx,dimy,wx_new,wy_new)
#             xh_sa_new[i]=xhat(x_new[i],y_new[i],N,dt,xh0,RoptNM,dimx)
#             if KB_flag:
#                 # !!! Here should use sigNM in NM model since actual model is unknown.
#                 xh_kb_new[i],R_new,F_new=kal_bucy(x_new[i],y_new[i],N,dt,sig,sigNM,xh0,R0,dimx,dimy)
#             if DF_flag:
#                 new_data,new_label=format_data(x_new[i],y_new[i],n0)
#                 new_data=(new_data-tr_d_mean)/tr_d_std  #!!! normalized the data
#                 xh_df_tmp=model.predict(new_data).reshape((N-n0+2,dimx))
#                 xh_df0_tmp=np.array([x0 for i in range(n0-1)])
#                 xh_df_new[i]=np.vstack((xh_df0_tmp,xh_df_tmp))
                
#         cost_tuple=cost_new(x_new,xh_sa_new,xh_kb_new,xh_df_new,KB_flag,DF_flag)
#         print(cost_tuple)
                   
# #====================================
# #  Robustness: Fixing AM (testing)
# #====================================

# if __name__=="__main__":
    
#     dimx=2; dimy=2; dim_wx=2; dim_wy=2
#     sAM=0.5
#     sNMs=[0.0,0.1,0.5,1.0,1.5,2.0]
    
#     DF_flag=True
#     A=np.array([[1,-1],[1,1]])
#     H=np.array([[1,0],[0,1]])
#     sig=np.array([[1,0],[0,1]])
#     sigAM=np.array([[sAM,0],[0,1]])
    
#     T=1; N=500; dt=T/N
#     x0=y0=xh0=np.array([0,0])
#     R0=np.ones((dimx,dimx))
#     M=500; dlt=0.5; epi=0.1; maxitr=500
#     level=10**(-8); n0=50
    
#     # For SA_update and DF, generate random samples for Monte Carlo.
#     for sNM in sNMs:
#         if(sNM==0):
#             KB_flag= False
#         else:
#             KB_flag= True
            
#         sigNM=np.array([[sNM,0],[0,1]])

#         x=np.zeros((M,N+1,dimx))
#         y=np.zeros((M,N+1,dimy))
#         mu_wx=np.zeros(dim_wx); cov_wx=np.eye(dim_wx)
#         mu_wy=np.zeros(dim_wy); cov_wy=np.eye(dim_wy)

#         for i in range(M):
#             wx=np.random.multivariate_normal(mu_wx,cov_wx,N)
#             wy=np.random.multivariate_normal(mu_wy,cov_wy,N)
#             x[i],y[i]=state_sim(A,H,x0,y0,sig,sigNM,N,dt,dimx,dimy,wx,wy)
#         # DF
#         if DF_flag:
#             tr_data, te_data, tr_label, te_label,tr_d_mean, tr_d_std= tr_sample(M, N, n0, x, y)
#             model = build_model(); epoch=200
#             mymodel = model.fit(tr_data, tr_label, validation_split=0.2,epochs=epoch,
#                             batch_size=64,verbose=0)
#             hist_dict = mymodel.history
#             loss_val = hist_dict['loss']
#             valid_loss = hist_dict['val_loss']
#             epochs = range(1, epoch+1)

#             fig5 = plt.figure()
#             plt.plot(epochs, loss_val, epochs, valid_loss)
#             plt.legend(["Train Loss", "Valid Loss"])
#             plt.xlabel("Epochs")
#             plt.ylabel("MSE")
#             plt.show()

#         # SA
#         RoptNM,R,c=SA_update(M,dlt,epi,R0,x,y,maxitr,verbose=False)
#         print("RoptNM: {}".format(RoptNM))
#         # R_costPlot(R,c)

#         # To compute cost on new data (using AM model)
#         x_new=np.zeros((M,N+1,dimx))
#         y_new=np.zeros((M,N+1,dimy))

#         xh_sa_new=np.zeros((M,N+1,dimx))
#         xh_kb_new=np.zeros((M,N+1,dimx))
#         xh_df_new=np.zeros((M,N+1,dimx))

#         for i in range(M):
#             wx_new=np.random.multivariate_normal(mu_wx,cov_wx,N)
#             wy_new=np.random.multivariate_normal(mu_wy,cov_wy,N)
#             x_new[i],y_new[i]=state_sim(A,H,x0,y0,sig,sigAM,N,dt,dimx,dimy,wx_new,wy_new)

#             xh_sa_new[i]=xhat(x_new[i],y_new[i],N,dt,xh0,RoptNM,dimx)

#             if KB_flag:
#                 # !!! Here should use sigNM in NM model since actual model is unknown.
#                 xh_kb_new[i],R_new,F_new=kal_bucy(x[i],y[i],N,dt,sig,sigNM,xh0,R0,dimx,dimy)
#             if DF_flag:
#                 new_data,new_label=format_data(x_new[i],y_new[i],n0)
#                 new_data=(new_data-tr_d_mean)/tr_d_std  #!!! normalized the data
#                 xh_df_tmp=model.predict(new_data).reshape((N-n0+2,dimx))
#                 xh_df0_tmp=np.array([x0 for i in range(n0-1)])
#                 xh_df_new[i]=np.vstack((xh_df0_tmp,xh_df_tmp))
                
#         cost_tuple=cost_new(x_new,xh_sa_new,xh_kb_new,xh_df_new,KB_flag,DF_flag)
#         print(cost_tuple)    
        

    
    
    





