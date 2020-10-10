"""
robust_analysis.py
robust_analysis_NM function
robust_analysis_AM function
"""

#----------------------------
# Robustness Analysis for NM 
#----------------------------
import numpy as np

def robust_analysis_NM():
    sigma0_AM=0.5;  sigma0_NM=np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5])
    
    # For kalman Filter, we use NM model with recursive formula and Riccati equation.
    # But when applying KF procedure, use observation from AM that is y_n is from AM model.
    
    R0_NM=np.zeros((len(sigma0_NM),dimY,dimY))
    
    for i in range(len(sigma0_NM)):
        
        R0_NM[i]=sigma0_NM[i]*sigma0_NM[i]*R0
        # first train a DF model with NM noise for use
        datas, labels, x_hats, x_bars, x_raws, y_raws=sample_generator(Q0,R0_NM[i],sigma0_NM[i]) #!!!
        model, data_mean, data_std, label_mean, label_std=deep_filtering(datas,labels,x_hats,x_bars,x_raws,y_raws)
        
        # second generating samples with AM model noise
        x_raw_AM, y_raw_AM=ekf_mc(F0,h,u,v,x0,sigma0_AM,N)
        
        # Riccati equation with NM, observation with AM
        x_hat, x_bar=extended_kf(f,g,h,F,G,H,Q0,R0_NM[i],x0,y_raw_AM,N)
        
        x_raw_AM=x_raw_AM.reshape(N+1,dimX)
        y_raw_AM=y_raw_AM.reshape(N+1,dimY)
        data_new_AM=np.zeros((N-n0+2,n0,dimY))
        
        for k in range(N-n0+2):
            data_new_AM[k]=y_raw_AM[k:k+n0]
            
        data_new_AM=data_new_AM.reshape(N-n0+2,n0*dimY)
        data_new_AM=pd.DataFrame(data_new_AM)
        # input normalization
        data_new_AM=(data_new_AM-data_mean)/data_std # data_mean and data_std come from NM noise.
        df_pred=model.predict(data_new_AM)
        from sklearn.metrics import mean_squared_error
        # Deep Filtering Error
        df_mse_err=mean_squared_error((x_raw_AM[n0-1:]-label_mean)/label_std,df_pred)
        x_bar=x_bar.reshape(N+1,dimX)
        kf_mse_err=mean_squared_error(x_raw_AM[n0-1:],x_bar[n0-1:])
        print("For fixed sigma0_AM and sigma0_NM {}, the mse errs of df and kf are:{:.2%},{:.2%}".format(sigma0_NM[i],df_mse_err, kf_mse_err))

#----------------------------
# Robustness Analysis for AM
#----------------------------

def robust_analysis_AM():
    sigma0_AM=np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5]); sigma0_NM=0.5  
    
    # For kalman Filter, we use NM model with recursive formula and Riccati equation.
    # But when applying KF procedure, use observation from AM that is y_n is from AM model.
    
    R0_NM=sigma0_NM*sigma0_NM*R0
    R0_AM=np.zeros((len(sigma0_NM),dimY,dimY))
    
    for i in range(len(sigma0_AM)):
        
        R0_AM[i]=sigma0_AM[i]*sigma0_AM[i]*R0
        # first train a DF model with NM noise for use
        datas, labels, x_hats, x_bars, x_raws, y_raws=sample_generator(Q0,R0_NM,sigma0_NM) #!!!
        model, data_mean, data_std, label_mean, label_std=deep_filtering(datas,labels,x_hats,x_bars,x_raws,y_raws)
        
        # second generating samples with AM model noise
        x_raw_AM, y_raw_AM=ekf_mc(F0,h,u,v,x0,sigma0_AM[i],N)
        
        # Riccati equation with NM, observation with AM
        x_hat, x_bar=extended_kf(f,g,h,F,G,H,Q0,R0_NM,x0,y_raw_AM,N)
        
        x_raw_AM=x_raw_AM.reshape(N+1,dimX)
        y_raw_AM=y_raw_AM.reshape(N+1,dimY)
        data_new_AM=np.zeros((N-n0+2,n0,dimY))
        
        for k in range(N-n0+2):
            data_new_AM[k]=y_raw_AM[k:k+n0]
            
        data_new_AM=data_new_AM.reshape(N-n0+2,n0*dimY)
        data_new_AM=pd.DataFrame(data_new_AM)
        # input normalization
        data_new_AM=(data_new_AM-data_mean)/data_std # data_mean and data_std come from NM noise.
        df_pred=model.predict(data_new_AM)
        
        from sklearn.metrics import mean_squared_error
        # Deep Filtering Error
        df_mse_err=mean_squared_error((x_raw_AM[n0-1:]-label_mean)/label_std,df_pred)
        x_bar=x_bar.reshape(N+1,dimX)
        kf_mse_err=mean_squared_error(x_raw_AM[n0-1:],x_bar[n0-1:])
        print("For fixed sigma0_NM and sigma0_AM {}, the mse errs of df and kf are:{:.2%},{:.2%}".format(sigma0_AM[i],df_mse_err, kf_mse_err))  