"""
sample_generator.py
Input:Q0,R0,sigma0_train
Output:datas,labels,x_hats,x_bars,x_raws,y_raws
"""
#-------------------------------------------------------------------
# Generating tons of samples
#-------------------------------------------------------------------

def sample_generator(Q0,R0,sigma0_train):
    
    datas=np.zeros(((N-n0+2)*N_sample,n0,dimY)) #for each sample path, we have N-n0+2 data
    labels=np.zeros(((N-n0+2)*N_sample,dimX))
    
    x_bars=np.zeros(((N-n0+2)*N_sample,dimX)) #store Kalman filtering estimation value.
    x_hats=np.zeros(((N-n0+2)*N_sample,dimX))
    
    x_raws=np.zeros((N_sample, N+1, dimX, 1))
    y_raws=np.zeros((N_sample, N+1, dimY, 1))
    
    for i in range(N_sample):
        data=np.zeros((N-n0+2,n0,dimY)) #store data for each sample
        label=np.zeros((N-n0+2,dimX))
        # call ekf_mc function to generate sample
        x_raw,y_raw=ekf_mc(F0,h,u,v,x0,sigma0_train,N)
        x_raws[i]=x_raw; y_raws[i]=y_raw
        
        # call extended_kf function to compute estimation
        # make sure here y_raw to be column vector
        x_hat, x_bar=extended_kf(f,g,h,F,G,H,Q0,R0,x0,y_raw,N)
        
        # convert x_raw... into row vector
        x_raw=x_raw.reshape(N+1,dimX) 
        y_raw=y_raw.reshape(N+1,dimY)
        x_hat=x_hat.reshape(N+1,dimX)
        x_bar=x_bar.reshape(N+1,dimX)
        
        # make data and label for each sample
        for k in range(N-n0+2):
            data[k]=y_raw[k:k+n0]
            label[k]=x_raw[k+n0-1]
            
        # put data and label into datas and labels with i representing sample number
        datas[i*(N-n0+2):(i+1)*(N-n0+2)]=data
        labels[i*(N-n0+2):(i+1)*(N-n0+2)]=label
        x_hats[i*(N-n0+2):(i+1)*(N-n0+2)]=x_hat[n0-1:]
        x_bars[i*(N-n0+2):(i+1)*(N-n0+2)]=x_bar[n0-1:]
    
    return datas,labels,x_hats,x_bars,x_raws,y_raws