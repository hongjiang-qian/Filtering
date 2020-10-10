"""
ekf_mc.py
Input:F0,h,u,v,x0,sigma0_train,N
Output: x_raw,y_raw
"""
#-------------------------------------
# Extended KF Monte Carlo
#-------------------------------------
def ekf_mc(F0,h,u,v,x0,sigma0_train,N):
    x_raw=np.zeros((N+1,dimX,1)); x_raw[0]=x0
    y_raw=np.zeros((N+1,dimY,1))
    y_raw[0]=h(x_raw[0])+sigma0_train*v[0]

    for k in range(N):
        x_raw[k+1]=F0@x_raw[k]+u[k]  #!!! here is u[k]
        y_raw[k+1]=h(x_raw[k+1])+sigma0_train*v[k+1]  #!!! add sigma0_train
        
    return x_raw,y_raw