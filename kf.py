"""
kf.py
Function name is kalman_filtering
Input:F,G,H,Q0,R0,x0,y_raw,N
Output: x_hat, x_bar
"""
#---------------------------------------
# Kalman Filtering Algorithm
#---------------------------------------
def kalman_filtering(F,G,H,Q0,R0,x0,y_raw,N):
    """Kalman Filtering Algorithm"""
    #caution: need to specific x is dimX x 1 to be column vector
    x_hat=np.zeros((N+1,dimX,1)); x_hat[0]=x0
    R=np.zeros((N+1,dimX,dimX)); R[0]=np.zeros((dimX,dimX))  #!!!
    
    for k in range(N):
        #y_raw has to be column array or vector.
        inv=np.linalg.inv(H@R[k]@H.T+R0)
        x_hat[k+1]=F@x_hat[k]+F@R[k]@H.T@inv@(y_raw[k]-H@x_hat[k]) #!!!
        R[k+1]=F@(R[k]-R[k]@H.T@inv@H@R[k])@F.T+G@Q0@G.T           #!!!
        
    x_bar=[x_hat[k]+R[k]@H.T@np.linalg.inv(H@R[k]@H.T+R0)@(y_raw[k]-H@x_hat[k]) for k in range(N+1)]
    x_bar=np.array(x_bar) #make list to np.array
    
    return x_hat, x_bar

