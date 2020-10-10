"""
ekf.py
Function name is Extended_kf
Input: f,g,h,F,G,H,Q0,R0,x0,y_raw,N
Output: x_hat,x_bar
"""
#-------------------------------------
# Extended Kalman Filter Algorithm
#-------------------------------------

def extended_kf(f,g,h,F,G,H,Q0,R0,x0,y_raw,N):
    """
    f,g,h,F,G,H are all functions.
    Q0: covariance matrix of u_n
    R0: covariance matrix of v_n """
    
    x_hat=np.zeros((N+1,dimX,1)); x_hat[0]=x0
    R=np.zeros((N+1,dimX,dimX)); R[0]=np.eye(dimX) #!!!!!!
    x_bar=np.zeros((N+1,dimX,1)) 
    x_bar[0]=x_hat[0]+R[0]@H(x_hat[0]).T@np.linalg.inv(H(x_hat[0])@R[0]@H(x_hat[0]).T+R0)@(y_raw[0]-h(x_hat[0]))
    
    for k in range(N):   
        x_hat[k+1]=f(x_bar[k])
        inv_pre=np.linalg.inv(H(x_hat[k])@R[k]@H(x_hat[k]).T+R0)
        R[k+1]=F(x_bar[k])@(R[k]-R[k]@H(x_hat[k]).T@inv_pre@H(x_hat[k])@R[k])@F(x_bar[k]).T+G(x_bar[k])@Q0@G(x_bar[k]).T
        inv_pos=np.linalg.inv(H(x_hat[k+1])@R[k+1]@H(x_hat[k+1]).T+R0)
        x_bar[k+1]=x_hat[k+1]+R[k+1]@H(x_hat[k+1]).T@inv_pos@(y_raw[k+1]-h(x_hat[k+1]))
        
    return x_hat,x_bar