"""
kf_mc.py
Function name is mc_simulation
Input:F,G,H,u,v,sigma0_train,N
Output:x_raw, y_raw
"""

#--------------------------------------
# Monte Carlo Simulation for once
#--------------------------------------
def mc_simulation(F,G,H,u,v,sigma0_train,N):
    """Monte Carlo Simulation
       N: time step horizon.
       sigma0_train: observation noise."""
    x_raw=np.zeros((N+1,dimX,1)); x_raw[0]=x0                                     
    y_raw=np.zeros((N+1,dimY,1)); y_raw[0]=H@x0+sigma0_train*v[0] #!!!
    for k in range(N):
        x_raw[k+1]=F@x_raw[k]+G@u[k]  #!!!
        y_raw[k+1]=H@x_raw[k+1]+sigma0_train*v[k+1] #!!!
    return x_raw, y_raw