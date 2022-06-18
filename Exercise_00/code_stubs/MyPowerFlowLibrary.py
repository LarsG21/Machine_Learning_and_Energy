import numpy as np



def compute_currents(volt, res):
    v = np.array([volt])
    r = np.array([[res[0]+res[3],-res[3],-res[0]],
              [res[3]       ,-res[3] -res[1],0],
              [-res[0]      ,0,res[0]+res[2]]]
                                            )

    rt = np.linalg.inv(r)
    i = rt @ v.T
    return i 



