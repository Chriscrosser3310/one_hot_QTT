import numpy as np

def _as_np(x):
    return np.asarray(x, dtype=complex)

P0 = _as_np([[1, 0],
             [0, 0]])
P1 = _as_np([[0, 0],
             [0, 1]])
PauliI = _as_np([[1, 0],
                 [0, 1]]) 
PauliX = _as_np([[0, 1],
                 [1, 0]]) 
PauliY = _as_np([[0, -1j],
                 [1j, 0]]) 
PauliZ = _as_np([[1, 0],
                 [0, -1]]) 

def exp_i_theta_X(theta):
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    return _as_np([
        [c,   s],
        [s,   c]
    ])

def exp_i_theta_Z(theta):
    return _as_np([
        [np.exp(1j* theta), 0.0],
        [0.0,  np.exp(-1j* theta)],
    ])

#exp(i theta (XX + YY)/2)
def exp_i_theta_xx_yy_over_two(theta):
    c = np.cos(theta)
    s = 1j * np.sin(theta)
    return _as_np([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,   c,   s, 0.0],
        [0.0,   s,   c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    
def exp_i_theta_ZZ(theta):
    return _as_np([
        [np.exp(1j* theta), 0.0, 0.0, 0.0],
        [0.0, np.exp(-1j* theta), 0.0, 0.0],
        [0.0, 0.0, np.exp(-1j* theta), 0.0],
        [0.0, 0.0, 0.0, np.exp(1j* theta)],
    ])
