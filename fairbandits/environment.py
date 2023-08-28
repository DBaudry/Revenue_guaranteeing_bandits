import numpy as np

def mab_environment(p, mus, rng:np.random.RandomState):
    """
    Sample an arm, pull the arm and return the corresponding reward
    Parameters
    ----------
    p: np array of size K
       allocation 
    mus: np array of size K
       mean rewards 
    rng: RandomState
        Random state

    Return
    ---------
    k: int
        chosen arm
    r: float
        reward
    """
    K = len(mus) # number of arms
    k = rng.choice(np.arange(K), p=p) # sample an arm according to p
    r = rng.choice([0, 1], p=[1-mus[k], mus[k]]) # play this arm and give a reward
    return k, r
    


