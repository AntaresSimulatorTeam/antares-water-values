import numpy as np
from numpy.linalg import norm

def BezierAv(
    t:float, 
    pv0:np.ndarray,
    pv1:np.ndarray,
    method:int=0, 
    last_axify:bool=True
        ) -> np.ndarray:
    """"""
    n0, n1 = norm(pv0[1]), norm(pv1[1])
    s0, s1 = np.abs(pv0[1,-1]), np.abs(pv1[1,-1])
    p0,v0 = pv0[0],pv0[1]
    p1,v1 = pv1[0],pv1[1]
    diff_rate = s1/(s0+s1)
    partage=6/7
    if (method==0):
        cos_left = min(np.dot(v0,p1-p0)/(norm(v0)*norm(p1-p0)+1e-9),1)   ## avoiding round-up errors
        cos_left = max(cos_left,-1)                                 ## -"""-
        cos_right = min(np.dot(v1,p1-p0)/(norm(v1)*norm(p1-p0)+1e-9),1)  ## -"""-
        cos_right = max(cos_right,-1)                               ## -"""-
        theta = (np.arccos(cos_left)+np.arccos(cos_right))/4
    else:
        cos_theta = min(np.dot(v0,v1)/(norm(v0)*norm(v1)),1)        ## -"""-
        cos_theta = max(cos_theta,-1)                               ## -"""-
        theta = np.arccos(cos_theta)/4
    if (np.cos(theta)>1e-12):
        alpha = norm(p0-p1)/(3*np.cos(theta)**2)
    else:
        alpha = 0
    p = ((1-t)**2*(1+2*t)*p0 + t**2*(3-2*t)*p1) + 3*alpha*t*(1-t)*((1-t)*v0 - t*v1)
    v = (2*t*(t-1)*p0 + 2*t*(1-t)*p1) + alpha*((3*t**2-4*t+1)*v0 - t*(2-3*t)*v1)
    if last_axify:
        p[:-1] = (p0[:-1] + p1[:-1])/2
        p[-1] = p0[-1]*diff_rate + p1[-1]*(1-diff_rate)
        # v_sum = np.sum(v[:-1])
        # v[-1] /= v_sum
        v = v0*(t) + v1*(1-t)
        v[-1] = v0[-1]*(t*(1-partage)+partage*diff_rate) + v1[-1]*((1-t)*(1-partage) + partage*(1-diff_rate))
    elif (norm(v)>1e-12):
        # v = v / sum(v[:-1])
        v = v/norm(v) * (n0+n1)/2
    return (np.array([p,v]))

def AvScheme(f:np.ndarray,num_iterations:int=1,average:int=0,periodic:bool=True) -> np.ndarray:
    """Averages between all points

    Args:
        f (np.ndarray): data
        num_iterations (int, optional): number of iterations. Defaults to 1.
        average (int, optional): . Defaults to 0.
        periodic (bool, optional): Find a circular trajectory. Defaults to True.

    Returns:
        np.ndarray: interpolated data
    """
    if (num_iterations==0):
        return f
    else:
        new = []
        for i in range(len(f)-1):
            new.append(f[i])
            new.append(BezierAv(0.5,f[i],f[i+1],average))
        new.append(f[len(f)-1])
        if (periodic):
            new.append(BezierAv(0.5,f[len(f)-1],f[0],average))
        return AvScheme(np.array(new),num_iterations-1,average,periodic)


# Lane-Risenfeld with m-1 smoothing steps.
# f is the data to refine.
def LR(m:int,f:np.ndarray,iterations:int=8,periodic:bool=True) -> np.ndarray:
    """ Recursively splits the data and finds average between successive pts

    Args:
        m (int): smoothing parameter
        f (np.ndarray): data
        iterations (int, optional): number of in between interps. Defaults to 8.
        periodic (bool, optional): Find a circular trajectory. Defaults to True.

    Returns:
        np.ndarray: interpolated data
    """
    if (iterations==0):
        return f
    else:
        f = AvScheme(f,1,periodic)
        return LR(m,f,iterations-1,periodic)
    
chosen_pts = [0, -1]
po2=3

def interpolate_between(data:np.ndarray, n_splits:int=3)-> tuple[np.ndarray, np.ndarray]:
    """Iteratively adds mid points between every point

    Args:
        data (np.ndarray): Positions and angles of our "arrows"
        n_splits (int, optional): Number of times we double the number of interpolants (log2(nb_interpolants)). Defaults to 3.

    Returns:
        tuple[np.ndarray, np.ndarray]: interpolated data / data
    """
    n = data.shape[0]
    mlr=data
    for _ in range(n_splits):
        mlr = LR(1,mlr,1,periodic=False)
    mlr = mlr[:int((mlr.shape[0])/(n/(n-1)))+1]
    return mlr, data

mlrs, datas = [], []

def enrich_by_interpolation(
        controls_init:np.ndarray, 
        costs:np.ndarray, 
        slopes:np.ndarray, 
        n_splits:int=3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        controls_init (np.ndarray): Controls between which we wish to interpolate
        costs (np.ndarray): Costs associated
        slopes (np.ndarray): Slopes associated
        n_splits (int, optional): Number of times we double the number of interpolants (log2(nb_interpolants)). Defaults to 3.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: positions, costs and slopes
    """
    n_pts = controls_init.shape[0]
    ps = np.append(controls_init, costs[:,None], axis=1)
    vs = np.append(slopes , np.sum(np.power(slopes,2), axis=1)[:, None], axis=1) / np.sum(slopes, axis=1)[:, None]
    data = np.array([[p,v] for p, v in zip(ps, vs)])
    mlrs=[]
    for i in range(n_pts):
        for j in range(i, n_pts):
            mlr, _ = interpolate_between(np.array([data[i], data[j]]), n_splits=n_splits)
            mlrs.append(mlr)
    mlrs = np.concatenate(mlrs)
    positions = mlrs[:,0, :-1]
    costs = mlrs[:,0,-1]
    false_slopes = mlrs[:,1]
    real_slopes = false_slopes[:,:-1]*(false_slopes[:,-1]/(np.sum(np.power(false_slopes[:,:-1],2), axis=1)))[:,None]
    return positions, costs, real_slopes