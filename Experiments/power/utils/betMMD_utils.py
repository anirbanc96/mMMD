# this code is adapted from Sekhar and Ramdas "Nonparametric Two-Sample Testing by Betting"

from time import time 
import numpy as np 
import scipy.stats as stats 
from scipy.spatial.distance import cdist, pdist
from numpy.random import default_rng
from tqdm import tqdm 

from tqdm import tqdm
from functools import partial
from math import sqrt
from time import time 

def truncateRows(X, radius=1):
    """
    Given a 2d array X, find the rows whose 
    norm is larger than radius, and scale those 
    rows to make their norm equal to radius 
    """
    assert radius>0 
    normX = np.linalg.norm(X, axis=1) 
    idx = np.where(normX>radius) 
    X[idx] = (X[idx]*radius)/(normX[idx].reshape((-1,1))) 
    return X 

def getGaussianSourceparams(d=10, epsilon_mean=0.5, epsilon_var=0.0,
                            num_perturbations_mean=2,
                            num_perturbations_var=2): 
    meanX, meanY = np.zeros((d,)), np.zeros((d,))
    meanY[:num_perturbations_mean] = epsilon_mean 
    # get the covariance matrices 
    covX = np.eye(d)
    diagY = np.ones((d,)) 
    diagY[:num_perturbations_var] = epsilon_var 
    covY = np.diag(diagY) 
    return meanX, meanY, covX, covY
    


def GaussianSource(meanX=None, meanY=None, covX=None, covY=None,
                truncated=False, radius=None, epsilon=0.5, rng_X = None, rng_Y = None):
    if meanX is None: # set all params to default
        d=5
        meanX = np.zeros((d,)) 
        meanY = np.ones((d,))*epsilon
        covX = np.eye(d)
        covY = np.eye(d)
    
    if truncated:
        radius = 1 if radius is None else radius 
        assert radius > 0 
    def Source(n, m=None, truncated=truncated, radius=radius, rng_X = rng_X, rng_Y = rng_Y):
        m = n if m is None else m
        X = stats.multivariate_normal.rvs(mean=meanX, cov=covX, size=n, random_state=rng_X)
        Y = stats.multivariate_normal.rvs(mean=meanY, cov=covY, size=m, random_state=rng_Y)
        if truncated:
            X = truncateRows(X, radius=radius)
            Y = truncateRows(Y, radius=radius)
        return X, Y
    return Source 


def TdistSource(df1=1, df2=1, scale1=1.0, scale2=1.0, 
                loc1=0.0, loc2=0.0):

    def Source(n, m=None):
        m = n if m is None else m 
        X = stats.t.rvs(size=n, loc=loc1, df=df1, scale=scale1)
        Y = stats.t.rvs(size=n, loc=loc2, df=df2, scale=scale2)
        return X, Y 
    return Source 

def median_heuristic(Z):
    # compute the pairwise distance between the elements of Z 
    dists_ = pdist(Z)
    # obtain the median of these pairwise distances 
    sig = np.median(dists_)
    return sig


def RBFkernel(x, y=None, bw=1.0):
    y = x if y is None else y 
    dists = cdist(x, y, 'euclidean') 
    sq_dists = dists * dists 
    K = np.exp(-sq_dists/(2*bw*bw))
    return K 

def LinearKernel(x, y=None):
    y = x if y is None else y 
    K = np.einsum('ji, ki ->jk', x, y) 
    return K 

def PolynomialKernel(x, y=None, c=1.0, p=2):
    L = LinearKernel(x, y) 
    K = (c + L)**p 
    return K 

def permuteXY(X, Y, perm=None):
    Z = np.concatenate((X, Y), axis=0)
    nZ, nX = len(Z), len(X)
    if perm is None:
        perm = np.random.permutation(nZ) 
    idxX, idxY = perm[:nX], perm[nX:]
    X_, Y_ = Z[idxX], Z[idxY]
    return X_, Y_

def permutationTwoSampleTest(X, Y, statfunc, params=None, num_perms=200):
    params = {} if params is None else params 
    stat = statfunc(X, Y, **params)

    V = np.zeros((num_perms,))
    nZ = len(X) + len(Y)
    for i in range(num_perms):
        perm = np.random.permutation(nZ)
        X_, Y_ = permuteXY(X, Y, perm=perm) 
        val = statfunc(X_, Y_, **params)
        V[i] = val
    # compute the p-value 
    p = len(V[V>=stat])/num_perms
    return p 


def runBatchTwoSampleTest(Source, statfunc, params=None, num_perms=200,
                        alpha=0.05, Nmax=200, num_steps=20, initial=10, 
                        num_trials=200, progress_bar=False, store_times=False):
    # generate the different sample-sizes to be used in the power curve 
    NN = np.linspace(start=initial, stop=Nmax, num=num_steps, dtype=int) 
    # initialize the array to hold power values 
    Power = np.zeros(NN.shape) 
    # initialize the array to store average running times 
    if store_times:
        Times = np.zeros(NN.shape)

    range_ = range(num_trials) 
    range_ = tqdm(range_) if progress_bar else range_

    for trial in range_: 
        for i, n in enumerate(NN): 
            # do one trial of the test with sample-size equal to n 
            X, Y = Source(n) 
            t0 = time()
            p = permutationTwoSampleTest(X, Y, statfunc, num_perms=num_perms,
                                            params=params) 
            t1 = time() - t0
            # update the power value 
            if p<=alpha:
                Power[i] += 1 
            # update the running time 
            if store_times:
                Times[i] += t1 
    Power /= num_trials 
    if store_times:
        Times /= num_trials 
        return Power, Times, NN
    else:
        return Power, NN 


def get_power_from_stopping_times(StoppingTimes, N):
    num_trials = len(StoppingTimes) 
    S = StoppingTimes[StoppingTimes<N] 
    S = np.sort(S)
    Power = np.zeros((N,))
    for s in S: 
        Power[int(s-1):] += 1
    Power /= num_trials 
    return Power 


def ONSstrategy(F, lambda_max=0.5):
    """
    Compute the bets corresponding to the observations F 

    Parameters:
        F           :numpy array   of size (N,)
        lambda_max  :float positive real number in range (0,1)
    Returns:
        Lambda  :numpy array of size (N,) containing the bets
    
    Note that we use the wealth update rule:
    K_{t} = K_{t-1} \times (1 + \lambda_t F_t), 
    with a "+" instead of "-" used by Cutkosky & Orabona (2018). 
    """
    N = len(F)
    assert N>2
    Lambda = np.zeros((N,))
    A = 1
    c = 2/(2- np.log(3))
    for i in range(N-1):
        #update the z term 
        z = -F[i] / (1 + Lambda[i]*F[i])
        # update the A term 
        A += z**2 
        # get the new betting fraction 
        Lambda[i+1] = max(min(
            Lambda[i] - c * z / A, lambda_max
         ), -lambda_max)
    return Lambda 

def KellyBettingApprox(F, lambda_max=0.8):
    N = len(F) 
    assert N>2
    Lambda = np.zeros((N,))
    F2 = F*F 
    for i in range(1, N):
        lambda_ = F[:i].sum() / (F2[:i].sum() + 1e-10) 
        lambda_ = min(lambda_max, max(-lambda_max, lambda_)) 
        Lambda[i] = lambda_
    return Lambda



def get_stopping_time_from_wealth(W, alpha=0.05):
    th = 1/alpha 
    idx = np.where(W>=th)[0] 
    if len(idx)==0: # no stopping 
        stopped = False 
        stopping_time = len(W) 
    else:
        stopped = True 
        stopping_time = idx[0]+1  
    return stopped, stopping_time 


def computeMMD(X, Y, kernel=None, perm=None, biased=True):
    """
    Compute the quadratic time MMD statistic based on gram-matrix K. 

    X       :ndarray    (nX, ndims) size observations
    Y       :ndarray    (nY, ndims) size observations
    kernel  :callable   kernel function 
    perm    :ndarray    the permutation array 
    biased  :bool       if True, compute the biased MMD statistic 

    returns 
    -------
    mmd     :float      the quadratic-time MMD statistic. 
    """
    Z = np.concatenate((X, Y), axis=0)
    nX, nZ = len(X), len(Z)
    # default kernel is RBF kernel 
    if kernel is None:
        bw = median_heuristic(Z)
        kernel = partial(RBFkernel, bw=bw)
    # # default value of perm is the indentity permutation
    if perm is None: 
        perm = np.arange(nZ)
    # obtain the X and Y indices 
    idxX, idxY = perm[:nX], perm[nX:]
    # permuted rows of observations 
    X_, Y_ = Z[idxX], Z[idxY]
    # extract the required matrices 
    KXX = kernel(X_, X_)
    KYY = kernel(Y_, Y_)
    KXY = kernel(X_, Y_)
    # compute the mmd statistic 
    nY = nZ - nX
    nY2, nX2, nXY = nY*nY, nX*nX, nX*nY
    assert nY>0 
    if biased:
        mmd = sqrt((1/nX2)*KXX.sum() + (1/nY2)*KYY.sum() - (2/nXY)*KXY.sum()) 
    else:#TODO: implement the unbiased mmd statistic 
        raise NotImplementedError
    # return the mmd statistic
    return mmd 


def kernelMMDprediction(X, Y, kernel=None, post_processing=None):
    nX, nY = len(X), len(Y) 
    assert nX==nY # only works with paired observations 
    assert nX>20
    # default kernel is RBF kernel 
    if kernel is None:
        # use the first 20 pairs of observations for bandwidth selection
        # TODO: get rid of this hardcoding and update the bandwidth 
        # after every block of observations 
        bw = median_heuristic(np.concatenate((X, Y), axis=0))
        kernel = partial(RBFkernel, bw=bw)
    KXX = kernel(X, X)
    KYY = kernel(Y, Y)
    KXY = kernel(X, Y) 
    F = np.zeros((nX,)) 
    F_ = np.zeros((nX,))
    for i in range(1, nX):
        termX = np.mean((KXX[i, :i] - KXY[i, :i]))
        termY = np.mean((KXY[:i, i] - KYY[:i, i]))
        F_[i] = (termX - termY)
        F[i] = (termX - termY)
        ### a heuristic that significantly improve the 
        ### practical performance
        if i>10:
            i0 = max(0, i-50)
            max_val = np.max(F_[:i]) 
            F[i] =F_[i] / max_val 
    if post_processing=='sinh':
        F = np.sinh(F) 
    elif post_processing=='tanh':
        F = np.tanh(F)
    elif post_processing=='arctan':
        F = (2/np.pi)*np.arctan(F)  
    elif post_processing=='delapena':
        F = deLaPenaMartingale(F)
    return F 

def runSequentialTest(Source, Prediction, Betting, alpha=0.05,
                            pred_params=None, bet_params=None, 
                            Nmax=1000, num_trials=50, progress_bar=False, 
                            hedge=False, hedge_weights=None, 
                            return_wealth=False,
                            seeds = None):

    Power = np.zeros((Nmax,)) 
    StoppingTimes = np.zeros((num_trials,))
    Stopped = np.zeros((num_trials,))
    pred_params = {} if pred_params is None else pred_params
    bet_params = {} if bet_params is None else bet_params

    range_ = range(num_trials)
    range_ = tqdm(range_) if progress_bar else range_

    for trial in range_:

        rng = default_rng(seeds[trial])

        X, Y = Source(Nmax, rng_X = rng, rng_Y = rng) 
        # get the wealth process 
        if not hedge: # no hedgeing over different prediction strategies
            # get the payoff values 
            F = Prediction(X, Y, **pred_params)
            # get the betting fractions 
            Lambda = Betting(F, **bet_params) 
            W = np.cumprod(1 + Lambda*F) 
        else: #hedge over different prediction strategies 
            # some sanity checking 
            assert isinstance(Prediction, list) 
            assert isinstance(Betting, list) 
            nP = len(Prediction)
            assert nP==len(Betting)  
            if hedge_weights is None:
                # default weights are uniform 
                hedge_weights = np.ones((nP,)) 
            else:
                assert len(hedge_weights)==nP
            hedge_weights /= hedge_weights.sum()
            for j in range(nP):
                Fj = Prediction[j](X, Y, **pred_params)
                Lambdaj = Betting[j](Fj, **bet_params)
                if j==0:
                    W = hedge_weights[j]*np.cumprod(1 + Lambdaj*Fj) 
                else: 
                    W += hedge_weights[j]*np.cumprod(1 + Lambdaj*Fj) 
        # get the stopping_time 
        stopped, stopping_time = get_stopping_time_from_wealth(W, alpha)
        # update the results 
        Stopped[trial] = stopped 
        StoppingTimes[trial] = stopping_time 
        if stopped: 
            Power[stopping_time:] += 1
    Power /= num_trials 
    if return_wealth:
        return Power, Stopped, StoppingTimes, W 
    else:
        return Power, Stopped, StoppingTimes 


def deLaPenaMartingale(F):
    f_plus = np.exp(F - F*F/2) 
    f_minus = np.exp(-F - F*F/2) 
    idx1 = np.where(f_plus>=f_minus)[0]
    idx2 = np.where(f_plus<f_minus)[0] 
    f= np.zeros(F.shape) 
    f[idx1] = f_plus[idx1]
    f[idx2] = 2 - f_minus[idx2] 
    F_new = f - 1 
    return F_new 

def power_betting_test(N_batch=200, N_betting=600, d=10, num_perms=10,num_trials=20,
            alpha=0.05, epsilon_mean=0.35, epsilon_var=1.5, num_pert_mean=1, 
            num_pert_var=0, num_steps_batch=10, progress_bar=False, seeds = None):
   
    # Maximum sample size 
    N_max = N_betting
    meanX, meanY, covX, covY = getGaussianSourceparams(d=d, epsilon_mean=epsilon_mean, 
                                    epsilon_var=epsilon_var,
                                    num_perturbations_mean=num_pert_mean, 
                                    num_perturbations_var=num_pert_var)
    # generate the source 
    Source = GaussianSource(meanX=meanX, meanY=meanY, 
                            covX=covX, covY=covY, truncated=False)

    ####=========================================================
    # Do the betting based sequential kernel-MMD test 
    t0 = time()
    Prediction = kernelMMDprediction
    Betting = ONSstrategy
    pred_params=None 
    bet_params=None
    PowerBetting, StoppedBetting, StoppingTimesBetting = runSequentialTest(Source, Prediction, Betting, 
                                                            alpha=alpha, Nmax=N_max,
                                                            pred_params=pred_params, bet_params=bet_params,
                                                            num_trials=num_trials, seeds = seeds)
    mean_tau_betting = StoppingTimesBetting.mean()
    NNBetting = np.arange(1, N_betting+1)
    ## Prepare the data for plotting 
    Data = {}
    Data['betting']=(PowerBetting, NNBetting, mean_tau_betting, StoppingTimesBetting) 

    return PowerBetting[-1]