#Compute Laplace approximation to p^{*}(w) = exp(-w^2 + cos(\alpha w ) )
#Compare against naive quadrature - only possible in simple low dimensional examples.

from matplotlib import pylab as plt

import numpy as np
from IPython import embed

def get_linear_spaced():
    start = -5.
    stop = 5.
    linear_spaced = np.linspace( start, stop, num=10000, endpoint = True  )
    return linear_spaced

def target(alpha, x):
    exponent = -x**2 + np.cos( alpha * x )
    return np.exp( exponent ) 

def quadrature_normalizer(alpha):
    linear_spaced = get_linear_spaced()

    delta = np.diff(linear_spaced)[0]
    targets = target(alpha, linear_spaced)
    normalizer = np.sum(targets)* delta
    return normalizer

def laplace_approx(alpha, x):
    exponent = -0.5*(2. + alpha**2) * x**2
    constant = np.sqrt( (2. + alpha**2)/2./np.pi )
    approx_normalized = constant * np.exp( exponent )
    approx_raw = approx_normalized * np.exp(1) / constant
    return approx_normalized, approx_raw

def run_demo():
    plt.rc('font', size=25)  
    fig,axes = plt.subplots(2,3)


    alphas = [1.,5.,10.]
    linear_spaced = get_linear_spaced()

    axes[0,0].set_ylabel("Unnormalized")
    axes[1,0].set_ylabel("Normalized")
    lines = []

    for index in range(len(alphas)):
        alpha = alphas[index]
        current_normalizer = quadrature_normalizer(alpha)
        standard = target(alpha, linear_spaced)
        normalized = standard / current_normalizer
        laplace_normalized, laplace_raw = laplace_approx(alpha, linear_spaced)
        if index==0:
            lines = lines + axes[0,index].plot( linear_spaced, standard, 'b', label = 'Exact') 
            lines = lines + axes[0,index].plot( linear_spaced, laplace_raw, 'g', label = 'Laplace') 
        else:
            axes[0,index].plot( linear_spaced, standard, 'b') 
            axes[0,index].plot( linear_spaced, laplace_raw, 'g')
        axes[0,index].set_title('Alpha = '+str(alpha))
        axes[1,index].plot( linear_spaced, normalized, 'b')
        axes[1,index].plot( linear_spaced, laplace_normalized, 'g')
        axes[1,index].set_ylim( [0.,2.21])

    plt.subplots_adjust(right=0.7)
    plt.legend( handles = lines, loc= 7, bbox_to_anchor=(1.75, 1.1) )
    
    plt.show()

run_demo()
