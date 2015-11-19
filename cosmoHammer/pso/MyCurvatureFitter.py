'''
Created on Oct 30, 2013

@author: J.Akeret
'''
from __future__ import print_function, division, absolute_import, \
    unicode_literals

from numpy.linalg.linalg import norm
from scipy.optimize.minpack import leastsq
from scipy.optimize import minimize
import numpy
import sys

def parabola(p, theta):
    """
    Computation of the paraboloid for the given curvature matrix and samples.
    :param p: list of samples
    :param theta: vector containing the lower triangle of the matrix and the offset from the true mean
    
    :return: vector y from f(x,p)
    """    
    # if(any(p==0)):
    #     return numpy.ones(dim)*numpy.inf

    leng, dim = theta.shape
    # corrm, v, mu = transform(dim, p)

    
#     _cov = corr2cov(corrm, v)
#     R = numpy.linalg.inv(_cov)
    # if(any(v==0)):
    #     vi = v
    # else:
    #     vi = numpy.diag(1/v)
        
    
        
    # R = numpy.dot(vi, numpy.dot(numpy.linalg.inv(corrm), vi))
    R = numpy.diag(p)

    
    v = numpy.zeros(leng)
    for i,thetaj in enumerate(theta):
        #print(thetaj)
        #thetaj = thetaj / thetabar
        v[i] = numpy.dot(thetaj.T,numpy.dot(R, thetaj)) #+ numpy.dot(thetaj,  mu)
        # He dos not want to estimate the mean which is assumed to be gbest.position

    return numpy.array(v)


def errfunc(p,theta,delta):
    """
    Error function defined by f(theta) - delta
    :param p: list of samples
    :param theta: the curvature matrix. see parabola def
    :param delta: the measured values
    """
    return parabola(p, theta) - delta

def errfunc2(p,theta,delta, thetabar):
    """
    Error function defined by f(theta) - delta
    :param p: the curvature matrix. see parabola def
    :param theta: list of samples
    :param delta: the measured values
    """
    return sum((parabola(p, theta, thetabar) - delta)**2)

def transform(dim, p):
    """
    Transforms a vector containg the lower triangle of a matrix into a symmetric matrix
    
    :param p: the vector
    
    :return: the matrix and left over values

    * Note: Comented to work only with variances, not with covariances.
    """


    # corrm = numpy.identity(dim)
    # k=0
    # for i in range(1,dim):
    #     for j in range(0,i):
    #         corrm[i,j]= p[k]
    #         k +=1
            
    # corrm += corrm.T - numpy.diag(corrm.diagonal())
    
    # vars = p[k:k+dim]
    # mu = p[k+dim:]
    # return corrm, vars, mu
    return numpy.identity(dim), p, numpy.zeros(dim)

def reverse(dim, R, vars):
    """
    Transforms a symmetric matrix into a vector containig the lower triangle
    
    :param R: the symmetric matrix
    
    :return: the vector
    """
    p = numpy.zeros(dim*(dim-1)/2)
    k=0
    for i in range(1,dim):
        for j in range(0,i):
            p[k] = R[i,j]
            k +=1
            
    p = numpy.append(p, vars)
    return numpy.append(p, numpy.zeros_like(vars))

def bound(x):
    dim = int(1./2 * (numpy.sqrt(8*len(x)+1)-1))
    _, stds, _ = transform(dim, x)
    return stds


class CurvatureFitter(object):
    '''
    Fits a paraboloid centered around the global best fit of the PSO by estimating a curvarture
    matrix with the particle given in the swarm
    
    :param swarm: list of particles
    :param gbest: the global best particle at the last iteration
    '''


    def __init__(self, swarm, gbest,maxNorm=1):
        '''
        Constructor
        '''
        self.swarm = swarm
        self.gbest = gbest
        self.maxnorm = maxNorm
        
    def fit(self):
        """
        Fits the paraboloid to the swarm particles
        
        :return: the mean = global best position and the estimated covariance matrix
        """
        
        scale = 10**0
        dim = len(self.gbest.position) 
        
        x = numpy.array([particle.position * scale for particle in self.swarm])
        theta = (x - self.gbest.position * scale) #/ (self.gbest.position * scale)       
        fitness = numpy.array([particle.fitness * scale for particle in self.swarm])
        norms = numpy.array(list(map(norm, theta)))
        #---------------------------------------------------------
        print("Max norm",max(norms))

        b = (norms < self.maxnorm)
        theta = theta[b]
        fitness =  fitness[b]

        delta = -2*(fitness - self.gbest.fitness * scale)
        print("Standar deviation of fitness ", numpy.std(delta/2))
        
        _cov = self.minimize1(dim, theta, delta)
        #_cov = self.minimize2(dim, theta, delta)

        return self.gbest.position, _cov
    
    def minimize1(self, dim, theta, delta):
        # p0Cor = numpy.random.uniform(-1,1,dim**2).reshape(dim, dim)
        # p0Cor = p0Cor - numpy.diag(p0Cor) + numpy.identity(dim)
        
        #p0 = reverse(dim, numpy.identity(dim), numpy.ones(dim)/20)
        p0  = numpy.ones(dim)
        popt, _,infodict,mesg,_ = leastsq(errfunc, p0, args=(theta, delta),full_output=True)
        print(mesg)
         
         
        ss_err=(infodict['fvec']**2).sum()
        ss_tot=((delta-delta.mean())**2).sum()
        rsquared=1-(ss_err/ss_tot)
        print("rsquared", rsquared)
        
        # corrm, var, mu = transform(dim, popt)
        # var = var * self.gbest.position
        # _cov = corr2cov(corrm, var)
        print("found diag_inv_cov:\n", abs(popt))
        
        _cov = numpy.linalg.inv(numpy.diag(abs(popt)))

        #print("used mu:", mu)
        print("found _cov:\n", _cov)

        #sigma = numpy.sqrt(numpy.diag(_cov))
        #print( "=> found sigma:", sigma)
        
        return _cov
    
    def minimize2(self, dim, theta, delta):
        cons = (
         {'type': 'ineq',
          'fun' : lambda x: bound(x)})
        
        p0 = reverse(dim, numpy.identity(dim), numpy.ones(dim)*self.gbest.position/10)
        res = minimize(errfunc2, p0, args=(theta, delta, self.gbest.position), 
            constraints=cons, method='SLSQP', options={'disp': True, "ftol":10**-3})
        popt=res.x

        corrm, var, mu = transform(dim, popt)
        var = var * self.gbest.position
        _cov = corr2cov(corrm, var)
        
        print("used mu:", mu)
        print("found _cov:\n", _cov)
        
        sigma = numpy.sqrt(numpy.diag(_cov))
        print( "=> found sigma:", sigma)

        return _cov

def corr2cov(corrm, var):
    dim = len(var)
    covm = numpy.zeros((dim,dim))
    for i in range(len(corrm)):
        for j in range(len(corrm)):
            covm[i,j] = corrm[i,j]*var[i]*var[j]
            
    return covm

def rescale(_cov, v, dim):
        #rescaling
        cov2 = numpy.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                cov2[i,j] = _cov[i,j] * v[i] * v[j]
        return cov2