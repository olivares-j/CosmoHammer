'''
Created on Sep 30, 2013

@author: J. Akeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from copy import copy
from math import floor
import math
import multiprocessing
import numpy
import pandas

class ParticleSwarmOptimizer(object):
    '''
    Optimizer using a swarm of particles
    
    :param func:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param low: array of the lower bound of the parameter space
    :param high: array of the upper bound of the parameter space
    :param particleCount: the number of particles to use. 
    :param threads: (optional)
        The number of threads to use for parallelization. If ``threads == 1``,
        then the ``multiprocessing`` module is not used but if
        ``threads > 1``, then a ``Pool`` object is created and calls to
        ``lnpostfn`` are run in parallel.

    :param pool: (optional)
        An alternative method of using the parallelized algorithm. If
        provided, the value of ``threads`` is ignored and the
        object provided by ``pool`` is used for all parallelization. It
        can be any object with a ``map`` method that follows the same
        calling sequence as the built-in ``map`` function.
    
    '''


    def __init__(self, func, low, high, particleCount=25, threads=1, pool=None, fSwarm=None):
        '''
        Constructor
        '''
        print('My PSO')
        self.func = func
        self.low = low
        self.high = high
        self.particleCount = particleCount
        self.threads = threads
        self.pool = pool
        
        if self.threads > 1 and self.pool is None:
            self.pool = multiprocessing.Pool(self.threads)
        
        self.paramCount = len(self.low)

        if fSwarm is None :
            self.swarm = self._initSwarm()
        else:
            self.swarm = self._readSwarm(fSwarm)
            print("Position read")
        
        self.gbest = Particle.create(self.paramCount)
        # for particle in self.swarm:
        #     print(particle.position)
        # stop

        # self.gbest = Particle(numpy.random.uniform(self.low, self.high, size=self.paramCount), 
        #     numpy.zeros(self.paramCount),-numpy.Inf)

        # print(self.gbest.position.shape)
        # for particle in self.swarm:
        #     print(particle.position.shape)
        # stop
        
    def _initSwarm(self):
        swarm = []
        for _ in range(self.particleCount):
            swarm.append(Particle(position=numpy.random.uniform(self.low, self.high, size=self.paramCount),
                                  velocity=numpy.zeros(self.paramCount)))
        return swarm

    def _readSwarm(self,fswrm):
        # The file must have, as second entry, the fitness, the rest must be particle position.
        nswarm = numpy.array(pandas.read_csv(fswrm,sep='\t',comment='#',header=None))
        swarm = []
        for i in range(self.particleCount):
            swarm.append(Particle(position=nswarm[-i,2:],
                                  velocity=numpy.zeros(self.paramCount)))
        return swarm
        
    def sample(self, maxIter=1000, c1=1.193, c2=1.193, p=0.7, m=10**-3, n=10**-2):
        """
        Launches the PSO. Yields the complete swarm per iteration
        
        :param maxIter: maximum iterations
        :param c1: cognitive weight
        :param c2: social weight
        :param p: stop criterion, percentage of particles to use 
        :param m: stop criterion, difference between mean fitness and global best
        :param n: stop criterion, difference between norm of the particle vector and norm of the global best
        """
        self._get_fitness(self.swarm)

        i = 0
        while True:
            
            
            for particle in self.swarm:
                if ((self.gbest.fitness)<particle.fitness):
                    
                    self.gbest = particle.copy()
                    # if(self.isMaster()):
                    #   print("new global best found %i %s"%(i, self.gbest.__str__()))
                    
                if (particle.fitness > particle.pbest.fitness):
                    particle.updatePBest()

            if(i>=maxIter):
                print("max iteration reached! stoping")
                return
            #--------------------------------------
            if(self._convergedFit(p=p, m=m)):
                self._run_emcee(i,it=10)
            #--------------------------------------

            if(self._converged(i, p=p,m=m, n=n)):
                if(self.isMaster()):
                    print("converged after %s iterations!"%i)
                    print("best fit found: ", self.gbest.fitness, self.gbest.position)
                return
            
            
            for particle in self.swarm:
                #This is a PSO with inertia
                #w = 0.5 + numpy.random.uniform(0,1,size=self.paramCount)/2
                #----- This is PSO of Clerc and Kennedy 2002 
                c1 = 1.49618
                c2 = 1.49618
                w  = 0.7298
                #----- This is my pso. 0.2% of the realisations will produce a velocity factor grater than 1
                w  = w + numpy.absolute(numpy.random.uniform(0,0.09006,size=self.paramCount))
                #----------------------------------------------------------
                part_vel = w * particle.velocity
                cog_vel = c1 * numpy.random.uniform(0,1,size=self.paramCount) * (particle.pbest.position - particle.position)
                soc_vel = c2 * numpy.random.uniform(0,1,size=self.paramCount) * (self.gbest.position - particle.position)
                particle.velocity = part_vel + cog_vel + soc_vel
                particle.position = particle.position + particle.velocity
            
            self._get_fitness(self.swarm)

            swarm = []
            for particle in self.swarm:
                swarm.append(particle.copy()) 
            yield swarm
            
            i+=1
        
    def optimize(self, maxIter=1000, c1=1.193, c2=1.193, p=0.7, m=10**-3, n=10**-2):
        """
        Runs the complete optimiziation.
        
        :param maxIter: maximum iterations
        :param c1: cognitive weight
        :param c2: social weight
        :param p: stop criterion, percentage of particles to use 
        :param m: stop criterion, difference between mean fitness and global best
        :param n: stop criterion, difference between norm of the particle vector and norm of the global best

        :return swarms, gBests: the swarms and the global bests of all iterations
        """
        
        swarms = []
        gBests = []
        for swarm in self.sample(maxIter,c1,c2,p,m,n):
            swarms.append(swarm)
            gBests.append(self.gbest.copy())
        
        return swarms, gBests
        
    def _get_fitness(self,swarm):
        
        # If the `pool` property of the pso has been set (i.e. we want
        # to use `multiprocessing`), use the `pool`'s map method. Otherwise,
        # just use the built-in `map` function.
        if self.pool is not None:
            mapFunction = self.pool.map
        else:
            mapFunction = map
        
        pos = numpy.array([part.position for part in swarm])
        results =  mapFunction(self.func, pos)
        lnprob = numpy.array([l[0] for l in results])
        for i, particle in enumerate(swarm):
            particle.fitness = lnprob[i]

    def _converged(self,it, p, m, n):
#        test = self._convergedSpace2(p=p)
#        print(test)
        fit = self._convergedFit(p=p, m=m)
        if(fit):
            space = self._convergedSpace(p=p, m=n)
            return space
        else:
            return False
        
    def _convergedFit(self, p, m):
        bestSort = numpy.sort([particle.pbest.fitness for particle in self.swarm])[::-1]
        meanFit = numpy.mean(bestSort[1:math.floor(self.particleCount*p)])
        # print( "best , meanFit, ration %",self.gbest.fitness, meanFit, (self.gbest.fitness-meanFit))
        return ((self.gbest.fitness-meanFit)<m)

    # def _convergedFit2(self, it, p, m):
    #     bestSort = numpy.sort([particle.pbest.fitness for particle in self.swarm])[::-1]
    #     maxFit = max(bestSort[1:math.floor(self.particleCount*p)])
    #     minFit = min(bestSort[1:math.floor(self.particleCount*p)])
    #     cond1  = abs(self.gbest.fitness-maxFit)<m
    #     cond2  = abs(self.gbest.fitness-minFit)<m
    #     #print( "best , meanFit, ration %",self.gbest.fitness, meanFit, (self.gbest.fitness-meanFit))
    #     return (cond1 and cond2)
    
    def _convergedSpace(self, p, m):
        sortedSwarm = [particle for particle in self.swarm]
        sortedSwarm.sort(key=lambda part: -part.fitness)
        bestOfBest = sortedSwarm[0:int(floor(self.particleCount*p))]
        
        diffs = []
        for particle in bestOfBest:
            diffs.append(self.gbest.position-particle.position)
            
        maxNorm = max(list(map(numpy.linalg.norm, diffs)))
        print('Max Norm',maxNorm,'Best fitness',self.gbest.fitness)
        return (abs(maxNorm)<m)

    def _convergedSpace2(self, p):
        #Andres N. Ruiz et al.
        sortedSwarm = [particle for particle in self.swarm]
        sortedSwarm.sort(key=lambda part: -part.fitness)
        bestOfBest = sortedSwarm[0:int(floor(self.particleCount*p))]
        
        positions = [particle.position for particle in bestOfBest]
        means = numpy.mean(positions, axis=0)
        delta = numpy.mean((means-self.gbest.position)/self.gbest.position)
        return numpy.log10(delta) < -3.0


    def isMaster(self):
        return True   

    def _run_emcee(self,current_it,it=10):
        print("Inside emcee. Iteration ",current_it)
        p = numpy.array([part.position for part in self.swarm])
        lnprob = numpy.array([part.fitness  for part in self.swarm])
        for j in range(int(it)):
            # print("Iteration ",j)
            #--------- modified version -------
            aranged = numpy.arange(self.particleCount)
            halfk = int(self.particleCount/2)
            numpy.random.shuffle(aranged)
            first  = aranged[0:halfk]
            second = aranged[halfk:self.particleCount]
            #----------------------------------

            for S0, S1 in [(first, second), (second, first)]:
                q, newlnp, acc = self._propose_stretch(p[S0], p[S1], lnprob[S0])
                if numpy.any(acc):
                            #------ modified version ---------
                            temp = lnprob[S0]
                            temp[acc] = newlnp[acc]
                            lnprob[S0]= temp

                            temp = p[S0]
                            temp[acc]= q[acc]
                            p[S0]=temp
        l = 0
        for particle in self.swarm:
            particle.position = p[l]
            particle.fitness  = lnprob[l]
            l +=1

        pass

    def _propose_stretch(self,p0, p1, lnprob0,a=1.5):
        """
        Propose a new position for one sub-ensemble given the positions of
        another.

        :param p0:
            The positions from which to jump.

        :param p1:
            The positions of the other ensemble.

        :param lnprob0:
            The log-probabilities at ``p0``.

        This method returns:

        * ``q`` - The new proposed positions for the walkers in ``ensemble``.

        * ``newlnprob`` - The vector of log-probabilities at the positions
          given by ``q``.

        * ``accept`` - A vector of type ``bool`` indicating whether or not
          the proposed position for each walker should be accepted.

        * ``blob`` - The new meta data blobs or ``None`` if nothing was
          returned by ``lnprobfn``.

        """
        s = numpy.atleast_2d(p0)
        Ns = len(s)
        c = numpy.atleast_2d(p1)
        Nc = len(c)

        # Generate the vectors of random numbers that will produce the
        # proposal.
        zz = ((a - 1.) * numpy.random.rand(Ns) + 1) ** 2. / a
        rint = numpy.random.randint(Nc, size=(Ns,))

        # Calculate the proposed positions and the log-probability there.
        q = c[rint] - zz[:, numpy.newaxis] * (c[rint] - s)
        #--new prob 
        results = self.pool.map(self.func, q)
        newlnprob = numpy.array([l[0] for l in results])

        # Decide whether or not the proposals should be accepted.
        lnpdiff = (self.paramCount - 1.) * numpy.log(zz) + newlnprob - lnprob0
        accept = (lnpdiff > numpy.log(numpy.random.rand(len(lnpdiff))))

        return q, newlnprob,accept

class Particle(object):
    """
    Implementation of a single particle
    
    :param position: the position of the particle in the parameter space
    :param velocity: the velocity of the particle
    :param fitness: the current fitness of the particle
    
    """
    
    
    def __init__(self, position, velocity, fitness = 0):
        self.position = position
        self.velocity = velocity
        
        self.fitness = fitness
        self.paramCount = len(self.position)
        self.pbest = self

    @classmethod
    def create(cls, paramCount):
        """
        Creates a new particle without position, velocity and -inf as fitness
        """
        # return Particle(numpy.array([[]]*paramCount),
        #          numpy.array([[]]*paramCount),
        #          -numpy.Inf)

        return Particle(numpy.zeros(paramCount),numpy.zeros(paramCount),-numpy.Inf)
        
    def updatePBest(self):
        """
        Sets the current particle representation as personal best
        """
        self.pbest = self.copy()
        
    def copy(self):
        """
        Creates a copy of itself
        """
        return Particle(copy(self.position),
                        copy(self.velocity),
                        self.fitness)
        
    def __str__(self):
        return "%f, pos: %s velo: %s"%(self.fitness, self.position, self.velocity)
    
    def __unicode__(self):
        return self.__str__()
