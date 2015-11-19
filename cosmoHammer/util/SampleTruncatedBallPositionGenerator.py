
import numpy
from scipy.stats import truncnorm as truncnorm
class SampleTruncatedBallPositionGenerator(object):
    """
        Generates samples in a very thight n-dimensional ball 
    """
    def __init__(self, low=-numpy.inf,upp=numpy.inf):
        self.hlow=low
        self.hupp=upp
    
    def setup(self, sampler):
        """
            setup the generator
        """
        self.sampler = sampler
    
    def generate(self):
        """
            generates the positions
        """
        # The first walker carries the inital parameter values
        # samples = numpy.row_stack([self.sampler.paramValues,[self.sampler.paramValues+numpy.random.normal(size=
        #     self.sampler.paramCount)*self.sampler.paramWidths for i in range((self.sampler.nwalkers-1))]])

        walkers_pos = numpy.zeros((self.sampler.nwalkers-1,self.sampler.paramCount))
        
        for j in range(len(walkers_pos)):
            for i in range(self.sampler.paramCount):
                a= (self.hlow[i] - self.sampler.paramValues[i]) / self.sampler.paramWidths[i]
                b= (self.hupp[i] - self.sampler.paramValues[i]) / self.sampler.paramWidths[i]
                pos = truncnorm.rvs(a,b,loc=self.sampler.paramValues[i],
                                    scale=self.sampler.paramWidths[i],size=1)[0] 
                # while numpy.isinf(pos):
                #     pos = truncnorm.rvs(self.sampler.paramMins[i],self.sampler.paramMaxs[i],
                #             loc=self.sampler.paramValues[i],
                #             scale=self.sampler.paramWidths[i],size=1) 
                #     print 
                #     print self.sampler.paramMins[i],self.sampler.paramMaxs[i]
                #     print self.sampler.paramValues[i],self.sampler.paramWidths[i]
                walkers_pos[j,i] = pos
        return numpy.row_stack([self.sampler.paramValues,walkers_pos])
    
    def __str__(self, *args, **kwargs):
        return "SampleBallPositionGenerator"