
import sys
import numpy as np


class UniformIntervalPositionGenerator(object):
    """
        Generates samples in a very thight n-dimensional ball 
    """
    
    def setup(self, sampler):
        """
            setup the generator
        """
        self.sampler = sampler
    
    def generate(self):
        """
            generates the positions
        """
        #print self.sampler.nwalkers
        # The first walker carries the inital parameter values
        pos=np.row_stack([self.sampler.paramValues,[np.random.uniform(
            low=self.sampler.paramMins,
            high=self.sampler.paramMaxs,
            size=self.sampler.paramCount) 
            for i in range((self.sampler.nwalkers-1))]])
        if (pos < self.sampler.paramMins).any(): sys.exit("Position ERROR")
        if (pos > self.sampler.paramMaxs).any(): sys.exit("Position ERROR")
        return pos
    
    def __str__(self, *args, **kwargs):
        return "UniformIntervalPositionGenerator"