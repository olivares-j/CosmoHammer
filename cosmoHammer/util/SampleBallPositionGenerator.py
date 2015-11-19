
import numpy

class SampleBallPositionGenerator(object):
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

        #samples = numpy.random.multivariate_normal(pso.gbest.position, _cov, self.sampler.nwalkers)
#         print numpy.std(samples, axis=0)
#        return samples
        # The first walker carries the inital parameter values
        # return np.row_stack([self.sampler.paramValues,[self.sampler.paramValues+np.random.normal(size=
        #     self.sampler.paramCount)*self.sampler.paramWidths for i in range((self.sampler.nwalkers-1))]])

        return numpy.row_stack([self.sampler.paramValues,
            numpy.random.multivariate_normal(self.sampler.paramValues,
                            numpy.diag(self.sampler.paramWidths),
                            self.sampler.nwalkers-1)])
    
    def __str__(self, *args, **kwargs):
        return "SampleBallPositionGenerator"