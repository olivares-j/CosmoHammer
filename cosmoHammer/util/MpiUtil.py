import itertools
import os
from cosmoHammer import getLogger
import time
import numpy as np

# If mpi4py is installed, import it.
try:
    from mpi4py import MPI
    MPI = MPI
except ImportError:
    MPI = None

class MpiPool(object):
    """
    Implementation of a mpi based pool. Currently it supports only the map function.
    
    :param mapFunction: the map function to apply on the mpi nodes
    
    """
    def __init__(self, mapFunction):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.mapFunction = mapFunction
    
    def map(self, function, sequence):
        """
        Emulates a pool map function using Mpi.
        Retrieves the number of mpi processes and splits the sequence of walker position 
        in order to allow each process its block
        
        :param function: the function to apply on the items of the sequence
        :param sequence: a sequence of items
        
        :returns sequence: sequence of results
        """
        
        (rank,size) = (MPI.COMM_WORLD.Get_rank(),MPI.COMM_WORLD.Get_size())
        #sync
        sequence = mpiBCast(sequence)
        
        getLogger().debug("Rank: %s, pid: %s MpiPool: starts processing iteration" %(rank, os.getpid()))
        #split, process and merge the sequence
        mergedList = mergeList(MPI.COMM_WORLD.allgather(
                                                  self.mapFunction(function, splitList(sequence,size)[rank])))
        getLogger().debug("Rank: %s, pid: %s MpiPool: done processing iteration"%(rank, os.getpid()))
#         time.sleep(10)
        return mergedList
    
    def isMaster(self):
        """
        Returns true if the rank is 0
        """
        return (self.rank==0)
    
def mpiBCast(value):
    """
    Mpi bcasts the value and Returns the value from the master (rank = 0).
    """
    getLogger().debug("Rank: %s, pid: %s MpiPool: bcast", MPI.COMM_WORLD.Get_rank(), os.getpid())
    return MPI.COMM_WORLD.bcast(value,root=0)

def mpiBarrier():
    """
    Mpi bcasts the value and Returns the value from the master (rank = 0).
    """
    getLogger().debug("Rank: %s, pid: %s MpiPool: barrier", MPI.COMM_WORLD.Get_rank(), os.getpid())
    return MPI.COMM_WORLD.barrier()

def mpiMean(value):
    """
    Mpi gather the value and Returns the value from the master (rank = 0).
    """
    total =  np.zeros_like(value)
    value = np.asarray(value)
    getLogger().debug("Rank: %s, pid: %s MpiPool: reduce", MPI.COMM_WORLD.Get_rank(), os.getpid())
    MPI.COMM_WORLD.Reduce([value, MPI.DOUBLE],[total, MPI.DOUBLE],op = MPI.SUM,root=0)
    return total/MPI.COMM_WORLD.Get_size()


def splitList(list, n):
    """
    Splits the list into block of equal sizes (listlength/n)
    
    :param list: a sequence of items
    :param n: the number of blocks to create
    
    :returns sequence: a list of blocks
    """
    getLogger().debug("Rank: %s, pid: %s MpiPool: splitList", MPI.COMM_WORLD.Get_rank(), os.getpid())
    blockLen = len(list) / float(n)
    return [list[int(round(blockLen * i)): int(round(blockLen * (i + 1)))] for i in range(n)]    

def mergeList(lists):
    """
    Merges the lists into one single list
    
    :param lists: a list of lists
    
    :returns list: the merged list
    """
    getLogger().debug("Rank: %s, pid: %s MpiPool: mergeList", MPI.COMM_WORLD.Get_rank(), os.getpid())
    return list(itertools.chain(*lists))

################## Modified #################
# def wait(self):
#         """
#         If this isn't the master process, wait for instructions.

#         """
#         self.debug = True
#         if self.isMaster():
#             raise RuntimeError("Master node told to await jobs.")

#         status = MPI.Status()

#         while True:
#             # Event loop.
#             # Sit here and await instructions.
#             if self.debug:
#                 print("Worker {0} waiting for task.".format(self.rank))

#             # Blocking receive to wait for instructions.
#             task = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)
#             if self.debug:
#                 print("Worker {0} got task {1} with tag {2}."
#                       .format(self.rank, task, status.tag))

#             # Check if message is special sentinel signaling end.
#             # If so, stop.
#             if isinstance(task, _close_pool_message):
#                 if self.debug:
#                     print("Worker {0} told to quit.".format(self.rank))
#                 break

#             # Check if message is special type containing new function
#             # to be applied
#             if isinstance(task, _function_wrapper):
#                 self.function = task.function
#                 if self.debug:
#                     print("Worker {0} replaced its task function: {1}."
#                           .format(self.rank, self.function))
#                 continue

#             # If not a special message, just run the known function on
#             # the input and return it asynchronously.
#             result = self.function(task)
#             if self.debug:
#                 print("Worker {0} sending answer {1} with tag {2}."
#                       .format(self.rank, result, status.tag))
#             MPI.COMM_WORLD.isend(result, dest=0, tag=status.tag)

# class _close_pool_message(object):
#     def __repr__(self):
#         return "<Close pool message>"


# class _function_wrapper(object):
#     def __init__(self, function):
#         self.function = function



