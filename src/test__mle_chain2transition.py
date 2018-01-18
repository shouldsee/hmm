#### Question 2
####### MLE for a single internal state chain
# from util import *
import util
import matplotlib.pyplot as plt
import collections
import numpy as np
def mle_tmat_internalChain(chain, Hn = None,epsilon = 0):
    '''
    epsilon: an adjustable pesudo count
    '''
    Hn = max(chain) + 1
    trans = collections.Counter([tuple(chain[i:i+2]) for i in range(len(chain)-1)])

    mat = np.zeros((Hn,Hn))
    for k,v in trans.iteritems():
        mat[k]=float(v) + epsilon
    mat /= mat.sum(axis = 1,keepdims = 1)    
    return mat

# import inspect
# print inspect.getsource(mle_tmat_internalChain)
h1 = util.Var_hmm.read_model(fname = 'Q3.hmm')
chain = util.read_chain(fname='test.echain',dtype=int)[0]
chain = list(chain[:20])
mle_tmat = mle_tmat_internalChain(chain)

print "Truncating chain to 20 elements:",chain 
print 'MLE for transition matrix is:\n',mle_tmat
print 'TRUE transition matrix is:\n', h1.transition
print 'MAE of the estimator is:\n', abs(mle_tmat - h1.transition)
