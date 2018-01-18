import util
import collections, functools
import numpy as np
fname = 'emission_test.hmm'

####################################################
### Forward-backward algorithm #####################
####################################################


def _make_alpha_initial(self):
    '''
    Return a matrix M where M(i,j) = b_i( Y_j ) * \mu( s_i ) 
    '''
    mu = self.internal_initial
    b = self.emission_mat
    M = mu.T*(b)
#    print mu.shape, b.shape
#    print M
    self.alpha_initial = M
    return M
util.Var_hmm._make_alpha_initial = _make_alpha_initial

if __name__ =='__main__':
    h1 = util.Var_hmm.read_model(fname = fname)
    # self.
    a_ini = h1._make_alpha_initial()
    print 'Initial alpha( s_i, Y_j) is:\n',a_ini
    print 

def _make_emission_initial(self):
    '''
    Marginalise initial \alpha(s_i,Y_j) over i (all states) to give \nu(Y_j)
    '''
    if not hasattr(self,'alpha_initial'):
        self._make_alpha_initial()
    a = self.alpha_initial
    nu = a.sum(axis = 0,keepdims = 1)
    #print nu.shape
    self.emission_initial = nu
    return nu
#    print self.internal_initial.shape
#    pass
util.Var_hmm._make_emission_initial = _make_emission_initial

if __name__ =='__main__':
    h1 = util.Var_hmm.read_model(fname = fname)
    nu = h1._make_emission_initial()
    print 'Initial emission distribution is:\n', nu
    print 'Summed to:',nu.sum()
    print

def _make_Nalpha_initial(self):
    '''
    Return the matrix M(i,j) = normAlpha_{0}(s_i,Y_j)
    '''
    if not hasattr(self,'emission_initial'):
        self._make_emission_initial()
        self.c_initial = 1
    nu = self.emission_initial
    a  = self.alpha_initial
    #print a.shape,nu.shape
    Na = a / nu
    self.Nalpha_initial = Na
    return Na
util.Var_hmm._make_Nalpha_initial = _make_Nalpha_initial

if __name__ =='__main__':
    h1 = util.Var_hmm.read_model(fname = fname)
    Na = h1._make_Nalpha_initial()
    print 'Initial normalised alpha is:\n',Na
    print 'summed over hidden states:\n',Na.sum(axis = 0)
    print

def _forward(self, emission_idx,
                 return_c = False,
                 as_norm = True,
                 rescale = True,
                hold = False):
    '''
    Input an emission index, based on which "self.Nalpha" and "self.c" is updated
    Params:
        hold: BOOL, whether time step shall be updated
        return_c: whether return c as ouput
        as_norm : if returning alpha, whether it should be rescaled
        rescale : if True, use rescaled implementation. Otherwise use naive implementation
    '''
    if not rescale:
        raise Exception("Naive forward algorith is not implemented")
    if self.t == -1:
        if not hasattr(self,'Nalpha_initial'):
            self._make_Nalpha_initial()
        #Na = self.Nalpha_initial
        #c  = self.c_initial
        cNa_mat = self.alpha_initial 
#         Na_curr = self.c_initial * Na
        cNa = cNa_mat[:,emission_idx:emission_idx+1].T ## keep the matrix shape
    else:
        Na_curr = self.Nalpha
        cNa = self.emission_mat[:,emission_idx:emission_idx+1].T * Na_curr.dot(self.transition)
    self.cNalpha = cNa
    c = cNa.sum()
    self.Nalpha = self.cNalpha/c
    
    #self.c = c if not self.t == -1 else 1 ##### Force initial c to be 1 so that sum of log is meaningful because log(1)=0 
    self.c = c
    if not hold:
        self.t += 1
    if return_c:
        return self.c
    if as_norm:
        return self.Nalpha
    else:
        return cNa
util.Var_hmm._forward = _forward

if __name__ =='__main__':
    h1 = util.Var_hmm.read_model(fname = fname)
    print 'Transition matrix is:\n',h1.transition
    print 'Emission matrix is:\n',h1.emission_mat

    #h1.sample(hold = 1)
    #h1.emit(as_idx = 1)
    h1.emission = 0
    print "Emission index is:", h1.emission
    print "Emitted state is:", h1.emission_list[h1.emission]
    cNa = h1._forward(h1.emission,as_norm=0)
    print 'un-normalised alpha is:',cNa
    print 'Normalised alpha is:',h1.Nalpha
    print 'c_{n+1}= sum(raw_alpha):',h1.c

    print 
    #h1.sample(hold = 0)
    h1.emission = 2
    print "Emission index is:", h1.emission
    print "Emitted state is:", h1.emission_list[h1.emission]
    cNa = h1._forward(h1.emission,as_norm=0)
    print 'un-normalised alpha is:',cNa
    print 'Normalised alpha is:',h1.Nalpha
    print 'c_{n+1}= sum(raw_alpha):',h1.c

    
import warnings
def emission_likelihood(self, echains = None, fname = None, as_log = True, debug = 0): 
    if echains is None:
        echains = util.read_chain(fname, dtype = 'int',debug = debug)
    cs = []
    print echains 
    for echain in echains:
        self.reset(t = -1)
        print echain.shape
        cs += [[self._forward(E,return_c  = 1) for E in echain]]
    cs = np.array(cs)
    logL = np.log(cs).sum(axis = 1)    
    if debug:
        print "Log-likelihood:", logL
    if as_log:
        return logL
    else:
        if logL < -700:
            warnings.warn("[Underflow]:Trying to raise exp(X) of an X < -700")
        return np.exp(logL)
util.Var_hmm.emission_likelihood = emission_likelihood

if __name__=='__main__':
    h1 = util.Var_hmm.read_model(fname = fname)
    echain = util.read_chain('test.echain',dtype = 'int')[0:1]
    print "Calculating likelihood for chain:",echain
    _ = h1.emission_likelihood(echain,debug=1)
    
    
# from forward import *
def _make_Nbeta_initial(self):
    self.beta_initial = np.ones((1,self.internal_space),dtype = 'float')
    self.Nbeta_initial = self.beta_initial / self.internal_space
    pass
util.Var_hmm._make_Nbeta_initial = _make_Nbeta_initial
if __name__=='__main__':
    h1 = util.Var_hmm()
    h1._make_Nbeta_initial()
    print "initial beta:\n",util.mat2str(h1.beta_initial)
    print "initial normed beta:\n",util.mat2str(h1.Nbeta_initial)

def _backward(self, emission_idx,
                 return_d = False,
                 as_norm = True,
                 rescale = True,
                hold = False,
             debug = 0):
    '''
    Input an emission index, based on which "self.Nbeta" and "self.d" is updated
    Params:
        hold: BOOL, whether time step shall be updated
        return_d: whether return d as ouput
        as_norm : if returning beta, whether it should be rescaled
        rescale : if True, use rescaled implementation. Otherwise use naive implementation
    '''
    if not rescale:
        raise Exception("Naive forward algorith is not implemented")
    if self.t == -1: 
        #### Here "t" denotes time step from the end of the chain
        if not hasattr(self,'Nbeta_initial'):
            self._make_Nbeta_initial()
        #Na = self.Nalpha_initial
        #c  = self.c_initial
        dNb_mat = self.beta_initial
        dNb = dNb_mat
#         Na_curr = self.c_initial * Na
#        cNb = cNb_mat[:,emission_idx:emission_idx+1].T ## keep the matrix shape
    else:
        #assert hasattr(self,'last_emission')
        assert self.last_emission is not None
        last_emission = self.last_emission
        if debug:
            print "last emission index was:",last_emission
#         emission_idx = last
        Nb_curr = self.Nbeta
        bjk = self.emission_mat[:,last_emission:last_emission+1]
        Abeta = np.matmul( Nb_curr * bjk.T, self.transition.T )
        if debug:
            print Nb_curr.shape,bjk.T.shape
            print Nb_curr,bjk.T
        #print Abeta.shape
#        print bjk.shape, Abeta.shape
        dNb = Abeta
#        dNb = bjk * Abeta
    self.dNbeta = dNb
    d = dNb.sum() 
    self.Nbeta = self.dNbeta/d
#     self.d = d if not self.t == -1 else 1
    self.d = d
    self.last_emission = emission_idx
    
    if debug:
        print "non-normalised beta * d was:",dNb
        print "normalised beta was:",self.Nbeta
        print "sum(raw_beta) was:",self.d
    if not hold:
        self.t += 1
    if return_d:
        return self.d
    if as_norm:
        return self.Nbeta
    else:
        return dNb
util.Var_hmm._backward = _backward

if __name__=='__main__':
    print '\nTesting _backward(self)\n'
    h1 = util.Var_hmm.read_model(fname ='emission_test.hmm')
    print h1
    print h1._backward(0,debug = 1)
    print 
    print h1._backward(2,debug = 1)
    print h1._backward(1,debug = 1)


    
####################################################
### Viterbi algorithm   ############################
####################################################
####################################################
def _make_delta_initial(self):
    self._make_Nalpha_initial()
    self.delta_initial    = self.alpha_initial #### This is actully \nu( s_i, Y = v_j )
    self.logDelta_initial = np.log(self.delta_initial)
    self.logT = np.log(self.transition)
    self.logE = np.log(self.emission_mat)
    return self.delta_initial
util.Var_hmm._make_delta_initial = _make_delta_initial

def _forward_viterbi(self, emission_idx,
                 return_trackback = False,
                 as_norm = True,
#                  rescale = True,
                hold = False,
                    debug = False):
    '''
    Input an emission index, based on which "self.logDelta" and "self.psi" is updated
    Params:
        hold: BOOL, whether time step shall be updated
        return_c: whether return c as ouput
        as_norm : if returning alpha, whether it should be rescaled
        rescale : if True, use rescaled implementation. Otherwise use naive implementation
    '''
#     if not rescale:
#         raise Exception("Naive forward algorith is not implemented")
    if self.t == -1:
        if not hasattr(self,'delta_initial'):
            self._make_delta_initial()
        logD_mat = self.logDelta_initial
        #Na = self.Nalpha_initial
        #c  = self.c_initial
#         cNa_mat = self.alpha_initial 
#         Na_curr = self.c_initial * Na
        logD = logD_mat[:,emission_idx:emission_idx+1].T ## keep the matrix shape
    else:
        logD = self.logDelta.T
#         logD 
#         print "logTransition shape:",self.logT.shape
#         print self.logDelta.shape
        logD_logT = logD + self.logT
        psi = np.argmax(logD_logT,axis = 0)
        idx = (psi, np.arange(self.internal_space))
        idx = np.ravel_multi_index(idx, logD_logT.shape)        
        logD = logD_logT.flat[idx][None,:]
        evct = self.logE[:,emission_idx:emission_idx+1].T
        logD += evct        
        self.psi = psi.T
        
        if debug:
            print 'Raised logD :\n',logD_logT
            print 'Maximum row index for each column:\n', psi
#             print 'log emission prob: \n',evct 
            print 'Emission prob: \n', np.exp(evct)  ### Prefer non-log prob for readability
    self.logDelta = logD
    
    if debug:
        print "timestep:%d \t emission_idx:%d \t Delta:%s"%(self.t, emission_idx, np.exp(logD))
        print "logDelta shape:", logD.shape
        print 
    if not hold:
        self.t += 1
    if return_trackback:
        return self.psi
    else:
        return logD
util.Var_hmm._forward_viterbi = _forward_viterbi

if __name__=='__main__':
    print '\nTesting _forward_viterbi(self)\n'
    h1 = util.Var_hmm.read_model(fname ='Q3.hmm')
    print h1
    print h1._forward_viterbi(1,debug = 1)
    print h1._forward_viterbi(3,debug = 1)
    print h1._forward_viterbi(3,debug = 1)
    print h1._forward_viterbi(1,debug = 1)

    
def MLE_viterbi(self,echain = None,
               debug = 0):
    self.reset()
    traceback = []
    for E in echain:
        self._forward_viterbi(E)
        if self.t==0:
            continue
        else:
            traceback += [self.psi]
    traceback = np.array(traceback)
    
    traceback
    if debug > 1:
        print "Traceback for MLE:\n",traceback
        print self.logDelta
    if debug:
        print "Loglik of MLE hidden sequence is:", self.logDelta.max()
    self.mle_internal_end = np.argmax(self.logDelta)
    mle_internal_seq = []

    self.mle_internal = self.mle_internal_end
    mle_internal_seq += [self.mle_internal]
    for trace in traceback[::-1,:]:
        self.mle_internal = trace[self.mle_internal]
        mle_internal_seq += [self.mle_internal]    
    self.mle_internal_seq = np.array(mle_internal_seq[::-1])
    return self.mle_internal_seq
util.Var_hmm.MLE_viterbi = MLE_viterbi
if __name__=='__main__':
    print '\nTesting MLE_viterbi(self)\n'
    h1 = util.Var_hmm.read_model(fname ='Q3.hmm')
    echain = util.read_chain('Q3.echain',dtype=int)[0]
    mle_seq = h1.MLE_viterbi(echain, debug =1)
#     print _ 

