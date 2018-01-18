import util
from forward import *
import warnings
import util
from forward import *
import warnings

class chains(object):
    def __init__(self,fname = None,model = None):
        if fname is not None:
            self.echains = util.read_chain(fname, dtype = 'int',debug = 1)
        self.model = model
        pass

def bulk_forward(self, echains = None, fname = None, as_log = True, debug = 0): 
    mdl = self.model
    assert self.model is not None, 'No model associated with these chains'
    echains = self.echains
#     if echains is None:
#         echains = util.read_chain(fname, dtype = 'int',debug = debug)
    cs  = []
    Nalphas = []
    for echain in echains:
        mdl.reset(t = -1)
        cs_curr = []
        Nalphas_curr = []
        for E in echain:
            _ = mdl._forward(E)  #### Create mdl.c and mdl.Nalpha
            cs_curr     += [mdl.c]
            Nalphas_curr += [mdl.Nalpha]
            
        Nalphas += [Nalphas_curr]
        cs      += [cs_curr] 
    cs = np.array(cs)
    Nalphas = np.array(Nalphas)
    self.cs = cs
    self.Nalphas = Nalphas
    
    log_cs = np.log(cs)
    self.logProb_0toM = np.cumsum(log_cs,axis = 1)
    self.logAlpha = self.logProb_0toM[:,:,None,None] + np.log(Nalphas)
    logL = self.logProb_0toM[:,-1:]
    self.forward_logL = logL
#     ba
#     return Nalphas 
#     print Nalphas.shape
#     print cs.shape
    if debug:
        print "Log-likelihood:", logL
    if as_log:
        return logL
    else:
        if logL < -700:
            warnings.warn("[Underflow]:Trying to raise exp(X) of an X < -700")
        return np.exp(logL)
if __name__=='__main__':
    mdl = util.Var_hmm.read_model(fname = 'Q3.hmm',
           emission_list = [str(x) for x in range(1,6)])
    chains.bulk_forward = bulk_forward
    chs = chains(fname = 'test.echain',model = mdl)
    chs.bulk_forward()




def bulk_backward(self, echains = None, fname = None, as_log = True, debug = 0): 
    mdl = self.model
    assert self.model is not None, 'No model associated with these chains'
    echains = self.echains
#     if echains is None:
#         echains = util.read_chain(fname, dtype = 'int',debug = debug)
    ds  = []
    Nbetas = []
    for echain in echains[:,::-1]:
        Lc = echain.size
        mdl.reset(t = -1)
        ds_curr = []
        Nbetas_curr = []
        for E in echain:
            _ = mdl._backward(E)  #### Create mdl.c and mdl.Nalpha
            ds_curr      += [mdl.d]
            Nbetas_curr += [mdl.Nbeta]
            
        Nbetas += [Nbetas_curr]
        ds      += [ds_curr] 
    ds = np.array(ds)
    Nbetas = np.array(Nbetas)
    Nbetas = Nbetas[:,::-1,:,:]
    self.Nbetas = Nbetas
    
    log_ds = np.log(ds)
#     self.logProb_Mp1toN = np.cumsum(log_ds,axis = 1)[:,-2::-1]
    self.logProb_Mp1toN = np.cumsum(log_ds,axis = 1)[:,::-1]
    SHAPE = list(self.logProb_Mp1toN.shape)
    SHAPE[1] = 1
#     self.logProb_Mp1toN = np.append(self.logProb_Mp1toN,axis = 1,values = np.zeros( shape=  SHAPE ))
    self.ds = ds[::-1]
    self.logBeta = self.logProb_Mp1toN[:,:,None,None] + np.log(Nbetas)
#     print np.exp(self.logBeta[:,-1:-5:-1])

    logL = self.logProb_Mp1toN[:,:1]
    self.backward_logL = logL
    return logL
chains.bulk_forward = bulk_forward
chains.bulk_backward= bulk_backward

if __name__=='__main__':
    mdl = util.Var_hmm.read_model(fname = 'Q3.hmm',
           emission_list = [str(x) for x in range(1,6)])
    print mdl
    chs = chains(fname = 'Q3.echain',model = mdl)

    print 'Forward logL is:\n',chs.bulk_forward()
    print 'Forward variables are:\n',np.exp(chs.logAlpha[:,0:5])
    print 'Backward logL is:\n',chs.bulk_backward()
    print 'Backward variables are:\n',np.exp(chs.logBeta[:,-5:])
#emission_likelihood(mdl,fname = 'test.echain' )

def MLE_BW(self,debug = False,echains = None):
    self.bulk_forward()
    self.bulk_backward()
    if echains is None:
        echains = self.echains
#     print echains.shape
#     echains = echains[:,1:]
    bjk = self.model.emission_mat[:,echains[:,1:]]
    bjk = np.moveaxis(bjk[:,:,:,None],0,3)
    log_bjk = np.log(bjk)
    A = self.logAlpha[:,:-1]
    A = np.moveaxis(A,3,2)
    B = self.logBeta[:,1:]
    logT = np.log(self.model.transition)[None,None,:,:]
    logL = (self.forward_logL )[:,:,None,None]
    logXi =(A + B + log_bjk + logT - logL)
#     print 'logXi is:\n',logXi
    xi = np.exp(logXi)
    gamma = xi.sum(axis = 3,keepdims = 1)
    gammaSum = gamma.sum(axis = 1,keepdims =1 ) 

    if debug:
        print 'Forward logL is:\n',self.forward_logL
        print 'Backward logL is:\n',self.backward_logL
        print 'Gamma Shape is:\n',gamma.shape
        print 'Gamma Sum is:\n',gamma.sum(axis = 2,keepdims = 1)[:,:3]

        print echains.shape
        print logL.shape
        print bjk.shape
        print 'Shape of b_{s_j}(Y_{m+1})\n',bjk.shape
        print 'Shape of alpha\n',A.shape
        print 'Shape of beta\n',B.shape
        print 'Shape of transition\n',logT.shape
        print 'Shape of Xi is:',xi.shape
    A_hat  = xi.sum(axis = 1,keepdims =1)/gammaSum
    mu_hat = gamma[:,0:1,:,:]
#     mu_hat = mu_hat / mu_hat.sum(axis = 2,keepdims = 1)
    
#     print "Sum of Xi_mat is:",gammaSum.sum(axis = (2,))
#     print A_hat.shape
#     print gamma.shape,mu_hat.shape
#     print echains.shape
    outprod = np.equal(echains[:,:-1,None,None], np.arange(self.model.emission_space).reshape((1,1,1,-1)))
    raw_bjk = (outprod * gamma).sum(axis = 1,keepdims = 1)
    emission_mat_hat = raw_bjk/gammaSum
    
    if debug:
        print "Shape of outer product is",outprod.shape
        print "Sum of Xi_mat is:",xi.sum(axis = (3,2))
        print "Shape of gamma is",gamma.shape,mu_hat.shape

        print emission_mat_hat.shape
        print A_hat
    #     print gammaSum
        print 'Mu hat is:\n',mu_hat.shape,mu_hat
        print mu_hat.sum(axis  = -1)
        print emission_mat_hat.shape
#     print outprod.shape
#     muhat = 
#     xi =  + B + np.log(self.model.transition) 
    Nc = len(echains)
    A_hat = A_hat[:,0,:,:]
    mu_hat = mu_hat[:,0,:,:]
    emission_mat_hat = emission_mat_hat[:,0,:,:]
    return (A_hat, mu_hat,emission_mat_hat)
    pass
chains.MLE_BW = MLE_BW
if __name__=='__main__':
    mdl = util.Var_hmm.read_model(fname = 'Q3.hmm',
           emission_list = [str(x) for x in range(1,6)])
    chs = chains(fname = 'Q3.echain',model = mdl)
    chs.echains = chs.echains[0:1,:]
    print chs.echains.shape
    # chs.MLE_BW()
    [x.shape for x in chs.MLE_BW()]

def single_chain_BW(
    chs,
    cidx = 0,
    iter_min = 5,
    iter_max = 200,
    thres = 0.005,
    print_pd = 10,
    debug = 0):
    logLs = []
    all_chains = chs.echains
    chs.echains = all_chains[cidx:cidx+1,:]
#     print all_chains.shape
#     print chs.echains.shape
    for i in range(iter_max):
        mle = chs.MLE_BW(debug = 0)
    #     print mle
        if debug:
            print( [x.shape for x in mle ])
        chs.model.transition = mle[0][0]
        chs.model.initial = mle[1][0]
        chs.model.emission_mat = mle[2][0]
        logL = chs.forward_logL

        if not i %  print_pd:
            print "Iteration %d,\t log-likelihood is:"%i,logL
        if i > iter_min:
            dL = abs(logL - logLs[-1])
    #         mdev = np.std(logLs[-iter_min:])
    #         print mdev
    #             print dL
#             print dL
            if dL < thres:
                print "The change in logL is less than threshold: %f"%thres
                break
        logLs.append(logL.tolist()[0][0])
    chs.logLs  = np.array([logLs])
    chs.echains = all_chains
    return chs
chains.single_chain_BW = single_chain_BW
