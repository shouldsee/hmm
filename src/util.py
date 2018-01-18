import numpy as np
import collections, functools
import re

'''
### To-Do:
1. write parser for transition matrix and initial distribution
'''

def mat2str(mat):
    mat = mat.astype('str')
    lst = mat.tolist()
    if mat.ndim == 1:
        lst = [lst]
    elif mat.ndim == 2:
        pass
    s = '\n'.join('\t'.join(x) for x in lst )
    return s


def state2dist(x,state_space = None):
    '''
    x is a list of states to be converted into a distribution
    '''
    if state_space is None:
        state_space = max(x)+1
    if isinstance(x,np.ndarray):
        x = x.tolist()
    ct = collections.Counter(x)
#    ct.update()
    lst = [0]*state_space
    for k,v in ct.items():
        lst[k] = v
    final_dist = np.array(lst,dtype='float')
    final_dist = final_dist/final_dist.sum()
#    print final_dist
    return final_dist
print state2dist(np.random.randint(0,5,100))



def check_transition(mat,epsilon = 1E-10):
    #print mat
    assert np.all(abs(mat.sum(axis = 1) - 1) < epsilon)
    
    

def find_stationary(b):
    w,v = np.linalg.eig(b.T)
    lv = v.T
    same_sign = (lv > 0).all(axis = 1) | (lv < 0).all(axis = 1)
    stat_idx, = np.where( same_sign & (w > 0 ))
    stat_vct = lv.take(stat_idx,axis = 0)
    stat_vct = stat_vct / stat_vct.sum()
    return stat_vct



######## We start with a discrete version of hmm
class Var_hmm(object):
    def __init__(self,
                 fname = None,
                 transition = np.array([[0.1,.9],[.2,.8]]),
                 emission_mat = None,
                 internal_initial = np.array([[.8, .2]]),
                 t = -1, 
                 internal_list = None,
                 emission_list = None,
                **kwargs):
        if fname is not None:
            self = self.read_model(fname = fname)
            return 
        if internal_list is None:
            internal_list = list(range(transition.shape[0]))
        if emission_list is None:
            if emission_mat is not None:
                emission_list = list(range(emission_mat.shape[1]))

        self.internal_space = len(internal_list)
        if emission_list is not None:
            self.emission_space = len(emission_list)

        self.transition = np.array(transition)
        check_transition(self.transition)
        assert self.transition.shape   == (self.internal_space, self.internal_space)
        if emission_mat is not None:
            emission_mat = np.array(emission_mat)
            assert emission_mat.shape == (self.internal_space, self.emission_space),'%s %s'%(emission_mat.shape,[self.internal_space, self.emission_space])
            check_transition(emission_mat)
        self.emission_mat = emission_mat
        self.internal_initial = np.array(internal_initial)
        assert self.internal_initial.size == self.internal_space
        #self.emission_initial = None
        #self.alpha_initial = None
#        self.nu_initial = None
#        self.internal_dict = 
        self.internal_list = internal_list
        self.emission_list = emission_list
        self.emission_mat  = emission_mat
        self.t = t
        self.internal = None
        self.emssion  = None
#        self.sample()
        pass
    @classmethod
    def read_model(cls, fname = None, text = None, debug = False,
                  **kwargs):
        if fname is not None:
            text = open(fname,'r').read()
        #buf = []
        buf = re.split('([a-zA-Z].+)\n',text.rstrip('\n'))
        header = buf.pop(0)
        buf = dict(zip(buf[::2],buf[1::2]))
        for k,v in buf.iteritems():
            buf[k] = np.vstack(read_chain(text = v, dtype ='float',debug = 0))
    #        if k == 'TRANSITION':
    #            read_chain 
    #            pass
    #        elif k=='INITIAL':
    #            ini_db = read_chain(text = v)
    #        pass
        buf['internal_initial'] = buf.pop('INITIAL')
        buf['transition'] = buf.pop('TRANSITION')
        if 'EMISSION' in buf.keys(): 
            buf['emission_mat'] = buf.pop('EMISSION')
        kwargs.update(buf)
        obj = cls(**kwargs)
        return obj

    def save_model(self,fname = 'test.hmm'):
        buf = {'TRANSITION':self.transition,
              'INITIAL': self.internal_initial}
        if self.emission_mat is not None:
            buf.update( {'EMISSION':self.emission_mat } )
        kv = sorted(buf.items(),key = lambda x:x[0])
        text = '\n'.join('%s\n%s' %( k, mat2str(v)) for k,v in kv)
        print >>open(fname,'w'),text
        return text    
    def __unicode__(self):
        s = '\n'.join( str(x) for x in [
            '--------',
            'Internal state space size:%s '%self.internal_space,
            'Emission state space size:%s '%self.emission_space,'',
            'Initial distribution of internal states is:',self.internal_initial,'',
            'Transition matrix is:',self.transition,'',
            'Emission matrix is:',self.emission_mat,
            '--------',
        ])
        return s
    def __str__(self):
        return unicode(self).encode('utf-8')
    def sample(self, as_idx = False,
               debug = False,
               hold = False,
               **kwargs):
        if self.t == -1:
#        if self.internal is None:
            distrib = self.internal_initial
        else:
            distrib = self.transition[self.internal,:]
        distrib = distrib.ravel()
        if debug:
            print distrib
        
        internal = np.random.choice( self.internal_space, size = 1, p = distrib)[0]
        self.internal = internal
        if not hold: 
            self.t += 1
        if not as_idx:
            internal = self.internal_list[internal]
        return internal
    def run_for(self, T = 10,debug = False,**kwargs):
#        chain = [ self.sample() for i in range(T)]
 #       return chain 
  #      chain = [None]*T
        chain = np.zeros([T],dtype=int)
        
        for i in range(T):
            chain[i] = self.sample(debug = debug ,**kwargs)
#            chain[i] += [h1.sample()]
        return(chain)

    def _emit(self, internal, as_idx = False):
        distrib = self.emission_mat[ internal,:]
        distrib = distrib.ravel()
        emission = np.random.choice( self.emission_space, size = 1, p = distrib)[0]
        if not as_idx:
            emission = self.emission_list[emission]
        return emission
        
    def emit(self, as_idx = False):
        emission = self._emit(self.internal, as_idx = as_idx)
        self.emission = emission
        return self.emission
    def bulk_emit(self,chain,as_idx = False):
        echain = map(functools.partial(self._emit, as_idx = as_idx), 
                     chain)
        return echain
    def find_stationary(self):
        return find_stationary(self.transition)
    def reset(self, t = -1 ):
        self.t = t
        self.last_emission = None
        
def cross_entropy(y, t, epsilon = 1E-16):
    return - np.sum( t * np.log(y+epsilon) - t * np.log(t+epsilon),axis = -1)

def MAE(y,t):
    return abs(y - t).sum(axis = -1)

if __name__=='__main__':
    h1 = Var_hmm()
    ch = h1.run_for(50)
    #ch = h1.run_for(100,debug = 1)
    print h1.emit()
    print  ch 
    print h1.find_stationary()

    print "Done"
    

def read_chain(fname=None,text = None,dtype = 'float',debug = True):
    '''
    Read in a tab-delimited file, return a list of chains, as separated by line.
    Use dtype='int' to prepare chain as index
    '''
    if fname is not None:
        text = open(fname,'r').read()
    buf = []
    for line in text.splitlines():
        lst = line.split('\t')
#        if len(lst)<=1:
        lst = [ float(x.strip()) for x in lst if x]
        if lst:
            buf.append( lst )
    c = np.array(buf,dtype = dtype)
    if debug:
        print "Reading chains from file '%s' :\n" % fname, c 
    return  c

buf = '''
1	1	0	1	1	0	1	0	1	1	1	1	1	1	0	0	1	0	0	0	0	0	1	1	0	1	1	0	0	0	0	
'''

if __name__=='__main__':
    chain = read_chain(text = buf,dtype = 'int',debug = 1)[0]
    
    
def chain2bed(echain, gr,  fname = 'test.bed',chrom = None):
    gr = np.array(gr,dtype='int')
    L_gr = len(gr)
    assert( len(echain) in [L_gr,L_gr +1])
    if len(echain) == L_gr:
        rad = (gr[1] - gr[0])//2
        l = gr - rad
        r = gr + rad
        lst = zip([chrom]*len(l), l.tolist(),r.tolist(),echain.tolist())
        lines = ('\t'.join( str(x) for x in e ) for e  in lst)
    
    with open(fname,'w') as f:
        for line in lines:
            print>>f,line
    #        print line
    print "Saved track to file:%s" % fname