####### Read fasta file
import re, collections
import numpy as np
def read_fasta(fname = None,text = None):
    if fname is not None:
        text = open(fname,'r').read()
    s = text
    buf = re.split('(>.+)\n',s)
    fileheader = buf.pop(0)
    assert not len(buf) % 2, 'Some headers are not matched with the sequences'
    buf = [x.replace('\n','') for x in buf]
    buf = zip(buf[::2],buf[1::2])
#    seq =  buf[0][1].replace('\n','')
    return buf

def to_chunks(lst, wid, stride = None,debug = 0):
    if stride is None:
        stride = wid
    L = len(lst)
    idx = range(0,L-( wid - 1), stride)
    if debug:
        print "Creating %d chunks of length %d from %d elements" % (len(idx),wid,L)
    return zip(*[ (float(i+i+wid)/2,lst[i:i+wid]) for i in idx ])

def CG_density(x):
    CG = 0
    ALL = 0
    for k,v in collections.Counter(x).items():
        if k in ['C','G']:
            CG += v
        ALL += v
    return float(CG)/ALL

if __name__ == '__main__':
    seqs = read_fasta('Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.III.fa')
    mids,chunks = to_chunks(seqs[0][1],100,100)
    CG_densi = [CG_density(x) for x in chunks]

    
##### Use the obtained borders to encode a sequences of density
scheme1 = np.array([0.355, 0.415, 0.475, 0.575])
scheme2  = np.array([0.26,  0.355, 0.415, 0.475])
def construct_LR(egs):
    legs = np.array([-0.01] + egs.tolist())[None,:]
    regs = np.array(egs.tolist() +  [1.01])[None,:]
    return {'legs':legs,'regs':regs}

def classify(val,legs = None, regs = None):
    '''
    Take a 1D vector and report the index of interval for each value
    legs: left edges of the interval
    regs: right edges of the interval
    '''
    #### Disable test for speed
    assert np.all(legs[1:]==regs[:-1])
    val = np.array(val).reshape((-1,1))
    BOOL = (val >= legs)  & (val < regs)
    return np.where( BOOL )[1]
if __name__=='__main__':
    egs_kw = construct_LR(egs)
    B = classify(np.linspace(0,1,50),**egs_kw)
    print B

#len(edges)