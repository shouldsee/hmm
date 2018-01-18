import util
from forward import *
import warnings

if __name__=='__main__':
    h1 = util.Var_hmm.read_model("Q3.hmm")
    echain = util.read_chain('test.echain',dtype = 'int')
    #print "Calculating likelihood for chain:",echain
    _ = h1.emission_likelihood(echain,debug=1)
