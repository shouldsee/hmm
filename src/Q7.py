import util, util_genome
from forward import *
import matplotlib.pyplot as plt
#%matplotlib inline

if __name__ == '__main__':
    seqs = util_genome.read_fasta('Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.III.fa')
    gr,chunks = util_genome.to_chunks(seqs[0][1],100,100)
    CG_densi = [util_genome.CG_density(x) for x in chunks]
    
if __name__=='__main__':
    print '\nViterbi MLE for scheme 1'
#     data = {'Q6_scheme1.hmm':None,'Q6_scheme2.hmm':None}
    h1 = util.Var_hmm.read_model(fname ='Q6_scheme1.hmm')
    echain = util.read_chain('Q3.echain',dtype=int)[0]
    mle_seq1 = h1.MLE_viterbi(echain,debug = 1)

    print '\nViterbi MLE for scheme 2'
    h2 = util.Var_hmm.read_model(fname ='Q6_scheme2.hmm')
    echain = util.read_chain('Q3.echain',dtype=int)[1]
    mle_seq2 = h2.MLE_viterbi(echain,debug = 1)
    gr = np.array(gr)/1000.

    plt.figure(figsize = [10,8])
    plt.subplot(211)
#     plt.figure(figsize = [10,4])
    plt.plot(gr,mle_seq1,  label = 'internal_state_scheme1')
    plt.plot(gr,mle_seq2/2.,  label = 'internal_state_scheme2')
    plt.plot(gr,CG_densi,label = 'local_CG_density')
    plt.xlim(0,5E1)
    plt.xlabel('Genomic Position (Kbp)')
    plt.legend()
#    plt.savefig('Q7_50K.pdf')

#     plt.figure(figsize = [10,4])
    plt.subplot(212)
    plt.plot(gr,mle_seq1,  label = 'internal_state_scheme1')
    plt.plot(gr,mle_seq2/2.,  label = 'internal_state_scheme2')
    plt.plot(gr,CG_densi,label = 'local_CG_density')
    plt.xlim(0,10)
    plt.xlabel('Genomic Position (Kbp)')
    plt.legend()
    plt.savefig('Q7.pdf')
#     plt.savefig('Q7')
