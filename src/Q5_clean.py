
# coding: utf-8

# In[1]:

import util
from forward import *
from util_genome import *
import collections, functools
import numpy as np

# In[2]:

# In[48]:

# In[3]:

if __name__ == '__main__':
    seqs = read_fasta('Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.III.fa')
    seq = seqs[0][1]
    gr,chunks = to_chunks(seq,100,100,debug = 1)
    CG_densi = [CG_density(x) for x in chunks]
    print len(seq)

# In[10]:

import matplotlib.pyplot as plt

plt.figure(figsize = [14,10])
plt.subplot(211)
plt.plot(np.array(gr)/1000, CG_densi)
plt.xlabel('Genomic coordinate (Kbp)')
plt.ylabel('Windowed CG density (%)')

plt.title('Choice of binning scheme')
plt.subplot(212)
# plt.figure()
#which()
ct,edges,_ = plt.hist(CG_densi,bins = np.linspace(0,1,101),normed=1)
mids = (edges[:-1] + edges[1:])/2
egs = mids[np.where( ct == 0 )]
egs = egs[(egs > 0.3) & (egs < 0.6)]
scheme1 = egs.copy()
scheme2 = np.array([0.26] + egs[:-1].tolist() )
plt.vlines(scheme1, 0,10,'g',label = 'scheme1')
plt.vlines(scheme2 - 0.0005,0,7,'r',label = 'scheme2')
plt.ylabel('Normalised count')
plt.xlabel('CG density (%)')
plt.xlim(0,0.7)
plt.legend()
print 'Scheme 1:'
print "The border of bins are:", scheme1

print 'Scheme 2:'
print "The border of bins are:", scheme2

figname = 'Q5_fig1.pdf'
print "Saving figure to: %s"%figname
plt.savefig(figname)

# In[50]:

plt.figure(figsize = [6,4])
coded_CG_densi = classify(CG_densi,**construct_LR(scheme1))
_ = plt.hist(coded_CG_densi,alpha = 0.3, bins = np.arange(-0.5,5.5,0.25),label = 'scheme1')
coded_CG_densi = classify(CG_densi,**construct_LR(scheme2))
_ = plt.hist(coded_CG_densi,alpha = 0.3,bins = np.arange(-0.5,5.5,0.25),label ='scheme2')
plt.legend()
plt.ylabel('Count of emission')
plt.xlabel('Emission index')

figname = 'Q5_fig2.pdf'
print "Saving figure to: %s"%figname
plt.savefig(figname)

# In[9]:

mdl = util.Var_hmm.read_model(fname = 'Q3.hmm',
           emission_list = [str(x) for x in range(1,6)])
print mdl

lst = []
print '\nScheme 1'
coded_CG_densi = classify(CG_densi,**construct_LR(scheme1))
mdl.emission_likelihood([coded_CG_densi],debug = 1)
lst.append(coded_CG_densi)

print '\nScheme 2'
coded_CG_densi = classify(CG_densi,**construct_LR(scheme2))
_ = mdl.emission_likelihood([coded_CG_densi],debug = 1)
lst.append(coded_CG_densi)

print >>open('Q3.echain','w'),util.mat2str(np.vstack(lst))
#mdl.unicode()

