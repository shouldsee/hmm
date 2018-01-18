########## Functionality test for BW-algo
##### Although there seems to be numeric instability, the c_(m+1) = d_m  is visually observable

import util
from forward import *
from bulk import chains
import matplotlib.pyplot as plt
h1 = util.Var_hmm.read_model(fname ='Q3.hmm')
chs = chains(fname = 'Q3.echain',model = h1)
i = 0
plt.figure(figsize = [12,8])
#for N in np.arange(2,100,20).astype(int):
N = 100
if 1:
    i += 1
    echain = chs.echains[0][:N]
    h1.reset()
    cs = [h1._forward(E,return_c=1) for E in echain ]
    cs = np.round(cs[:],decimals=5)
    print 'shape',cs.shape
    print 'c_m:',cs[:30]
    cs = cs[1:]
    print cs.size

#    print np.log(cs)
    h1.reset()
    ds = [h1._backward(E,return_d=1) for E in echain[::-1] ]
    ds =  np.round(ds[::-1],decimals=5)
    print 'd_m',ds[:30]
    ds = ds[:-1]
#     print(zip(cs,ds)[:10])
    C = np.corrcoef(cs,ds)
    print C[0,1]
    plt.figure(figsize = [12,5])
    plt.subplot(1,1,i)
    clab = '$c_{m+1}$'
    dlab = '$d_{m}$'
    plt.plot(cs,label = clab)
    plt.plot(ds,label = dlab)
#     plt.ylabel()
    plt.xlabel('time step')
    plt.legend()
figname = 'Q6_FBQC.pdf'
print "Saving figure to: %s"%figname
plt.savefig(figname)

###########################


import copy
mdl = util.Var_hmm.read_model(fname = 'Q3.hmm',
       emission_list = [str(x) for x in range(1,6)])

plt.figure(figsize = [12,5])
print "Scheme 1"
chs = chains(fname = 'Q3.echain',model = copy.copy(mdl))
chs.single_chain_BW(cidx = 0)
chs.model.save_model('Q6_scheme1.hmm')
plt.plot(chs.logLs[0],label = 'scheme_1')

print "Scheme 2"
chs = chains(fname = 'Q3.echain',model = copy.copy(mdl))
chs.single_chain_BW(cidx = 1)
chs.model.save_model('Q6_scheme2.hmm')
plt.plot(chs.logLs[0])
plt.plot(chs.logLs[0],label = 'scheme_2')

plt.xlabel('Iteration number')
plt.ylabel('$\log{P(Y_0^N)}$')
plt.legend()

figname = 'Q6_BWQC.pdf'
print "Saving figure to: %s"%figname
plt.savefig(figname)
