#### Question 3
import util
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
h1 = util.Var_hmm(
    internal_list = [0,1],
    emission_list = [ (x) for x in [1,2,3,4,5]],
    transition = np.array([[0.8, 0.2],[0.1,0.9]]),
    emission_mat = np.array([[0.2, 0.5, 0.2, 0.1, 0],[ 0, 0.1, 0.4 ,0.4 , 0.1]])
)
n = 115
chain = h1.run_for(n,as_idx = 1)
emission_chain = h1.bulk_emit(chain, as_idx = False)

fig = plt.figure(figsize= [12,4])
plt.subplot(111)
plt.plot(chain,label = 'hidden_state')
plt.plot(emission_chain, label ='emission_state')
plt.legend(bbox_to_anchor=(1.19, 1.00))
plt.xlabel('time step')
plt.ylabel('Value')

#plt.show()

figname = __file__.rsplit('.',1)[0]+'.pdf'
print 'Saving figure to: %s' % figname
plt.savefig(figname)
#h1.bulk_emit([1,1,0,0,0],as_idx = 1)

