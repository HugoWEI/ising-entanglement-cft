import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with open('fourth_order_convergence.json') as f:d=json.load(f)
h=np.array(d['steps'])
fig,ax=plt.subplots(figsize=(8,5))
for orient in ['forward','reversed']:
    xx=np.array(d[orient]['source_majorant'])
    ax.loglog(h,xx,'o-',label=f'{orient}: source-only error bound')
    yy=np.array(d[orient]['measured_error_from_exact_hamiltonian'])
    ax.loglog(h,yy,'x--',label=f'{orient}: measured error (validation)')
ax.set_xlabel('Time step h')
ax.set_ylabel('Occupied-amplitude operator-norm error')
ax.set_title('Fourth-order two-site collocation: a posteriori source bounds')
ax.legend();ax.grid(True,which='both',alpha=.15);fig.tight_layout()
fig.savefig('figures/fourth_order_residual_convergence.png',dpi=180)
