import numpy as np
import matplotlib.pyplot as plt
from run_report import evaluate
from pathlib import Path
HERE=Path(__file__).parent
for a,b in [(1.2,.35),(.35,1.2)]:
    d=np.load(HERE/f'airy_witness_{a}_{b}.npz')
    t=d['t'];ap=d['airy'];w=d['witness'];eps=d['eps']
    result=evaluate(a,b); pk=result['airy_peak'];radius=result['chosen_radius']['radius']
    fig,ax=plt.subplots(figsize=(9,5))
    mask=(t>=pk-2.5)&(t<=pk+2.5)
    ax.plot(t[mask],ap[mask],label='Airy first-return prediction',lw=2)
    ax.plot(t[mask],w[mask],label='Independent two-site source-only witness',ls='--',lw=1.7)
    ax.axvspan(pk-radius,pk+radius,alpha=.15,label='Conditional local peak interval')
    ax.set_title(f'L=48 — two-site defect ({a}, {b})')
    ax.set_xlabel('Time (hopping units)')
    ax.set_ylabel('Capacity of entanglement')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(HERE/'figures'/f'airy_first_peak_{a}_{b}.png',dpi=170)
    plt.close(fig)
