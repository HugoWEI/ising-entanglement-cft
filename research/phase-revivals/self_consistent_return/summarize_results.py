"""Create diagnostic table and figure from reproducible per-grid capacity arrays."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from self_consistent_return import smoothed_peak
ROOT=Path(__file__).resolve().parent
pairs=[(1.2,.35),(.35,1.2),(.5,.2)];steps=[.1,.05,.025,.0125]
out=[]
for pair in pairs:
    values={}
    for h in steps:
        file=ROOT/f'curves_48_{pair[0]}_{pair[1]}_dt{h}.npz'
        if not file.exists():continue
        with np.load(file) as z:
            values[str(h)]={k:smoothed_peak(z['t'],z[k]) for k in ('base','posthoc_airy','airy','bessel','full_bessel','hermite')}
    r={'pair':pair,'step_peaks':values,'richardson':{}}
    for coarse,fine in zip(steps[:-1],steps[1:]):
        if str(coarse) not in values or str(fine) not in values:continue
        ra=values[str(fine)]['airy'];rb=values[str(coarse)]['airy']
        estimate=(4*ra-rb)/3
        r['richardson'][str(fine)]={'airy_peak_extrapolated':estimate,'error_vs_fine_grid_hermite':estimate-values[str(fine)]['hermite']}
    out.append(r)
(ROOT/'CONVERGENCE_RESULTS.json').write_text(json.dumps(out,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=False)
for ax,which in zip(axes,out[:2]):
    xs=[];errors=[];fb=[]
    for h in steps:
        d=which['step_peaks'].get(str(h))
        if d:
            xs.append(h);errors.append(abs(d['airy']-d['hermite']));fb.append(abs(d['airy']-d['posthoc_airy']))
    ax.loglog(xs,errors,'o-',label='Self-consistent Airy: peak error')
    ax.loglog(xs,fb,'s--',label='Return-wave feedback peak shift')
    ax.loglog(xs,np.array(xs)**2*errors[-1]/xs[-1]**2,':',label=r'$O(h^2)$ guide')
    ax.set_title(f'Defect ({which["pair"][0]}, {which["pair"][1]})')
    ax.set_xlabel('Volterra time step h')
    ax.set_ylabel('Absolute time shift')
    ax.grid(True,which='both',alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Self-consistent first return: feedback versus quadrature')
fig.tight_layout()
fig.savefig(ROOT/'first_return_convergence.png',dpi=165)
print('Saved',ROOT/'CONVERGENCE_RESULTS.json')
