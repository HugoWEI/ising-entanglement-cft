"""Build reproducible diagnostics (floating point, not interval arithmetic)."""
import os,sys,json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
from defect_residual import diagnostic,defect_drive_response,hermite_residual_integral_majorant,clean_modes
from airy_error_budget import run as airy_test
from published_baseline import capizzi_eisler_capacity,mode_capacity
import volterra_scattering_revival as vol
from phase_revival_research import hopping

ROOT=os.path.dirname(__file__)
def main():
    rows=[]
    for pair in [(1.2,.35),(.35,1.2)]:
        for dt in [.2,.1,.05,.025]:
            raw,t,_,_=diagnostic(dt=dt,pair=pair)
            L=48; _,v=np.linalg.eigh(hopping(L)+np.diag(2.5*(-1.)**np.arange(L)));Q=v[:,:L//2]
            Z,Y=vol.defect_response(L,t,*pair,Q,1,return_full=True,return_defect=True)
            F,r=defect_drive_response(L,t,Q,pair,Y)
            raw['continuous_hermite_majorant_evaluated_in_float']=hermite_residual_integral_majorant(L,t,Q,pair,Y,F)
            rows.append(raw)
            print('residual',pair,dt,raw['continuous_hermite_majorant_evaluated_in_float'],flush=True)
    airy=[]
    for pair in [(1.2,.35),(.35,1.2)]:
        out,*_=airy_test(pair=pair,dt=.1,T=30)
        airy.append(out)
    T=38.4;L=48;m=2.5;t=np.arange(0,T+1e-6,.2)
    _,q=np.linalg.eigh(hopping(L)+np.diag(m*(-1.)**np.arange(L)));Q=q[:,:L//2]
    e,S=clean_modes(L);D=S.T@Q
    cap=[]
    ent=[]
    for ti in t:
        F=S@(np.exp(-1j*e*ti)[:,None]*D)
        p=np.clip(np.linalg.eigvalsh(F[:L//2]@F[:L//2].conj().T),1e-14,1-1e-14)
        cap.append(mode_capacity(p).sum())
        ent.append(-(p*np.log(p)+(1-p)*np.log1p(-p)).sum())
    cap=np.array(cap);ent=np.array(ent)
    ce=capizzi_eisler_capacity(L,m,t,lam=1)
    pub=dict(L=48,mass=2.5,lambda_conformal=1,
             capacity_rmse=float(np.sqrt(np.mean((cap-cap[0]-ce)**2))),
             capacity_max_abs=float(np.max(abs(cap-cap[0]-ce))),
             note='Conformal homogeneous validation only; NOT onsite-defect predictor.')
    js={'residual':rows,'airy_components':airy,'published_baseline':pub}
    with open(os.path.join(ROOT,'diagnostics.json'),'w') as f:json.dump(js,f,indent=2)
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    for pair in [(1.2,.35),(.35,1.2)]:
        r=[x for x in rows if tuple(x['pair'])==pair]
        label=f'(u,v)=({pair[0]:g}, {pair[1]:g})'
        ax.loglog([x['dt'] for x in r],[x['continuous_hermite_majorant_evaluated_in_float'] for x in r],'-o',label=label+' Hermite majorant')
        ax.loglog([x['dt'] for x in r],[x['source_only_bound_diagnostic'] for x in r],'--x',label=label+' sampled integral')
    ax.set_xlabel('Time step ∆t');ax.set_ylabel('Upper-bound scale (operator norm)')
    ax.set_title('Two-site source residual: second-order convergence')
    ax.grid(True,alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(os.path.join(ROOT,'figures','source_residual_convergence.png'),dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    ax.plot(t,cap-cap[0],label='Homogeneous chain, exact Gaussian capacity change')
    ax.plot(t,ce,'--',label='Capizzi–Eisler Eq. 64 differentiated at n=1')
    ax.set_title('Published coherent-revival baseline, applicable conformal limit')
    ax.set_xlabel('Time (our hopping units)');ax.set_ylabel('Capacity change')
    ax.grid(True,alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(os.path.join(ROOT,'figures','published_capacity_baseline.png'),dpi=170);plt.close(fig)
    print(json.dumps({'published':pub,'airy':airy},indent=2))
if __name__=='__main__':main()
