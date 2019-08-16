import numpy as np
import pandas as pd
import argparse, math, sys
import scipy
import scipy.stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# main function
def main():

    # simulation parameters
    nsim = 1000
    nsnp = 50000
    prop_neut = 0.90
    prop_sel = 1.0 - prop_neut

    ne1 = 14269
    ne2 = 11418

    # eyre-walker parameters
    delta_prob = 0.5
    S_weak = 1e-5
    S_strong = 1e-4
    t1 = float(sys.argv[1])
    t2 = float(sys.argv[1])
    sigmasq_eps = 1.0
    
    # extended eyre-walker parameters
    all_sigmasq_delta = (np.arange(0.0,11)/10).tolist()

    # record results
    print 'VAR', 'QUINTILE', 'GCOR_MEAN', 'GCOR_SD', 'HSQ1_MEAN', 'HSQ1_SD', 'HSQ2_MEAN', 'HSQ2_SD'
    for sigmasq_delta in all_sigmasq_delta:
       
        gw_result = []
        qt_result = []
        qt_hsq1_result = []
        qt_hsq2_result = []
        for i in range(nsim):
            
            # draw sign of effect
            delta = 2*np.random.binomial(1, delta_prob, size=nsnp)-1
            
            # draw selection coefficients of ancestral populations
            neut_sel = np.random.binomial(1, prop_sel, size=nsnp)
            S0 = np.zeros(nsnp)
            S0[neut_sel == 0] = S_weak
            S0[neut_sel == 1] = S_strong
            nsel = np.sum(neut_sel)

            # draw selection coefficient of descendent population
            S1 = S0.copy(); S2 = S0.copy()
            S1[neut_sel==1] += S0[neut_sel==1]*np.random.normal(
                scale=np.sqrt(sigmasq_delta), size=nsel)
            S2[neut_sel==1] += S0[neut_sel==1]*np.random.normal(
                scale=np.sqrt(sigmasq_delta), size=nsel)

            # make sure everything is non negative
            S1[S1<0] = 0.0; S2[S2<0] = 0.0
            
            # draw random noise
            eps = np.random.normal(scale=np.sqrt(sigmasq_eps), size=nsnp)
            
            # draw effect
            b1 = delta*np.power(4*ne1*S1, t1)*(1.0 + eps)
            b2 = delta*np.power(4*ne2*S2, t2)*(1.0 + eps)

            # get genome-wide gcor and heritability
            gw_gcor = np.corrcoef(b1, b2)[0,1]
            
            gw_hsq1 = np.sum(np.square(b1)) / nsnp
            gw_hsq2 = np.sum(np.square(b2)) / nsnp
            
            # get stratified gcor and heritability
            nbins = 2
            cut = pd.cut(S0, nbins, retbins=True)[1]
            qt_gcor_en = []
            qt_hsq1_en = []
            qt_hsq2_en = []
            for j in range(nbins):
                
                low = cut[j]
                high = cut[j+1]
                idx = np.where((S0>low) & (S0<=high))[0]

                tmp = np.square(np.corrcoef(b1[idx], b2[idx])[0,1]/gw_gcor)
                qt_gcor_en.append(tmp)
                
                tmp = np.sum(np.square(b1[idx])/idx.shape[0]/gw_hsq1)
                qt_hsq1_en.append(tmp)
                
                tmp = np.sum(np.square(b2[idx])/idx.shape[0]/gw_hsq2)
                qt_hsq2_en.append(tmp)

            gw_result.append(gw_gcor)
            qt_result.append(qt_gcor_en)
            qt_hsq1_result.append(qt_hsq1_en)
            qt_hsq2_result.append(qt_hsq2_en)

        qt_result = np.array(qt_result)
        qt_hsq1_result = np.array(qt_hsq1_result)
        qt_hsq2_result = np.array(qt_hsq2_result)

        # compute mean and output
        for j in range(nbins):
            gcor_mu,gcor_sd = np.mean(qt_result[:,j]),np.std(qt_result[:,j])
            hsq1_mu = np.mean(qt_hsq1_result[:,j])
            hsq1_sd = np.std(qt_hsq1_result[:,j])
            hsq2_mu = np.mean(qt_hsq2_result[:,j])
            hsq2_sd = np.std(qt_hsq2_result[:,j])
            print sigmasq_delta,j+1,gcor_mu,gcor_sd,hsq1_mu,hsq1_sd,hsq2_mu,hsq2_sd

if(__name__ == '__main__'):
    main()
