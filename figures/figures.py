### Figures

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# formatting parameters
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# ---------- Phonetic Clustering Score ----------
def plot_pcs(pcs_H_data_bsa, pcs_HS_data_bsa, pcs_HR_data_bsa, pcs_J_data_bsa, pcs_JFL_data_bsa, pcs_stats):
    # merge dataframes
    pcs_H_data_bsa.rename(columns={'pcs': 'pcs_H'}, inplace=True)
    pcs_HS_data_bsa.rename(columns={'pcs': 'pcs_HS'}, inplace=True)
    pcs_HR_data_bsa.rename(columns={'pcs': 'pcs_HR'}, inplace=True)

    dfH = pd.merge(pcs_H_data_bsa, 
                    pd.merge(pcs_HS_data_bsa, pcs_HR_data_bsa, 
                             how='outer', on=['subject', 'experiment']),
                    how='outer', on=['subject', 'experiment'])

    dfJ = pd.merge(pcs_J_data_bsa, pcs_JFL_data_bsa, how='outer', 
                  on=['subject', 'experiment'], suffixes=('_J', '_JFL'))

    # re-organize dataframes
    dfmH = pd.melt(dfH, id_vars=['subject', 'experiment'], value_vars=['pcs_H', 'pcs_HS', 'pcs_HR'],
                   var_name='metric', value_name='pcs')

    dfmJ = pd.melt(dfJ, id_vars=['subject', 'experiment'], value_vars=['pcs_J', 'pcs_JFL'], 
                   var_name='metric', value_name='pcs')

    # statistics
    p = {'H': 0, 'HS': 0, 'HR': 0, 'J': 1, 'JFL': 1}
    i = {'H': 0, 'HS': 1, 'HR': 2, 'J': 0, 'JFL': 1}
    h = {'H': 0.024, 'HS': 0.024, 'HR': 0.024, 'J': 0.524, 'JFL': 0.524}
    pcs_stats['plt'] = [p.get(row.metric) for _, row in pcs_stats.iterrows()]
    pcs_stats['idx'] = [i.get(row.metric) for _, row in pcs_stats.iterrows()]
    pcs_stats['h'] = [h.get(row.metric) for _, row in pcs_stats.iterrows()]


    fig, ax = plt.subplots(1, 2, figsize=(5.5, 4), width_ratios=[3, 2])

    sns.pointplot(dfmH, x='metric', y='pcs', order=['pcs_H', 'pcs_HS', 'pcs_HR'],
                  hue='metric',  palette='Blues_d', errorbar=('se', 1.96), ax=ax[0])
    ax[0].axhline(0, color='black', linestyle='dotted')

    sns.pointplot(dfmJ, x='metric', y='pcs', order=['pcs_J', 'pcs_JFL'],
                  hue='metric', palette='Greens_d', errorbar=('se', 1.96), ax=ax[1])
    ax[1].axhline(0.5, color='black', linestyle='dotted')

    # statistical significance
    for _, row in pcs_stats.iterrows():
        if row.p_val_fdr < 0.001:
            ax[row.plt].annotate('***', (row.idx, row.h), ha='center', fontsize=14)
        elif row.p_val_fdr < 0.01:
            ax[row.plt].annotate('**', (row.idx, row.h), ha='center', fontsize=14)
        elif row.p_val_fdr < 0.05:
            ax[row.plt].annotate('*', (row.idx, row.h), ha='center', fontsize=14)

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='', ylim=(-0.01, 0.027))
    ax[0].set_xticks(np.arange(3), labels=['A/R', 'Alliteration', 'Rhyme'])
    #ax[0].set_xticks(np.arange(3), labels=['Start/Rhyme', 'Start', 'Rhyme'])

    ax[1].spines[['left', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='', ylim=(0.49, 0.527))
    ax[1].set_xticks(np.arange(2), labels=['Jaccard', 'J A/R'])
    ax[1].yaxis.tick_right()

    fig.supxlabel('Similarity Metric', x=0.6)
    fig.supylabel('Phonological Clustering Score', y=0.55)
    plt.tight_layout(w_pad=3)
    plt.savefig('figures/gallery/pcs.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Correlation of Phonetic Clustering Scores ----------
def plot_pcs_correlations(pcs_corrs_bsa):

    fix, ax = plt.subplots(figsize=(3, 4))

    dfm = pd.melt(pcs_corrs_bsa, id_vars=['subject', 'experiment'], value_vars=['r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL'], 
                  var_name='comparison', value_name='r_pcs')

    sns.barplot(dfm, x='comparison', y='r_pcs', order=['r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL'], 
                hue='comparison', palette='Greys', errorbar=('se', 1.96), ax=ax)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Metric Comparison', ylabel='Pearson R Correlation')
    ax.set_xticks(np.arange(3), labels=['A/R - J', 'A/R - J A/R', 'J - J A/R'])

    plt.savefig('figures/gallery/pcs_corrs.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Temporal and Semantic Clustering Scores ----------
def plot_tcs_scs(tcs_data_bsa, scs_data_bsa):
    fig, ax = plt.subplots(1, 2, figsize=(4, 3))

    sns.pointplot(tcs_data_bsa, y='tcs', color='darkorange', errorbar=('se', 1.96), ax=ax[0])
    ax[0].axhline(0.5, color='black', linestyle='dotted')


    sns.pointplot(scs_data_bsa, y='scs', color='purple', errorbar=('se', 1.96), ax=ax[1])
    ax[1].axhline(0.5, color='black', linestyle='dotted')


    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(ylabel='Temporal Clustering Score', ylim=(0.48, 0.78))

    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(ylabel='Semantic Clustering Score', ylim=(0.495, 0.565))

    plt.tight_layout(w_pad=3)
    plt.savefig('figures/gallery/tcs_scs.pdf', bbox_inches='tight')
    plt.show()
    

# ---------- Phonetic Intrusions ----------
def plot_psim_intr_l(psim_intr_l_data_bsa, metric):
    colors = np.load('figures/palette/blue_green.npy')
    if metric == 'H':
        palette1 = [colors[0], 'lightgray', colors[1], 'darkgray']
        palette2 ='Blues_d'
        yb1 = 0.06; yt1 = 0.12
        yb2 = -0.005; yt2 = 0.03
    elif metric == 'J':
        palette1 = [colors[2], 'lightgray', colors[3], 'darkgray']
        palette2 = 'Greens_d'
        yb1 = 0.1; yt1 = 0.115
        yb2 = -0.001; yt2 = 0.012
    else:
        raise ValueError(f'{metric} not a valid metric')
        
    # re-organize dataframe
    dfm = pd.melt(psim_intr_l_data_bsa, id_vars=['subject', 'experiment'],
                  value_vars=['pli_psim', 'pli_control', 'pli_delta', 'eli_psim', 'eli_control', 'eli_delta'],
                  var_name='intr_type', value_name='psim')

    fig, ax = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[2, 1])

    # raw scores
    sns.barplot(dfm, x='experiment', y='psim', hue='intr_type', hue_order=['pli_psim', 'pli_control', 'eli_psim', 'eli_control'], palette=palette1, 
                errorbar=('se', 1.96), gap=0.1, ax=ax[0])

    # deltas
    sns.pointplot(dfm, x='experiment', y='psim', hue='intr_type', hue_order=['pli_delta', 'eli_delta'], palette=palette2, 
                  errorbar=('se', 1.96), dodge=0.6, linestyle='none',  ax=ax[1])
    ax[1].axhline(0, color='black', linestyle='dotted')

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='Mean Phonological Similarity', ylim=(yb1, yt1))
    ax[0].set_xticks([])

    labels=['PLI', 'PLI Control', 'ELI', 'ELI Control']
    handles, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels, title='', shadow=True, ncols=2, loc='upper center', bbox_to_anchor=(0.5, 1.05))

    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='Difference in Phonological Similarity\n(Observed - Control)', ylim=(yb2, yt2))
    ax[1].set_xticks([])

    labels=['PLI Delta', 'ELI Delta']
    handles, _ = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, title='', shadow=True, ncols=1, loc='upper center', bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.savefig(f'figures/gallery/psim_intr_l_{metric}.pdf', bbox_inches='tight')
    plt.show()
    

def plot_psim_intr_r(psim_intr_r_data_bsa, metric):
    colors = np.load('figures/palette/blue_green.npy')
    if metric == 'H':
        palette1 = ['silver', 'black', colors[0], colors[1]]
        palette2 = ['black', colors[0], colors[1]]
        yb1 = 0.06; yt1 = 0.15
        yb2 = -0.01; yt2 = 0.05
    elif metric == 'J':
        palette1 = ['silver', 'black', colors[2], colors[3]]
        palette2 = ['black', colors[2], colors[3]]
        yb1 = 0.1; yt1 = 0.125
        yb2 = -0.01; yt2 = 0.02
    else:
        raise ValueError(f'{metric} not a valid metric')
        
    fig, ax = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[4, 3])
    
    sns.barplot(psim_intr_r_data_bsa, x='experiment', y='psim', hue='resp_type', hue_order=['control', 'cr', 'pli', 'eli'], 
                palette=palette1, errorbar=('se', 1.96), gap=0.1, ax=ax[0])
    
    sns.pointplot(psim_intr_r_data_bsa.query("resp_type != 'control'"), x='experiment', y='delta', hue='resp_type', hue_order=['cr', 'pli', 'eli'], 
                  palette=palette2, errorbar=('se', 1.96), dodge=0.4, linestyle='none', ax=ax[1])
    ax[1].axhline(0, color='black', linestyle='dotted')
    
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='Phonological Similarity', ylim=(yb1, yt1))
    ax[0].set_xticks([])
    
    labels = ['Control', 'Correct Recall', 'PLI', 'ELI']
    handles, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels, title='', shadow=True, ncols=2, loc='upper center', bbox_to_anchor=(0.5, 1.05))
    
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='Difference in Phonological Similarity\n(Observed - Control)', ylim=(yb2, yt2))
    ax[1].set_xticks([])
    
    labels=['CR Delta', 'PLI Delta', 'ELI Delta']
    handles, _ = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, title='', shadow=True, ncols=1, loc='upper center', bbox_to_anchor=(0.5, 1.05))
    
    plt.tight_layout()
    plt.savefig(f'figures/gallery/psim_intr_r_{metric}.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Correlations with Recall Probability ----------
def plot_p_recall_correlations(p_recall_corrs_bsa):
    # re-organize dataframe
    dfm = pd.melt(p_recall_corrs_bsa, id_vars=['subject', 'experiment'], value_vars=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'],
                  var_name='label', value_name='correlation')
    
    fig, ax = plt.subplots(figsize=(4, 5))
    
    sns.barplot(dfm, x='label', y='correlation', order=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'], 
                hue='label', hue_order=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'], palette=['steelblue', 'forestgreen', 'darkorange', 'purple'], 
                errorbar=('se', 1.96), ax=ax, alpha=0.9)
    sns.stripplot(dfm, x='label', y='correlation', order=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'], 
                  hue='label', hue_order=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'], palette=['steelblue', 'forestgreen', 'darkorange', 'purple'], 
                  ax=ax, alpha=0.3)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Clustering Score', ylabel='Correlation with Recall Probability', ylim=(-0.3, 0.48))   # omits one outlier
    ax.set_xticks(np.arange(4), labels=['PCS A/R', 'PCS J', 'TCS', 'SCS'])
    
    plt.savefig('figures/gallery/p_recall_corrs.pdf', bbox_inches='tight')
    plt.show()

def plot_cl_pr_slopes(cl_pr_H_fe, cl_pr_H_re, cl_pr_J_fe, cl_pr_J_re):
    # add random slopes to fixed effects
    m_pcs_H = cl_pr_H_fe.query("effect == 'pcs_H'").iloc[0].Estimate
    m_scs_H = cl_pr_H_fe.query("effect == 'scs'").iloc[0].Estimate
    m_tcs_H = cl_pr_H_fe.query("effect == 'tcs'").iloc[0].Estimate

    m_pcs_J = cl_pr_J_fe.query("effect == 'pcs_J'").iloc[0].Estimate
    m_scs_J = cl_pr_J_fe.query("effect == 'scs'").iloc[0].Estimate
    m_tcs_J = cl_pr_J_fe.query("effect == 'tcs'").iloc[0].Estimate
    
    df = pd.merge(cl_pr_H_re, cl_pr_J_re, on='subject', suffixes=('_H', '_J'))
    
    df['m_pcs_H'] = df['pcs_H'] + m_pcs_H
    df['m_scs_H'] = df['scs_H'] + m_scs_H
    df['m_tcs_H'] = df['tcs_H'] + m_tcs_H

    df['m_pcs_J'] = df['pcs_J'] + m_pcs_J
    df['m_scs_J'] = df['scs_J'] + m_scs_J
    df['m_tcs_J'] = df['tcs_J'] + m_tcs_J
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    print(df.m_pcs_H.max())
    # re-organize dataframe
    dfm = pd.melt(df, id_vars='subject', value_vars=['m_pcs_H', 'm_pcs_J'], var_name='metric', value_name='slope')

    sns.pointplot(dfm, x='slope', y='metric', order=['m_pcs_H', 'm_pcs_J'], hue='metric', hue_order=['m_pcs_H', 'm_pcs_J'], palette=['steelblue', 'forestgreen'], errorbar=('se', 1.96), orient='h', ms=7, ax=ax)
    sns.stripplot(dfm, x='slope', y='metric', order=['m_pcs_H', 'm_pcs_J'], hue='metric', hue_order=['m_pcs_H', 'm_pcs_J'], palette=['steelblue', 'forestgreen'], alpha=0.1, orient='h', ax=ax)
    ax.axvline(0, color='black', linestyle='dotted', alpha=0.8)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Model Slope', ylabel='', xlim=(-0.095, 0.165))      # cuts off one outlier
    ax.set_yticks(np.arange(2), labels=['PCS A/R', 'PCS J'])
    
    plt.savefig('figures/gallery/psim_pr_slopes.pdf', bbox_inches='tight')
    plt.show()


# ---------- Phonetic-CRL/IRT ----------
# loftus-masson error bars
def plot_psim_crl_lm(psim_crl_H_data_tr, psim_crl_H_data_bsa):
    # calcualte loftus-masson error bars
    psim_crl_H_lm = pd.merge(psim_crl_H_data_tr, 
                         psim_crl_H_data_bsa[['subject', 'lag', 'psim', 'crl']], 
                         on=['subject', 'lag', 'psim'], 
                         suffixes=('', '_subj_avg')).merge(psim_crl_H_data_bsa.groupby(['lag', 'psim'])['crl'].mean().reset_index(), 
                                                           on=['lag', 'psim'], suffixes=('', '_grand_avg'))
    psim_crl_H_lm['crl_lm'] = psim_crl_H_lm['crl'] - psim_crl_H_lm['crl_subj_avg'] + psim_crl_H_lm['crl_grand_avg']
    psim_crl_H_lm_bsa = psim_crl_H_lm.groupby(['subject', 'experiment', 'lag', 'psim'])['crl_lm'].mean().reset_index()
    
    lm = psim_crl_H_lm_bsa.groupby(['lag', 'psim']).agg(n=('subject', 'count'), stdev=('crl_lm', 'std')).reset_index()
    lm['sem'] = lm['stdev'] / np.sqrt(lm['n'])
    lm['ci'] = 1.96 * lm['sem']
    lm = lm.sort_values(['psim', 'lag'])
    
    crl_mu = psim_crl_H_data_bsa.groupby(['psim', 'lag'])['crl'].mean().reset_index().sort_values(['psim', 'lag']).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # psim
    ax.errorbar(x=np.arange(-7, 0), y=crl_mu.query("psim == 1 and lag >= -7 and lag < 0").crl, yerr=lm.query("psim == 1 and lag >= -7 and lag < 0").ci, marker='o', ms=5, color='steelblue')
    ax.errorbar(x=np.arange(1, 8), y=crl_mu.query("psim == 1 and lag > 0 and lag <= 7").crl, yerr=lm.query("psim == 1 and lag > 0 and lag <= 7").ci, marker='o', ms=5, color='steelblue', label='Phonologically Similar')

    # non-psim
    ax.errorbar(x=np.arange(-7, 0), y=crl_mu.query("psim == 0 and lag >= -7 and lag < 0").crl, yerr=lm.query("psim == 0 and lag >= -7 and lag < 0").ci, marker='o', ms=5, color='gray')
    ax.errorbar(x=np.arange(1, 8), y=crl_mu.query("psim == 0 and lag > 0 and lag <= 7").crl, yerr=lm.query("psim == 0 and lag > 0 and lag <= 7").ci, marker='o', ms=5, color='gray', label='Not Phonologically Similar')

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Lag', ylabel='Inter-Response Time (ms)')
    #ax.set(xlabel='Lag', ylabel='Conditional Response Latency (ms)')

    labels = ['Phonologically Similar', 'Not Phonologically Similar']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=1, loc='upper right')

    plt.savefig('figures/gallery/psim_crl.pdf', bbox_inches='tight')
    plt.show()
    
    
# loftus-masson error bars
def plot_psim_irt_lm(psim_irt_J_data_tr, psim_irt_J_data_bsa):
    # calcualte loftus-masson error bars
    psim_irt_J_lm = pd.merge(psim_irt_J_data_tr,
                             psim_irt_J_data_bsa[['subject', 'bin', 'irt']],
                             on=['subject', 'bin'],
                             suffixes=('', '_subj_avg')).merge(psim_irt_J_data_bsa.groupby('bin')['irt'].mean().reset_index(),
                                                               on='bin', suffixes=('', '_grand_avg'))
    psim_irt_J_lm['irt_lm'] = psim_irt_J_lm['irt'] - psim_irt_J_lm['irt_subj_avg'] + psim_irt_J_lm['irt_grand_avg']
    psim_irt_J_lm_bsa = psim_irt_J_lm.groupby(['subject', 'experiment', 'bin'])['irt_lm'].mean().reset_index()

    lm = psim_irt_J_lm_bsa.groupby('bin').agg(n=('subject', 'count'), stdev=('irt_lm', 'std')).reset_index()
    lm['sem'] = lm['stdev'] / np.sqrt(lm['n'])
    lm['ci'] = 1.96 * lm['sem']
    lm = lm.sort_values('bin')

    irt_mu = psim_irt_J_data_bsa.groupby('bin')['irt'].mean().reset_index().sort_values('bin').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.errorbar(x=np.arange(0, 6), y=irt_mu.irt, yerr=lm.ci, color='forestgreen', marker='o', ms=5)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Phonological Similarity Bin', ylabel='Inter-Response Time (ms)')

    plt.savefig('figures/gallery/psim_irt.pdf', bbox_inches='tight')
    plt.show()


# bin phonetic similarities
def bin_phonetic_similarities(psim_crl_v2_data_tr):
    mask = psim_crl_v2_data_tr['psim'] == 0
    # psim = 0 all in one bin
    psim_crl_v2_data_tr.loc[mask, 'bin'] = 0

    # 5 quantiles
    psim_crl_v2_data_tr.loc[~mask, 'bin'] = pd.qcut(psim_crl_v2_data_tr.loc[~mask, 'psim'], q=5, labels=False) + 1

    return psim_crl_v2_data_tr

# bins semantic similarities
def bin_semantic_similarities(df):
    # 6 bins
    df['sem_bin'] = pd.qcut(df.ssim, q=6, labels=False)
    
    return df


def plot_crl_psim_lag_H(psim_crl_H_pred):
    data = psim_crl_H_pred.groupby(['subject', 'experiment', 'session', 'abs_lag', 'psim'])[['log_crl_pred', 'crl_pred']].mean().reset_index()
    data_bsa = data.groupby(['subject', 'experiment', 'abs_lag', 'psim'])[['log_crl_pred', 'crl_pred']].mean().reset_index()
    
    psim_crl_lm = pd.merge(psim_crl_H_pred, data_bsa, 
                       on=['subject', 'abs_lag', 'psim'], 
                       suffixes=('', '_subj_avg')).merge(data_bsa.groupby(['abs_lag', 'psim'])['log_crl_pred'].mean().reset_index(), 
                                                         on=['abs_lag', 'psim'], suffixes=('', '_grand_avg'))
    psim_crl_lm['log_crl_pred_lm'] = psim_crl_lm['log_crl_pred'] - psim_crl_lm['log_crl_pred_subj_avg'] + psim_crl_lm['log_crl_pred_grand_avg']
    psim_crl_lm_bsa = psim_crl_lm.groupby(['subject', 'experiment', 'abs_lag', 'psim'])['log_crl_pred_lm'].mean().reset_index()

    lm = psim_crl_lm_bsa.groupby(['abs_lag', 'psim']).agg(n=('subject', 'count'), stdev=('log_crl_pred_lm', 'std')).reset_index()
    lm['sem'] = lm['stdev'] / np.sqrt(lm['n'])
    lm['ci'] = 1.96 * lm['sem']
    lm = lm.sort_values(['psim', 'abs_lag'])

    log_crl_mu = data_bsa.groupby(['psim', 'abs_lag'])['log_crl_pred'].mean().reset_index().sort_values(['psim', 'abs_lag']).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(4, 4))

    # psim
    ax.errorbar(x=np.arange(1, 24), y=log_crl_mu.query('psim == 1').log_crl_pred, yerr=lm.query("psim == 1").ci, color='steelblue', marker='o', ms=5, label='Phonologically Similar')

    # non-psim
    ax.errorbar(x=np.arange(1, 24), y=log_crl_mu.query('psim == 0').log_crl_pred, yerr=lm.query("psim == 0").ci, color='gray', marker='o', ms=5, label='Not Phonologically Similar')

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Absolute Lag', ylabel='Predicted ln(IRT)')
    #ax.set(xlabel='Absolute Lag', ylabel='Predicted ln(CRL)')
    ax.legend(shadow=True, loc='upper left', bbox_to_anchor=(0, 1.1))
    plt.savefig('figures/gallery/crl_psim_lag_H.pdf', bbox_inches='tight')
    plt.show()


def plot_irt_psim_lag_J(psim_irt_J_pred):
    # bin phonological similarities
    psim_irt_J_pred = bin_phonetic_similarities(psim_irt_J_pred)
    
    # calculate loftus-masson errorbars
    data = psim_irt_J_pred.groupby(['subject', 'experiment', 'session', 'abs_lag', 'bin'])[['log_irt_pred', 'irt_pred']].mean().reset_index()
    data_bsa = data.groupby(['subject', 'experiment', 'abs_lag', 'bin'])[['log_irt_pred', 'irt_pred']].mean().reset_index()

    psim_irt_lm = pd.merge(psim_irt_J_pred, data_bsa,
                           on=['subject', 'abs_lag', 'bin'],
                           suffixes=('', '_subj_avg')).merge(data_bsa.groupby(['abs_lag', 'bin'])['log_irt_pred'].mean().reset_index(),
                                                             on=['abs_lag', 'bin'], suffixes=('', '_grand_avg'))
    psim_irt_lm['log_irt_pred_lm'] = psim_irt_lm['log_irt_pred'] - psim_irt_lm['log_irt_pred_subj_avg'] + psim_irt_lm['log_irt_pred_grand_avg']
    psim_irt_lm_bsa = psim_irt_lm.groupby(['subject', 'experiment', 'abs_lag', 'bin'])['log_irt_pred_lm'].mean().reset_index()

    lm = psim_irt_lm_bsa.groupby(['abs_lag', 'bin']).agg(n=('subject', 'count'), stdev=('log_irt_pred_lm', 'std')).reset_index()
    lm['sem'] = lm['stdev'] / np.sqrt(lm['n'])
    lm['ci'] = 1.96 * lm['sem']
    lm = lm.sort_values(['bin', 'abs_lag'])

    log_irt_mu = data_bsa.groupby(['bin', 'abs_lag'])['log_irt_pred'].mean().reset_index().sort_values(['bin', 'abs_lag']).reset_index(drop=True)
    
    
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = np.load('figures/palette/blue_green.npy')
    
    # only plot bins 0, 2, 5 for clarity
    ax.errorbar(x=np.arange(1, 24), y=log_irt_mu.query("bin == 0").log_irt_pred, yerr=lm.query("bin == 0").ci, color='gray', marker='o', ms=5, ls='solid', label=str(0))
    ax.errorbar(x=np.arange(1, 24), y=log_irt_mu.query("bin == 2").log_irt_pred, yerr=lm.query("bin == 2").ci, color=colors[2], marker='o', ms=5, ls='solid', label=str(2))
    ax.errorbar(x=np.arange(1, 24), y=log_irt_mu.query("bin == 5").log_irt_pred, yerr=lm.query("bin == 5").ci, color=colors[3], marker='o', ms=5, ls='solid', label=str(5))
    
    """
    palette = sns.color_palette("Greens_d")
    hex_colors = palette.as_hex()
    colors = ['gray'] + hex_colors
    ls = 2 * ['solid', 'dashed', 'dotted']
    
    for i, b in enumerate(log_irt_mu['bin'].unique()):
        ax.errorbar(x=np.arange(1, 24), y=log_irt_mu.query("bin == @b").log_irt_pred, yerr=lm.query("bin == @b").ci, color=colors[i], marker='o', ms=5, ls=ls[i], label=str(b))
    """

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Absolute Lag', ylabel='Predicted ln(IRT)')
    ax.legend(title='Phonological Similarity Bin', shadow=True, ncols=3, loc='upper left', bbox_to_anchor=(0, 1.1))
    plt.savefig('figures/gallery/irt_psim_lag_J.pdf', bbox_inches='tight')
    plt.show()


def plot_irt_psim_ssim_J(psim_irt_J_pred):
    # bin phonological similarities
    psim_irt_J_pred = bin_phonetic_similarities(psim_irt_J_pred)
    # bin semantic similarities
    psim_irt_J_pred = bin_semantic_similarities(psim_irt_J_pred)

    # calculate loftus-masson errorbars
    data = psim_irt_J_pred.groupby(['subject', 'experiment', 'session', 'sem_bin', 'bin'])[['log_irt_pred', 'irt_pred']].mean().reset_index()
    data_bsa = data.groupby(['subject', 'experiment', 'sem_bin', 'bin'])[['log_irt_pred', 'irt_pred']].mean().reset_index()

    psim_irt_lm = pd.merge(psim_irt_J_pred, data_bsa,
                           on=['subject', 'sem_bin', 'bin'],
                           suffixes=('', '_subj_avg')).merge(data_bsa.groupby(['sem_bin', 'bin'])['log_irt_pred'].mean().reset_index(),
                                                             on=['sem_bin', 'bin'], suffixes=('', '_grand_avg'))
    psim_irt_lm['log_irt_pred_lm'] = psim_irt_lm['log_irt_pred'] - psim_irt_lm['log_irt_pred_subj_avg'] + psim_irt_lm['log_irt_pred_grand_avg']
    psim_irt_lm_bsa = psim_irt_lm.groupby(['subject', 'experiment', 'sem_bin', 'bin'])['log_irt_pred_lm'].mean().reset_index()

    lm = psim_irt_lm_bsa.groupby(['sem_bin', 'bin']).agg(n=('subject', 'count'), stdev=('log_irt_pred_lm', 'std')).reset_index()
    lm['sem'] = lm['stdev'] / np.sqrt(lm['n'])
    lm['ci'] = 1.96 * lm['sem']
    lm = lm.sort_values(['bin', 'sem_bin'])

    log_irt_mu = data_bsa.groupby(['bin', 'sem_bin'])['log_irt_pred'].mean().reset_index().sort_values(['bin', 'sem_bin']).reset_index(drop=True)


    fig, ax = plt.subplots(figsize=(4, 4))
    colors = np.load('figures/palette/blue_green.npy')
    
    # only plot bins 0, 2, 5 for clarity
    ax.errorbar(x=np.arange(0, 6), y=log_irt_mu.query("bin == 0").log_irt_pred, yerr=lm.query("bin == 0").ci, color='gray', marker='o', ms=5, ls='solid', label=str(0))
    ax.errorbar(x=np.arange(0, 6), y=log_irt_mu.query("bin == 2").log_irt_pred, yerr=lm.query("bin == 2").ci, color=colors[2], marker='o', ms=5, ls='solid', label=str(2))
    ax.errorbar(x=np.arange(0, 6), y=log_irt_mu.query("bin == 5").log_irt_pred, yerr=lm.query("bin == 5").ci, color=colors[3], marker='o', ms=5, ls='solid', label=str(5))
    
    """
    palette = sns.color_palette("Greens_d")
    hex_colors = palette.as_hex()
    colors = ['gray'] + hex_colors
    ls = 2 * ['solid', 'dashed', 'dotted']

    for i, b in enumerate(log_irt_mu['bin'].unique()):
        ax.errorbar(x=np.arange(0, 6), y=log_irt_mu.query("bin == @b").log_irt_pred, yerr=lm.query("bin == @b").ci, color=colors[i], marker='o', ms=5, ls=ls[i], label=str(b))
    """

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Semantic Similarity Bin', ylabel='Predicted ln(IRT)')
    ax.legend(title='Phonological Similarity Bin', shadow=True, ncols=3, loc='upper right', bbox_to_anchor=(1, 1))
    plt.savefig('figures/gallery/irt_psim_ssim_J.pdf', bbox_inches='tight')
    plt.show()