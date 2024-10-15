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
def plot_pcs(pcs_H_data_bsa, pcs_HS_data_bsa, pcs_HR_data_bsa, pcs_J_data_bsa, pcs_JFL_data_bsa, exp_type):
    # merge dataframes
    pcs_H_data_bsa.rename(columns={'pcs': 'pcs_H'}, inplace=True)
    pcs_HS_data_bsa.rename(columns={'pcs': 'pcs_HS'}, inplace=True)
    pcs_HR_data_bsa.rename(columns={'pcs': 'pcs_HR'}, inplace=True)
    
    dfH = pd.merge(pcs_H_data_bsa, 
                    pd.merge(pcs_HS_data_bsa, pcs_HR_data_bsa, 
                             how='outer', on=['subject', 'exp_type', 'experiment']),
                    how='outer', on=['subject', 'exp_type', 'experiment'])
    
    dfJ = pd.merge(pcs_J_data_bsa, pcs_JFL_data_bsa, how='outer', 
                  on=['subject', 'exp_type', 'experiment'], suffixes=('_J', '_JFL'))

    # re-organize dataframes
    dfmH = pd.melt(dfH, id_vars=['subject', 'exp_type', 'experiment'], value_vars=['pcs_H', 'pcs_HS', 'pcs_HR'],
                   var_name='metric', value_name='pcs')
    
    dfmJ = pd.melt(dfJ, id_vars=['subject', 'exp_type', 'experiment'], value_vars=['pcs_J', 'pcs_JFL'], 
                   var_name='metric', value_name='pcs')
    
    
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 4), width_ratios=[3, 2])
    
    sns.pointplot(dfmH.query("exp_type == @exp_type"), x='metric', y='pcs', order=['pcs_H', 'pcs_HS', 'pcs_HR'],
                  hue='metric',  palette='Blues_d', errorbar=('se', 1.96), ax=ax[0])
    ax[0].axhline(0, color='black', linestyle='dotted')
    
    sns.pointplot(dfmJ.query("exp_type == @exp_type"), x='metric', y='pcs', order=['pcs_J', 'pcs_JFL'],
                  hue='metric', palette='Greens_d', errorbar=('se', 1.96), ax=ax[1])
    ax[1].axhline(0.5, color='black', linestyle='dotted')
    
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='', ylim=(-0.02, 0.02))
    ax[0].set_xticks(np.arange(3), labels=['Start/Rhyme', 'Start', 'Rhyme'])
    
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='', ylim=(0.48, 0.52))
    ax[1].set_xticks(np.arange(2), labels=['Jaccard', 'Jaccard First/Last'])
    
    fig.supxlabel('Similarity Metric', x=0.6)
    fig.supylabel('Phonological Clustering Score')
    fig.suptitle(f'{exp_type.capitalize()} Data', x=0.6, y=1.02)
    plt.tight_layout(w_pad=3)
    plt.savefig(f'figures/gallery/pcs_{exp_type[0].upper()}.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Correlation of Phonetic Clustering Scores ----------
def plot_pcs_correlations(pcs_corrs_bsa):
    fig, ax = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
    
    # re-organize dataframe
    dfm = pd.melt(pcs_corrs_bsa, id_vars=['subject', 'exp_type', 'experiment'],
                  value_vars=['r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL'], var_name='comparison', value_name='r_pcs')
    
    sns.barplot(dfm.query("exp_type == 'scalp'"), x='comparison', y='r_pcs', order=['r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL'],
                hue='comparison', palette='Greys', errorbar=('se', 1.96), linestyle='none', ax=ax[0])
    
    sns.barplot(dfm.query("exp_type == 'intracranial'"), x='comparison', y='r_pcs', order=['r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL'],
                hue='comparison', palette='Greys', errorbar=('se', 1.96), linestyle='none', ax=ax[1])
    
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='Pearson R Correlation', ylim=(0, 1), title='Scalp Data')
    ax[0].set_xticks(np.arange(0, 3), labels=['S/R - J', 'S/R - J F/L', 'J - J F/L'])
    
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', title='Intracranial Data')
    ax[1].set_xticks(np.arange(0, 3), labels=['S/R - J', 'S/R - J F/L', 'J - J F/L'])
    
    fig.supxlabel("Metric Comparison", x=0.53)
    plt.tight_layout()
    plt.savefig('figures/gallery/pcs_corrs.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Temporal and Semantic Clustering Scores ----------
def plot_tcs_scs(tcs_data_bsa, scs_data_bsa):
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))

    sns.pointplot(tcs_data_bsa, x='exp_type', y='tcs', order=['scalp', 'intracranial'], 
                      hue='exp_type', hue_order=['scalp', 'intracranial'], errorbar=('se', 1.96), palette='Greys_d', ax=ax[0])
    #ax[0].axhline(0.5, color='black', linestyle='dotted')

    sns.pointplot(scs_data_bsa, x='exp_type', y='scs', order=['scalp', 'intracranial'],
                      hue='exp_type', hue_order=['scalp', 'intracranial'], errorbar=('se', 1.96), palette='Purples_d', ax=ax[1])
    ax[1].axhline(0.5, color='black', linestyle='dotted')

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='Temporal Clustering Score', ylim=(0.6, 0.78))
    ax[0].set_xticks(np.arange(2), labels=['Scalp', 'Intracranial'])

    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='Semantic Clustering Score', ylim=(0.49, 0.55))
    ax[1].set_xticks(np.arange(2), labels=['Scalp', 'Intracranial'])

    fig.supxlabel('Experiment Type', x=0.53)
    fig.tight_layout(w_pad=3)
    plt.savefig('figures/gallery/tcs_scs.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Phonetic-CRL ----------
def plot_psim_crl(psim_crl_H_data_bsa):
    # map 1, 0 to psim_crl, non_crl
    psim_crl_H_data_bsa['type'] = ['psim_crl' if x == 1 else 'non_crl' for x in psim_crl_H_data_bsa.psim]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    colors = np.load('figures/palette/blue_green.npy')

    # scalp
    df_scalp = psim_crl_H_data_bsa.query("exp_type == 'scalp'")
    sns.lineplot(df_scalp.query("lag >= -7 and lag < 0"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=[colors[0], 'gray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[0])
    sns.lineplot(df_scalp.query("lag > 0 and lag <= 7"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=[colors[0], 'gray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[0], legend=False)
    
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='', title='Scalp')
    
    labels = ['Phonologically Similar', 'Not Phonologically Similar']
    handles, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels, title='', shadow=True, ncols=1, loc='upper right')
    
    # intracranial
    df_intrac = psim_crl_H_data_bsa.query("exp_type == 'intracranial'")
    sns.lineplot(df_intrac.query("lag >= -7 and lag < 0"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=[colors[1], 'dimgray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[1])
    sns.lineplot(df_intrac.query("lag > 0 and lag <= 7"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=[colors[1], 'dimgray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[1], legend=False)
    
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='', title='Intracranial')
    
    labels = ['Phonologically Similar', 'Not Phonologically Similar']
    handles, _ = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, title='', shadow=True, ncols=1, loc='upper right')

    
    fig.supxlabel("Lag", x=0.53)
    fig.supylabel("Conditional Response Latency (ms)")
    
    plt.tight_layout()
    plt.savefig('figures/gallery/psim_crl.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_psim_irt(psim_crl_J_data_bsa):
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.pointplot(psim_crl_J_data_bsa, x='bin', y='irt', hue='exp_type', hue_order=['scalp', 'intracranial'], 
                  palette="Greens_d", errorbar=('se', 1.96))

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Phonological Similarity Bin', ylabel='Inter-Response Time (ms)', ylim=(2300, 3800))
    ax.set_xticks(np.arange(6), labels=np.arange(6))

    labels = ['Scalp', 'Intracranial']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=2, loc='upper right', bbox_to_anchor=(1, 1.05))

    plt.savefig('figures/gallery/psim_irt.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Phonetic Intrusions ----------
def plot_psim_intr_l(psim_intr_l_data_bsa, metric):
    colors = np.load('figures/palette/blue_green.npy')
    if metric == 'H':
        palette1 = [colors[0], 'lightgray', colors[1], 'darkgray']
        palette2 ='Blues_d'
        yb1 = 0.12; yt1 = 0.2
        yb2 = -0.005; yt2 = 0.05
    elif metric == 'J':
        palette1 = [colors[2], 'lightgray', colors[3], 'darkgray']
        palette2 = 'Greens_d'
        yb1 = 0.075; yt1 = 0.12
        yb2 = -0.001; yt2 = 0.02
    else:
        raise ValueError(f'{metric} not a valid metric')
    
    # re-organize dataframe
    dfm = pd.melt(psim_intr_l_data_bsa, id_vars=['subject', 'exp_type', 'experiment'],
                  value_vars=['pli_psim', 'pli_control', 'pli_delta', 'eli_psim', 'eli_control', 'eli_delta'], 
                  var_name='intr_type', value_name='probability')
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[2, 1])

    # raw scores
    sns.barplot(dfm, x='exp_type', y='probability', order=['scalp', 'intracranial'], hue='intr_type', 
                hue_order=['pli_psim', 'pli_control', 'eli_psim', 'eli_control'], 
                palette=palette1, errorbar=('se', 1.96), gap=0.1, ax=ax[0])
    
    # deltas
    sns.pointplot(dfm, x='exp_type', y='probability', order=['scalp', 'intracranial'], hue='intr_type',
                  hue_order=['pli_delta', 'eli_delta'], palette=palette2, errorbar=('se', 1.96), 
                  dodge=0.2, linestyle='none',  ax=ax[1])
    
    ax[1].axhline(0, color='black', linestyle='dotted')

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='Mean Phonological Similarity\nto List Items', ylim=(yb1, yt1))
    ax[0].set_xticks(np.arange(2), labels=['Scalp', 'Intracranial'])
    
    labels=['PLI', 'PLI Control', 'ELI', 'ELI Control']
    handles, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels, title='', shadow=True, ncols=4, loc='upper right', bbox_to_anchor=(1, 1.05))
    
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='Difference in Phonological Similarity\n(Observed - Control)', ylim=(yb2, yt2))
    ax[1].set_xticks(np.arange(2), labels=['Scalp', 'Intracranial'])
    
    labels=['PLI Delta', 'ELI Delta']
    handles, _ = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, title='', shadow=True, ncols=2, loc='upper right', bbox_to_anchor=(1, 1.05))
    
    plt.tight_layout()
    plt.savefig(f'figures/gallery/psim_intr_l_{metric}.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_psim_intr_r(psim_intr_r_data_bsa, metric):
    colors = np.load('figures/palette/blue_green.npy')
    if metric == 'H':
        palette = ['silver', colors[0], colors[1]]
        yb = 0.1
        yt = 0.22
    elif metric == 'J':
        palette = ['silver', colors[2], colors[3]]
        yb = 0.07
        yt = 0.13
    else:
        raise ValueError(f'{metric} not a valid metric')
        
    fig, ax = plt.subplots(figsize=(7,5))
    
    sns.barplot(psim_intr_r_data_bsa, x='exp_type', y='psim', order=['scalp', 'intracranial'], hue='resp_type', 
                hue_order=['cr', 'pli', 'eli'], palette=palette, errorbar=('se', 1.96), gap=0.1, ax=ax)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Phonological Similarity', ylim=(yb, yt))
    ax.set_xticks([0, 1], labels=['Scalp', 'Intracranial'])
    
    labels = ['Correct Recall', 'PLI', 'ELI']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=3, loc='upper right', bbox_to_anchor=(1, 1.05))
    
    plt.savefig(f'figures/gallery/psim_intr_r_{metric}.pdf', bbox_inches='tight')
    plt.show()

    

# ---------- Correlations with Recall Probability ----------
def plot_p_recall_correlations(p_recall_corrs_bsa):
    # re-organize dataframe
    dfm = pd.melt(p_recall_corrs_bsa, id_vars=['subject', 'exp_type', 'experiment'], value_vars=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'],
                  var_name='label', value_name='correlation')
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.barplot(dfm, x='exp_type', y='correlation', order=['scalp', 'intracranial'], hue='label', hue_order=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'],
                palette=['steelblue', 'forestgreen', 'silver', 'purple'], errorbar=('se', 1.96), ax=ax, dodge=True, gap=0.1)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Correlation with Recall Probability', ylim=(-0.07, 0.11))
    ax.set_xticks(np.arange(2), labels=['Scalp', 'Intracranial'])
    
    labels = ['PCS S/R', 'PCS J', 'TCS', 'SCS']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=4, loc='upper right', bbox_to_anchor=(1, 1.10))
    
    plt.savefig('figures/gallery/p_recall_corrs.pdf', bbox_inches='tight')
    plt.show()