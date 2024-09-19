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
def plot_pcs_v1(pcs_v1_data_bsa):
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.pointplot(pcs_v1_data_bsa, x='exp_type', y='pcs', order=['intracranial', 'scalp'],
                  hue='exp_type', hue_order=['intracranial', 'scalp'], errorbar=('se', 1.96), palette="Blues_d", ax=ax)
    ax.axhline(0, color='black', linestyle='dotted')

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Phonetic Clustering Score', ylim=(-0.03, 0.03))
    ax.set_xticks([0, 1], labels=['Intracranial', 'Scalp'])
    
    plt.savefig('figures/gallery/pcs_v1.pdf', bbox_inches='tight')
    plt.show()
    

def plot_pcs_v2(pcs_v2_data_bsa):
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.pointplot(pcs_v2_data_bsa, x='exp_type', y='pcs', order=['intracranial', 'scalp'],
                  hue='exp_type', hue_order=['intracranial', 'scalp'], errorbar=('se', 1.96), palette="Greens_d", ax=ax)
    ax.axhline(0.5, color='black', linestyle='dotted')

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Phonetic Clustering Score', ylim=(0.48, 0.52))
    ax.set_xticks([0, 1], labels=['Intracranial', 'Scalp'])
    
    plt.savefig('figures/gallery/pcs_v2.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Correlation of Phonetic Clustering Scores ----------
def plot_pcs_correlations(pcs_corrs_bsa):
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.pointplot(pcs_corrs_bsa, x='exp_type', y='r_pcs', order=['intracranial', 'scalp'], errorbar=('se', 1.96), 
                  color='black', linestyles='none', ax=ax)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Correlation of\nPhonetic Clustering Scores', ylim=(0.3, 0.6))
    ax.set_xticks([0, 1], labels=['Intracranial', 'Scalp'])

    plt.savefig('figures/gallery/pcs_corrs.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Phonetic-CRL ----------
def plot_psim_crl_v1(psim_crl_v1_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(psim_crl_v1_data_bsa, id_vars=['subject', 'exp_type', 'experiment', 'lag'],
                  value_vars=['psim_crl', 'non_crl'], var_name='type', value_name='crl')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    colors = np.load('figures/palette/blue_green.npy')
    
    # intracranial
    dfm_intrac = dfm.query("exp_type == 'intracranial'")
    sns.lineplot(dfm_intrac.query("lag >= -7 and lag < 0"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=[colors[0], 'dimgray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[0])
    sns.lineplot(dfm_intrac.query("lag > 0 and lag <= 7"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=[colors[0], 'dimgray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[0], legend=False)
    
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='')
    
    labels = ['Phonetically Similar', 'Not Phonetically Similar']
    handles, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels, title='', shadow=True, ncols=1, loc='upper right')

    # scalp
    dfm_scalp = dfm.query("exp_type == 'scalp'")
    sns.lineplot(dfm_scalp.query("lag >= -7 and lag < 0"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=[colors[1], 'dimgray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[1])
    sns.lineplot(dfm_scalp.query("lag > 0 and lag <= 7"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=[colors[1], 'dimgray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[1], legend=False)
    
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='')
    
    labels = ['Phonetically Similar', 'Not Phonetically Similar']
    handles, _ = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, title='', shadow=True, ncols=1, loc='upper right')

    
    fig.supxlabel("Lag", x=0.53)
    fig.supylabel("Conditional Response Latency (ms)")
    
    plt.tight_layout()
    plt.savefig('figures/gallery/psim_crl_v1.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_psim_crl_v2(psim_crl_v2_data_bsa):
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.pointplot(psim_crl_v2_data_bsa, x='bin', y='irt', hue='exp_type', hue_order=['intracranial', 'scalp'], 
                  palette="Greens_d", errorbar=('se', 1.96))

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Phonetic Similarity Bin', ylabel='Inter-Response Time (ms)', ylim=(2000, 4000))

    labels = ['Intracranial', 'Scalp']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=2, loc='upper right', bbox_to_anchor=(1, 1.05))

    plt.savefig('figures/gallery/psim_crl_v2.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- Phonetic Intrusions ----------
def plot_psim_intr_v1(psim_intr_v1_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(psim_intr_v1_data_bsa, id_vars=['subject', 'exp_type', 'experiment'],
                  value_vars=['psim_pli', 'psim_pli_control', 'psim_eli', 'psim_eli_control'], 
                  var_name='intr_type', value_name='probability')
    
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = np.load('figures/palette/blue_green.npy')

    sns.barplot(dfm, x='exp_type', y='probability', order=['intracranial', 'scalp'], hue='intr_type', 
                hue_order=['psim_pli', 'psim_pli_control', 'psim_eli', 'psim_eli_control'], 
                palette=[colors[0], 'lightgray', colors[1], 'darkgray'], errorbar=('se', 1.96), gap=0.1)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Proportion of Intrusions\nPhonetically Similiar to List Item', ylim=(0.7, 1))
    ax.set_xticks([0, 1], labels=['Intracranial', 'Scalp'])
    
    labels=['PLI', 'PLI Control', 'ELI', 'ELI Control']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=4, loc='upper right', bbox_to_anchor=(1, 1.05))
    
    plt.savefig('figures/gallery/psim_intr_v1.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_psim_intr_v2(psim_intr_v2_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(psim_intr_v2_data_bsa, id_vars=['subject', 'exp_type', 'experiment'],
                  value_vars=['cr_psim', 'pli_psim', 'eli_psim'], var_name='resp_type', value_name='psim')
    
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = np.load('figures/palette/blue_green.npy')

    sns.barplot(dfm, x='exp_type', y='psim', order=['intracranial', 'scalp'], hue='resp_type', 
                hue_order=['cr_psim', 'pli_psim', 'eli_psim'], palette=['silver', colors[2], colors[3]],
                errorbar=('se', 1.96), gap=0.1)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Phonetic Similarity', ylim=(0.1, 0.2))
    ax.set_xticks([0, 1], labels=['Intracranial', 'Scalp'])
    
    labels = ['Correct Recall', 'PLI', 'ELI']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=3, loc='upper right', bbox_to_anchor=(1, 1.05))
    
    plt.savefig('figures/gallery/psim_intr_v2.pdf', bbox_inches='tight')
    plt.show()
    
    
# ---------- First Phoneme Recall Probability ----------
def plot_ph1_recall(ph1_recall_data_bsa, buffer):
    # sort by phoneme
    phonemes = ['AA1', 'AE1', 'AH0', 'AH1', 'AO1', 'AW1', 'AY1', 'B', 'CH', 'D', 'EH1', 'ER1', 'EY1', 'F', 'G', 'HH', 
                'IH1', 'IY1', 'JH', 'K', 'L', 'M', 'N', 'OW1', 'OY1', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z']
    ph1_recall_data_bsa['ph1'] = pd.Categorical(ph1_recall_data_bsa['ph1'], categories=phonemes)
    ph1_recall_data_bsa = ph1_recall_data_bsa.sort_values(by='ph1', ignore_index=True)
    
    # select middle serial positions
    ph1_recall_data_bsa['middle'] = (ph1_recall_data_bsa.serial_position > buffer) & (ph1_recall_data_bsa.serial_position <= ph1_recall_data_bsa.l_length - buffer)
    df_intrac = ph1_recall_data_bsa.query("middle == True and exp_type == 'intracranial'")
    df_scalp = ph1_recall_data_bsa.query("middle == True and exp_type == 'scalp'")
    
    # between-subject averages (normalize within exp_type)
    avg_intrac = df_intrac.groupby('ph1')['p_recall'].mean().reset_index()
    avg_scalp = df_scalp.groupby('ph1')['p_recall'].mean().reset_index()
    norm_intrac = plt.Normalize(avg_intrac['p_recall'].min() - 0.01, avg_intrac['p_recall'].max() + 0.01)
    norm_scalp = plt.Normalize(avg_scalp['p_recall'].min() - 0.01, avg_scalp['p_recall'].max() + 0.01)
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    cmap_intrac = plt.get_cmap("bone_r")
    cmap_scalp = plt.get_cmap("bone_r")

    # intracranial
    df_intrac = ph1_recall_data_bsa.query("exp_type == 'intracranial'")
    bars = sns.barplot(df_intrac.query("middle == True"), x='ph1', y='p_recall', errorbar=('se', 1.96), ax=ax[0], legend=False)
    
    for bar in bars.patches:
        bar.set_color(cmap_intrac(norm_intrac(bar.get_height())))
    
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel='', ylim=(0, 0.59))
    
    # scalp
    df_scalp = ph1_recall_data_bsa.query("exp_type == 'scalp'")
    bars = sns.barplot(df_scalp.query("middle == True"), x='ph1', y='p_recall', errorbar=('se', 1.96), ax=ax[1], legend=False)
    
    for bar in bars.patches:
        bar.set_color(cmap_scalp(norm_scalp(bar.get_height())))
    
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='', ylim=(0, 0.59))
    
    fig.supxlabel('First Phoneme')
    fig.supylabel('Recall Probability')
    
    plt.tight_layout()
    plt.savefig('figures/gallery/ph1.pdf', bbox_inches='tight')
    plt.show()
    

# ---------- Temporal Clustering Score ----------
def plot_tcs(tcs_data_bsa):
    fig, ax = plt.subplots(figsize=(3, 4))
    
    sns.pointplot(tcs_data_bsa, x='exp_type', y='tcs', order=['intracranial', 'scalp'], 
                  hue='exp_type', hue_order=['intracranial', 'scalp'], errorbar=('se', 1.96), palette='Greys_d', ax=ax)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Temporal Clustering Score', ylim=(0.6, 0.78))
    ax.set_xticks([0, 1], labels=['Intracranial', 'Scalp'])
    
    plt.savefig('figures/gallery/tcs.pdf', bbox_inches='tight')
    plt.show()
    

# ---------- Semantic Clustering Score ----------
def plot_scs(scs_data_bsa):
    fig, ax = plt.subplots(figsize=(3, 4))
    
    sns.pointplot(scs_data_bsa, x='exp_type', y='scs', order=['intracranial', 'scalp'],
                  hue='exp_type', hue_order=['intracranial', 'scalp'], errorbar=('se', 1.96), palette='Purples_d', ax=ax)
    ax.axhline(0.5, color='black', linestyle='dotted')
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Semantic Clustering Score', ylim=(0.48, 0.56))
    ax.set_xticks([0, 1], labels=['Intracranial', 'Scalp'])
    
    plt.savefig('figures/gallery/scs.pdf', bbox_inches='tight')
    plt.show()
    

# ---------- Correlations with Recall Probability ----------
def plot_p_recall_correlations(p_recall_corrs_bsa):
    # re-organize dataframe
    dfm = pd.melt(p_recall_corrs_bsa, id_vars=['subject', 'exp_type', 'experiment', 'l_length'], value_vars=['r_pcs_v1', 'r_pcs_v2', 'r_tcs', 'r_scs'],
                  var_name='label', value_name='correlation')
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.barplot(dfm, x='exp_type', y='correlation', order=['intracranial', 'scalp'], hue='label', hue_order=['r_pcs_v1', 'r_pcs_v2', 'r_tcs', 'r_scs'],
                palette=['steelblue', 'forestgreen', 'silver', 'purple'], errorbar=('se', 1.96), ax=ax, dodge=True, gap=0.1)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Experiment Type', ylabel='Correlation with Recall Probability')
    ax.set_xticks([0, 1], labels=['Intracranial', 'Scalp'])
    
    labels = ['PCS v1', 'PCS v2', 'TCS', 'SCS']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=4, loc='upper right', bbox_to_anchor=(1, 1.10))
    
    plt.savefig('figures/gallery/p_recall_corrs.pdf', bbox_inches='tight')
    plt.show()