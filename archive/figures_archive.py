# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    plt.savefig('figures/gallery/pcs_v1.png', bbox_inches='tight')
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
    plt.savefig('figures/gallery/pcs_v2.png', bbox_inches='tight')
    plt.show()
    

# ---------- Correlation of Phonetic Clustering Scores ----------
def plot_pcs_correlations(pcs_corrs_bsa, exp_type):
    fig, ax = plt.subplots(figsize=(4.5, 4))

    # re-organize dataframe
    dfm = pd.melt(pcs_corrs_bsa, id_vars=['subject', 'exp_type', 'experiment'],
                  value_vars=['r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL'], var_name='comparison', value_name='r_pcs')

    sns.barplot(dfm.query("exp_type == @exp_type"), x='comparison', y='r_pcs', order=['r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL'],
                  hue='comparison', palette='Greys_d', errorbar=('se', 1.96), linestyle='none', ax=ax)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='', ylabel='Pearson R Correlation', ylim=(0, 1), title=f'{exp_type.capitalize()} Data')
    ax.set_xticks(np.arange(0, 3), labels=['S/R - J', 'S/R - J F/L', 'J - J F/L'])
    
    plt.savefig(f'figures/gallery/pcs_corrs_{exp_type[0].upper()}.pdf', bbox_inches='tight')
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
    plt.savefig('figures/gallery/tcs.png', bbox_inches='tight')
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
    plt.savefig('figures/gallery/scs.png', bbox_inches='tight')
    plt.show()


# ---------- Phonetic-CRL ----------
def plot_psim_crl_v1(psim_crl_H_data_bsa, exp_type):
    # map 1, 0 to psim_crl, non_crl
    psim_crl_H_data_bsa['type'] = ['psim_crl' if x == 1 else 'non_crl' for x in psim_crl_H_data_bsa.psim]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = np.load('figures/palette/blue_green.npy')
    
    # experiment type
    df = psim_crl_H_data_bsa.query("exp_type == @exp_type")
    sns.lineplot(df.query("lag >= -7 and lag < 0"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=['steelblue', 'dimgray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax)
    sns.lineplot(df.query("lag > 0 and lag <= 7"), x='lag', y='crl', hue='type', hue_order=['psim_crl', 'non_crl'],
                 palette=['steelblue', 'dimgray'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax, legend=False)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Lag', ylabel='Conditional Response Latency (ms)')
    ax.set_title(f'{exp_type.capitalize()} Data', y=1.02)
    
    labels = ['Phonologically Similar', 'Not Phonologically Similar']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', shadow=True, ncols=1, loc='upper right')
    
    plt.savefig(f'figures/gallery/psim_crl_v1_{exp_type[0].upper()}.pdf', bbox_inches='tight')
    plt.show()
    

def plot_psim_crl_v2(psim_crl_J_data_bsa, exp_type):
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.pointplot(psim_crl_J_data_bsa.query("exp_type == @exp_type"), x='bin', y='irt', color="forestgreen", errorbar=('se', 1.96), ax=ax)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Phonological Similarity Bin', ylabel='Inter-Response Time (ms)')
    ax.set_title(f'{exp_type.capitalize()} Data', y=1.03)

    plt.savefig(f'figures/gallery/psim_crl_v2_{exp_type[0].upper()}.pdf', bbox_inches='tight')
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
    plt.savefig('figures/gallery/psim_crl_v2.png', bbox_inches='tight')
    plt.show()


# ---------- Phonetic Intrusions ----------    
def plot_psim_intr_l(psim_intr_l_data_bsa, exp_type, metric):
    colors = np.load('figures/palette/blue_green.npy')
    if metric == 'H':
        palette1 = [colors[0], 'lightgray', colors[1], 'darkgray']
        palette2 ='Blues_d'
        yb = 0.7 if exp_type == 'intracranial' else 0.85
        yt = 1
        ylbl = 'Proportion of Intrusions\nPhonologically Similiar to List Item'
    elif metric == 'J':
        palette1 = [colors[2], 'lightgray', colors[3], 'darkgray']
        palette2 = 'Greens_d'
        yb = 0.075 if exp_type == 'intracranial' else 0.095
        yt = 0.11 if exp_type == 'intracranial' else 0.12
        ylbl = 'Mean Phonological Similarity\nto List Items'
    else:
        raise ValueError(f'{metric} not a valid metric')
        

    # re-organize dataframe
    dfm = pd.melt(psim_intr_l_data_bsa, id_vars=['subject', 'exp_type', 'experiment'],
                  value_vars=['pli_psim', 'pli_control', 'pli_delta', 'eli_psim', 'eli_control', 'eli_delta'], 
                  var_name='intr_type', value_name='probability')
    
    fig, ax = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[2, 1.3])

    # raw scores
    sns.barplot(dfm.query("exp_type == @exp_type"), x='intr_type', y='probability', order=['pli_psim', 'pli_control', 'eli_psim', 'eli_control'], 
                hue='intr_type', hue_order=['pli_psim', 'pli_control', 'eli_psim', 'eli_control'], 
                palette=palette1, errorbar=('se', 1.96), ax=ax[0])
    
    # deltas
    sns.pointplot(dfm.query("exp_type == @exp_type"), x='intr_type', y='probability', order=['pli_delta', 'eli_delta'],
                  hue='intr_type', hue_order=['pli_delta', 'eli_delta'], palette=palette2, errorbar=('se', 1.96), ax=ax[1])
    ax[1].axhline(0, color='black', linestyle='dotted')

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].set(xlabel='', ylabel=ylbl, ylim=(yb, yt))
    ax[0].set_xticks(np.arange(4), labels=['PLI', 'PLI Control', 'ELI', 'ELI Control'])
    
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set(xlabel='', ylabel='Delta Phonological Similarity\n(Observed - Control)') #, ylim=(-0.01, 0.04))
    ax[1].set_xticks(np.arange(2), labels=['PLI', 'ELI'])
    
    fig.supxlabel("Intrusion Type", x=0.6)
    fig.suptitle(f'{exp_type.capitalize()} Data', x=0.6, y=1.02)
    plt.tight_layout(w_pad=3)
    plt.savefig(f'figures/gallery/psim_intr_l_{metric}_{exp_type[0].upper()}.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_psim_intr_r(psim_intr_r_data_bsa, exp_type, metric):
    colors = np.load('figures/palette/blue_green.npy')
    if metric == 'H':
        palette = ['silver', colors[0], colors[1]]
        yb = 0.1
        yt = 0.23 if exp_type == 'intracranial' else 0.2
    elif metric == 'J':
        palette = ['silver', colors[2], colors[3]]
        yb = 0.06 if exp_type == 'intracranial' else 0.09
        yt = 0.13
    else:
        raise ValueError(f'{metric} not a valid metric')

    fig, ax = plt.subplots(figsize=(4, 5))

    sns.barplot(psim_intr_r_data_bsa.query("exp_type == @exp_type"), x='resp_type', y='psim', order=['cr', 'pli', 'eli'], 
                hue='resp_type', hue_order=['cr', 'pli', 'eli'], palette=palette, errorbar=('se', 1.96), ax=ax)

    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Response Type', ylabel='Phonological Similarity', ylim=(yb, yt))
    ax.set_xticks(np.arange(3), labels=['CR', 'PLI', 'ELI'])
    #ax.set_yticks(np.arange(yb, yt, 0.01))
    ax.set_title(f'{exp_type.capitalize()} Data')
    
    plt.savefig(f'figures/gallery/psim_intr_r_{metric}_{exp_type[0].upper()}.pdf', bbox_inches='tight')
    plt.show()
    

# ---------- Correlations with Recall Probability ----------
def plot_p_recall_correlations(p_recall_corrs_bsa, exp_type):
    yb = -0.06 if exp_type == 'intracranial' else 0
    
    # re-organize dataframe
    dfm = pd.melt(p_recall_corrs_bsa, id_vars=['subject', 'exp_type', 'experiment'], value_vars=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'],
                  var_name='label', value_name='correlation')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.barplot(dfm.query("exp_type == @exp_type"), x='label', y='correlation', order=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'], 
                hue='label', hue_order=['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'], palette=['steelblue', 'forestgreen', 'silver', 'purple'], 
                errorbar=('se', 1.96), ax=ax)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Organization Type', ylabel='Correlation with Recall Probability', ylim=(yb, 0.11))
    ax.set_xticks(np.arange(4), labels=['Phonological S/R', 'Phonological J', 'Temporal', 'Semantic'])
    ax.set_title(f'{exp_type.capitalize()} Data')
    
    plt.savefig(f'figures/gallery/p_recall_corrs_{exp_type[0].upper()}.pdf', bbox_inches='tight')
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
    plt.savefig('figures/gallery/ph1.png', bbox_inches='tight')
    plt.show()