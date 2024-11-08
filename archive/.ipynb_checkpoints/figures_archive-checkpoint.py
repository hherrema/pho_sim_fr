# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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