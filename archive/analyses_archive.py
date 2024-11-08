# imports
import numpy as np
import pandas as pd
import cmlreaders as cml
import scipy.stats
import json
from tqdm.notebook import tqdm
import itertools
import xarray as xr
from cmldask import CMLDask as da
from dask.distributed import wait
import nltk
from nltk.corpus import cmudict

# download CMU pronouncing dictionary
nltk.download('cmudict')

# ---------- Phonetic Similarity Calculations ----------

# number of shared unique phonemes divided by average number of unique phonemes
def phonetic_sim_v2_H(w1, w2, pronouncing_cml):
    p1 = pronouncing_cml.get(w1.upper())
    p2 = pronouncing_cml.get(w2.upper())
    
    # one of the words does not have a phoneme breakdown
    if not p1 or not p2:
        raise RuntimeError("No phoneme breakdown")
        
    # iterate over all pairs of pronounciations
    psim = []
    for x in p1:
        for y in p2:
            # unique phonemes
            xu = np.unique(x)
            yu = np.unique(y)
            
            p_shared = len(np.intersect1d(xu, yu))
            p_avg = np.mean([len(xu), len(yu)])
            
            psim.append(p_shared / p_avg)
            
    return np.mean(psim)

# normalied Damerau-Levenshtein distance = ratio of edit distance / max(p1, p2)
# toggle = True, use unique phonemes
def phonetic_sim_v2_DL(w1, w2, pronouncing_cml, toggle=False):
    p1 = pronouncing_cml.get(w1.upper())
    p2 = pronouncing_cml.get(w2.upper())
    
    # one of the words does not have a phoneme breakdown
    if not p1 or not p2:
        raise RuntimeError("No phoneme breakdown")
        
    # iterate over all pairs of pronounciations
    psim = []
    for x in p1:
        for y in p2:
            if toggle:
                ndld = normalized_damerau_levenshtein_distance(np.unique(x), np.unique(y))
            else:
                ndld = normalized_damerau_levenshtein_distance(x, y)
                
            psim.append(1 - ndld)     # subtract for similarity
            
    return np.mean(psim)


# ---------- First Phoneme Recall Probability ----------
def ph1_recall_sess(evs, lt, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
    
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]      # don't analyze practice list
    max_sp = max(word_evs.serialpos)
    
    # initial phonemes data
    ph1_data = []
    for _, row in word_evs.iterrows():
        if pyFR_toggle:
            phones = pronouncing_cml.get(row['item'].upper())
        else:
            phones = pronouncing_cml.get(row['item_name'].upper())
        
        # loop over all pronounciations
        for ph in phones:
            ph1_data.append((ph[0], row[lt], row.serialpos, max_sp, row.recalled))

    ph1_data = pd.DataFrame(ph1_data, columns=['ph1', 'list', 'serial_position', 'l_length', 'p_recall'])
    
    return ph1_data.groupby(['ph1', 'serial_position', 'l_length'])['p_recall'].mean().reset_index()

def ph1_recall_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
        
    ph1_recall_df = ph1_recall_sess(evs, lt, exp=='pyFR')
    ph1_recall_df = apply_md(ph1_recall_df, sub, exp, sess, loc, mont)
    
    return ph1_recall_df

def ph1_recall_btwn_subj_avg(ph1_recall_data):
    ph1_recall_data_bsa = ph1_recall_data.groupby(['subject', 'exp_type', 'experiment', 'ph1', 'serial_position', 'l_length'])['p_recall'].mean().reset_index()
    
    # sort by experiment
    ph1_recall_data_bsa = sort_by_experiment(ph1_recall_data_bsa)
    
    return ph1_recall_data_bsa