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


# ---------- Phonetic Similarity Utility ----------
# return whether recall is phonetically similar to any word in list
# or average phonetic similarity to list items
def psim_to_list_item(r, words, pronouncing_cml, psim_fxn):
    if psim_fxn == phonetic_sim_H:
        for w in words:
            sim_start = psim_fxn(r, w, pronouncing_cml, False)
            rhyme = psim_fxn(r, w, pronouncing_cml, True)

            if sim_start or rhyme:
                return True
            
        return False
    else:
        psim = []
        for w in words:
            psim.append(psim_fxn(r, w, pronouncing_cml))
            
        return np.mean(psim)


# phonetic intrusions permutation control
# 'all_words' does not contain current list words
def psim_intr_pc(r, all_words, max_sp, n_perms, pronouncing_cml, psim_fxn, seed):
    np.random.seed(seed)
    psim_pc = []
    
    if psim_fxn == phonetic_sim_H:
        for n in range(n_perms):
            psim_toggle = 0
            words_control = np.random.choice(all_words, size=max_sp, replace=False)      # select list length words from session at random
            for w in words_control:
                sim_start = phonetic_sim_H(r, w, pronouncing_cml, False)
                rhyme = phonetic_sim_H(r, w, pronouncing_cml, True)

                if sim_start or rhyme:
                    psim_toggle = 1
                    break                     # stop search if found phonetically similar word
                
            psim_pc.append(psim_toggle)
        
    else:
        for n in range(n_perms):
            psim = []
            words_control = np.random.choice(all_words, size=max_sp, replace=False)
            for w in words_control:
                psim.append(psim_fxn(r, w, pronouncing_cml))
                
            psim_pc.append(np.mean(psim))
    
    return np.mean(psim_pc)

# ---------- Correlation of Phonetic Clustering Scores ----------
def pcs_correlations(pcs_v1_data_lst, pcs_v2_data_lst):
    # pcs v1 & pcs v2
    df = pd.merge(pcs_v1_data_lst, pcs_v2_data_lst, on=['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage', 'list'],
                  how='inner', suffixes=('_v1', '_v2'))
    df = df.drop_duplicates(subset=['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage', 'list'], keep=False)
    
    stats = []
    for (sub, et, exp, sess, loc, mont), data in tqdm(df.groupby(['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage'])):
        if len(data) > 1:
            r_pcs, _ = scipy.stats.pearsonr(data.pcs_v1, data.pcs_v2, alternative='two-sided')

            stats.append((sub, et, exp, sess, loc, mont, r_pcs))

    # save results as dataframe
    return pd.DataFrame(stats, columns=['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage', 'r_pcs'])

def pcs_corr_btwn_subj_avg(pcs_corrs):
    pcs_corrs_bsa = pcs_corrs.groupby(['subject', 'exp_type', 'experiment'])['r_pcs'].mean().reset_index()
    
    return pcs_corrs_bsa


# ----------- Phonetic-CRL ----------

def psim_crl_sess_v1(evs, lt, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    psim_crl = np.zeros(2 * max_sp - 1)
    psim_count = np.zeros_like(psim_crl)
    non_crl = np.zeros(2 * max_sp - 1)
    non_count = np.zeros_like(non_crl)
    
    for i in rec_evs[lt].unique():
        w_evs = word_evs[word_evs[lt] == i]
        r_evs = rec_evs[rec_evs[lt] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs, pyFR_toggle)
        rts = r_evs['rectime'].to_numpy()
        
        # only analyse lists with at least 2 recalls
        if len(recs) >= 2:
            
            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:      # correct recall
                    words_left.remove(recs[j])                               # can't transition to already recalled word
                    
                    if 1<= sp[j+1] <= max_sp and recs[j+1] in words_left:           # transition to correct recall
                        sim_start = phonetic_sim_v1(recs[j], recs[j+1], pronouncing_cml, False)
                        rhyme = phonetic_sim_v1(recs[j], recs[j+1], pronouncing_cml, True)
                        lag = sp[j+1] - sp[j]
                        irt = rts[j+1] - rts[j]
                        
                        if irt <= 30000:                                           # exclude IRTs over 30 s (standardize across experiments, min recall phase duration = 30 s)
                            if sim_start or rhyme:
                                psim_crl[lag + max_sp - 1] += irt
                                psim_count[lag + max_sp - 1] += 1
                            else:
                                non_crl[lag + max_sp - 1] += irt
                                non_count[lag + max_sp - 1] += 1
                            
    lags = np.arange(-(max_sp - 1), max_sp, 1)
    
    return pd.DataFrame({'lag':lags, 'psim_crl': psim_crl / psim_count, 'non_crl': non_crl / non_count})

def psim_crl_btwn_subj_avg_v1(pcs_crl_v1_data):
    psim_crl_v1_data_bsa = pcs_crl_v1_data.groupby(['subject', 'exp_type', 'experiment', 'lag'])[['psim_crl', 'non_crl']].mean().reset_index()
    
    # sort by experiment
    psim_crl_v1_data_bsa = sort_by_experiment(psim_crl_v1_data_bsa)
    
    return psim_crl_v1_data_bsa


def psim_intr_sess_v1(evs, lt, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    n_pli = 0; n_psim_pli = 0; psim_pli_control = []               # PLI counters
    n_eli = 0; n_psim_eli = 0; psim_eli_control = []               # ELI counters
    
    for i in rec_evs[lt].unique():
        w_evs = word_evs[word_evs[lt] == i]
        r_evs = rec_evs[rec_evs[lt] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs, pyFR_toggle)
        
        # loop over recalls
        for j in range(len(recs)):
            # intrusion (intracranial experiments):
            if lt == 'list' and sp[j] == -999 and recs[j].upper() in pronouncing_cml.keys():         # only analyze intrusions with phoneme breakdown
                
                # determine if PLI
                if pyFR_toggle:
                    pli_toggle = recs[j].upper() in word_evs[word_evs[lt] < i]['item'].unique()
                    all_words = word_evs[word_evs[lt] != i]['item'].to_numpy()
                else:
                    pli_toggle = recs[j].upper() in word_evs[word_evs[lt] < i]['item_name'].unique()
                    all_words = word_evs[word_evs[lt] != i]['item_name'].to_numpy()
                    
                psim_toggle = psim_to_list_item(recs[j], words, pronouncing_cml)           # phonetically similar to list item
                psim_pc = psim_intr_pc(recs[j], all_words, max_sp, 100, pronouncing_cml)                 # permutation control
                
                if pli_toggle:
                    n_pli += 1
                    psim_pli_control.append(psim_pc)
                    if psim_toggle:
                        n_psim_pli += 1
                else:
                    n_eli += 1
                    psim_eli_control.append(psim_pc)
                    if psim_toggle:
                        n_psim_eli += 1
                           
            # intrusion(ltpFR2)
            elif lt == 'trial' and recs[j] not in words and recs[j].upper() in pronouncing_cml.keys():     # only analyze intrusions with phoneme breakdown
                
                # determine if PLI
                pli_toggle = recs[j].upper() in word_evs[word_evs[lt] < i]['item_name'].unique()
                
                psim_toggle = psim_to_list_item(recs[j], words, pronouncing_cml)           # phonetically similar to list item
                
                all_words = word_evs[word_evs[lt] != i]['item_name'].to_numpy()
                psim_pc = psim_intr_pc(recs[j], all_words, max_sp, 100, pronouncing_cml)                 # permutation control
                
                if pli_toggle:
                    n_pli += 1
                    psim_pli_control.append(psim_pc)
                    if psim_toggle:
                        n_psim_pli += 1
                else:
                    n_eli += 1
                    psim_eli_control.append(psim_pc)
                    if psim_toggle:
                        n_psim_eli += 1
                        
    # no intrusions, return NaN
    if n_pli == 0:
        n_pli = np.nan
    if n_eli == 0:
        n_eli = np.nan
        
    psim_intr = pd.DataFrame({'psim_pli': n_psim_pli / n_pli, 'psim_eli': n_psim_eli / n_eli, 
                              'psim_pli_control': np.mean(psim_pli_control), 'psim_eli_control': np.mean(psim_eli_control)}, index=[0])
    
    # take difference from control
    psim_intr['delta_pli'] = psim_intr['psim_pli'] - psim_intr['psim_pli_control']
    psim_intr['delta_eli'] = psim_intr['psim_eli'] - psim_intr['psim_eli_control']
    
    return psim_intr

def psim_intr_btwn_subj_avg_v1(psim_intr_v1_data):
    psim_intr_v1_data_bsa = psim_intr_v1_data.groupby(['subject', 'exp_type', 'experiment'])[['psim_pli', 'psim_eli', 'psim_pli_control', 'psim_eli_control', 'delta_pli', 'delta_eli']].mean().reset_index()
    
    # sort by experiment
    psim_intr_v1_data_bsa = sort_by_experiment(psim_intr_v1_data_bsa)
    
    return psim_intr_v1_data_bsa

def psim_intr_sess_v2(evs, lt, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    cr_psim = []
    pli_psim = []
    eli_psim = []
    
    for i in rec_evs[lt].unique():
        w_evs = word_evs[word_evs[lt] == i]
        r_evs = rec_evs[rec_evs[lt] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs, pyFR_toggle)
        
        # loop over recalls
        words_left = list(words)
        for j in range(len(recs) - 1):
            if 1 <= sp[j] <= max_sp and recs[j] in words_left:               # correct recall
                words_left.remove(recs[j])                                   # can't transition to already realled word
                
                # transition to correct recall
                if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:
                    psim = phonetic_sim_v2(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # phonetic similarity
                    cr_psim.append(psim)
                    
                # transition to intrusion (intracranial experiments)
                elif lt == 'list' and sp[j+1] == -999 and recs[j+1].upper() in pronouncing_cml.keys():      # only analyze intrusions with phoneme breakdown
                    psim = phonetic_sim_v2(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # phonetic similarity
                    
                    # determine if PLI
                    if pyFR_toggle:
                        pli_toggle = recs[j+1].upper() in word_evs[word_evs[lt] < i]['item'].unique()
                    else:
                        pli_toggle = recs[j+1].upper() in word_evs[word_evs[lt] < i]['item_name'].unique()
                        
                    if pli_toggle:
                        pli_psim.append(psim)
                    else:
                        eli_psim.append(psim)
                       
                # transition to intrusion (ltpFR2)
                elif lt == 'trial' and recs[j+1] not in words and recs[j+1].upper() in pronouncing_cml.keys():
                    psim = phonetic_sim_v2(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # phonetic similarity
                    
                    # determine if PLI
                    pli_toggle = recs[j+1].upper() in word_evs[word_evs[lt] < i]['item_name'].unique()
                    if pli_toggle:
                        pli_psim.append(psim)
                    else:
                        eli_psim.append(psim)
                                    
    return pd.DataFrame({'cr_psim': np.mean(cr_psim), 'pli_psim': np.mean(pli_psim), 'eli_psim': np.mean(eli_psim)}, index=[0])

def psim_intr_btwn_subj_avg_v2(psim_intr_v2_data):
    psim_intr_v2_data_bsa = psim_intr_v2_data.groupby(['subject', 'exp_type', 'experiment'])[['cr_psim', 'pli_psim', 'eli_psim']].mean().reset_index()
    
    # sort by experiment
    psim_intr_v2_data_bsa = sort_by_experiment(psim_intr_v2_data_bsa)
    
    return psim_intr_v2_data_bsa


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


# ---------- All Transitions ----------
def all_transitions_sess(evs, lt, pyFR_toggle):
    # phoneme breakdowns
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
    
    # word2vec semantic similarites
    sem_sim_cml = pd.read_csv('semantic_similarity/semantic_cml.csv')
    
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    # filter all semantic similarities, convert to xarray for query speed
    if pyFR_toggle:
        try:
            all_words = pd.concat([word_evs, rec_evs])['item'].unique()
        except TypeError:
            all_words = np.unique([x for x in pd.concat([word_evs, rec_evs])['item'] if type(x) == str])
        sem_sim_cml = sem_sim_cml[(sem_sim_cml.word_i.isin(all_words)) & (sem_sim_cml.word_j.isin(all_words))]
    else:
        all_words = pd.concat([word_evs, rec_evs])['item_name'].unique()
        sem_sim_cml = sem_sim_cml[(sem_sim_cml.word_i.isin(all_words)) & (sem_sim_cml.word_j.isin(all_words))]
    sem_sim_cml = xr.DataArray(sem_sim_cml.pivot(index='word_i', columns='word_j', values='cosine_similarity'))
    
    all_tr_data = []
    for i in rec_evs[lt].unique():
        w_evs = word_evs[word_evs[lt] == i]
        r_evs = rec_evs[rec_evs[lt] == i]

        words, recs, sp = apply_serial_positions(w_evs, r_evs, pyFR_toggle)
        rts = r_evs['rectime'].to_numpy()
        
        # only analyse lists with at least 2 recalls
        if len(recs) >= 2:
            
            # number of correct recalls
            ncr = len(np.unique(sp[(1 <= sp) & (sp <= max_sp)]))

            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):

                # only transitions between words with phoneme breakdowns
                if recs[j].upper() in pronouncing_cml.keys() and recs[j+1].upper() in pronouncing_cml.keys():

                    # v1
                    sim_start = phonetic_sim_H(recs[j], recs[j+1], pronouncing_cml, False)
                    rhyme = phonetic_sim_H(recs[j], recs[j+1], pronouncing_cml, True)
                    psim_v1 = int(sim_start or rhyme)

                    # v2 (Jaccard)
                    psim_v2_J = phonetic_sim_J(recs[j], recs[j+1], pronouncing_cml)

                    # v2 (Jaccard, only first and last phoneme)
                    psim_v2_JFL = phonetic_sim_JFL(recs[j], recs[j+1], pronouncing_cml)

                    # v2 (DamLev normalized, sets)
                    psim_v2_DLS = phonetic_sim_v2_DL(recs[j], recs[j+1], pronouncing_cml, True)

                    # v2 (DamLev normalized, all phonemes)
                    psim_v2_DLA = phonetic_sim_v2_DL(recs[j], recs[j+1], pronouncing_cml, False)


                    # recall 1 response type
                    if 1 <= sp[j] <= max_sp and recs[j] in words_left:           # correct recall
                        resp_type_i = 'cr'
                        words_left.remove(recs[j])                               # can't transition to already recalled word

                    elif 1 <= sp[j] <= max_sp and recs[j] not in words_left:     # repetition
                        resp_type_i = 'rep'

                    elif pyFR_toggle and recs[j].upper() in word_evs[word_evs[lt] < i]['item'].unique():    # PLI
                        resp_type_i = 'pli'
                    elif not pyFR_toggle and recs[j].upper() in word_evs[word_evs[lt] < i]['item_name'].unique():   # PLI
                        resp_type_i = 'pli'

                    else:    # ELI
                        resp_type_i = 'eli'

                    # recall 2 response type
                    if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:           # correct recall
                        resp_type_j = 'cr'

                    elif 1 <= sp[j+1] <= max_sp and recs[j+1] not in words_left:     # repetition
                        resp_type_j = 'rep'

                    elif pyFR_toggle and recs[j+1].upper() in word_evs[word_evs[lt] < i]['item'].unique():    # PLI
                        resp_type_j = 'pli'
                    elif not pyFR_toggle and recs[j+1].upper() in word_evs[word_evs[lt] < i]['item_name'].unique():   # PLI
                        resp_type_j = 'pli'

                    else:    # ELI
                        resp_type_j = 'eli'


                    # lag
                    if resp_type_i == 'cr' and resp_type_j == 'cr':
                        lag = sp[j+1] - sp[j]
                    else:
                        lag = np.nan

                    # semantic similarity
                    try:
                        ssim = float(sem_sim_cml.loc[recs[j], recs[j+1]].values)
                    except KeyError:
                        ssim = np.nan

                    # IRT
                    irt = rts[j+1] - rts[j]
                    if irt > 0:
                        log_irt = np.log(irt)
                    else:
                        log_irt = irt

                    # length of first word
                    len_sp = len(recs[j])                         # letters
                    ph = pronouncing_cml.get(recs[j].upper())
                    len_ph = np.mean([len(x) for x in ph])        # phonemes
                    
                    # output position
                    outpos = j + 1

                    all_tr_data.append((i, resp_type_i, resp_type_j, psim_v1, psim_v2_J, psim_v2_JFL, 
                                        psim_v2_DLS, psim_v2_DLA,lag, abs(lag), ssim, irt, log_irt, 
                                        len_sp, len_ph, outpos, ncr))


    all_tr_data = pd.DataFrame(all_tr_data, columns=['list', 'resp_type_i', 'resp_type_j', 'psim_v1', 'psim_v2_J', 'psim_v2_JFL', 
                                                     'psim_v2_DLS', 'psim_v2_DLA', 'lag', 'abs_lag', 'ssim', 'irt', 'log_irt', 
                                                     'len_sp', 'len_ph', 'outpos', 'ncr'])
    return all_tr_data

def all_transitions_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
    
    all_tr_df = all_transitions_sess(evs, lt, exp=='pyFR')
    all_tr_df = apply_md(all_tr_df, sub, exp, sess, loc,  mont)
    
    return all_tr_df