### Analyses

# imports
import numpy as np
import pandas as pd
import cmlreaders as cml
import scipy.stats
import json
from tqdm.notebook import tqdm
import itertools
import xarray as xr
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from cmldask import CMLDask as da
from dask.distributed import wait
import nltk
from nltk.corpus import cmudict

# download CMU pronouncing dictionary
nltk.download('cmudict')



# ---------- Utility ----------
# apply serial positions to recall events
# change serial position of PLIs to -999
def apply_serial_positions(w_evs, r_evs):
    words = np.array(w_evs.item_name)
    recs = np.array(r_evs.item_name)
    sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
        
    return words, recs, sp

# apply subject, experiment, session to dataframe
def apply_md(df, sub, exp, sess):
    cols = list(df.columns)
    df['subject'] = sub
    df['experiment'] = exp
    df['session'] = sess
    
    df = df[['subject', 'experiment', 'session'] + cols]
    return df


# ---------- Parallel Computing ----------
# create dask client for parallel computing
# provide job name, Gv1 per job, number of jobs
def create_client(j_name, gb, nj):
    return da.new_dask_client_slurm(j_name, gb, max_n_jobs=nj, walltime='10000', log_directory='/home1/hherrema/research_projects/pho_sim_fr/logs/')

# run sessions in parallel
def run_parallel_sessions(client, fxn, sub_iter, exp_iter, sess_iter, *args):
    futures = client.map(fxn, sub_iter, exp_iter, sess_iter, *args)
    wait(futures)
    errors = da.get_exceptions(futures, [f'{x[0]}_{x[1]}_{x[2]}' for x in zip(sub_iter, exp_iter, sess_iter)])
    good_futures = da.filter_futures(futures)
    results = client.gather(good_futures)

    return errors, results

# create iterables from data index
def build_iterables(df_select):
    return list(df_select.subject), list(df_select.experiment), list(df_select.session)


# ---------- Phonetic Similarity Calculations ----------
# return whether two words are phonetically similar
# 'rhyme' argument to toggle last phomeme (True) or first phoneme (False)
def phonetic_sim_H(w1, w2, pronouncing_cml, rhyme):
    p1 = pronouncing_cml.get(w1.upper())
    p2 = pronouncing_cml.get(w2.upper())
    
    # one of the words does not have phoneme breakdown
    if not p1 or not p2:
        raise RuntimeError("No phoneme breakdown")
    
    for x in p1:
        for y in p2:
            if rhyme:
                if x[-2:] == y[-2:]:
                    return True
            else:
                if x[0] == y[0]:
                    return True
    
    return False


# Jaccard index = intersection / union (sets = unique phonemes)
def phonetic_sim_J(w1, w2, pronouncing_cml):
    p1 = pronouncing_cml.get(w1.upper())
    p2 = pronouncing_cml.get(w2.upper())
    
    # one of the words does not have a phoneme breakdown
    if not p1 or not p2:
        raise RuntimeError("No phoneme breakdown")
        
    # iterate over all pairs of pronounciations
    psim = []
    for x in p1:
        for y in p2:
            # unique phonemes (sets)
            xu = set(x)
            yu = set(y)
            
            j_idx = len(set.intersection(xu, yu)) / len(set.union(xu, yu))
            
            psim.append(j_idx)
            
    return np.mean(psim)

# Jaccard index (only first and last phonemes)
def phonetic_sim_JFL(w1, w2, pronouncing_cml):
    p1 = pronouncing_cml.get(w1.upper())
    p2 = pronouncing_cml.get(w2.upper())
    
    # one of the words does not have a phoneme breakdown
    if not p1 or not p2:
        raise RuntimeError("No phoneme breakdown")
        
    # iterate over all pairs of pronounciations
    psim = []
    for x in p1:
        for y in p2:       
            # first phoneme and last 2 phonemes
            xfl = set([x[0]] + x[-2:])
            yfl = set([y[0]] + y[-2:])
            
            j_idx = len(set.intersection(xfl, yfl)) / len(set.union(xfl, yfl))
            
            psim.append(j_idx)
    
    return np.mean(psim)


# ---------- Phonetic Similarity Utility ----------
# return all pairs of phonetically similar words in list of presented words
# return dictionary mapping word to number of phonetic neighbors
def evaluate_list_ps(words, pronouncing_cml, method):
    if method not in ['sim_start', 'rhyme', 'both']:
        raise ValueError(f"{method} not a valid phonetic similarity method")
    # make list of tuples of all pairs of phonetically similar words
    ps_pairs = []
    for w1, w2 in itertools.combinations(words, 2):
        sim_start = phonetic_sim_H(w1, w2, pronouncing_cml, False)
        rhyme = phonetic_sim_H(w1, w2, pronouncing_cml, True)
        
        if method == 'sim_start' and sim_start:
            ps_pairs.append((w1, w2))
        elif method == 'rhyme' and rhyme:
            ps_pairs.append((w1, w2))
        elif method == 'both' and (sim_start or rhyme):
            ps_pairs.append((w1, w2))
            
    # make dictionary mapping words to number of phonetic neighbors
    ps_dict = {}
    for word in words:
        n_sim = 0
        for pair in ps_pairs:
            if word in pair:
                n_sim += 1
        ps_dict[word] = n_sim
        
    return ps_pairs, ps_dict

# phonetic percentile rank
def percentile_rank_P(actual, possible):
    if len(possible) < 2:
        return None

    # sort possible transitions from lowest to highest similarity
    possible.sort()

    # get indices of possible transitions with same similarity as actual transition
    matches = np.where(possible == actual)[0]
    if len(matches) > 0:
        # get number of possible transition that were less similar than the actual transition
        rank = np.mean(matches)
        # convert rank to proportion of possible transitions that were less similar than the actual transition
        ptile_rank = rank / (len(possible) - 1.0)
    else:
        ptile_rank = None

    return ptile_rank

# average phonetic similarity to list items
def psim_to_list_item(r, words, pronouncing_cml, psim_fxn):
    psim = []
    if psim_fxn == phonetic_sim_H:
        for w in words:
            sim_start = psim_fxn(r, w, pronouncing_cml, False)
            rhyme = psim_fxn(r, w, pronouncing_cml, True)

            if sim_start or rhyme:
                psim.append(1)
            else:
                psim.append(0)
    else:
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
            psim = []
            ssim = []
            words_control = np.random.choice(all_words, size=max_sp, replace=False)      # select list length words from session at random
            for w in words_control:
                sim_start = phonetic_sim_H(r, w, pronouncing_cml, False)
                rhyme = phonetic_sim_H(r, w, pronouncing_cml, True)

                if sim_start or rhyme:
                    psim.append(1)
                else:
                    psim.append(0)
                     
            psim_pc.append(np.mean(psim))
        
    else:
        for n in range(n_perms):
            psim = []
            words_control = np.random.choice(all_words, size=max_sp, replace=False)
            for w in words_control:
                psim.append(psim_fxn(r, w, pronouncing_cml))
                
            psim_pc.append(np.mean(psim))
    
    return np.mean(psim_pc)


# mean phonetic similarity of list items (all pairwise comparisons)
def psim_of_list(words, pronouncing_cml, psim_fxn):
    psim = []
    if psim_fxn == phonetic_sim_H:
        for w1, w2, in itertools.combinations(words, 2):
            sim_start = psim_fxn(w1, w2, pronouncing_cml, False)
            rhyme = psim_fxn(w1, w2, pronouncing_cml, True)
            
            if sim_start or rhyme:
                psim.append(1)
            else:
                psim.append(0)
    else:
        for w1, w2 in itertools.combinations(words, 2):
            psim.append(psim_fxn(w1, w2, pronouncing_cml))
            
    return np.mean(psim)


# ---------- Phonetic Clustering Score ----------
def pcs_sess_v1(evs, method):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs['trial'] > 0)]
    max_sp = max(word_evs.serialpos)
    
    pcs_list = []
    
    for i in rec_evs['trial'].unique():
        pcs_vals = []                     # phonetic clustering score for each transition
        
        w_evs = word_evs[word_evs['trial'] == i]
        r_evs = rec_evs[rec_evs['trial'] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs)
        
        # only analyse lists with at least 2 recalls
        if len(recs) >= 2:
            
            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:                           # correct recall
                    
                    # phonetically similar pairs in list, map of words to number of neighbors
                    ps_pairs, ps_dict = evaluate_list_ps(words_left, pronouncing_cml, method)
                    ps_neighbors = ps_dict.get(recs[j])                                      # number of phonetic neighbors to current recall
                    words_left.remove(recs[j])                                               # can't transition to already recalled word
                    
                    if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:                    # transition to correct recall
                        
                        # possible to transition to phonetically similar word
                        if ps_neighbors and ps_neighbors > 0:
                            
                            # transition to phonetically similar word
                            if (recs[j], recs[j+1]) in ps_pairs or (recs[j+1], recs[j]) in ps_pairs:
                                pcs = 1 - (ps_neighbors / len(words_left))
                                
                            # transition to not phonetically similar word
                            else:
                                pcs = 0 - (ps_neighbors / len(words_left))
                        
                            pcs_vals.append(pcs)
        
        # calculate phonetic clustering score for each list
        if len(pcs_vals) > 0:
            pcs_list.append((i, np.mean(pcs_vals)))
        
    return pd.DataFrame(pcs_list, columns=['list', 'pcs'])

def pcs_parallel_v1(sub, exp, sess, method):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
        
    pcs_df = pcs_sess_v1(evs, method)
    pcs_df = apply_md(pcs_df, sub, exp, sess)
    
    return pcs_df


def pcs_sess_v2(evs, psim_fxn):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs['trial'] > 0)]
    max_sp = max(word_evs.serialpos)
    
    pcs_list = []
    
    for i in rec_evs['trial'].unique():
        pcs_vals = []                     # phonetic clustering score for each transition
        
        w_evs = word_evs[word_evs['trial'] == i]
        r_evs = rec_evs[rec_evs['trial'] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs)
        
        # only analyse lists with at least 2 recalls
        if len(recs) >= 2:
            
            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:              # correct recall
                    words_left.remove(recs[j])                                  # can't transition to already recalled word
                    
                    if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:      # transition to correct recall
                        psim = psim_fxn(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # actual transition phonetic similarity
                        poss = []
                        for l in range(len(words_left)):
                            poss_psim = psim_fxn(recs[j].upper(), words_left[l].upper(), pronouncing_cml)      # includes actual phonetic similarity
                            poss.append(poss_psim)
                            
                        ptile_rank = percentile_rank_P(psim, poss)
                        if ptile_rank is not None:
                            pcs_vals.append(ptile_rank)
                            
        # calculate phonetic clustering score for each list
        if len(pcs_vals) > 0:
            pcs_list.append((i, np.mean(pcs_vals)))
            
    return pd.DataFrame(pcs_list, columns=['list', 'pcs'])

def pcs_parallel_v2(sub, exp, sess, psim_fxn):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
        
    pcs_df = pcs_sess_v2(evs, psim_fxn)
    pcs_df = apply_md(pcs_df, sub, exp, sess)
    
    return pcs_df

def pcs_sess_avg(pcs_data_lst):
    pcs_data = pcs_data_lst.groupby(['subject', 'experiment', 'session'])['pcs'].mean().reset_index()
    
    return pcs_data
    
def pcs_btwn_subj_avg(pcs_data):
    pcs_data_bsa = pcs_data.groupby(['subject', 'experiment'])['pcs'].mean().reset_index()
    pcs_data_bsa = pcs_data_bsa.sort_values('subject')
    
    return pcs_data_bsa


# ---------- Temporal Clustering Score ----------
# temporal percentile rank
def percentile_rank_T(actual, possible):
    if len(possible) < 2:
        return None

    # sort possible transitions from largest to smallest lag
    possible.sort(reverse=True)

    # get indices of the one or more possible transitions with the same lag as the actual transition
    matches = np.where(possible == actual)[0]

    if len(matches) > 0:
        # get number of posible lags that were more distance than actual lag
        rank = np.mean(matches)
        # convert rank to proportion of possible lags that were greater than actual lag
        ptile_rank = rank / (len(possible) - 1.0)
    else:
        ptile_rank = None

    return ptile_rank

def tcs_sess(evs):
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs['trial'] > 0)]
    max_sp = max(word_evs.serialpos)
    
    tcs_list = []
    
    for i in rec_evs['trial'].unique():
        tcs_vals = []                                 # temporal clustering scores for each transition
        
        w_evs = word_evs[word_evs['trial'] == i]
        r_evs = rec_evs[rec_evs['trial'] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs)
        
        # only analyze lists with at least 2 recalls
        if len(recs) >= 2:
            
            # loop over recalls
            sps_left = [x for x in range(1, max_sp+1)]                    # array with remaining serial position, subtract from
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and sp[j] in sps_left:            # correct recall
                    sps_left.remove(sp[j])                                # can't transition to already recalled word
                    
                    if 1 <= sp[j+1] <= max_sp and sp[j+1] in sps_left:    # transition to correct recall
                        poss = []
                        lag = abs(sp[j+1] - sp[j])                        # actual transition lag
                        for l in range(len(sps_left)):
                            poss_lag = abs(sps_left[l] - sp[j])
                            poss.append(poss_lag)                         # includes actual lag
                            
                        ptile_rank = percentile_rank_T(lag, poss)
                        if ptile_rank is not None:
                            tcs_vals.append(ptile_rank)
                            
        # calculate temporal clustering score for each list
        if len(tcs_vals) > 0:
            tcs_list.append((i, np.mean(tcs_vals)))
            
    return pd.DataFrame(tcs_list, columns=['list', 'tcs'])

def tcs_parallel(sub, exp, sess):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
        
    tcs_df = tcs_sess(evs)
    tcs_df = apply_md(tcs_df, sub, exp, sess)
    
    return tcs_df

def tcs_sess_avg(tcs_data_lst):
    tcs_data = tcs_data_lst.groupby(['subject', 'experiment', 'session'])['tcs'].mean().reset_index()
    
    return tcs_data

def tcs_btwn_subj_avg(tcs_data):
    tcs_data_bsa = tcs_data.groupby(['subject', 'experiment'])['tcs'].mean().reset_index()
    tcs_data_bsa = tcs_data_bsa.sort_values('subject')
    
    return tcs_data_bsa


# ---------- Semantic Clustering Score ----------
# semantic percentile rank
def percentile_rank_S(actual, possible):
    if len(possible) < 2:
        return None

    # sort possible transitions from lowest to highest similarity
    possible.sort()

    # get indices of possible transitions with same similarity as actual transition
    matches = np.where(possible == actual)[0]
    if len(matches) > 0:
        # get number of possible transition that were less similar than the actual transition
        rank = np.mean(matches)
        # convert rank to proportion of possible transitions that were less similar than the actual transition
        ptile_rank = rank / (len(possible) - 1.0)
    else:
        ptile_rank = None

    return ptile_rank

def scs_sess(evs):
    sem_sim_cml = pd.read_csv('semantic_similarity/semantic_cml.csv')              # word2vec semantic similarites
    
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs['trial'] > 0)]
    max_sp = max(word_evs.serialpos)
    
    # filter all semantic similarities, convert to xarray for query speed
    sem_sim_cml = sem_sim_cml[(sem_sim_cml.word_i.isin(word_evs['item_name'].unique())) & (sem_sim_cml.word_j.isin(word_evs['item_name'].unique()))]
    sem_sim_cml = xr.DataArray(sem_sim_cml.pivot(index='word_i', columns='word_j', values='cosine_similarity'))
    
    scs_list = []
    for i in rec_evs['trial'].unique():
        scs_vals = []                              # semantic clustering scores for each transition
        
        w_evs = word_evs[word_evs['trial'] == i]
        r_evs = rec_evs[rec_evs['trial'] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs)
        
        # only analyze lists with at least 2 recalls
        if len(recs) >= 2:
            
            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:                       # correct recall
                    words_left.remove(recs[j])                                           # can't transition to already recalled word
                    
                    if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:               # transition to correct recall
                        ssim = float(sem_sim_cml.loc[recs[j], recs[j+1]].values)         # actual transition semantic similarity

                        poss = []
                        for l in range(len(words_left)):
                            poss_ssim = float(sem_sim_cml.loc[recs[j], words_left[l]].values)      # includes actual semantic similarity
                            poss.append(poss_ssim)
                        
                        ptile_rank = percentile_rank_S(ssim, np.array(poss))             # requires array instead of list
                        if ptile_rank is not None:
                            scs_vals.append(ptile_rank)
                            
        # calculate semantic clustering score for each list
        if len(scs_vals) > 0:
            scs_list.append((i, np.mean(scs_vals)))
    
    return pd.DataFrame(scs_list, columns=['list', 'scs'])

def scs_parallel(sub, exp, sess):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
    
    scs_df = scs_sess(evs)
    scs_df = apply_md(scs_df, sub, exp, sess)
    
    return scs_df

def scs_sess_avg(scs_data_lst):
    scs_data = scs_data_lst.groupby(['subject', 'experiment', 'session'])['scs'].mean().reset_index()
    
    return scs_data

def scs_btwn_subj_avg(scs_data):
    scs_data_bsa = scs_data.groupby(['subject', 'experiment'])['scs'].mean().reset_index()
    scs_data_bsa = scs_data_bsa.sort_values('subject')
    
    return scs_data_bsa


# ---------- Correlation of Clustering Scores ----------
# correlation of pcs and scs/tcs at list-level within session
def find_correlated_sessions(df1, df2, cs):
    df = pd.merge(df1, df2, how='inner', on=['subject', 'experiment', 'session', 'list'])
    
    corrs = []
    for (sub, exp, sess), data in tqdm(df.groupby(['subject', 'experiment', 'session'])):
        if len(data) > 1:
            r, p = scipy.stats.pearsonr(data.pcs, data[cs], alternative='two-sided')
            if p <= 1:      # one session with p = 1.000....2
                corrs.append((sub, exp, sess, r, p))
        
    corrs = pd.DataFrame(corrs, columns=['subject', 'experiment', 'session', 'pearson_r', 'p_val'])
    
    # FDR correction
    corrs['p_val_fdr'] = scipy.stats.false_discovery_control(corrs.p_val, method='bh')
    
    return corrs

# correlation of phonetic clustering scores
def pcs_correlations(pcs_H_data_lst, pcs_J_data_lst, pcs_JFL_data_lst):
    pcs_H_data_lst.rename(columns={'pcs': 'pcs_H'}, inplace=True)
    pcs_J_data_lst.rename(columns={'pcs': 'pcs_J'}, inplace=True)
    pcs_JFL_data_lst.rename(columns={'pcs': 'pcs_JFL'}, inplace=True)
    
    on_cols = ['subject', 'experiment', 'session', 'list']
    df = pd.merge(pcs_H_data_lst, pd.merge(pcs_J_data_lst, pcs_JFL_data_lst, on=on_cols, how='inner'), on=on_cols, how='inner')
    df = df.drop_duplicates(subset=on_cols, keep=False)
    
    stats = []
    for (sub, exp, sess), data in tqdm(df.groupby(['subject', 'experiment', 'session'])):
        if len(data) > 1:
            r_pcs_HJ, _ = scipy.stats.pearsonr(data.pcs_H, data.pcs_J, alternative='two-sided')
            r_pcs_HJFL, _ = scipy.stats.pearsonr(data.pcs_H, data.pcs_JFL, alternative='two-sided')
            r_pcs_JJFL, _ = scipy.stats.pearsonr(data.pcs_J, data.pcs_JFL, alternative='two-sided')

            stats.append((sub, exp, sess, r_pcs_HJ, r_pcs_HJFL, r_pcs_JJFL))

    # save results as dataframe
    return pd.DataFrame(stats, columns=['subject', 'experiment', 'session', 'r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL'])

def pcs_corr_btwn_subj_avg(pcs_corrs):
    pcs_corrs_bsa = pcs_corrs.groupby(['subject', 'experiment'])[['r_pcs_HJ', 'r_pcs_HJFL', 'r_pcs_JJFL']].mean().reset_index()
    
    return pcs_corrs_bsa


# ---------- Phonetic Intrusions ----------
# phonetic similarity list words
def psim_intr_sess_l(evs, psim_fxn, seed):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs['trial'] > 0)]
    max_sp = max(word_evs.serialpos)
    
    psim_intr = []
    
    for i in rec_evs['trial'].unique():
        w_evs = word_evs[word_evs['trial'] == i]
        r_evs = rec_evs[rec_evs['trial'] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs)
        
        # loop over recalls
        for j in range(len(recs)):
            if recs[j] not in words and recs[j].upper() in pronouncing_cml.keys():     # only analyze intrusions with phoneme breakdown
                
                # determine if PLI
                pli_toggle = recs[j].upper() in word_evs[word_evs['trial'] < i]['item_name'].unique()
                
                psim = psim_to_list_item(recs[j], words, pronouncing_cml, psim_fxn)           # phonetic similarity to list items
                
                all_words = word_evs[word_evs['trial'] != i]['item_name'].to_numpy()
                psim_pc = psim_intr_pc(recs[j], all_words, max_sp, 100, pronouncing_cml, psim_fxn, seed+10000)       # permutation control
                
                if pli_toggle:
                    psim_intr.append((i, 'pli', psim, psim_pc, psim - psim_pc))
                else:
                    psim_intr.append((i, 'eli', psim, psim_pc, psim - psim_pc))             
                        
        
    return pd.DataFrame(psim_intr, columns=['list', 'intr_type', 'psim', 'control', 'delta'])

def psim_intr_parallel_l(sub, exp, sess, psim_fxn, seed):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
    
    psim_intr_df = psim_intr_sess_l(evs, psim_fxn, seed)
    psim_intr_df = apply_md(psim_intr_df, sub, exp, sess)
    
    return psim_intr_df

def psim_intr_sess_avg_l(psim_intr_l_data_at):
    psim_intr_l_data = psim_intr_l_data_at.groupby(['subject', 'experiment', 'session', 'intr_type'])[['psim', 'control', 'delta']].mean().reset_index()
    
    return psim_intr_l_data
    
def psim_intr_btwn_subj_avg_l(psim_intr_l_data):
    psim_intr_l_data_bsa = psim_intr_l_data.groupby(['subject', 'experiment', 'intr_type'])[['psim', 'control', 'delta']].mean().reset_index()
    
    # re-organize dataframe
    psim_intr_l_data_bsa = psim_intr_l_data_bsa.pivot(index=['subject', 'experiment'], columns='intr_type', values=['psim', 'control', 'delta'])
    psim_intr_l_data_bsa.columns = [f'{col[1]}_{col[0]}' for col in psim_intr_l_data_bsa.columns]
    psim_intr_l_data_bsa = psim_intr_l_data_bsa.reset_index()
    psim_intr_l_data_bsa = psim_intr_l_data_bsa[['subject', 'experiment', 'pli_psim', 'pli_control', 'pli_delta', 'eli_psim', 'eli_control', 'eli_delta']]
    psim_intr_l_data_bsa[['pli_psim', 'pli_control', 'pli_delta', 'eli_psim', 'eli_control', 'eli_delta']] = psim_intr_l_data_bsa[['pli_psim', 'pli_control', 'pli_delta', 'eli_psim', 'eli_control', 'eli_delta']].astype(float)
    psim_intr_l_data_bsa = psim_intr_l_data_bsa.sort_values('subject')
    
    return psim_intr_l_data_bsa


# phonetic similarity to previous recall
def psim_intr_sess_r(evs, psim_fxn):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs['trial'] > 0)]
    max_sp = max(word_evs.serialpos)
    
    psim_intr = []
    
    for i in rec_evs['trial'].unique():
        w_evs = word_evs[word_evs['trial'] == i]
        r_evs = rec_evs[rec_evs['trial'] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs)
        
        l_toggle = False                                                     # valid transition on list
        psim_control = psim_of_list(words, pronouncing_cml, psim_fxn)        # mean phonetic similarity of list words
        
        # loop over recalls
        words_left = list(words)
        for j in range(len(recs) - 1):
            if 1 <= sp[j] <= max_sp and recs[j] in words_left:               # correct recall
                words_left.remove(recs[j])                                   # can't transition to already realled word
                
                # transition to correct recall
                if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:
                    l_toggle = True
                    if psim_fxn == phonetic_sim_H:
                        sim_start = psim_fxn(recs[j].upper(), recs[j+1].upper(), pronouncing_cml, False)
                        rhyme = psim_fxn(recs[j].upper(), recs[j+1].upper(), pronouncing_cml, True)
                        psim = int(sim_start or rhyme)
                    else:
                        psim = psim_fxn(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # phonetic similarity
                    
                    psim_intr.append((i, 'cr', psim, psim_control, psim - psim_control))
                       
                # transition to intrusion
                elif recs[j+1] not in words and recs[j+1].upper() in pronouncing_cml.keys():
                    l_toggle = True
                    if psim_fxn == phonetic_sim_H:
                        sim_start = psim_fxn(recs[j].upper(), recs[j+1].upper(), pronouncing_cml, False)
                        rhyme = psim_fxn(recs[j].upper(), recs[j+1].upper(), pronouncing_cml, True)
                        psim = int(sim_start or rhyme)
                    else:
                        psim = psim_fxn(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # phonetic similarity
                        
                    
                    # determine if PLI
                    pli_toggle = recs[j+1].upper() in word_evs[word_evs['trial'] < i]['item_name'].unique()
                    if pli_toggle:
                        psim_intr.append((i, 'pli', psim, psim_control, psim - psim_control))
                    else:
                        psim_intr.append((i, 'eli', psim, psim_control, psim - psim_control))
                        
        if l_toggle:
            psim_intr.append((i, 'control', psim_control, np.nan, np.nan))
                                    
    return pd.DataFrame(psim_intr, columns=['list', 'resp_type', 'psim', 'control', 'delta'])

def psim_intr_parallel_r(sub, exp, sess, psim_fxn):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
    
    psim_intr_df = psim_intr_sess_r(evs, psim_fxn)
    psim_intr_df = apply_md(psim_intr_df, sub, exp, sess)
    
    return psim_intr_df

def psim_intr_sess_avg_r(psim_intr_r_data_at):
    psim_intr_r_data = psim_intr_r_data_at.groupby(['subject', 'experiment', 'session', 'resp_type'])[['psim', 'control', 'delta']].mean().reset_index()
    
    return psim_intr_r_data

def psim_intr_btwn_subj_avg_r(psim_intr_r_data):
    psim_intr_r_data_bsa = psim_intr_r_data.groupby(['subject', 'experiment', 'resp_type'])[['psim', 'control', 'delta']].mean().reset_index()
    psim_intr_r_data_bsa = psim_intr_r_data_bsa.sort_values('subject')
    
    return psim_intr_r_data_bsa


# ---------- Recall Probability ----------
def p_recall_sess(evs):        
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]      # don't analyze practice list
    p_recall_df = word_evs.groupby('trial').agg(l_length=('serialpos', 'size'), ncr=('recalled', 'sum'), p_recall=('recalled', 'mean')).reset_index()
    p_recall_df = p_recall_df.rename(columns={'trial': 'list'})
    
    return p_recall_df

def p_recall_parallel(sub, exp, sess):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
        
    p_recall_df = p_recall_sess(evs)
    p_recall_df = apply_md(p_recall_df, sub, exp, sess)
    
    return p_recall_df

def p_recall_sess_avg(p_recall_data_lst):
    p_recall_data = p_recall_data_lst.groupby(['subject', 'experiment', 'session', 'l_length'])['p_recall'].mean().reset_index()
    
    return p_recall_data

def p_recall_btwn_subj_avg(p_recall_data):
    p_recall_data_bsa = p_recall_data.groupby(['subject', 'experiment', 'l_length'])['p_recall'].mean().reset_index()
    p_recall_data_bsa = p_recall_data_bsa.sort_values('subject')
    
    return p_recall_data_bsa


# ---------- Correlations with Recall Probability ----------
def aggregate_data_beh(pcs_H_data_lst, pcs_J_data_lst, tcs_data_lst, scs_data_lst, p_recall_data_lst):
    id_cols = ['subject', 'experiment', 'session', 'list']
    
    # pcs_H & pcs_J
    df = pd.merge(pcs_H_data_lst, pcs_J_data_lst, on=id_cols, how='outer', suffixes=('_H', '_J'))
    df = df.drop_duplicates(subset=id_cols, keep=False)
    
    # tcs
    df = pd.merge(df, tcs_data_lst, on=id_cols, how='outer')
    df = df.drop_duplicates(subset=id_cols, keep=False)

    # scs
    df = pd.merge(df, scs_data_lst, on=id_cols, how='outer')
    df = df.drop_duplicates(subset=id_cols, keep=False)

    # p_recall
    df = pd.merge(df, p_recall_data_lst, on=id_cols, how='outer')
    df = df.drop_duplicates(subset=id_cols, keep=False)
    
    return df


# correlate with p_recall over lists within sessions
def p_recall_correlations(df):
    stats = []
    for (sub, exp, sess), data in tqdm(df.groupby(['subject', 'experiment', 'session'])):
        if len(data) > 1:
            # remove NaN separately (maximize used data)
            d_pcs_H = data[['pcs_H', 'p_recall']].dropna()
            d_pcs_J = data[['pcs_J', 'p_recall']].dropna()
            d_tcs = data[['tcs', 'p_recall']].dropna()
            d_scs = data[['scs', 'p_recall']].dropna()
            
            # correlations with p_recall
            r_pcs_H, _ = scipy.stats.pearsonr(d_pcs_H.pcs_H, d_pcs_H.p_recall, alternative='two-sided')
            r_pcs_J, _ = scipy.stats.pearsonr(d_pcs_J.pcs_J, d_pcs_J.p_recall, alternative='two-sided')
            r_tcs, _ = scipy.stats.pearsonr(d_tcs.tcs, d_tcs.p_recall, alternative='two-sided')
            r_scs, _ = scipy.stats.pearsonr(d_scs.scs, d_scs.p_recall, alternative='two-sided')

            stats.append((sub, exp, sess, r_pcs_H, r_pcs_J, r_tcs, r_scs))

    # save results as dataframe
    return pd.DataFrame(stats, columns=['subject', 'experiment', 'session', 'r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs'])


# p_recall correlations between-subject average
def p_recall_corr_btwn_subj_avg(p_recall_corrs):
    p_recall_corrs_bsa = p_recall_corrs.groupby(['subject', 'experiment'])[['r_pcs_H', 'r_pcs_J', 'r_tcs', 'r_scs']].mean().reset_index()
    
    return p_recall_corrs_bsa


# ---------- Phonetic-CRL/IRT ----------
def psim_crl_sess_v1(evs):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    sem_sim_cml = pd.read_csv('semantic_similarity/semantic_cml.csv')              # word2vec semantic similarites
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs['trial'] > 0)]
    max_sp = max(word_evs.serialpos)
    
    # filter all semantic similarities, convert to xarray for query speed
    sem_sim_cml = sem_sim_cml[(sem_sim_cml.word_i.isin(word_evs['item_name'].unique())) & (sem_sim_cml.word_j.isin(word_evs['item_name'].unique()))]
    sem_sim_cml = xr.DataArray(sem_sim_cml.pivot(index='word_i', columns='word_j', values='cosine_similarity'))
    
    psim_crl = []
    
    for i in rec_evs['trial'].unique():
        w_evs = word_evs[word_evs['trial'] == i]
        r_evs = rec_evs[rec_evs['trial'] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs)
        rts = np.array(r_evs.rectime)
        
        # only analyse lists with at least 2 recalls
        if len(recs) >= 2:
            
            ncr = len(np.unique(sp[(1 <= sp) & (sp <= max_sp)]))     # number of correct recalls
            
            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:      # correct recall
                    words_left.remove(recs[j])                               # can't transition to already recalled word
                    
                    if 1<= sp[j+1] <= max_sp and recs[j+1] in words_left:           # transition to correct recall
                        sim_start = phonetic_sim_H(recs[j], recs[j+1], pronouncing_cml, False)
                        rhyme = phonetic_sim_H(recs[j], recs[j+1], pronouncing_cml, True)
                        ssim = float(sem_sim_cml.loc[recs[j], recs[j+1]].values)                  # semantic similarity
                        lag = sp[j+1] - sp[j]
                        abs_lag = abs(lag)                                                        # |lag|
                        outpos = j + 1                                                            # output position
                        ph = pronouncing_cml.get(recs[j].upper())
                        len_ph = np.mean([len(x) for x in ph])                                    # length of first word (phonemes)
                        
                        irt = rts[j+1] - rts[j]
                        log_irt = np.log(irt) if irt > 0 else irt                                 # ln(irt)
                        
                        if irt <= 30000:                                                          # omit IRTs > 30s
                            if sim_start or rhyme:
                                psim_crl.append((i, lag, abs_lag, 1, ssim, outpos, ncr, len_ph, irt, log_irt))
                            else:
                                psim_crl.append((i, lag, abs_lag, 0, ssim, outpos, ncr, len_ph, irt, log_irt))
    
    return pd.DataFrame(psim_crl, columns=['list', 'lag', 'abs_lag', 'psim', 'ssim', 'outpos', 'ncr', 'len_ph', 'crl', 'log_crl'])

def psim_crl_parallel_v1(sub, exp, sess):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
    
    psim_crl_df = psim_crl_sess_v1(evs)
    psim_crl_df = apply_md(psim_crl_df, sub, exp, sess)
    
    return psim_crl_df

def psim_crl_sess_avg_v1(psim_crl_v1_data_at):
    psim_crl_v1_data = psim_crl_v1_data_at.groupby(['subject', 'experiment', 'session', 'lag', 'psim'])[['crl', 'log_crl']].mean().reset_index()
    psim_crl_v1_data['abs_lag'] = psim_crl_v1_data['lag'].abs()
    
    return psim_crl_v1_data

def psim_crl_btwn_subj_avg_v1(psim_crl_v1_data):
    psim_crl_v1_data_bsa = psim_crl_v1_data.groupby(['subject', 'experiment', 'lag', 'psim'])[['crl', 'log_crl']].mean().reset_index()
    psim_crl_v1_data_bsa['abs_lag'] = psim_crl_v1_data_bsa['lag'].abs()
    psim_crl_v1_data_bsa = psim_crl_v1_data_bsa.sort_values('subject')
    
    return psim_crl_v1_data_bsa


# bin phonetic similarities
def bin_phonetic_similarities(psim_crl_v2_data_tr):
    mask = psim_crl_v2_data_tr['psim'] == 0
    # psim = 0 all in one bin
    psim_crl_v2_data_tr.loc[mask, 'bin'] = 0

    # 5 quantiles
    psim_crl_v2_data_tr.loc[~mask, 'bin'] = pd.qcut(psim_crl_v2_data_tr.loc[~mask, 'psim'], q=5, labels=False) + 1

    return psim_crl_v2_data_tr


def psim_irt_sess_v2(evs, psim_fxn):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    sem_sim_cml = pd.read_csv('semantic_similarity/semantic_cml.csv')              # word2vec semantic similarites
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs['trial'] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs['trial'] > 0)]
    max_sp = max(word_evs.serialpos)
    
    # filter all semantic similarities, convert to xarray for query speed
    sem_sim_cml = sem_sim_cml[(sem_sim_cml.word_i.isin(word_evs['item_name'].unique())) & (sem_sim_cml.word_j.isin(word_evs['item_name'].unique()))]
    sem_sim_cml = xr.DataArray(sem_sim_cml.pivot(index='word_i', columns='word_j', values='cosine_similarity'))
    
    psim_irt = []
    
    for i in rec_evs['trial'].unique():
        w_evs = word_evs[word_evs['trial'] == i]
        r_evs = rec_evs[rec_evs['trial'] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs)
        rts = np.array(r_evs.rectime)
        
        # only analyse lists with at least 2 recalls
        if len(recs) >= 2:
            
            ncr = len(np.unique(sp[(1 <= sp) & (sp <= max_sp)]))     # number of correct recalls
            
            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:                  # correct recall
                    words_left.remove(recs[j])                                      # can't transition to already recalled word
                    
                    if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:           # transition to correct recall
                        psim = psim_fxn(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # phonetic similarity
                        ssim = float(sem_sim_cml.loc[recs[j], recs[j+1]].values)                  # semantic similarity
                        lag = sp[j+1] - sp[j]
                        abs_lag = abs(lag)                                                        # |lag|
                        outpos = j + 1                                                            # output position
                        ph = pronouncing_cml.get(recs[j].upper())
                        len_ph = np.mean([len(x) for x in ph])                                    # length of first word (phonemes)
                        irt = rts[j+1] - rts[j]
                        log_irt = np.log(irt) if irt > 0 else irt                                 # ln(irt)
                        
                        if irt <= 30000:                                                          # omit IRTs > 30s
                            psim_irt.append((i, psim, ssim, lag, abs_lag, outpos, ncr, len_ph, irt, log_irt))
                        
                        
    return pd.DataFrame(psim_irt, columns=['list', 'psim', 'ssim', 'lag', 'abs_lag', 'outpos', 'ncr', 'len_ph', 'irt', 'log_irt'])

def psim_irt_parallel_v2(sub, exp, sess, psim_fxn):
    reader = cml.CMLReader(sub, exp, sess)
    evs = reader.load('events')
        
    psim_irt_df = psim_irt_sess_v2(evs, psim_fxn)
    psim_irt_df = apply_md(psim_irt_df, sub, exp, sess)
    
    return psim_irt_df

def psim_irt_sess_avg_v2(psim_irt_v2_data_at):
    psim_irt_v2_data = psim_irt_v2_data_at.groupby(['subject', 'experiment', 'session', 'bin'])[['psim', 'irt']].mean().reset_index()
    
    return psim_irt_v2_data

def psim_irt_btwn_subj_avg_v2(psim_irt_v2_data):
    psim_irt_v2_data_bsa = psim_irt_v2_data.groupby(['subject', 'experiment', 'bin'])[['psim', 'irt']].mean().reset_index()
    psim_irt_v2_data_bsa = psim_irt_v2_data_bsa.sort_values('subject')
    
    return psim_irt_v2_data_bsa