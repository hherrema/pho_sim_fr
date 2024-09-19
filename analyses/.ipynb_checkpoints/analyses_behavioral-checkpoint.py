### Analyses (Behavioral)

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



# ---------- Utility ----------
# apply serial positions to recall events
def apply_serial_positions(w_evs, r_evs, pyFR_toggle):
    # pyFR uses item field, manually apply serial positions to recalls
    if pyFR_toggle:
        words = np.array(w_evs['item'])
        recs = np.array([r for r in r_evs['item'] if r != '<>'])      # remove vocalizations
        sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
        
    # recall events all given default serial position == -999
    elif len(r_evs.serialpos.unique()) == 1 and r_evs.serialpos.unique()[0] == -999:
        words = np.array(w_evs.item_name)
        recs = np.array(r_evs.item_name)
        sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
    
    # serial positions already assigned
    else:
        words = np.array(w_evs.item_name)
        recs = np.array(r_evs.item_name)
        sp = np.array(r_evs.serialpos)
        
    return words, recs, sp

# apply subject, experiment, session, localization, montage to dataframe
def apply_md(df, sub, exp, sess, loc, mont):
    cols = list(df.columns)
    df['subject'] = sub
    df['experiment'] = exp
    df['exp_type'] = 'scalp' if exp == 'ltpFR2' else 'intracranial'
    df['session'] = sess
    df['localization'] = loc
    df['montage'] = sess
    
    
    df = df[['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage'] + cols]
    return df

# sort dataframe by experiment
def sort_by_experiment(df):
    exps = ['pyFR', 'FR1', 'IFR1', 'ltpFR2']
    
    df['experiment'] = pd.Categorical(df['experiment'], categories=exps)
    df = df.sort_values(by=['experiment', 'subject'], ignore_index=True)
    
    return df


# ---------- Parallel Computing ----------
# create dask client for parallel computing
# provide job name, GB per job, number of jobs
def create_client(j_name, gb, nj):
    return da.new_dask_client_slurm(j_name, gb, max_n_jobs=nj, walltime='10000', log_directory='/home1/hherrema/research_projects/pho_sim_fr/logs/')

# run sessions in parallel
def run_parallel_sessions(client, fxn, sub_iter, exp_iter, sess_iter, loc_iter, mont_iter):
    futures = client.map(fxn, sub_iter, exp_iter, sess_iter, loc_iter, mont_iter)
    wait(futures)
    errors = da.get_exceptions(futures, [f'{x[0]}_{x[1]}_{x[2]}' for x in zip(sub_iter, exp_iter, sess_iter)])
    good_futures = da.filter_futures(futures)
    results = client.gather(good_futures)

    return errors, results

# create iterables from data index
def build_iterables(df_select):
    sub_iter = []; exp_iter = []; sess_iter = []
    loc_iter = []; mont_iter = []
    
    spanish_subs = ['R1039M', 'R1070T', 'R1094T', 'R1134T', 'R1331T', 'R1461T', 'R1499T']

    for _, row in tqdm(df_select.iterrows()):
        # only english subjects
        if row.subject in spanish_subs or (row.experiment == 'pyFR' and row.subject[:2] not in ['UP', 'TJ', 'CP', 'BW', 'CH']):
            continue
        #elif row.subject == 'R1719S' and row.experiment == 'ICatFR1' and row.session == 2:
        #    continue
        
        # skip duplicate sessions
        elif row.subject == 'R1100D' and row.experiment == 'FR1' and row.session == 0:
            continue
        #elif row.subject == 'R1486J' and row.experiment == 'catFR1' and row.session in [4, 5, 6, 7]:
        #    continue
        elif row.subject == 'R1275D' and row.experiment == 'FR1' and row.session == 3:
            continue
        #elif row.subject == 'R1310J' and row.experiment == 'catFR1' and row.session == 1:
        #    continue
            
        sub_iter.append(row.subject); exp_iter.append(row.experiment); sess_iter.append(row.session)
        loc_iter.append(row.localization); mont_iter.append(row.montage)
        
    return sub_iter, exp_iter, sess_iter, loc_iter, mont_iter


# ---------- Phonetic Similarity Calculations ----------
# return whether two words are phonetically similar
# 'rhyme' argument to toggle last phomeme (True) or first phoneme (False)
def phonetic_sim_v1(w1, w2, pronouncing_cml, rhyme):
    p1 = pronouncing_cml.get(w1.upper())
    p2 = pronouncing_cml.get(w2.upper())
    
    # one of the words does not have phoneme breakdown
    if not p1 or not p2:
        raise RuntimeError("No phoneme breakdown")
        
    if rhyme:
        idx = -1
    else:
        idx = 0
        
    # iterate over all pairs of pronounciations
    for x in p1:
        for y in p2:
            if x[idx] == y[idx]:
                return True
    
    return False

# number of shared unique phonemes divided by average number of unique phonemes
def phonetic_sim_v2(w1, w2, pronouncing_cml):
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


# ---------- Phonetic Similarity Utility ----------
# return all pairs of phonetically similar words in list of presented words
# return dictionary mapping word to number of phonetic neighbors
def evaluate_list_ps(words, pronouncing_cml):
    # make list of tuples of all pairs of phonetically similar words
    ps_pairs = []
    for w1, w2 in itertools.combinations(words, 2):
        sim_start = phonetic_sim_v1(w1, w2, pronouncing_cml, False)
        rhyme = phonetic_sim_v1(w1, w2, pronouncing_cml, True)
        
        if sim_start or rhyme:
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

# return whether recall is phonetically similar to any word in list
def psim_to_list_item(r, words, pronouncing_cml):
    for w in words:
        sim_start = phonetic_sim_v1(r, w, pronouncing_cml, False)
        rhyme = phonetic_sim_v1(r, w, pronouncing_cml, True)
        
        if sim_start or rhyme:
            return True
        
    return False

# phonetic intrusions permutation control
# 'all_words' does not contain current list words
def psim_intr_pc(r, all_words, max_sp, n_perms, pronouncing_cml):
    psim_pc = []
    for n in range(n_perms):
        psim_toggle = False
        words_control = np.random.choice(all_words, size=max_sp, replace=False)      # select list length words from session at random
        for w in words_control:
            sim_start = phonetic_sim_v1(r, w, pronouncing_cml, False)
            rhyme = phonetic_sim_v1(r, w, pronouncing_cml, True)
            
            if sim_start or rhyme:
                psim_toggle=True
                break                    # stop search if found phonetically similar word
                
        psim_pc.append(psim_toggle)
    
    return np.mean(psim_pc)


# ---------- Phonetic Clustering Score ----------
def pcs_sess_v1(evs, lt, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    pcs_list = []
    
    for i in rec_evs[lt].unique():
        pcs_vals = []                     # phonetic clustering score for each transition
        
        w_evs = word_evs[word_evs[lt] == i]
        r_evs = rec_evs[rec_evs[lt] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs, pyFR_toggle)
        
        # only analyse lists with at least 2 recalls
        if len(recs) >= 2:
            
            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:                           # correct recall
                    
                    # phonetically similar pairs in list, map of words to number of neighbors
                    ps_pairs, ps_dict = evaluate_list_ps(words_left, pronouncing_cml)
                    ps_neighbors = ps_dict.get(recs[j])                                      # number of phonetic neighbors to current recall
                    words_left.remove(recs[j])                                               # can't transition to already recalled word
                    
                    if 1<= sp[j+1] <= max_sp and recs[j+1] in words_left:                    # transition to correct recall
                        
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

def pcs_parallel_v1(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
        
    pcs_df = pcs_sess_v1(evs, lt, exp=='pyFR')
    pcs_df = apply_md(pcs_df, sub, exp, sess, loc, mont)
    
    return pcs_df
    

def pcs_sess_v2(evs, lt, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    pcs_list = []
    
    for i in rec_evs[lt].unique():
        pcs_vals = []                     # phonetic clustering score for each transition
        
        w_evs = word_evs[word_evs[lt] == i]
        r_evs = rec_evs[rec_evs[lt] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs, pyFR_toggle)
        
        # only analyse lists with at least 2 recalls
        if len(recs) >= 2:
            
            # loop over recalls
            words_left = list(words)
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:              # correct recall
                    words_left.remove(recs[j])                                  # can't transition to already recalled word
                    
                    if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:      # transition to correct recall
                        psim = phonetic_sim_v2(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # actual transition phonetic similarity
                        poss = []
                        for l in range(len(words_left)):
                            poss_psim = phonetic_sim_v2(recs[j].upper(), words_left[l].upper(), pronouncing_cml)      # includes actual phonetic similarity
                            poss.append(poss_psim)
                            
                        ptile_rank = percentile_rank_P(psim, poss)
                        if ptile_rank is not None:
                            pcs_vals.append(ptile_rank)
                            
        # calculate phonetic clustering score for each list
        if len(pcs_vals) > 0:
            pcs_list.append((i, np.mean(pcs_vals)))
            
    return pd.DataFrame(pcs_list, columns=['list', 'pcs'])


def pcs_parallel_v2(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
        
    pcs_df = pcs_sess_v2(evs, lt, exp=='pyFR')
    pcs_df = apply_md(pcs_df, sub, exp, sess, loc, mont)
    
    return pcs_df


def pcs_btwn_subj_avg(pcs_data):
    pcs_data_bsa = pcs_data.groupby(['subject', 'exp_type', 'experiment'])['pcs'].mean().reset_index()
    
    # sort by experiment
    pcs_data_bsa = sort_by_experiment(pcs_data_bsa)
    
    return pcs_data_bsa

# ---------- Phonetic-CRL ----------
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

def psim_crl_parallel_v1(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
    
    psim_crl_df = psim_crl_sess_v1(evs, lt, exp=='pyFR')
    psim_crl_df = apply_md(psim_crl_df, sub, exp, sess, loc, mont)
    
    return psim_crl_df

def psim_crl_btwn_subj_avg_v1(pcs_crl_v1_data):
    psim_crl_v1_data_bsa = pcs_crl_v1_data.groupby(['subject', 'exp_type', 'experiment', 'lag'])[['psim_crl', 'non_crl']].mean().reset_index()
    
    # sort by experiment
    psim_crl_v1_data_bsa = sort_by_experiment(psim_crl_v1_data_bsa)
    
    return psim_crl_v1_data_bsa
                
    
def psim_crl_sess_v2(evs, lt, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    psim_irt = []
    
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
                if 1 <= sp[j] <= max_sp and recs[j] in words_left:                  # correct recall
                    words_left.remove(recs[j])                                      # can't transition to already recalled word
                    
                    if 1 <= sp[j+1] <= max_sp and recs[j+1] in words_left:           # transition to correct recall
                        psim = phonetic_sim_v2(recs[j].upper(), recs[j+1].upper(), pronouncing_cml)      # phonetic similarity
                        irt = rts[j+1] - rts[j]
                        if irt <= 30000:                                            # exclude IRTs over 30 s (standardize across experiments, min recall phase duration = 30 s)
                            psim_irt.append((psim, irt))
                        
                        
    return pd.DataFrame(psim_irt, columns=['psim', 'irt'])


def psim_crl_parallel_v2(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
        
    psim_crl_df = psim_crl_sess_v2(evs, lt, exp=='pyFR')
    psim_crl_df = apply_md(psim_crl_df, sub, exp, sess, loc, mont)
    
    return psim_crl_df

def psim_crl_btwn_subj_avg_v2(psim_crl_v2_data):
    psim_crl_v2_data_bsa = psim_crl_v2_data.groupby(['subject', 'exp_type', 'experiment', 'bin'])[['psim', 'irt']].mean().reset_index()
    
    # sort by experiment
    psim_crl_v2_data_bsa = sort_by_experiment(psim_crl_v2_data_bsa)
    
    return psim_crl_v2_data_bsa

# ---------- Phonetic Intrusions ----------
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
        
    return pd.DataFrame({'psim_pli': n_psim_pli / n_pli, 'psim_eli': n_psim_eli / n_eli, 
                         'psim_pli_control': np.mean(psim_pli_control), 'psim_eli_control': np.mean(psim_eli_control)}, index=[0])

def psim_intr_parallel_v1(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
    
    psim_intr_df = psim_intr_sess_v1(evs, lt, exp=='pyFR')
    psim_intr_df = apply_md(psim_intr_df, sub, exp, sess, loc, mont)
    
    return psim_intr_df

def psim_intr_btwn_subj_avg_v1(psim_intr_v1_data):
    psim_intr_v1_data_bsa = psim_intr_v1_data.groupby(['subject', 'exp_type', 'experiment'])[['psim_pli', 'psim_eli', 'psim_pli_control', 'psim_eli_control']].mean().reset_index()
    
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

def psim_intr_parallel_v2(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
    
    psim_intr_df = psim_intr_sess_v2(evs, lt, exp=='pyFR')
    psim_intr_df = apply_md(psim_intr_df, sub, exp, sess, loc, mont)
    
    return psim_intr_df

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

# ---------- Recall Probability ----------
def p_recall_sess(evs, lt):
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]      # don't analyze practice list
    p_recall_df = word_evs.groupby(lt).agg(l_length=('serialpos', 'size'), ncr=('recalled', 'sum'), p_recall=('recalled', 'mean')).reset_index()
    
    if lt == 'trial':
        p_recall_df = p_recall_df.rename(columns={'trial': 'list'})
    
    return p_recall_df

def p_recall_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
        
    p_recall_df = p_recall_sess(evs, lt)
    p_recall_df = apply_md(p_recall_df, sub, exp, sess, loc, mont)
    
    return p_recall_df

def p_recall_btwn_subj_avg(p_recall_data):
    p_recall_data_bsa = p_recall_data.groupby(['subject', 'exp_type', 'experiment', 'l_length'])['p_recall'].mean().reset_index()
    
    # sort by experiment
    p_recall_data_bsa = sort_by_experiment(p_recall_data_bsa)
    
    return p_recall_data_bsa


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

def tcs_sess(evs, lt, pyFR_toggle):
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    tcs_list = []
    
    for i in rec_evs[lt].unique():
        tcs_vals = []                                 # temporal clustering scores for each transition
        
        w_evs = word_evs[word_evs[lt] == i]
        r_evs = rec_evs[rec_evs[lt] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs, pyFR_toggle)
        
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

def tcs_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
        
    tcs_df = tcs_sess(evs, lt, exp=='pyFR')
    tcs_df = apply_md(tcs_df, sub, exp, sess, loc, mont)
    
    return tcs_df

def tcs_btwn_subj_avg(tcs_data):
    tcs_data_bsa = tcs_data.groupby(['subject', 'exp_type', 'experiment'])['tcs'].mean().reset_index()
    
    # sort by experiment
    tcs_data_bsa = sort_by_experiment(tcs_data_bsa)
    
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

def scs_sess(evs, lt, pyFR_toggle):
    sem_sim_cml = pd.read_csv('semantic_similarity/semantic_cml.csv')              # word2vec semantic similarites
    
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]         # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    # filter all semantic similarities, convert to xarray for query speed
    if pyFR_toggle:
        sem_sim_cml = sem_sim_cml[(sem_sim_cml.word_i.isin(word_evs['item'].unique())) & (sem_sim_cml.word_j.isin(word_evs['item'].unique()))]
    else:
        sem_sim_cml = sem_sim_cml[(sem_sim_cml.word_i.isin(word_evs['item_name'].unique())) & (sem_sim_cml.word_j.isin(word_evs['item_name'].unique()))]
    sem_sim_cml = xr.DataArray(sem_sim_cml.pivot(index='word_i', columns='word_j', values='cosine_similarity'))
    scs_list = []
    for i in rec_evs[lt].unique():
        scs_vals = []                              # semantic clustering scores for each transition
        
        w_evs = word_evs[word_evs[lt] == i]
        r_evs = rec_evs[rec_evs[lt] == i]
        
        words, recs, sp = apply_serial_positions(w_evs, r_evs, pyFR_toggle)
        
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

def scs_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    evs = reader.load('events')
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
    
    scs_df = scs_sess(evs, lt, exp=='pyFR')
    scs_df = apply_md(scs_df, sub, exp, sess, loc,  mont)
    
    return scs_df

def scs_btwn_subj_avg(scs_data):
    scs_data_bsa = scs_data.groupby(['subject', 'exp_type', 'experiment'])['scs'].mean().reset_index()
    
    # sort by experiment
    scs_data_bsa = sort_by_experiment(scs_data_bsa)
    
    return scs_data_bsa


# ---------- Correlation of Phonetic Clustering Scores ----------
def pcs_correlations(pcs_v1_data, pcs_v2_data):
    # pcs v1 & pcs v2
    df = pd.merge(pcs_v1_data, pcs_v2_data, on=['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage', 'list'],
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


# ---------- Correlations with Recall Probability ----------
def aggregate_data_beh(pcs_v1_data, pcs_v2_data, tcs_data, scs_data, p_recall_data):
    id_cols = ['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage', 'list']
    
    # pcs v1 & pcs v2
    df = pd.merge(pcs_v1_data, pcs_v2_data, on=id_cols, how='inner', suffixes=('_v1', '_v2'))
    df = df.drop_duplicates(subset=id_cols, keep=False)
    
    # tcs
    df = pd.merge(df, tcs_data, on=id_cols, how='inner')
    df = df.drop_duplicates(subset=id_cols, keep=False)

    # scs
    df = pd.merge(df, scs_data, on=id_cols, how='inner')
    df = df.drop_duplicates(subset=id_cols, keep=False)

    # p_recall
    df = pd.merge(df, p_recall_data, on=id_cols, how='inner')
    df = df.drop_duplicates(subset=id_cols, keep=False)
    
    return df

# correlate with p_recall over lists within sessions
def p_recall_correlations(df):
    stats = []
    for (sub, et, exp, sess, loc, mont), data in tqdm(df.groupby(['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage'])):
        if len(data) > 1:
            # correlations with p_recall
            r_pcs_v1, _ = scipy.stats.pearsonr(data.pcs_v1, data.p_recall, alternative='two-sided')
            r_pcs_v2, _ = scipy.stats.pearsonr(data.pcs_v2, data.p_recall, alternative='two-sided')
            r_tcs, _ = scipy.stats.pearsonr(data.tcs, data.p_recall, alternative='two-sided')
            r_scs, _ = scipy.stats.pearsonr(data.scs, data.p_recall, alternative='two-sided')

            stats.append((sub, et, exp, sess, loc, mont, r_pcs_v1, r_pcs_v2, r_tcs, r_scs))

    # save results as dataframe
    return pd.DataFrame(stats, columns=['subject', 'exp_type', 'experiment', 'session', 'localization', 'montage',
                                        'r_pcs_v1', 'r_pcs_v2', 'r_tcs', 'r_scs'])

# p_recall correlations between-subject average
def p_recall_corr_btwn_subj_avg(p_recall_corrs):
    p_recall_corrs_bsa = p_recall_corrs.groupby(['subject', 'exp_type', 'experiment'])[['r_pcs_v1', 'r_pcs_v2', 'r_tcs', 'r_scs']].mean().reset_index()
    
    return p_recall_corrs_bsa