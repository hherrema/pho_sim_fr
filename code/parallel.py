### parallelization code for auditory/phonetic similarity analyses

# imports
import numpy as np
import cmlreaders as cml
from tqdm import tqdm
from cmldask import CMLDask as da
from dask.distributed import wait, as_completed, progress
from .behavioral_analyses import phonetic_clustering_score, pho_sim_irt, pho_sim_crl, pho_blend_crl, pho_sim_blend_crl

# ---------- Utility ----------

# create dask client for parallel computing
# provide job name, GB per job, number of jobs
def create_client(j_name, gb, nj):
    return da.new_dask_client_slurm(j_name, gb, max_n_jobs=nj, walltime='10000', log_directory='/home1/hherrema/research_projects/auditory_clustering/logs/')


# run sessions in parallel
def run_parallel_sessions(client, fxn, sub_iter, exp_iter, sess_iter, loc_iter, mont_iter):
    futures = client.map(fxn, sub_iter, exp_iter, sess_iter, loc_iter, mont_iter)
    wait(futures)
    errors = da.get_exceptions(futures, sub_iter)
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
        elif row.subject == 'R1719S' and row.experiment == 'ICatFR1' and row.session == 2:
            continue
        # skip duplicate sessions
        elif row.subject == 'R1100D' and row.experiment == 'FR1' and row.session == 0:
            continue
        elif row.subject == 'R1486J' and row.experiment == 'catFR1' and row.session in [4, 5, 6, 7]:
            continue
            
        sub_iter.append(row.subject); exp_iter.append(row.experiment); sess_iter.append(row.session)
        loc_iter.append(row.localization); mont_iter.append(row.montage)
        
    return sub_iter, exp_iter, sess_iter, loc_iter, mont_iter

# ---------- Behavioral ----------

# phonetic clustering score
def pcs_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(subject=sub, experiment=exp, session=sess, localization=loc, montage=mont)
    evs = reader.load('events')
    
    # some pyFR sessions give errors for sorting eegfiles (don't contain errors)
    if exp != 'pyFR':
        evs = cml.sort_eegfiles(evs)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs.list > 0)]        # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs.list > 0)]
    max_sp = np.max(word_evs.serialpos)           # largest serial position

    # phonetic clustering score
    pcs_list = phonetic_clustering_score(word_evs, rec_evs, max_sp, exp == 'pyFR', rhyme_toggle=False)
    
    return sub, exp, sess, loc, mont, np.mean(pcs_list)


# inter-response times
def pho_sim_irt_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(subject=sub, experiment=exp, session=sess, localization=loc, montage=mont)
    evs = reader.load('events')

    # some pyFR sessions give errors for sorting eegfiles (don't contain errors)
    if exp != 'pyFR':
        evs = cml.sort_eegfiles(evs)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs.list > 0)]        # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs.list > 0)]
    max_sp = np.max(word_evs.serialpos)           # largest serial position
    
    # phonetic clustering inter-response times
    psim_irt, non_irt, psim_lag, non_lag = pho_sim_irt(word_evs, rec_evs, max_sp, exp == 'pyFR')

    return sub, exp, sess, loc, mont, psim_irt, non_irt, psim_lag, non_lag


# conditional response latency
def pho_sim_crl_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(subject=sub, experiment=exp, session=sess, localization=loc, montage=mont)
    evs = reader.load('events')

    # some pyFR sessions give errors for sorting eegfiles (don't contain errors)
    if exp != 'pyFR':
        evs = cml.sort_eegfiles(evs)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs.list > 0)]        # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs.list > 0)]
    max_sp = np.max(word_evs.serialpos)           # largest serial position
    
    # phonetic clustering inter-response times
    psim_crl, non_crl = pho_sim_crl(word_evs, rec_evs, max_sp, exp == 'pyFR')
    
    # only keep lags -7 to 7
    psim_crl = np.concatenate([psim_crl[(max_sp - 1) - 7: max_sp - 1], psim_crl[max_sp: max_sp + 7]])
    non_crl = np.concatenate([non_crl[(max_sp - 1) - 7: max_sp - 1], non_crl[max_sp: max_sp + 7]])

    return [sub, exp, sess, loc, mont] + list(psim_crl) + list(non_crl)


# conditional response latency (phonetic blends v. non phonetic neighbors)
def pho_blend_crl_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(subject=sub, experiment=exp, session=sess, localization=loc, montage=mont)
    evs = reader.load('events')

    # some pyFR sessions give errors for sorting eegfiles (don't contain errors)
    if exp != 'pyFR':
        evs = cml.sort_eegfiles(evs)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs.list > 0)]        # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs.list > 0)]
    max_sp = np.max(word_evs.serialpos)           # largest serial position
    
    # phonetic clustering inter-response times
    pbl_crl, non_crl = pho_blend_crl(word_evs, rec_evs, max_sp, exp == 'pyFR')
    
    # only keep lags -7 to 7
    pbl_crl = np.concatenate([pbl_crl[(max_sp - 1) - 7: max_sp - 1], pbl_crl[max_sp: max_sp + 7]])
    non_crl = np.concatenate([non_crl[(max_sp - 1) - 7: max_sp - 1], non_crl[max_sp: max_sp + 7]])

    return [sub, exp, sess, loc, mont] + list(pbl_crl) + list(non_crl)


# conditional response latency (including phonetic blends)
def pho_sim_blend_crl_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(subject=sub, experiment=exp, session=sess, localization=loc, montage=mont)
    evs = reader.load('events')

    # some pyFR sessions give errors for sorting eegfiles (don't contain errors)
    if exp != 'pyFR':
        evs = cml.sort_eegfiles(evs)
        
    word_evs = evs[(evs['type'] == 'WORD') & (evs.list > 0)]        # don't analyze practice list
    rec_evs = evs[(evs['type'] == 'REC_WORD') & (evs.list > 0)]
    max_sp = np.max(word_evs.serialpos)           # largest serial position
    
    # phonetic clustering inter-response times
    psim_crl, non_crl = pho_sim_blend_crl(word_evs, rec_evs, max_sp, exp == 'pyFR')
    
    # only keep lags -7 to 7
    psim_crl = np.concatenate([psim_crl[(max_sp - 1) - 7: max_sp - 1], psim_crl[max_sp: max_sp + 7]])
    non_crl = np.concatenate([non_crl[(max_sp - 1) - 7: max_sp - 1], non_crl[max_sp: max_sp + 7]])

    return [sub, exp, sess, loc, mont] + list(psim_crl) + list(non_crl)
