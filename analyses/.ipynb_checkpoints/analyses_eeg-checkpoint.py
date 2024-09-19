### Analyses (EEG)

# imports
import numpy as np
import pandas as pd
import cmlreaders as cml
from tqdm import tqdm
import itertools
from cmldask import CMLDask as da
from dask.distributed import wait
from ptsa.data.filters import morlet
from ptsa.data.filters import ButterworthFilter
import sys


# ---------- Utility ----------
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


# ---------- Neural Drift at Encoding ----------
def neural_drift_encoding_sess(reader, lt):
    # paramters
    freqs = np.logspace(np.log10(3), np.log10(180), 10)    # 10 log-spaced frequencies
    buf = 1000       # ms
    
    evs = reader.load('events')
    if reader.subject != evs.subject.unique()[0]:               # pyFR implants have wrong subject code in events dataframe
        evs['subject'] = reader.subject
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]     # don't include practice list
    
    # load in EEG from 750 ms ISI
    if lt == 'trial':
        eeg = reader.load_eeg(word_evs, rel_start=-buf-750, rel_stop=buf, clean=True)
    else:
        pairs = reader.load('pairs')
        eeg = reader.load_eeg(word_evs, rel_start=-buf-750, rel_stop=buf, scheme=pairs)
        
    # get events from EEG container (to ensure they line up with EEG)
    evs = eeg.events
    word_evs = evs[(evs['type'] == 'WORD') & (evs[lt] > 0)]
    max_sp = max(word_evs.serialpos)
    
    # convert to xarray timeseries
    eeg = eeg.to_ptsa()
    
    # filter out line noise with Butterworth filter
    freq_range = [58., 62.]
    b_filter = ButterworthFilter(freq_range=freq_range, filt_type='stop', order=4)
    eeg = b_filter.filter(eeg)
    
    # apply morelet wavelets
    morlet_reps = 5            # wavenumber
    wf = morlet.MorletWaveletFilter(freqs=freqs, width=morlet_reps, output='power', complete=True)
    powers = wf.filter(eeg)
    
    # remove buffers
    powers = powers.isel(time=powers.time>-750); powers = powers.isel(time=powers.time<0)
    
    # z-score (within channel and frequency)
    powers = (powers - powers.mean(['event', 'time'])) / powers.std(['event', 'time'])      # overwrite to save memory
    
    # average across ISI
    powers = powers.mean('time')
    
    nde = []
    # on each list, calculate cosine similarity between all epochs
    for l_idx, l in enumerate(word_evs[lt].unique()):
        idx = l_idx * max_sp
        for i, j in itertools.combinations(range(max_sp), 2):      # i < j
            if j-i <= 2:                                           # only care about serial distances 1 and 2

                cos_sim = np.dot(np.ravel(powers[:, idx+i, :]), np.ravel(powers[:, idx+j, :])) / (np.linalg.norm(np.ravel(powers[:, idx+i, :])) * np.linalg.norm(np.ravel(powers[:, idx+j, :])))     # cosine similarity

                nde.append((l, i+1, j+1, j-i, cos_sim))

        
    return pd.DataFrame(nde, columns=['list', 'sp_i', 'sp_j', 'serial_distance', 'cosine_similarity'])

def neural_drift_encoding_parallel(sub, exp, sess, loc, mont):
    reader = cml.CMLReader(sub, exp, sess, loc, mont)
    
    # trial v. list
    lt = 'trial' if exp == 'ltpFR2' else 'list'
    
    nde_df = neural_drift_encoding_sess(reader, lt)
    nde_df = apply_md(nde_df, sub, exp, sess, loc, mont)
    
    return nde_df


df = cml.get_data_index()
df_intrac = df[df.experiment.isin(['FR1', 'pyFR', 'IFR1'])]
df_scalp = df[(df['experiment'] == 'ltpFR2') & (df['session'] != 23)]
df_select = pd.concat([df_intrac, df_scalp], ignore_index=True)

idx_start = int(sys.argv[1])
idx_end = int(sys.argv[2])

# build iterables
sub_iter, exp_iter, sess_iter, loc_iter, mont_iter = build_iterables(df_select[idx_start: idx_end])
df_sel = pd.DataFrame({'subject': sub_iter, 'experiment': exp_iter, 'session': sess_iter, 'localization': loc_iter, 'montage': mont_iter})

nde_data = []
errors = []
for _, row in tqdm(df_sel.iterrows()):
    try:
        nde = neural_drift_encoding_parallel(row.subject, row.experiment, row.session, row.localization, row.montage)
        nde_data.append(nde)
    except BaseException as e:
        errors.append((row.subject, row.experiment, row.session, row.localization, row.montage, e))
        
nde_data = pd.concat(nde_data, ignore_index=True)
errors = pd.DataFrame(errors, columns=['subject', 'experiment', 'session', 'localization', 'montage', 'exception'])

nde_data.to_csv(f'temp/dataframes/nde_data_{idx_start}_{idx_end}.csv', index=False)
errors.to_csv(f'temp/errors/errors_{idx_start}_{idx_end}.csv', index=False)

"""
# build iterables
sub_iter, exp_iter, sess_iter, loc_iter, mont_iter = build_iterables(df_select)

# set up client
client = create_client('nde', '50GB', 50)

# run in parallel
errors, results = run_parallel_sessions(client, neural_drift_encoding_parallel, sub_iter, exp_iter, sess_iter, loc_iter, mont_iter)

# save out errors and results
nde_data = pd.concat(results, ignore_index=True)
nde_data.to_csv('nde_data.csv', index=False)
errors.to_csv('nde_errors.csv', index=True)
"""