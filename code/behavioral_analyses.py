### auditory/phonetic similarity behavioral analyses

# imports
import json
import numpy as np
from .utils import make_phonemes_cml, evaluate_list, phonetic_sim, phonetic_blend

# phonetic clustering score
def phonetic_clustering_score(word_evs, rec_evs, max_sp, pyFR_toggle, rhyme_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
    
    pcs_list = []        # phonetic clustering score for each list
    
    # iterate over lists
    for i in rec_evs.list.unique():
        pcs_vals = []    # phonetic clustering score for each transition
        w_evs = word_evs[word_evs['list'] == i]
        r_evs = rec_evs[rec_evs['list'] == i]
        
        # pyFR uses item field, manually apply serial positions to recalls
        if pyFR_toggle:
            words = np.array(w_evs.item)
            recs = np.array([r for r in r_evs.item if r != '<>'])      # remove vocalizations
            #recs = np.array(r_evs.item)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            
        # assign serial positions for sessions where recall events given defaul serial position = -999
        elif len(r_evs.serialpos.unique()) == 1 and r_evs.serialpos.unique()[0] == -999:
            words = np.array(w_evs.item_name)
            recs = np.array(r_evs.item_name)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
        
        # use assigned serial positions
        else:
            words = np.array(w_evs.item_name)         # current list words
            recs = np.array(r_evs.item_name)          # current list recalls
            sp = np.array(r_evs.serialpos)            # current list serial positions

        # dictionary mapping list words to phonemes
        #phones_dict = make_phonemes_dict(words)
        phones_dict = make_phonemes_cml(words, pronouncing_cml)
        
        # only analyze lists with at least 2 recalls
        if len(recs) >= 2:
            
            # loop over recalls
            words_left = list(words)
            words_recalled = []         # avoid including repetitions
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] not in words_recalled:      # correct recall
                    
                    # phonetically similar pairs in list, dictionary mapping words to number of phonetic neighbors
                    ps_pairs, ps_dict = evaluate_list(words_left, phones_dict, rhyme_toggle)
                    ps_neighbors = ps_dict.get(recs[j])                  # number of phonetic neighbors to current recall
                    words_left.remove(recs[j])                               # can't transition to already recalled word
                    words_recalled.append(recs[j])

                    if 1 <= sp[j+1] <= max_sp and recs[j+1] not in words_recalled:    # transition to correct recall

                        # possible to transition to phonetically similar word
                        if ps_neighbors and ps_neighbors > 0:
                            
                            # calculate phonetic clustering score for each transition
                            # subtract probability of randomly sampling phonetically similar word
                            
                            # actually transition to phonetically similar word
                            if (recs[j], recs[j+1]) in ps_pairs or (recs[j+1], recs[j]) in ps_pairs:
                                pcs = 1 - (ps_neighbors / len(words_left))
                                #pcs = 1 - (ps_neighbors / np.sum(list(ps_dict.values())))
                                
                            # transition to not phonetically similar word
                            else:
                                pcs = 0 - (ps_neighbors / len(words_left))
                                #pcs = 0 - (ps_neighbors / np.sum(list(ps_dict.values())))
                                
                            # put phonetic clustering scores for each transition in list
                            pcs_vals.append(pcs)

            # calculate phonetic clustering score for each list
            if len(pcs_vals) > 0:
                pcs_list.append(np.mean(pcs_vals))
                
    return pcs_list


# inter-response times
def pho_sim_irt(word_evs, rec_evs, max_sp, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
    
    psim_irt = []      # irts for transitions between phonetically similar words
    non_irt = []       # irts for transitions between non-phonetically similar words
    psim_lag = []      # lags for transitions between phonetically similar words
    non_lag = []       # lags for transitions between non-phonetically similar words
    
    for i in rec_evs.list.unique():
        w_evs = word_evs[word_evs['list'] == i]
        r_evs = rec_evs[rec_evs['list'] == i]
        
        # pyFR uses item field, manually apply serial positions to recalls
        if pyFR_toggle:
            words = np.array(w_evs.item)
            recs = np.array([r for r in r_evs.item if r != '<>'])      # remove vocalizations
            #recs = np.array(r_evs.item)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            rts = np.array(r_evs.rectime)
            
        # assign serial positions for sessions where recall events given defaul serial position = -999
        elif len(r_evs.serialpos.unique()) == 1 and r_evs.serialpos.unique()[0] == -999:
            words = np.array(w_evs.item_name)
            recs = np.array(r_evs.item_name)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            rts = np.array(r_evs.rectime)
        
        # use assigned serial positions
        else:
            words = np.array(w_evs.item_name)         # current list words
            recs = np.array(r_evs.item_name)          # current list recalls
            sp = np.array(r_evs.serialpos)            # current list serial positions
            rts = np.array(r_evs.rectime)             # current list response times
        
        # dictionary mapping list words to phonemes
        phones_dict = make_phonemes_cml(words, pronouncing_cml)

        # only analyze lists with at least 2 recalls
        if len(recs) >= 2:
            # loop over recalls
            words_left = list(words)
            words_recalled = []         # avoid including repetitions
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] not in words_recalled:      # correct recall
                    words_left.remove(recs[j])                               # can't transition to already recalled word
                    words_recalled.append(recs[j])

                    if 1 <= sp[j+1] <= max_sp and recs[j+1] not in words_recalled:    # transition to correct recall
                        sim_start = phonetic_sim(recs[j], recs[j+1], phones_dict, False)
                        rhyme = phonetic_sim(recs[j], recs[j+1], phones_dict, True)
                        lag = sp[j+1] - sp[j]

                        if sim_start or rhyme:
                            psim_irt.append(rts[j+1] - rts[j])
                            psim_lag.append(abs(sp[j+1] - sp[j]))
                        else:
                            non_irt.append(rts[j+1] - rts[j])
                            non_lag.append(abs(sp[j+1] - sp[j]))
    
    return np.mean(psim_irt), np.mean(non_irt), np.mean(psim_lag), np.mean(non_lag)


# conditional response latency
def pho_sim_crl(word_evs, rec_evs, max_sp, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
    
    psim_crl = np.zeros(2 * max_sp - 1)
    psim_count = np.zeros_like(psim_crl)
    non_crl = np.zeros(2 * max_sp - 1)
    non_count = np.zeros_like(non_crl)
    
    for i in rec_evs.list.unique():
        w_evs = word_evs[word_evs['list'] == i]
        r_evs = rec_evs[rec_evs['list'] == i]
        
        # pyFR uses item field, manually apply serial positions to recalls
        if pyFR_toggle:
            words = np.array(w_evs.item)
            recs = np.array([r for r in r_evs.item if r != '<>'])      # remove vocalizations
            #recs = np.array(r_evs.item)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            rts = np.array(r_evs.rectime)
            
        # assign serial positions for sessions where recall events given defaul serial position = -999
        elif len(r_evs.serialpos.unique()) == 1 and r_evs.serialpos.unique()[0] == -999:
            words = np.array(w_evs.item_name)
            recs = np.array(r_evs.item_name)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            rts = np.array(r_evs.rectime)
        
        # use assigned serial positions
        else:
            words = np.array(w_evs.item_name)         # current list words
            recs = np.array(r_evs.item_name)          # current list recalls
            sp = np.array(r_evs.serialpos)            # current list serial positions
            rts = np.array(r_evs.rectime)             # current list response times
        
        # dictionary mapping list words to phonemes
        phones_dict = make_phonemes_cml(words, pronouncing_cml)

        # only analyze lists with at least 2 recalls
        if len(recs) >= 2:
            # loop over recalls
            words_left = list(words)
            words_recalled = []         # avoid including repetitions
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] not in words_recalled:      # correct recall
                    words_left.remove(recs[j])                               # can't transition to already recalled word
                    words_recalled.append(recs[j])

                    if 1 <= sp[j+1] <= max_sp and recs[j+1] not in words_recalled:    # transition to correct recall
                        sim_start = phonetic_sim(recs[j], recs[j+1], phones_dict, False)
                        rhyme = phonetic_sim(recs[j], recs[j+1], phones_dict, True)
                        lag = sp[j+1] - sp[j]

                        if sim_start or rhyme:
                            psim_crl[lag + max_sp - 1] += rts[j+1] - rts[j]
                            psim_count[lag + max_sp - 1] += 1
                        else:
                            non_crl[lag + max_sp - 1] += rts[j+1] - rts[j]
                            non_count[lag + max_sp - 1] += 1
    
    return psim_crl / psim_count, non_crl / non_count


# conditional response latency (phonetic blends v. non phonetic neighbors)
def pho_blend_crl(word_evs, rec_evs, max_sp, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
    
    pbl_crl = np.zeros(2 * max_sp - 1)
    pbl_count = np.zeros_like(pbl_crl)
    non_crl = np.zeros(2 * max_sp - 1)
    non_count = np.zeros_like(non_crl)
    
    for i in rec_evs.list.unique():
        w_evs = word_evs[word_evs['list'] == i]
        r_evs = rec_evs[rec_evs['list'] == i]
        
        # pyFR uses item field, manually apply serial positions to recalls
        if pyFR_toggle:
            words = np.array(w_evs.item)
            recs = np.array([r for r in r_evs.item if r != '<>'])      # remove vocalizations
            #recs = np.array(r_evs.item)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            rts = np.array(r_evs.rectime)
            
        # assign serial positions for sessions where recall events given defaul serial position = -999
        elif len(r_evs.serialpos.unique()) == 1 and r_evs.serialpos.unique()[0] == -999:
            words = np.array(w_evs.item_name)
            recs = np.array(r_evs.item_name)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            rts = np.array(r_evs.rectime)
        
        # use assigned serial positions
        else:
            words = np.array(w_evs.item_name)         # current list words
            recs = np.array(r_evs.item_name)          # current list recalls
            sp = np.array(r_evs.serialpos)            # current list serial positions
            rts = np.array(r_evs.rectime)             # current list response times
        
        # dictionary mapping list words to phonemes
        phones_dict = make_phonemes_cml(words, pronouncing_cml)

        # only analyze lists with at least 2 recalls
        if len(recs) >= 2:
            # loop over recalls
            words_left = list(words)
            words_recalled = []         # avoid including repetitions
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] not in words_recalled:      # correct recall
                    words_left.remove(recs[j])                               # can't transition to already recalled word
                    words_recalled.append(recs[j])

                    if 1 <= sp[j+1] <= max_sp and recs[j+1] not in words_recalled:    # transition to correct recall
                        sim_start = phonetic_sim(recs[j], recs[j+1], phones_dict, False)
                        rhyme = phonetic_sim(recs[j], recs[j+1], phones_dict, True)
                        blend = phonetic_blend(recs[j], recs[j+1], phones_dict, one_way=True)    # only forward blends
                        lag = sp[j+1] - sp[j]
                        
                        # only compare phonetic blends v. non phonetically similar transitions
                        if blend:
                            pbl_crl[lag + max_sp - 1] += rts[j+1] - rts[j]
                            pbl_count[lag + max_sp - 1] += 1
                        elif sim_start == False and rhyme == False:
                            non_crl[lag + max_sp - 1] += rts[j+1] - rts[j]
                            non_count[lag + max_sp - 1] += 1
                        else:
                            pass
    
    return pbl_crl / pbl_count, non_crl / non_count


# conditional response latency (including phonetic blends)
# consider similar start, rhyme, and blends as phonetically similar
def pho_sim_blend_crl(word_evs, rec_evs, max_sp, pyFR_toggle):
    with open('pronouncing_dictionaries/pronouncing_cml.json', 'r') as f:
        pronouncing_cml = json.load(f)
    
    psim_crl = np.zeros(2 * max_sp - 1)
    psim_count = np.zeros_like(psim_crl)
    non_crl = np.zeros(2 * max_sp - 1)
    non_count = np.zeros_like(non_crl)
    
    for i in rec_evs.list.unique():
        w_evs = word_evs[word_evs['list'] == i]
        r_evs = rec_evs[rec_evs['list'] == i]
        
        # pyFR uses item field, manually apply serial positions to recalls
        if pyFR_toggle:
            words = np.array(w_evs.item)
            recs = np.array([r for r in r_evs.item if r != '<>'])      # remove vocalizations
            #recs = np.array(r_evs.item)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            rts = np.array(r_evs.rectime)
            
        # assign serial positions for sessions where recall events given defaul serial position = -999
        elif len(r_evs.serialpos.unique()) == 1 and r_evs.serialpos.unique()[0] == -999:
            words = np.array(w_evs.item_name)
            recs = np.array(r_evs.item_name)
            sp = np.array([np.argwhere(words == r)[0][0] + 1 if r in words else -999 for r in recs])
            rts = np.array(r_evs.rectime)
        
        # use assigned serial positions
        else:
            words = np.array(w_evs.item_name)         # current list words
            recs = np.array(r_evs.item_name)          # current list recalls
            sp = np.array(r_evs.serialpos)            # current list serial positions
            rts = np.array(r_evs.rectime)             # current list response times
        
        # dictionary mapping list words to phonemes
        phones_dict = make_phonemes_cml(words, pronouncing_cml)

        # only analyze lists with at least 2 recalls
        if len(recs) >= 2:
            # loop over recalls
            words_left = list(words)
            words_recalled = []         # avoid including repetitions
            for j in range(len(recs) - 1):
                if 1 <= sp[j] <= max_sp and recs[j] not in words_recalled:      # correct recall
                    words_left.remove(recs[j])                               # can't transition to already recalled word
                    words_recalled.append(recs[j])

                    if 1 <= sp[j+1] <= max_sp and recs[j+1] not in words_recalled:    # transition to correct recall
                        sim_start = phonetic_sim(recs[j], recs[j+1], phones_dict, False)
                        rhyme = phonetic_sim(recs[j], recs[j+1], phones_dict, True)
                        blend = phonetic_blend(recs[j], recs[j+1], phones_dict, one_way=True)
                        lag = sp[j+1] - sp[j]

                        if sim_start or rhyme or blend:
                            psim_crl[lag + max_sp - 1] += rts[j+1] - rts[j]
                            psim_count[lag + max_sp - 1] += 1
                        else:
                            non_crl[lag + max_sp - 1] += rts[j+1] - rts[j]
                            non_count[lag + max_sp - 1] += 1
    
    return psim_crl / psim_count, non_crl / non_count
