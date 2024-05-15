# auditory/phonetic similarity utility functions

# imports
import nltk
from nltk.corpus import cmudict
import itertools

# make dictionary mapping words in wordpool to phonemes
def make_phonemes_dict(words):
    # download CMU Pronouncing Dictionary
    nltk.download('cmudict')

    phone_dict = {}
    
    # load in the CMU Pronouncing Dictionary
    pronouncing_dict = cmudict.dict()
    
    for word in words:
        phonemes = pronouncing_dict.get(word.lower(), None)
        if phonemes:
            phone_dict[word] = phonemes
        elif word.lower() == 'strainer':       # manually assign phoneme breakdown
            phone_dict[word] = [['S', 'T', 'R', 'EY1', 'N', 'ER0']]
        else:
            print(f'No found phonemes for {word}.')
    
    return phone_dict


# make dictionary mapping words in CML wordpool to phonemes
def make_phonemes_cml(words, pronouncing_cml):
    phone_dict = {}
    
    for word in words:
        phonemes = pronouncing_cml.get(word.lower(), None)
        if phonemes:
            phone_dict[word] = phonemes
        elif word.lower() == 'strainer':       # manually assign phoneme breakdown
            phone_dict[word] = [['S', 'T', 'R', 'EY1', 'N', 'ER0']]
        else:
            print(f'No found phonemes for {word}.')
    
    return phone_dict


# return whether two words are phonetically similar
# 'rhyme' argument to toggle last phomeme (True) or first phoneme (False)
def phonetic_sim(w1, w2, phone_dict, rhyme):
    p1 = phone_dict.get(w1.upper())
    p2 = phone_dict.get(w2.upper())
    
    if rhyme:
        idx = -1
    else:
        idx = 0
        
    # iterate over all pairs of pronunciations
    for x in p1:
        for y in p2:
            if x[idx] == y[idx]:
                return True
    return False


# return whether two words for a phonetic blend
# last phoneme of one word same as first phoneme of the other
# 'one_way' argument to toggle only in the forwards direction
def phonetic_blend(w1, w2, phone_dict, one_way):
    p1 = phone_dict.get(w1.upper())
    p2 = phone_dict.get(w2.upper())
    
    if one_way:
        for x in p1:
            for y in p2:
                if x[-1] == y[0]:
                    return True
    else:
        for x in p1:
            for y in p2:
                if x[-1] == y[0] or x[0] == y[-1]:
                    return True
                
    return False

 
# return all pairs of phonetically similar words from a list of presented words
# also return dictionary mapping word to number of neighbors
# 'rhyme_toggle' argument to toggle whether to consider rhyming words as phonetic neighbors
def evaluate_list(words, phones_dict, rhyme_toggle=True):
    # make list of tuples of all pairs of phonetically similar words
    ps_pairs = []
    for w1, w2 in itertools.combinations(words, 2):
        sim_start = phonetic_sim(w1, w2, phones_dict, False)
        
        if rhyme_toggle:
            rhyme = phonetic_sim(w1, w2, phones_dict, True)
            if sim_start or rhyme:
                ps_pairs.append((w1, w2))
        else:
            if sim_start:
                ps_pairs.append((w1, w2))

    # make dictionary mapping word to number of neighbors
    ps_dict = {}
    for word in words:
        n_sim = 0
        for pair in ps_pairs:
            if word in pair:
                n_sim += 1
        ps_dict[word] = n_sim
        
    return ps_pairs, ps_dict
