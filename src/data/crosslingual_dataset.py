# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:22:34 2018

@author: pmfin
"""

from pathlib import Path
import pickle
from src.data.preprocess_data import preprocess_data

def get_crosslingual_split(test_language, source=None):
    path_to_split = 'data/processed/all_splits.pkl'
    p = Path(path_to_split)
    if p.exists():
        with open(p,'rb') as file:
            sample = pickle.load(file)
    else:
        print("No crosslingual split file. Generating in {}".format(p))
        preprocess_data()
        if p.exists():
            with open(p,'rb') as file:
                sample = pickle.load(file)
        else:
            print("ERROR: preprocessing data failed. Path checked was {}".format(p))
            return None
    
    # This is the case when we're training. The split SOURCE selected for testing
    # does not affect the training split (only the split LANGUAGE)
    if source == None:
        datasets_per_language = {"english": ["News", "WikiNews", "Wikipedia"],
                         "spanish": ["Spanish"],
                         "german": ["German"],
                         "french": ["French"]}
        source = datasets_per_language[test_language][0]
    
    test_lang_source = test_language + "_" + source
    trainset = sample[test_lang_source]['train']
    devset = sample[test_lang_source]['dev']
    testset = sample[test_lang_source]['test']
    
    return trainset, devset, testset

if __name__ == '__main__':
    get_crosslingual_split('english','Wikipedia')