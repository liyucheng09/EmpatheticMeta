import os
import pickle


class opinion_index:

    def __init__(self, cached_opinion_path):
        self.path = cached_opinion_path

    def _build_opinion_dict(self):
        opinion_lexicon_path = 'data/opinion_lexicon/opinion-words.pickle'
        assert os.path.exists(opinion_lexicon_path), f'No processed opinion word file found. Should be at {opinion_lexicon_path}'

        with open(opinion_lexicon_path, 'rb') as f:
            lexicon = pickle.load(f)
        
        self.positive_index = {}
        self.negative_index = {}
        self.index = {}
        
        for k,v in lexicon.items():
            if v == 'positive': self.positive_index[k] = {}
