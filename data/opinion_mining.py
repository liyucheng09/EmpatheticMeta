import spacy
from spacy.tokens import Doc
import os
import pickle
from spacy.symbols import VERB, AUX
import datasets
import sys

class OpinionMiner:
    """
    Lexicon based opinion mining engine.
    """

    def __init__(self, ):
        self.nlp = spacy.load("en_core_web_sm")
        self._load_opinion_lexicon()
    
    def _load_opinion_lexicon(self):
        negative_path = 'data/opinion_lexicon/negative-words.txt'
        positive_path = 'data/opinion_lexicon/positive-words.txt'
        cache_path = 'data/opinion_lexicon/opinion-words.pickle'

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.opinion_lexicon = pickle.load(f)
            return

        self.opinion_lexicon = {}
        with open(negative_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line or line.startswith(';'):
                    continue
                self.opinion_lexicon[line.strip()] = 'negative'
        with open(positive_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line or line.startswith(';'):
                    continue
                self.opinion_lexicon[line.strip()] = 'positive'
        
        with open(cache_path, 'wb') as f:
            pickle.dump(self.opinion_lexicon, f)
        
        print(f'Opinion lexicon saved to {cache_path}!')

        return
    
    def syntax_iterators(self, doc, comp = True, conj = True):
        labels = [
            'prep',
            'dobj',
            'pobj',
            'advmod',
            'obj',
            'amod',
            'advcl',
            'attr'
        ]

        # includes Complements in output?
        if comp:
            labels.extend(
                [
                    'pcomp',
                    'xcomp',
                    'acomp'
                ]
            )
        
        if conj:
            conj_labels = [
                'conj',
                'appos'
            ]
            labels.extend(conj_labels)
            conj_labels = [doc.vocab.strings.add(label) for label in conj_labels]

        np_deps = [doc.vocab.strings.add(label) for label in labels]
        conj = doc.vocab.strings.add("conj")

        def go_deeper(word):
            rl = []
            for child in word.children:
                if child.i <= word.i or child.dep not in np_deps:
                    continue                    
                r = go_deeper(child)

                if conj and child.dep in conj_labels:
                    # if child.pos in (VERB, AUX):
                    #     continue
                    if len([child for child in word.children if child.i > word.i and child.dep in np_deps]) == 1:
                        rl.append([word])
                    for line in r:
                        rl.append(line)
                else:
                    for line in r:
                        line.extend([word])
                        rl.append(line)
            if not rl: rl.append([word])
            return rl

        vps = []
        for i, word in enumerate(doc):
            if word.pos not in (VERB, AUX):
                continue
            if  word.head != word and word.head.pos in (VERB, AUX):
                continue
            r = go_deeper(word)
            vps.extend(r)
        
        vps_with_opinion = []
        for vp in vps:
            for token in vp:
                if token.lemma_ in self.opinion_lexicon:
                    vps_with_opinion.append([token.text for token in vp][::-1])
                    break
        
        return vps_with_opinion
    
    def __call__(self, chunk, titles, id, n_process = 1):
        docs = self.nlp.pipe(chunk, n_process = n_process)
        output_opinions = {}
        for title, doc in zip(titles, docs):
            opinions = self.syntax_iterators(doc)
            output_opinions[title] = opinions
        
        output_path = f'data/wiki_opinion/checkpoint_{id}.pickle'
        with open(output_path, 'wb') as f:
            pickle.dump(output_opinions, f)
        print(f'Opinion saved to {output_path}.')

if __name__ == '__main__':

    # start, end = sys.argv[1:]
    # start, end = int(start), int(end)

    opinion_miner = OpinionMiner()
    chunks_size_per_checkpoint = 60
    wiki = datasets.load_dataset("wikipedia", "20220301.en", split="train")

    for i in range(0, 60, chunks_size_per_checkpoint):
        chunk = wiki[i: i+chunks_size_per_checkpoint]
        titles = chunk['title']
        paragraphs = []
        for page in chunk['text']:
            first_para = page.split('\n\n', 1)[0]
            paragraphs.append(first_para)
        
        opinion_miner(paragraphs, titles, i, n_process=1)