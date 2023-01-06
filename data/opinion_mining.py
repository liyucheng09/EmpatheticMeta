import spacy
from spacy.tokens import Doc
import os
import pickle
from spacy.symbols import VERB, AUX
import datasets
import sys
import threading
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from pyopenie import OpenIE5


def get_first_few_paras(doc):
    paragraphs = doc.split('\n\n', 5)[:5]
    new_paras = []
    for i, para in enumerate(paragraphs):
        if len(para) < 60 and i != 0:
            break
        new_paras.append(para)
    return new_paras


class OpinionMiner:
    """
    Lexicon based opinion mining engine.
    """

    def __init__(self, merge_noun_chunks = True, left = True):
        self.nlp = spacy.load("en_core_web_sm", disable=['ner'])
        if merge_noun_chunks:
            self.nlp.add_pipe("merge_noun_chunks")
        self._load_opinion_lexicon()
        self.left = left
    
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
    
    def syntax_iterators(self, doc, comp = True, use_conj = True):
        labels = [
            'prep',
            'dobj',
            'pobj',
            'advmod',
            'obj',
            'amod',
            'advcl',
            'attr',
            'agent'
        ]

        if self.left:
            left_labels = [
                'amod',
                'advmod',
                'attr'
            ]
            left_label = [doc.vocab.strings.add(label) for label in left_labels]

        # includes Complements in output?
        if comp:
            labels.extend(
                [
                    'pcomp',
                    'xcomp',
                    'acomp'
                ]
            )
        
        if use_conj:
            conj_labels = [
                'conj',
                'appos'
            ]
            labels.extend(conj_labels)
            conj_labels = [doc.vocab.strings.add(label) for label in conj_labels]

        np_deps = [doc.vocab.strings.add(label) for label in labels]
        conj = doc.vocab.strings.add("conj")

        def go_deeper(word):
            ll = []
            rl = []
            for child in word.children:
                if child.dep not in np_deps:
                    continue
                if child.i <= word.i:
                    if not self.left: continue
                    
                    r = go_deeper(child)
                    for line in r:
                        # line2 = [word]
                        # line2.extend(line)
                        ll.append(line)
                else:
                    r = go_deeper(child)
                    if use_conj and child.dep in conj_labels:
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
            if not ll: ll.append([])
            final = []
            for lli in ll:
                for rli in rl:
                    final.append(rli + lli)
            return final

        vps = []
        for i, word in enumerate(doc):
            if word.pos not in (VERB, AUX):
                continue
            if  word.head != word and word.head.pos in (VERB, AUX):
                continue
            r = go_deeper(word)
            vps.extend(r)
        
        vps_with_opinion = []
        with self.nlp.select_pipes(enable=['tokenizer', 'lemmatizer']):
            for vp in vps:
                # all_tokens = []
                # for token in vp[::-1]:
                #     all_tokens.extend(self.nlp(token.lemma_))
                opinion_words = []
                # for t in all_tokens:
                for t in vp:
                    if t.lemma_ in self.opinion_lexicon:
                        opinion_words.append(t.lemma_)
                if opinion_words:
                    vps_with_opinion.append(([token.lemma_ for token in vp[::-1]], opinion_words))
                    # vps_with_opinion.append(([token.text for token in all_tokens], opinion_words))
        
        return vps_with_opinion
    
    def __call__(self, chunk, titles, id, n_process = 1):
        docs = self.nlp.pipe(chunk, n_process = n_process)
        output_opinions = {}
        for title, doc in tqdm(zip(titles, docs)):
            opinions = self.syntax_iterators(doc)
            output_opinions[title] = opinions
        
        output_path = f'data/wiki_opinion_test/checkpoint_{id}.pickle'
        with open(output_path, 'wb') as f:
            pickle.dump(output_opinions, f)
        print(f'Opinion saved to {output_path}.')

class OpenIEMiner:
    """A python wapper for the Java OpenIE miner.
    """
    def __init__(self):
        self.OpenIE = OpenIE5('http://localhost:8000')
        self._load_opinion_lexicon()

    def _load_opinion_lexicon(self):
        cache_path = 'data/opinion_lexicon/opinion-words.pickle'

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.opinion_lexicon = pickle.load(f)
            return
    
    def __call__(self, chunk, titles, id, output_dir, n_process = 1):
        output_opinions = {}
        for title, para in tqdm(zip(titles, chunk)):
            extractions = self._open_ie_miner(para)
            output_opinions[title] = extractions
        output_path = os.path.join(output_dir, f'checkpoint_{id}.pickle') 
        with open(output_path, 'wb') as f:
            pickle.dump(output_opinions, f)
        print(f'Opinion saved to {output_path}.')

    def _open_ie_miner(self, para):
        sents = sent_tokenize(para)
        triples = []
        output = []
        for sent in sents:
            sent = sent.encode('ascii', errors = 'ignore').decode()
            if len(sent)>240:
                continue
            for attempt in range(3):
                try:
                    triple = self.OpenIE.extract(sent)
                except:
                    continue
                else:
                    break
            else:
                continue
            triples.extend(triple)
        for line in triples:
            confidence = line['confidence']
            if confidence<0.8:
                continue
            triple = line['extraction']
            arg1 = triple['arg1']['text']
            rel = triple['rel']['text']
            arg2s = [ i['text'] for i in triple['arg2s'] ]
            triple = [arg1, rel] + arg2s
            output.append(triple)
        return output


if __name__ == '__main__':

    thread_id, output_dir = sys.argv[1:]
    thread_id = int(thread_id)

    chunks_size_per_thread = 613467
    chunks_size_per_checkpoint = 20000
    wiki = datasets.load_dataset("wikipedia", "20220301.en", split="train")

    start = 324000 + (chunks_size_per_thread*thread_id)
    end = start + chunks_size_per_thread

    for i in range(start, end, chunks_size_per_checkpoint):
        # opinion_miner = OpinionMiner(left=False)
        opinion_miner = OpenIEMiner()
        chunk = wiki[i: i+chunks_size_per_checkpoint]
        titles = []
        paragraphs = []
        for title, page in zip(chunk['title'], chunk['text']):
            if len(page) < 13000:
                continue
            titles.append(title)
            paras = get_first_few_paras(page)
            paragraphs.append('\n'.join(paras))
        print('Number of doc in this batch: ', len(titles))
        
        # x = threading.Thread(target=threading_wrapper, args=(paragraphs, titles, i))
        opinion_miner(paragraphs, titles, i, output_dir, n_process=1,)
        # threads.append(x)
        # print(f'thread id {i} starts.')
        # x.start()
    
    # for i, x in enumerate(threads):
    #     x.join()
    #     print(f'thread id {i} ends.')