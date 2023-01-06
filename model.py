import os
import pickle
from glob import glob
from collections import Counter
import datasets
import re
from tqdm import tqdm
import pyterrier as pt
pt.init()

class opinion_index:

    def __init__(self, wmodel = 'bm25', phrase = False, threads = 1):
        self.index_path = f'./index/{"phrase_" if phrase else ""}opinion.index'
        wmodels = {
            'bm25': 'BM25',
            'tf_idf': 'TF_IDF',
            'match': 'CoordinateMatch',
            'tf': 'Tf'
        }
        self.wmodel = wmodels[wmodel]
        self._build_index(self.index_path, threads = threads)

    def _opinion_looper(self, slice = None):
        opinion_path = 'data/wiki_opinion_small/checkpoint_*.pickle'
        count = 0
        files = sorted(glob(opinion_path), key=lambda x:float(re.findall("(\d+)",x)[0]))
        for file in files:
            count += 1
            print(file)
            if slice is not None and count > slice:
                break
            with open(file, 'rb') as f:
                data = pickle.load(f)
            for k, lines in data.items():
                # print(k)
                if len(k)>40 or not k: continue
                opinion_words = []
                last_opinion = None
                for line in lines:
                    tokenized_phrase = [t for t in line[0] if len(str.encode(t))<60]
                    opinions = line[1]
                    if last_opinion is None or last_opinion != opinions:
                        opinion_words.extend(tokenized_phrase)
                        last_opinion = opinions
                    else:
                        for t in tokenized_phrase:
                            if t not in opinion_words:
                                opinion_words.append(t)
                    # yield {'docno':k}
                if not opinion_words:
                    # print(k, lines)
                    continue
                yield {'docno':k, 'toks': Counter(opinion_words)}
    
    def _build_index(self, path, threads):
        if os.path.exists(path):
            self.index = pt.IndexFactory.of(path)
            self.retrieval = pt.BatchRetrieve(self.index, wmodel = self.wmodel)
            return
        
        # indexer = pt.IterDictIndexer(path, overwrite = True)
        indexer = pt.IterDictIndexer(path, overwrite = True, pretokenised = True, threads = threads)

        # doc_iter = self._opinion_looper(slice=1)
        indexref = indexer.index(self._opinion_looper())

        self.index = pt.IndexFactory.of(indexref)
        self.retrieval = pt.BatchRetrieve(self.index, wmodel = self.wmodel)

        return

class wiki_index:
    def __init__(self, wmodel, threads = 1):
        self.index_path = f"./index/wiki.index"
        self.wmodel = wmodel
        self._build_index(threads = threads)
    
    def _wiki_looper(self):
        ds = datasets.load_dataset("wikipedia", "20220301.en", split="train")
        for doc in tqdm(ds):
            title = doc['title']
            if len(title)>50 or len(doc['text'])<5000:
                continue
            paragraphs = doc['text'].split('\n\n', 5)[:5]
            new_paras = []
            for i, para in enumerate(paragraphs):
                if len(para) < 60 and i != 0:
                    break
                new_paras.append(para)
            yield {'docno': title, 'text':' '.join(new_paras)}

    def _build_index(self, threads = 1):
        if os.path.exists(self.index_path):
            self.index = pt.IndexFactory.of(self.index_path)
            self.retrieval = pt.BatchRetrieve(self.index, wmodel = self.wmodel)
            return
        
        indexer = pt.IterDictIndexer(self.index_path, overwrite = True, threads = threads)

        indexref = indexer.index(self._wiki_looper())

        self.index = pt.IndexFactory.of(indexref)
        self.retrieval = pt.BatchRetrieve(self.index, wmodel = self.wmodel)

        return

if __name__ == '__main__':

    # retrieve = wiki_index(wmodel = 'BM25')
    retrieve = opinion_index(wmodel='bm25')
    print(retrieve.retrieval.search('disaster'))
