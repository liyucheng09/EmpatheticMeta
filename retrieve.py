from glob import glob
import pickle

class RetrievalManager:
    """
    Retrieve potential concepts to consist a novel metaphor with the given concept.
    Three matching method are designed to score/rank concepts in the database.

    - word: only consider the opinion word when matching concepts
    - phrase: consider the entire verb phrase when matching concepts
    - semantic: leverage embeddings, should try static embedding and contextual embedding

    """

    def __init__(self, method):
        self.method = method
        assert self.method in ['word', 'phrase', 'semantic']
        self._prepare()
    
    def _prepare(self):

        # build an index on opinion words for all concepts
        if self.method == 'word':
            database = {}
            for file in glob('data/wiki_opinion/checkpoint_*.pickle'):
                data = pickle.load(f)
                database.update(data)
            


if __name__ == '__main__':
    pass