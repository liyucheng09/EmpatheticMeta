import pickle

def get_first_few_paras(doc):
    paragraphs = doc['text'].split('\n\n', 5)[:5]
    new_paras = []
    for i, para in enumerate(paragraphs):
        if len(para) < 60 and i != 0:
            break
        new_paras.append(para)
    return new_paras

if __name__ == '__main__':
    pass