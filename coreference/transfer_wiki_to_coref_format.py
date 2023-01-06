import datasets
from nltk.tokenize import sent_tokenize, word_tokenize
import json

def get_first_few_paras(doc):
    paragraphs = doc.split('\n\n', 5)[:5]
    new_paras = []
    for i, para in enumerate(paragraphs):
        if len(para) < 60 and i != 0:
            break
        new_paras.append(para)
    return new_paras

if __name__ == '__main__':
    
    chunks_size_per_checkpoint = 6000
    wiki = datasets.load_dataset("wikipedia", "20220301.en", split="train")

    for i in range(0, 6458670, chunks_size_per_checkpoint):
        output_file = f'coreference/output_batch_{i}.jsonlines'

        with open(output_file, 'w', encoding='utf-8') as f:
            chunk = wiki[i: i+chunks_size_per_checkpoint]
            titles = []
            paragraphs = []
            for title, page in zip(chunk['title'], chunk['text']):
                if len(page) < 13000:
                    continue
                titles.append(title)
                paras = get_first_few_paras(page)
                paragraphs.append('\n'.join(paras))
                sentences = [word_tokenize(sent) for sent in sent_tokenize(' '.join(paras))]

                line = {"doc_key": title, "sentences": sentences, "speakers": [], "cluster": []}
                output_file.write(json.dumps(line)+'\n')
                
        print(f'Finish write batch {i}.')


