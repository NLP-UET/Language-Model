from pyvi.ViTokenizer import tokenize
import re, os, string
import pandas as pd

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

# list stopwords
filename = './Language-Model/data/stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords'].tolist()

def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)
    return text2

def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents

def word_segment(sent):
    sent = tokenize(sent)
    return sent

# File path to the corpus
file_path = './Language-Model/wikipediacorpus/viwiki.txt'

with open('./Language-Model/data/datatrain.txt', 'w', encoding='utf-8') as f_w:
    with open(file_path, 'r', encoding='utf-8') as f_r:
        contents = f_r.read().strip().split('</doc>')
        for content in contents:
            if len(content) < 5:
                continue
            content = clean_text(content)
            sents = sentence_segment(content)
            for sent in sents:
                if sent:
                    sent = word_segment(sent)
                    sent = remove_stopword(normalize_text(sent))
                    if len(sent.split()) > 1:
                        f_w.write(sent + '\n')
            print("Done processing content.")
