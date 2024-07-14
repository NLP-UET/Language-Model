import gensim.models as gm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load the model
model = gm.Word2Vec.load('./Language-Model/model/word2vec_skipgram.model')
# model = gm.Word2Vec.load('../model/fasttext_gensim.model')

# Read words from file
pathfile = './Language-Model/words'
with open(pathfile, 'r', encoding='utf-8') as f:
    words = f.readlines()
    words = [word.strip() for word in words]

words_np = []
words_label = []

# Iterate over the words in the model
for word in model.wv.key_to_index:
    if word in words:
        words_np.append(model.wv[word])
        words_label.append(word)

words_np = np.array(words_np)

# Perform PCA
pca = PCA(n_components=2)
pca.fit(words_np)
reduced = pca.transform(words_np)

# Visualization function
def visualize():
    fig, ax = plt.subplots()

    for index, vec in enumerate(reduced):
        x, y = vec[0], vec[1]

        ax.scatter(x, y)
        ax.annotate(words_label[index], xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.show()
    # Save the plot to a file as a fallback
    fig.savefig('./Language-Model/result/word_vectors.png')

# Main function
if __name__ == '__main__':
    visualize()
