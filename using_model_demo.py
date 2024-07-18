from gensim.models import KeyedVectors, Word2Vec
import gensim
import numpy as np
from pyvi import ViTokenizer

# Check the version of gensim
print(f"gensim version: {gensim.__version__}")

# Load the pre-trained word2vec model
model_path = './Language-Model/model/word2vec_skipgram.model'

try:
    # Try loading the model using KeyedVectors.load
    print("Attempting to load the model using KeyedVectors.load...")
    model = Word2Vec.load(model_path)
    print("Model loaded successfully using KeyedVectors.load.")
except Exception as e:
    print(f"Error loading model with KeyedVectors.load: {e}")
    print("Attempting to load the model using KeyedVectors.load_word2vec_format...")
    try:
        # If the above fails, try loading with load_word2vec_format
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print("Model loaded successfully using KeyedVectors.load_word2vec_format.")
    except Exception as e:
        print(f"Error loading model with KeyedVectors.load_word2vec_format: {e}")
        print("Unable to load the model. Please check the model file format and path.")
        model = None

# Extract the KeyedVectors object if needed
if model:
    if hasattr(model, 'wv'):
        model = model.wv  # Extract the KeyedVectors object
        print("Extracted KeyedVectors from the Word2Vec model.")

# Open a file to write the output
output_file = open('./Language-Model/result/output.txt', 'w', encoding='utf-8')

# Function to find synonyms for given words
def find_synonyms(words):
    if model:
        for word in words:
            try:
                similar_words = model.most_similar(word)
                output_file.write(f"Most similar words to '{word}':\n")
                for sim_word, similarity in similar_words:
                    output_file.write(f"  {sim_word}: {similarity}\n")
            except Exception as e:
                output_file.write(f"Error finding similar words for '{word}': {e}\n")
    else:
        output_file.write("Model not loaded. Cannot find synonyms.\n")

# Function to calculate sentence similarity
def sentence_similarity(sent1, sent2):
    if model:
        try:
            vec1 = np.mean([model[word] for word in ViTokenizer.tokenize(sent1).split() if word in model], axis=0)
            vec2 = np.mean([model[word] for word in ViTokenizer.tokenize(sent2).split() if word in model], axis=0)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return similarity
        except Exception as e:
            output_file.write(f"Error calculating sentence similarity: {e}\n")
            return None
    else:
        output_file.write("Model not loaded. Cannot calculate sentence similarity.\n")
        return None

# Find and write the most similar words to given words
words_to_find = ["mặt_trời", "tiền"]
find_synonyms(words_to_find)

# Calculate and write the similarity between two given sentences
sentence1 = "Tôi yêu công việc của mình."
sentence2 = "Tôi thích công việc này."
similarity = sentence_similarity(sentence1, sentence2)
output_file.write(f"Similarity between '{sentence1}' and '{sentence2}': {similarity}\n")

# Close the output file
output_file.close()