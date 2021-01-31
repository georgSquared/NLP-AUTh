#############################
#                           #
#       RUN EXPERIMENT      #
#                           #
#############################


## --- Load modules ----
import time
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import gc

from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

# Import functions
from .define_functions import build_model, load_para, load_glove, load_fasttext
from .define_attenction_class import AttentionWeightedAverage


## -- Start experiment ----
start_time = time.time()

# 1. Load data
print("Loading data ...")

# train = pd.read_csv('../input/train.csv').fillna(' ')
# test = pd.read_csv('../input/test.csv').fillna(' ')

train = pd.read_csv('input/quora-insincere-questions-classification/train_sample.csv').fillna(' ')
test = pd.read_csv('input/quora-insincere-questions-classification/test.csv').fillna(' ')

train_text = train['question_text']
test_text = test['question_text']
text_list = pd.concat([train_text, test_text])
y = train['target'].values
num_train_data = y.shape[0]
print("--- %s seconds ---" % (time.time() - start_time))

# 2. Prepare questions
start_time = time.time()
print("Spacy NLP ...")
nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])

nlp = spacy.load('en_core_web_sm', disable=['parser','ner','tagger'])

nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
word_dict = {}
word_index = 1
lemma_dict = {}
docs = nlp.pipe(text_list, n_threads = 2)
word_sequences = []

for doc in tqdm(docs):
    word_seq = []
    for token in doc:
        if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
            word_dict[token.text] = word_index
            word_index += 1
            lemma_dict[token.text] = token.lemma_
        if token.pos_ is not "PUNCT":
            word_seq.append(word_dict[token.text])
    word_sequences.append(word_seq)
del docs
gc.collect()
train_word_sequences = word_sequences[:num_train_data]
test_word_sequences = word_sequences[num_train_data:]
print("--- %s seconds ---" % (time.time() - start_time))


# 3. Define hyperparameters
max_length = 55
embedding_size = 600
learning_rate = 0.001
batch_size = 512
num_epoch = 4

print("--- hyperparameters ----")
print(f"max_length:  {max_length}")
print(f"embedding_size:  {embedding_size}")
print(f"learning_rate:  {learning_rate}")
print(f"batch_size:  {batch_size}")
print(f"num_epoch: {num_epoch}")
print("-------------------------")


# 4. Fix size of sequences
train_word_sequences = pad_sequences(train_word_sequences, maxlen=max_length, padding='post')
test_word_sequences = pad_sequences(test_word_sequences, maxlen=max_length, padding='post')
print(train_word_sequences[:1])
print(test_word_sequences[:1])
pred_prob = np.zeros((len(test_word_sequences),), dtype=np.float32)


# 5 A. Train model - Make predictions

start_time = time.time()
print("Loading embedding matrix ...")
embedding_matrix_glove, nb_words = load_glove(word_dict, lemma_dict)
embedding_matrix_fasttext, nb_words = load_fasttext(word_dict, lemma_dict)
embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_fasttext), axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("Start training ...")
model = build_model(embedding_matrix, nb_words, embedding_size, max_length=max_length, learning_rate=learning_rate)
model.fit(train_word_sequences, y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)
pred_prob += 0.15*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))
model.fit(train_word_sequences, y, batch_size=batch_size, epochs=1, verbose=2)
pred_prob += 0.35*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))
del model, embedding_matrix_fasttext, embedding_matrix
gc.collect()
K.clear_session()
print("--- %s seconds ---" % (time.time() - start_time))


# 5 B. Train model - Make predictions
start_time = time.time()
print("Loading embedding matrix ...")
embedding_matrix_para, nb_words = load_para(word_dict, lemma_dict)
embedding_matrix = np.concatenate((embedding_matrix_glove, embedding_matrix_para), axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("Start training ...")
model = build_model(embedding_matrix, nb_words, embedding_size, max_length=max_length, learning_rate=learning_rate)
model.fit(train_word_sequences, y, batch_size=batch_size, epochs=num_epoch-1, verbose=2)
pred_prob += 0.15*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))
model.fit(train_word_sequences, y, batch_size=batch_size, epochs=1, verbose=2)
pred_prob += 0.35*np.squeeze(model.predict(test_word_sequences, batch_size=batch_size, verbose=2))
print("--- %s seconds ---" % (time.time() - start_time))


# 6. Create submission file
submission = pd.DataFrame.from_dict({'qid': test['qid']})
submission['prediction'] = (pred_prob>0.35).astype(int)
submission.to_csv('submission.csv', index=False)