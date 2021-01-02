import time
start = time.time()

import pandas as pd
import re
import nltk
import string

dft = pd.read_csv("train.csv")

dft["num_words"] = dft["question_text"].apply(lambda x: len(re.findall(r'\w+', x)))
dft["num_unique"] = dft["question_text"].apply(lambda x: len(nltk.FreqDist(re.findall(r'\w+', x))))
dft["num_chars"] = dft["question_text"].apply(lambda x: len(x))
translate_table = str.maketrans('', '', string.punctuation + ' ')
dft["num_chars_clean"] = dft["question_text"].apply(lambda x: len(x.translate(translate_table)))
stop_words = nltk.corpus.stopwords.words('english')
dft["num_stopwords"] = dft.apply(lambda x: x["num_words"] - len(set(re.findall(r'\w+', re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*').sub('', x["question_text"].lower())))), axis=1)
translate_table2 = str.maketrans('', '', string.punctuation)
dft["num_punctmrk"] = dft["question_text"].apply(lambda x: len(x) - len(x.translate(translate_table2)))
dft["num_upper_wrd"] = dft["question_text"].apply(lambda x: sum(map(str.isupper, x.translate(translate_table2).split())))
dft["num_titlecase"] = dft["question_text"].apply(lambda x: sum(map(str.istitle, x.translate(translate_table2).split())))
dft["num_sents"] = dft["question_text"].apply(lambda x: len(nltk.sent_tokenize(x)))
dft["word_avg_len"] = dft.apply(lambda x: x["num_chars_clean"] / x["num_words"] if x["num_words"] else 0, axis=1)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# print(dft.head(5))

dft.to_csv("data.csv")

end = time.time()
print(end - start)
