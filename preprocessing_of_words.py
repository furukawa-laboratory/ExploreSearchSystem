import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def preprocessing_tag_and_stopwords(s):
    lemmatizer = WordNetLemmatizer()

    need_tag = [['NN', 'NNS', 'NNP', 'NNPS'],['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                ['JJ', 'JJR', 'JJS'], ['RB', 'RBR', 'RBS']]
    p = ['n', 'v', 'a', 'r']  #'n':名詞，'v':動詞，'a':形容詞，'r':副詞
    stopWords = set(stopwords.words('english'))

    word = 'a'
    pos = []
    for i in range(len(s)):
        morph = nltk.word_tokenize(s[i])  #分かち書き
        l = nltk.pos_tag(morph)  #タグ付け
        l_new = []
        for i in l:
            if i[0] not in stopWords:
                for j in range(len(need_tag)):
                    if i[1] in need_tag[j]:
                        word = lemmatizer.lemmatize(i[0], pos=p[j])  #原型にする
                        l_new.append(word+' ')
        pos.append(''.join(l_new))
    return pos

def make_bow(df):
    s = df['snippet']
    pos = preprocessing_tag_and_stopwords(s)

    vectorizer = CountVectorizer(max_df=0.5, min_df=0.03)
    X = vectorizer.fit_transform(pos)
    X_array = X.toarray()
    word_label = np.array(vectorizer.get_feature_names())
    # Tfidf
    tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    tfidf = tfidf_transformer.fit_transform(X_array)

    features = np.array(tfidf.todense())

    word_num = 100
    print('最小出現回数＝' + str(min(np.sum(X_array[:, :word_num], axis=0))))
    word_order = np.argsort(np.sum(features, axis=0))[::-1]
    features = features[:, word_order][:, :word_num]
    word_label = word_label[word_order][:word_num]
    return features, word_label

if __name__ == '__main__':
    # Load file
    csv_df = pd.read_csv("~/Downloads/Unsupervised Kernel Regression.csv")
    features, word_label = make_bow(csv_df)
