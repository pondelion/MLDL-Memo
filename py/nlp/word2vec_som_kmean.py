import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from janome.tokenizer import Tokenizer
import re
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


DATAFILEPATH = 'test.txt'

def creaet_word_dictionary(wordlist, w2v_model, word_vec_dim):
    word_dic = {}
    for word in wordlist:
        if word not in word_dic:
            if word in w2v_model:
                word_dic[word] = w2v_model[word]
            else:
                word_dic[word] = np.zeros(word_vec_dim)
    return word_dic


class SOM():
    def __init__(self, num_col, num_row, word_vec_dim, learning_rate=0.1):
        # np.zeros(num_row, num_col)
        self.map = np.random.rand(num_row, num_col, word_vec_dim)
        self.learning_rate = learning_rate

    def train(self, vec, step, total_step):
        index = self.find_similar_vec(vec)
        for i in range(index[0]-1, index[0]+1):
            for j in range(index[1]-1, index[1]+1):
                try:
                    #self.map[i][j] += self.learning_rate * (1 - step/total_step) * (vec - self.map[index[0], index[1]])
                    self.map[i][j] += self.learning_rate * \
                        (vec - self.map[index[0], index[1]])
                    #self.map[i][j] += self.learning_rate * vec
                except Exception as e:
                    continue

    def find_similar_vec(self, vec):
        max_cos = -1
        index = (-1, -1)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                dot = np.dot(vec, self.map[i][j])
                norm1 = np.linalg.norm(vec)
                norm2 = np.linalg.norm(self.map[i][j])
                cos = dot / (norm1 * norm2)
                if cos > max_cos:
                    max_cos = cos
                    index = (i, j)
        return index


if __name__ == "__main__":

    # Word2Vec学習済みモデル読み込み
    print("Vord2Vec学習済みモデル読み込み")
    model_path = 'word2vec.gensim.model'  # 白ヤギコーポレーション
    model = Word2Vec.load(model_path)
    #print(model[u'ニュース'].shape)
    #sprint(model.corpus_count)
    word_vec_dim = model[u'ニュース'].shape[0]

    df = pd.read_csv(DATAFILEPATH, header=None, delimiter='\t', names=[
                     'CreatedMonth', 'CreatedDate', 'CaseNo', 'Product', 'QuestionerType', 'CustomerType', 'Type', 'Category', 'SubCategory', 'Subject', 'Inquiry', 'OralResponse'])

    t = Tokenizer()
    wvs = []
    print("トークン化")
    for i, line in enumerate(df['Inquiry'].dropna()):
        # 一つのページのワードのベクトル
        word_vector = []

        # 短すぎる場合は無視
        if len(line) < 10:
            continue
        # 記号以外はベクトル作成
        else:
            tokens = t.tokenize(line)

        for token in tokens:
            if token.part_of_speech[:2] == '名詞' and token.part_of_speech.split(',')[1] == '一般':
                #print(token)
                if token.base_form == "ボタン":
                    print(token.base_form)
                word_vector += [token.base_form]

        # データを連結
        wvs += [word_vector]

    word_list = []
    for words in wvs:
        for word in words:
            word_list.append(word)
    print("単語-単語ベクトル辞書作成")
    word_dict = creaet_word_dictionary(word_list, model, 50)
    word_list = list(word_dict.keys())
    word_vec_list = list(word_dict.values())

    font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
    font_prop = FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()

    som = SOM(num_col=100, num_row=100, word_vec_dim=50, learning_rate=0.01)

    print("Train start")
    for _ in range(3 * len(word_vec_list)):
        if _ % 100 == 0:
            print("Step {}".format(_))
        vec = word_vec_list[np.random.choice(range(len(word_vec_list)))]
        som.train(vec, step=_, total_step=len(word_vec_list))

    print(word_vec_list[0])
    print(som.find_similar_vec(word_vec_list[0]))

    word_dist_2d = []

    for word_vec in word_vec_list:
        word_dist_2d.append(np.array(som.find_similar_vec(word_vec)))

    print(word_dist_2d[:5])

    word_dist_2d = np.array(word_dist_2d)
    plt.scatter(word_dist_2d[:, 0], word_dist_2d[:, 1])
    for word, vec in zip(word_list, word_dist_2d):
        plt.annotate(word, xy=(vec[0], vec[1]))
    plt.show()

    print("K-Meansクラスタリング")
    cls = KMeans(n_clusters=200)
    pred = cls.fit_predict(word_dist_2d)

    for i in range(200):
        print("class {} : {}".format(i, np.array(word_list)[pred == i]))
