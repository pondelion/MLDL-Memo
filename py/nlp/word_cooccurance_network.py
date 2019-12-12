import math
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from janome.tokenizer import Tokenizer
import json
import numpy as np
import os.path
import random


STOP_WORDS = ["サイト", "よう", "ため", "://", "http", "---", "com"]


def filter_word(word_list):
    filtered_word_list = []
    for w in word_list:
        if len(w) < 2:
            continue
        if w.isdigit():
            continue
        if w in STOP_WORDS:
            continue
        filtered_word_list.append(w)
    return filtered_word_list

# 単語から単語idへ変換


def get_index_from_word(word, word_list):
    for i, w in enumerate(word_list):
        if word == w:
            return i
    return -1


def get_word_from_index(index, word_list):
    return word_list[index]

# word_listに含まれる全単語ペアの共起行列を全文から計算


def calc_cooccurrence_matrix(lines, word_list):
    print("calc_cooccurrence_matrix : len(word_list) = ", len(word_list))
    cooccurence_matrix = np.zeros((len(word_list), len(word_list)))
    t = Tokenizer()
    for line in lines:
        #print(line)
        malist = t.tokenize(line)
        for w1 in malist:
            word1 = w1.surface
            ps1 = w1.part_of_speech
            # word_listにあるもののみ共起行列の要素を計算
            if word1 not in word_list:
                continue
            for w2 in malist:
                word2 = w2.surface
                ps2 = w2.part_of_speech
                # word_listにあるもののみ共起行列の要素を計算
                if word1 == word2 or word2 not in word_list:
                    continue
                index1 = get_index_from_word(word1, word_list)
                index2 = get_index_from_word(word2, word_list)
                cooccurence_matrix[index1][index2] += 1
    return cooccurence_matrix


rss_data = pd.read_csv("feed_data.tsv", delimiter='\t')
content = rss_data['content'][rss_data['content'].notnull()]

if os.path.exists("./rss_data_word_freq.json"):
    f = open('./rss_data_word_freq.json', 'r')
    word_dic = json.load(f)
else:
    # 各単語の出現回数を計算
    t = Tokenizer()
    word_dic = {}
    for line in content:
        malist = t.tokenize(line)
        for w in malist:
            word = w.surface
            ps = w.part_of_speech  # 品詞
            if ps.find('名詞') < 0:
                continue  # 名詞だけカウント --- (※5)
            #if isJapanese(word):
            if not word in word_dic:
                word_dic[word] = 0
            word_dic[word] += 1  # カウント

    f = open('rss_data_word_freq.json', 'w')
    json.dump(word_dic, f)

# よく使われる単語を表示 --- (※6)
keys = sorted(word_dic.items(), key=lambda x: x[1], reverse=True)
for word, cnt in keys[:200]:
    print("{0}({1}) ".format(word, cnt), end="")

words = [w for w, cnt in keys]
words = filter_word(words)

NUM_DATA_TO_USE = 25000
NUM_WORD_TO_USE = 4000
word_list_to_use = words[10:NUM_WORD_TO_USE]


# 共起行列を計算してcsvファイルへ出力
if os.path.exists("./CooccurrenceMatrix_datanum-{}_wordnum-{}.csv".format(NUM_DATA_TO_USE, NUM_WORD_TO_USE)):
    print("\nReading CooccurrenceMatrix file..")
    cooccurence_matrix = pd.read_csv(
        "./CooccurrenceMatrix_datanum-{}_wordnum-{}.csv".format(NUM_DATA_TO_USE, NUM_WORD_TO_USE), index_col=0)
else:
    print("\nCalculating Co-Occurance Matrix..")
    cooccurence_matrix = pd.DataFrame(calc_cooccurrence_matrix(
        np.random.permutation(content.tolist())[:NUM_DATA_TO_USE], word_list_to_use))
    print("Calculating Co-Occurance Matrix done.")
    cooccurence_matrix.index = word_list_to_use
    cooccurence_matrix.columns = word_list_to_use
    cooccurence_matrix.to_csv(
        "CooccurrenceMatrix_datanum-{}.csv".format(NUM_DATA_TO_USE))

# 出現回数が多い20個の単語に対して
for word in words[10:30]:
    #index = get_index_from_word(word)
    print("単語 : {}, 出現回数 : ".format(word))
    print("共起単語 : ")
    # 共起数が多い単語を10個表示
    # print(cooccurence_matrix)
    #for co in cooccurence_matrix[word].sort_values(ascending=False)[:10]:
    #    print(co)
    print(cooccurence_matrix[word].sort_values(ascending=False).head(10))

'''
## ネットワーク描画(iGraph ver)
  
from igraph import *
vertices = word_list_to_use[:400]#words[10:400]
edges = []
print("Drawing Co-Occurance Networks..")
for i in range(len(vertices)):
    for j in range(len(vertices)):
        if int(cooccurence_matrix.ix[[i],[j]].values[0][0]) > 4000:
            #print(str(i), " : " , str(j))
            edges.append((i,j))
        #edges.append((i,j))

# 共起数でフィルターすると孤立したノードが多数描画されるため、エッジがあるノードのみ抽出
node_set = set()
for node1, node2 in edges:
    #print("node1 : ", node1)
    #print("node2 : ", node2)
    node_set.add(node1)
    node_set.add(node2)
    
vertices_to_draw = []
for node in sorted(node_set, reverse=True):
    word = vertices[node]#get_word_from_index(node, words[10:1000])
    if word not in vertices_to_draw:
        vertices_to_draw.append(word)

edges_to_draw = []

# 抽出したノードの中から更に共起数が閾値より大きいものにエッジを設ける
for i, w1 in enumerate(vertices_to_draw):
    for j, w2 in enumerate(vertices_to_draw):
        word_index1 = get_index_from_word(w1, word_list_to_use)
        word_index2 = get_index_from_word(w2, word_list_to_use)
        if int(cooccurence_matrix.ix[[word_index1],[word_index2]].values[0][0]) > 6600:
            edges_to_draw.append((i,j))
        

#g = Graph(vertex_attrs={"label": vertices}, edges=edges)#directed=True)
g = Graph(vertex_attrs={"label": vertices_to_draw}, edges=edges_to_draw)#directed=True)
plot(g)
'''

## ネットワーク描画(networkx ver)


font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

G = nx.Graph()
nodes = [(w, {'count': word_dic[w]})
         for w in word_list_to_use[:200]]  # words[10:400]
G.add_nodes_from(nodes)
print("Drawing Co-Occurance Networks..")
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        # 共起数が閾値以上
        if int(cooccurence_matrix.ix[[i], [j]].values[0][0]) > 1500:
            #print(str(i), " : " , str(j))
            G.add_edge(node1[0], node2[0], weight=math.sqrt(
                cooccurence_matrix.ix[[i], [j]].values[0][0])/20)
        #edges.append((i,j))
pos = nx.spring_layout(G)
node_size = [d['count']/5 for (n, d) in G.nodes(data=True)]
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#0000FF")
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_family=matplotlib.rcParams['font.family'])
plt.show()
cliques = nx.find_cliques(G)
#標準出力
#[print(c) for c in cliques]

cluster = nx.clustering(G)
print(cluster)
