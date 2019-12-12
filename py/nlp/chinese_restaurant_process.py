import numpy as np
import matplotlib.pyplot as plt


# 既存トピックが選択される確率：[トピックkの文書数]/([総文書数]-1+α)
# 新トピックが選択される確率:α/([総文書数]-1+α)
# トピックkの文書数、総文書数、パラメータαから次に選択されるトピックのインデックスを返す。
def dirichlet(doc_nums, alpha):
    all_doc_num = sum(doc_nums)
    normalized_factor = all_doc_num - 1 + alpha
    r = np.random.rand()
    s = 0
    for i, doc_num in enumerate(doc_nums):
        s += doc_num
        if r < (s/normalized_factor):
            return i
    # 新トピック
    return -1


# 各トピックの文書数　doc_nums[i]はi+1番目のトピックの文書数(各テーブルに座っている人の数　　doc_nums[i]はi+1番目のテーブルに座っている人の数)
doc_nums = []

# 100個の単語にトピックを割り当てる(100人の人をテーブルに割り当てる)
for i in range(100):
    index = dirichlet(doc_nums, alpha=5)
    # 新規トピック(新規テーブル)
    if index == -1:
        doc_nums.append(1)
    # 既存トピック(既存テーブル)
    else:
        doc_nums[index] += 1
    print(doc_nums)
plt.bar([i for i in range(len(doc_nums))], doc_nums)
plt.xlabel('topic number')
plt.ylabel('word count')
plt.show()
