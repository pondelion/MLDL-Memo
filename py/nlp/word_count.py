import numpy as np
import pandas as pd
from janome.tokenizer import Tokenizer


TEXTFILEPATH = 'test.txt'


def create_word_dictionary(word_list):
    word_dict = {}
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1
    return word_dict    


if __name__ == "__main__":
    
    df = pd.read_csv(TEXTFILEPATH, header=None, delimiter='\t', names=['CreatedMonth', 'CreatedDate', 'CaseNo', 'Product', 'QuestionerType', 'CustomerType', 'Type', 'Category', 'SubCategory', 'Subject', 'Inquiry', 'OralResponse'])

    t = Tokenizer()
    wvs = []
    print("トークン化中..")
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
                if token.base_form == "注入ボタン":
                    print(token.base_form)
                word_vector += [token.base_form]

        # データを連結
        wvs += [word_vector]
    
    word_list = []
    for words in wvs:
        for word in words:
            word_list.append(word)
            
    word_dict = create_word_dictionary(word_list)
    sorted_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_dict[:5])