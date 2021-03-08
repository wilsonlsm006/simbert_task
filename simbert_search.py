import faiss
import pandas as pd
import numpy as np
import keras
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer

# 数据只包含一列copywriter
df1 = pd.read_csv('/root/ad_title.csv', error_bad_lines=False)

max_len = 64

# simbert配置
simbert_model_path = "/root/simbert/"
config_path = simbert_model_path + '/chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = simbert_model_path + '/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = simbert_model_path + '/chinese_simbert_L-12_H-768_A-12/vocab.txt'

class data_generator(DataGenerator):
    '''新的迭代器，注意这里得到dict_path没有定义，需要外部定义'''
    def __iter__(self, random=False):
        tokenizer = Tokenizer(dict_path, do_lower_case=True)
        batch_token_ids, batch_segment_ids= [], []
        for is_end, text in self.sample(random):
            token_id, segment_id = tokenizer.encode(text, maxlen=max_len)
            batch_token_ids.append(token_id)
            batch_segment_ids.append(segment_id)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids]
                batch_token_ids, batch_segment_ids = [], []
    def forpred(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d

def buildSimbertEncoder():
    '''构建simbert的encoder'''
    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

    # 建立加载模型
    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        with_pool='linear',
        application='unilm',
        return_keras_model=False,
    )

    encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
    return encoder,tokenizer

def setIndex(dim, index_param):
    """
    设置faiss的index
    """
    if index_param[0:4] == 'HNSW' and ',' not in index_param:
        hnsw_num = int(index_param.split('HNSW')[-1])
        print(f'Index维度为{dim}，HNSW参数为{hnsw_num}')
        index = faiss.IndexHNSWFlat(dim, hnsw_num, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.index_factory(dim, index_param, faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    index.do_polysemous_training = False
    return index

def dumpIndex(index, index_save_path):
    """
    保存index索引
    """
    faiss.write_index(index, index_save_path)

def get_tag_data_vecs(tag_data, encoder, tokenizer):
    """
    根据文本数据得到768维向量
    """
    data_gen = data_generator(data=tag_data, batch_size=32)
    vecs = encoder.predict_generator(data_gen.forpred(), steps=len(data_gen), verbose=1)
    vecs = vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs 

encoder, tokenizer = buildSimbertEncoder()
dim, index_param = 768, 'Flat'
vecs_dic, index_dic = {}, {}
tag_list = list(df.prediction.unique()) + [-1]

tag_data = list(df1.copywriter.values)
data_vecs = get_tag_data_vecs(tag_data, encoder, tokenizer)

type(data_vecs)
data_vecs[:1]

ids = setIndex(dim, index_param)
ids.add(data_vecs)

def search_ad(key_word, df, topK=10):
    target_vecs = get_tag_data_vecs(key_word, encoder, tokenizer)
    C, I = ids.search(target_vecs, topK)
    df_tag = df1['copywriter'].reset_index(drop=True)
    print(df_tag[I[0]])
    print(C[0])
    
key_word = '传奇，爆率，装备'
search_ad(key_word, df)
