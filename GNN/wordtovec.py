from gensim.models import KeyedVectors

# 这需要下载大约 3.5 GB 的模型文件
model_path = 'C:/Users/30351/Desktop/数据挖掘/final_hw/conflict_data/GoogleNews-vectors-negative300.bin.gz'
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
vector = word_vectors['computer']
print(vector)