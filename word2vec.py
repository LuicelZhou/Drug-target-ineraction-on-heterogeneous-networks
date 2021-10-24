import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec

sentences=word2vec.Text8Corpus("E:\\n2v\\data_edge\\walkpath\\drugdrug_walkpath.txt")  #text8为语料库文件名
#sentences是训练所需预料，可通过该方式加载
model=Word2Vec(sentences,sg=1,size=128,window=5,min_count=2,negative=3,sample=0.001,hs=1,workers=4)

# min_count，是去除小于min_count的单词
# size，维数
# sg， 算法选择
# window， 句子中当前词与目标词之间的最大距离
# workers，线程数

model.wv.save_word2vec_format("E:\\n2v\\vectors\\drugdrug_vector", binary = False)  #通过该方式保存的模型，能通过文本格式打开，也能通过设置binary是否保存为二进制文件。但该模型在保存时丢弃了树的保存形式（详情参加word2vec构建过程，以类似哈夫曼树的形式保存词），所以在后续不能对模型进行追加训练