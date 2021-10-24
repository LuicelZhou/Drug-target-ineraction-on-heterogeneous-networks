import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


# 训练的语料
sentences=word2vec.Text8Corpus("E:\\n2v\\data_edge\\walkpath\\protein_walkpath.txt")
# 利用语料训练模型
model=Word2Vec(sentences,sg=1,size=128, window=5,min_count=2,negative=3,sample=0.001,hs=1,workers=4)

# 基于2d PCA拟合数据
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# 可视化展示
pyplot.scatter(result[:, 0], result[:, 1])
pyplot.title('protein Features Visualization')
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(" ", xy=(result[i, 0], result[i, 1]))
pyplot.show()
