import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_read = open('E:\\n2v\\data\\Networks\\Sim_proteinDisease_aq.txt', 'r')
# file_color = open('c.txt', 'r')
num = 0
matrix = np.empty([1915,1915], dtype=float)
# matrix = [[0 for x in range(732)] for y in range(732)]
i = 0
for line in file_read:
    line = line.replace('\n', '').split('\t')
    for h in range(1915):
        matrix[i][h] = float(line[h])
    #print(matrix[i])
    i = i + 1
matrix = np.array(matrix)

sns.set(font_scale=1.25)#字符大小设定
hm=sns.heatmap(matrix, cbar=True, annot=False, square=True)
plt.show()
