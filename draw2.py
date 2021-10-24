import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''file_read = open('E:\\n2v\\data_edge\\walkpath\\drugdrug_walkpath.txt', 'r')
# file_color = open('c.txt', 'r')
num = 0
matrix = np.empty([80,6890], dtype=int)
# matrix = [[0 for x in range(732)] for y in range(732)]
i = 0
for line in file_read:
    line = line.replace('\n', '').split('\t')
    for h in range(80):
        matrix[i][h] = int(line[h])
    #print(matrix[i])
    i = i + 1
matrix = np.array(matrix)'''
matrix = np.loadtxt("E:\\n2v\\data_edge\\walkpath\\drugdrug_walkpath.txt")
print(matrix)

plt.figure(figsize=(18,14))
sns.heatmap(matrix, cmap='Blues', annot=False)
plt.show()
