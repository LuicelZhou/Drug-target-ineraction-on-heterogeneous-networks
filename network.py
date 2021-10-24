import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

file_read = open('E:\\n2v\\data\\Networks\\drugDisease.txt', 'r')
#file_color = open('c.txt', 'r')
num = 0
try:

     matrix = np.empty([732, 732], dtype=int)
     # matrix = [[0 for x in range(732)] for y in range(732)]
     i = 0
     for line in file_read:
          line = line.replace('\n','').split('\t')
          for h in range(732):
               matrix[i][h] = int(line[h])
          print(matrix[i])
          i = i + 1
     matrix = np.array(matrix)

     G = nx.Graph()
     for x in range(0, len(matrix)):
          for y in range(0, len(matrix)):
               if matrix[x][y] == 1:
                    print("edge:", x, y)
                    G.add_edge(x, y)
     nx.draw(G, with_labels=True)
     plt.show()
finally:
     file_read.close()

