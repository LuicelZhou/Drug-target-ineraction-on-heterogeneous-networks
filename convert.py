import networkx as nx

# 抽取txt中的数据
def read_txt(data):
    g = nx.read_edgelist(data, create_using=nx.DiGraph())
    return g


g = read_txt('E:\\n2v\\data\\Networks\\drugDisease.txt')
nx.write_edgelist(g, 'edgelistFile.edgelist', delimiter=',')

