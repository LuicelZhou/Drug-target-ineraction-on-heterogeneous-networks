import utils
import argparse
import numpy as np
import networkx as nx
import node2vec
#from gensim.models import Word2Vec
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")


    parser.add_argument('--input', nargs='?', default='E:\\n2v\\data_edge\\networks\\drugdrug.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='E:\\n2v\\data_edge\\walkpath\\drugdrug_walkpath.txt',
                        help='Embeddings path')


    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')



    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')


    parser.add_argument('--num-walks', type=int, default=8,
                        help='Number of walks per source. Default is 10.')


    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')


    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')


    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 3.')


    parser.add_argument('--q', type=float, default=3,
                        help='Inout hyperparameter. Default is 2.')


    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')


    parser.add_argument('--unweighted', dest='unweighted', action='store_false')


    parser.set_defaults(weighted= True)


    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')


    parser.add_argument('--undirected', dest='undirected', action='store_false')


    parser.set_defaults(directed=False)


    return parser.parse_args()





def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_weighted_edgelist(args.input, create_using=nx.DiGraph(), nodetype=int)
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G







def write_walks(walks):
	fw=open(args.output,"w")
	for walk in walks:
		S=""
		for node in walk:
			S+=node+"\t"
		fw.write(S[:-1]+"\n")
	fw.flush()
	fw.close()





def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, 1, 3)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(10, 80)
    walks = [map(str, walk) for walk in walks]
    write_walks(walks)
    #word2vec????????????
    sentences = word2vec.Text8Corpus("E:\\n2v\\data_edge\\walkpath\\drugdrug_walkpath.txt")  # text8?????????????????????
    # sentences????????????????????????????????????????????????
    model = Word2Vec(sentences, sg=1, size=128, window=5, min_count=2, negative=3, sample=0.001, hs=1, workers=4)
    # min_count??????????????????min_count?????????
    # size?????????
    # sg??? ????????????
    # window??? ???????????????????????????????????????????????????
    # workers????????????
    model.wv.save_word2vec_format("E:\\n2v\\vectors\\drugdrug_vector", binary=False)



if __name__ == "__main__":
    args = parse_args()
    main(args)

