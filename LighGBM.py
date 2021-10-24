import random
from sklearn import svm
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
#from sklearn.cross_validation import cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
#from utils import *
from tflearn.activations import relu
from sklearn.model_selection import train_test_split,StratifiedKFold
import heapq
from sklearn import metrics
import pylab as plt
import warnings;warnings.filterwarnings('ignore')

def load_vec(filename):
    line_number=0
    node_vec={}
    #embedingdata mashup_embeding
    for i in open(filename).readlines():
    #for i in open("embedingdata").readlines():
        if line_number==0:
            line_number+=1
            continue
        newi=i.strip().split(' ')
        node_vec[str(line_number)]=newi[:]
        line_number+=1
    return node_vec

def load_mydata():
	drug_vec=load_vec("E:\\n2v\\features\\drug.txt")
	protein_vec=load_vec("E:\\n2v\\features\\protein.txt")
	fw=open("data_of_drug_protein_interaction","w")
	print("test1!")
	for i in open("result.txt").readlines():
		newi=i.strip().split('\t')
		vec1=drug_vec.get(newi[0])
		vec2=protein_vec.get(newi[1])
		if newi[2]=="1":
			label="1"
		else:
			label="0"
		if vec1!=None and vec2!=None:
			S=""
			for vec in vec1:
				S+=vec+"\t"
			for vec in vec2:
				S+=vec+"\t"
			S+=label+"\n"
			fw.write(S)
	fw.flush()
	fw.close()

def load_data():
    X_pos=[]
    X_neg=[]
    X=[]
    Y1_pos=[]
    Y1_neg=[]
    Y1=[]
    for i in open("data_of_drug_protein_interaction").readlines():
        newi=i.strip().split('\t')
        if int(newi[-1]) == 1:
            X_pos.append([float(x) for x in newi[:-1]])
            Y1_pos.append(int(newi[-1]))
        else:
            X_neg.append([float(x) for x in newi[:-1]])
            Y1_neg.append(int(newi[-1]))
            
    X = X_pos
    Y1 = Y1_pos
    
    random.seed(2021)

    for idx in random.sample(range(0,len(X_neg)-1),len(X_pos)):
        X.append(X_neg[idx])
        Y1.append(0)
    return np.array(X),np.array(Y1)

if __name__=="__main__":

    load_mydata()  #导入药物和蛋白质特征数据和监督数据
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'max_depth':5,
        'num_leaves':31,
        'learning_rate':0.05,
        'feature_fraction':0.9,
        'bagging_fraction':0.8,
        'bagging_freq':5,
        'verbose':-1
    }  
    
    X,Y=load_data()   #读取导入的数据
    print( X.shape,Y.shape)
    
    # index = np.random.permutation(X.shape[0])
    kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)
    
    test_auc_fold = []
    test_aupr_fold = []
    top = []
    top_idx = []
    vec_5fold = []
    vec = []
    label_5fold = []
    label = []
    top_num = 1000
    date = '_429.txt'
    fold = 0
    for train_index, test_index in kf.split(X, Y):
        X_train, Y_train = X[train_index,:], Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]
        train_data = lgb.Dataset(X_train, Y_train)
        eval_data = lgb.Dataset(X_test, Y_test)
        #开始训练
        gbm = lgb.train(params, train_data, num_boost_round=3000, valid_sets=eval_data, early_stopping_rounds=200)
        
        Y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        
        test_auc = roc_auc_score(Y_test, Y_pred)
        test_aupr = average_precision_score(Y_test, Y_pred)
        #存储五次分类结果
        np.savetxt('label'+str(fold), Y_test)
        np.savetxt('pred'+str(fold), Y_pred)
        fold += 1
        
        top += heapq.nlargest(top_num, Y_pred)
        top_idx = list(map(list(Y_pred).index, list(heapq.nlargest(top_num, Y_pred))))
        
        for index in top_idx:
             vec_5fold.append(X_test[index])
             label_5fold.append(Y_test[index])
        #存储完毕
        test_auc_fold.append(test_auc)
        test_aupr_fold.append(test_aupr)
        
    #top 1000 结果存储
    top_idx = list(map(top.index, heapq.nlargest(top_num, top)))
    
    
    for index in top_idx:
        vec.append(vec_5fold[index])
        label.append(label_5fold[index])
    print(vec)
    print(label)
    np.savetxt('top_pred_'+str(top_num)+date, heapq.nlargest(top_num, top))
    np.savetxt('top_vec_'+str(top_num)+date, vec)
    np.savetxt('top_label_'+str(top_num)+date, label)
    
    
    test_auc_fold.append(np.mean(test_auc_fold))
    test_aupr_fold.append(np.mean(test_aupr_fold))
    #注释



    print('AUROC:')
    print(test_auc_fold[:-1])
    print('AverageAUC:')
    print(test_auc_fold[-1])
    print('AUPR:')
    print(test_aupr_fold[:-1])
    print('AverageAUPR')
    print(test_aupr_fold[-1])
