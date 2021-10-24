import numpy as np
import random
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc



def load_vec():
    fw = open("data")
    ...
    print("test1!")
    for i in open("result.txt").readlines():
        newi = i.strip().split('\t')
        vec1 = drug_vec.get(newi[0])
        vec2 = protein_vec.get(newi[1])
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
    for i in open("data").readlines():
        newi = i.strip().split('\t')
        if int(newi[-1])==1:
            X_pos.append([float(x) for x in newi[:-1]])
            Y1_pos.append(int(newi[-1]))
        else:
            X_neg.append([float(x) for x in newi[:-1]])
            Y1_neg.append(int(newi[-1]))
    X=X_pos
    Y1=Y1_pos

    random.seed(2021)

    for idx in random.sample(range(o,len(X_neg)-1),len(X_pos)):
        X.append(X_neg[idx])
        Y1.append(0)
    return np.array (X),np.array(Y1)

if __name__== "__main__":
    params = {'num_leaves':31,
              'min_data_in_leaf': 30,
              'objective': 'binary',
              'max_depth': 5,
              'learning_rate': 0.05,
              "min_sum_hessian_in_leaf": 6,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 5,
              "bagging_fraction": 0.8,
              "bagging_seed": 11,
              "lambda_l1": 0.1,
              # 'lambda_l2': 0.001,
              "verbosity": -1,
              "nthread": -1,
              'metric': {'auc'},
              "random_state": 2019,
              # 'device': 'gpu'
              }
    X,Y = load_data()
    print(X.shape,Y.shape)
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

    test_auc_fold = []
    test_aupr_fold = []
    top=[]
    top_idx= []
    vec_5fold=[]
    vec=[]
    label=[]
    label_5fold=[]
    top_num=1000
    date="_223.txt"
    for train_index,test_index in kf.split(X,Y):
        X_train,Y_train = X[train_index,:],Y[train_index]
        X_test, Y_test = X[test_index,:], Y[test_index]
        train_data = lgb.Dataset(X.train,Y_train)
        eval_data = lgb.Dataset(X.test, Y_test)

        gbm = lgb.train(params, train_data, num_boost_round=3000, valid_sets = eval_data,early_stopping_round=200)

        Y_pred = gbm.predict(X_test,num_iteration=gbm.best_iteration)
        test_auc = roc_auc_score(Y_test,Y_pred)
        test_aupr = average_precision_score(Y_test, Y_pred)
        top += heapq.nlargest(top_num, Y_pred)
        top_idx = list(map(list(Y_pred).index,list(heapq.nlargest(top_num,Y_pred))))
