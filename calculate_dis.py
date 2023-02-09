import numpy as np
import sklearn.metrics as skm
from pos_score_calculator import get_scores
def custom_get_scores(neg_train_data, neg_test_data , pos_train_data ): #calculate Mahalanobis distance between two distance
    i,j,k = neg_train_data.shape
    neg_train_data = neg_train_data.reshape(i,j*k).T
    
    neg_test_data = neg_test_data.reshape(i,j*k).T
    
    pos_train_data = pos_train_data.reshape(i, j*k).T
    
    X=np.vstack([neg_train_data , neg_test_data])
    V=np.cov(X.T)
    VI= np.linalg.inv(V)
    dis_neg_train = np.diag(np.sqrt(np.dot(np.dot((neg_train_data-neg_test_data),VI),(neg_train_data-neg_test_data).T)))
    
    Z= np.vstack([neg_train_data, pos_train_data])
    V=np.cov(Z.T)
    ZI= np.linalg.inv(V)
    dis_pos_train = np.diag(np.sqrt(np.dot(np.dot((neg_train_data-pos_train_data),VI),(neg_train_data-pos_train_data).T)))
    return dis_neg_train, dis_pos_train
    
def get_scores_one_cluster(ftrain, ftest, food):
    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                ) 
            ).T,
            axis=-1,
        )
        for x in ftrain
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in ftrain
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood #Positive pair내에서의 positive score , Negative Pair내에서의 positive score 


def get_roc_sklearn(dtest, dood): #positive score를 통해 Positive pair, Negative pair 각각에서 roc점수를 계산
    din_label = [0]* len(din)
    dood_label = [1]*len(dood)
    auroc = skm.roc_auc_score(din_label, dtest)
    aupr = skm.roc_auc_score(dood_label, dood) 
    return auroc , aupr
    

def get_pr_sklearn(dtest, dood):
    din_label = [0]*len(din)
    dood_label = [1] * len(dood)
    get_clusters
    pr = skm.precision_score()

def fpr():
    NotImplemented

x= np.array([[1,2,3,4,5],
               [5,6,7,8,5],
               [5,6,7,8,5]])

y= np.array([[[31,32,33,34,5],
               [35,36,37,38,5],
               [5,6,7,8,5]],
              [[41,42,43,44,5],
               [45,46,47,48,5],
               [5,6,7,8,5]]])
print(get_scores_one_cluster(x,x,y))  