#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import cross_validation
import matplotlib.pyplot
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
import operator
from sklearn.metrics import fbeta_score, make_scorer

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'exercised_stock_options',
                 'bonus','restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
                 'expenses', 'loan_advances', 'other', 'director_fees', 'deferred_income',
                 'long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#number of data points
total_data_points = len(data_dict)
print 'Number of Data Points: ' + str(total_data_points)

#number of poi's
n_pois = 0;

for point in data_dict:
    if(data_dict[point]['poi'] == True):
        n_pois += 1

print 'Number of POIs: ' + str(n_pois)

#number of non-poi's
print 'Number of non-POIs: ' + str(total_data_points - n_pois)

#number of features
number_features = len(data_dict['SAVAGE FRANK'])
print 'Number of features: ' + str(number_features) 

print data_dict['SAVAGE FRANK']

na_value = 0;

for point in data_dict:
    for feat in data_dict[point]:
        num_feat_na = 0   
        if data_dict[point][feat] == 'NaN':
            na_value += 1
            num_feat_na +=1

print na_value

data_dict.pop("TOTAL", 0 )

#total_pay/salary ratio
for key in data_dict:
    salary = data_dict[key]['salary']
    total = data_dict[key]['total_payments']
    if salary != 'NaN' and total != 'NaN':
        data_dict[key]['total_to_salary'] = float(total) / (salary)
    else:
        data_dict[key]['total_to_salary'] = 0


features_list.append('total_to_salary')


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train = []
features_test  = []
labels_train   = []
labels_test    = []

split = StratifiedKFold(labels, 3, True, random_state=42)

for train_idx, test_idx in split:
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )



for point in data:
    salary = point[1]
    bonus = point[5]
    matplotlib.pyplot.scatter( salary,bonus )
	
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")


from sklearn.svm import SVC

#Feature Selection using KBest, Pipeline to do a series of transformations, PCA to perform dimension reduction  and GridSearch to automaticaly tune the parameters

from sklearn.neighbors import KNeighborsClassifier

scaler = MinMaxScaler()
skb = SelectKBest()
clf = KNeighborsClassifier()
pca = PCA()
#clf = SVC()
#clf = GaussianNB()

pipeline = Pipeline(steps =[('scaler', scaler),('kb', skb),('pca',pca), ('clf', clf)])
#pipeline = Pipeline(steps =[('kb', skb), ('clf', clf)])
#pipeline = Pipeline(steps =[('scaler',scaler),('kb', skb), ('clf', clf)])


pipeline_params = {
    'kb__k': [3,5,10], 
    'kb__score_func': [f_classif], 
    'clf__n_neighbors' : [3,4,5,6], 
    'clf__weights': ['distance','uniform'],
#    'clf__C': [1,5,10,50,100],
#    'clf__kernel':['rbf'],
    'pca__n_components':[1,2,3],
    'pca__whiten':[True,False]  
    }

ftwo_scorer = make_scorer(fbeta_score, beta=2)

clf = GridSearchCV(pipeline,param_grid=pipeline_params,verbose = 0, cv=split, scoring='f1')

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print clf.best_params_
print clf.best_score_ 
print accuracy_score(labels_test, pred)


print clf.best_estimator_.named_steps['kb'].scores_

features_selected_bool = clf.best_estimator_.named_steps['kb'].get_support()
features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]

print "The Selected Features Are:\n", features_selected_list 

best_features = []

for i in clf.best_estimator_.named_steps['kb'].get_support(indices = True):
     best_features.append(features_list[i+1]) 

d = defaultdict(int)
for i in best_features:
    d[i] += 1

sorted_d = sorted(d.items(), key=operator.itemgetter(1))
features_list = [x[0] for x in sorted_d[-len(best_features):]]

print "Precision: ", precision_score(labels_test,pred) 
print "Recall: ", recall_score(labels_test,pred)

clf = clf.best_estimator_

features_list.insert(0, 'poi')

dump_classifier_and_data(clf, my_dataset, features_list)
