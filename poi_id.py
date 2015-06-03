#!/usr/bin/python

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

import pickle
import numpy as np
import operator
import pprint as pp
import time
%matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import describe
from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from __future__ import print_function

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list =['poi',
                'expenses', 
                'total_stock_value', 
                'bonus', 
                'deferred_income', 
                 ]

### Task 2: Remove outliers
removed = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 'LOCKHART EUGENE E']
for name in removed:
    print("Removed: %s" %name)
    pp.pprint(data_dict[name])
    data_dict.pop(name)

##########################################################################
### Uncomment the codes if you want to see what new features were created
### and tested for feature importance
### Task 3: Create new feature(s)
'''
def compute_share(a,b,output):
    if person[a] == 'NaN' or person[b]=='NaN':
        person[output] = 'NaN'
    else:
        person[output]= round(float(person[a])/float(person[b]),4)

for person in data_dict.values():
    compute_share('bonus', 'salary', 'bonus_vs_salary')
    compute_share('total_stock_value','salary', 'stock_vs_salary')
    compute_share('long_term_incentive', 'salary', 'incentive_vs_salary')
    compute_share('expenses', 'salary', 'expenses_vs_salary')
    compute_share('exercised_stock_options', 'total_stock_value', 'exercised_share_stock')
    compute_share('deferral_payments', 'total_payments', 'deferral_share_total_payment')
    compute_share('from_poi_to_this_person', 'to_messages', 'received_from_poi_share')
    compute_share('from_this_person_to_poi', 'from_messages', 'sent_to_poi_share')
'''
###########################################################################
    
### Convert dictionary to dataframe
df = pd.DataFrame.from_dict(data_dict, orient='columns')
df = df.transpose()
### Treat NaN as 0
df = df.replace('NaN',0)
### Deferred income is stored as negative values
df['deferred_income'] = [abs(v) for v in df['deferred_income']]
### Scale features
min_max_scaler = preprocessing.MinMaxScaler()
for feature in features_list[1:]:
    df[feature] = min_max_scaler.fit_transform(df[feature].astype(float))
### Store to my_dataset for easy export below.
my_dataset = df.T.to_dict('dict')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Convert labels to integer
labels = map(int, labels)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC()
kn = KNeighborsClassifier()
dt =  DecisionTreeClassifier()
rfc = RandomForestClassifier()
adb = AdaBoostClassifier()

models = [(lr, 'Logistic'),
          (gnb, 'Naive Bayes'),
          (svc, 'Support Vector Classification'),
          (kn, 'K-Neighbors'),
          (dt, 'Decison Tree'),
          (rfc, 'Random Forest'),
          (adb, 'AdaBoost')]

for clf, name in models :
    test_classifier(clf, my_dataset, features_list, folds = 1000)

#########################################################################
####################Uncomment to see learning curve######################
'''
###Shortlist models
models_s = [(gnb, 'Naive Bayes'),
            (adb, "AdaBoost"),
            (dt, 'Decison Tree'),]
###Plot learning curves. 
plt.figure(figsize=(len(models_s)*5,len(models_s)*1.75))
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.tight_layout()
    return plt

cv = StratifiedKFold(labels, n_folds=3)

j = 0
for clf, name in models_s:
    title = "Learning Curves (%s)" %name
    ax = plt.subplot2grid((1,len(models_s)), (0, j))
    plot_learning_curve(clf, title, features, labels, (0.5, 1.01), cv = cv)
    j +=1
'''
#########################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#########################################################################
############Uncomment to see results for different parameters.###########
############Warning of slowness##########################################
'''
###Tune parameters
n_estimators = [10, 50, 100, 250, 1000]
learning_rate = [0.01, 0.1, 1.0, 1.5, 2]

### Iterate through all parameters
for i in range(len(n_estimators)):
    for j in range(len(learning_rate)):
        clf = AdaBoostClassifier(n_estimators = n_estimators[i],
                                 learning_rate = learning_rate[j])
        test_classifier(clf, my_dataset, features_list, folds = 1000)
'''
#########################################################################
### best parameters
clf = AdaBoostClassifier(n_estimators = 50, learning_rate = 1.0)
test_classifier(clf, my_dataset, features_list, folds = 1000)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
