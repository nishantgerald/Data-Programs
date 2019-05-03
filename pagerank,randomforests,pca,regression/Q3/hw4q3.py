## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('./eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100
# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=random_state,shuffle=True)
y_true_train=[];y_true_test=[];
for i in y_train:
	y_true_train.append(i)
for i in y_test:
	y_true_test.append(i)
# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
lin_reg = LinearRegression().fit(x_train,y_train)
y_predicted_train=lin_reg.predict(x_train);y_predicted_test=lin_reg.predict(x_test);
# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX
y_pred_train=[];y_pred_test=[];
for i in y_predicted_train:
	y_pred_train.append(int(round(i)))
for i in y_predicted_test:
	y_pred_test.append(int(round(i)))
print("Training set accuracy score using LR:",round(accuracy_score(y_true_train,y_pred_train,normalize=True),2))
print("Testing set accuracy score using LR:",round(accuracy_score(y_true_test,y_pred_test,normalize=True),2))
print()
# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX
rfc=RandomForestClassifier(n_estimators=10).fit(x_train,y_train)
y_predicted_train=rfc.predict(x_train);y_predicted_test=rfc.predict(x_test);
# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
y_pred_train=[];y_pred_test=[];
for i in y_predicted_train:
	y_pred_train.append(int(round(i)))
for i in y_predicted_test:
	y_pred_test.append(int(round(i)))
print("Training set accuracy score using RFC:",round(accuracy_score(y_true_train,y_pred_train,normalize=True),2))
print("Testing set accuracy score using RFC:",round(accuracy_score(y_true_test,y_pred_test,normalize=True),2))

# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX
importances=[]
for i in rfc.feature_importances_:
	importances.append(i)

print('\nFeatures ranked in descending order of importance:')
for i in np.argsort(importances)[::-1]:
	print('X'+str(i),end=' ')
print('\n')
most_important_feature='X'+str(np.argsort(importances)[-1])
least_important_feature='X'+str(np.argsort(importances)[0])
print("Most Important Feature evaluated by RFC:",most_important_feature)
print("Least Important Feature evaluated by RFC:",least_important_feature)
# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX
parameters={'n_estimators':[10,20,50,150,200],'max_depth':[1,4,7]}
clf=GridSearchCV(rfc,parameters,cv=10)
clf.fit(x_train,y_train)
print("Best score for training: ",clf.best_score_)
print("Best Parameters for training",clf.best_params_)
clf2=GridSearchCV(rfc,parameters,cv=10)
clf2.fit(x_test,y_test)
print("Best score for testing: ",clf2.best_score_)
print("Best Parameters for testing",clf2.best_params_)

rfc=RandomForestClassifier(n_estimators=150,max_depth=7).fit(x_train,y_train)
y_pred_test=rfc.predict(x_test)
print("Prediction accuracy of RFC after Tuning: ", round(accuracy_score(y_true_test,y_pred_test,normalize=True),2))
# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
pp_xtrain=normalize(x_train)
pp_xtest=normalize(x_test)
# TODO: Create a SVC classifier and train it.
# XXX
svm=SVC(gamma='auto').fit(pp_xtrain,y_true_train)
y_pred_train=svm.predict(pp_xtrain)
y_pred_test=svm.predict(pp_xtest)
# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print()
print("Training set accuracy score using SVM:",round(accuracy_score(y_true_train,y_pred_train),2))
print("Testing set accuracy score using SVM:",round(accuracy_score(y_true_test,y_pred_test),2))

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

svm_parameters = {'kernel':['linear', 'rbf'], 'C':[0.01, 0.1, 1]}
svm2 = SVC(gamma="auto")
clf3 = GridSearchCV(svm2, svm_parameters, cv=10).fit(pp_xtrain,y_train)
print("Best score for training: ",clf3.best_score_)
print("Best Parameters for training",clf3.best_params_)
print("CV Results:",clf3.cv_results_)
# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
# XXX
print()
pca=PCA(n_components=10,svd_solver='full')
pca.fit(x_data)

explained_variance_percentage=[]
for i in pca.explained_variance_ratio_:
	explained_variance_percentage.append(float(i)*100)
singular_values=[]
for i in pca.singular_values_:
	singular_values.append(i)
print('Percentage of variance explained by each component:',explained_variance_percentage)
print()
print('Singular values corresponding to each of the selected components:',singular_values)
