#!/usr/bin/python

import pandas as pd
import numpy as np
import xgboost
from sklearn import metrics
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from vecstack import stacking
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from sklearn.utils import resample

# load dataset
covid_df   = pd.read_csv("covid_data.csv")
covid_cols = ["SARSCov", "Hemoglobin", "Platelets", "Leukocytes",
                     "Lymphocytes", "Basophils", "Eosinophils", "Neutrophils",
                     "Monocytes", "Patient age quantile", "Urea", "Proteina C reativa mg/dL",
                     "Creatinine", "Potassium", "Sodium", "Alanine transaminase",
                     "Aspartate transaminase", "International normalized ratio (INR)",
                     "Albumin"]

covid_df = covid_df[covid_cols]

models = [
    ExtraTreesClassifier(n_estimators=300, max_depth=17),
    RandomForestClassifier(n_estimators=300, max_depth=17),
    LogisticRegression(solver = 'saga', max_iter= 500, multi_class='ovr',
                        class_weight='balanced')
]

# configure bootstrap
n_iterations = 1

accuracies_stats  = list()
AUC_stats         = list()
pre_stats         = list()
recall_stats      = list()
f1_stats          = list()
sensitivity_stats = list()
specificity_stats = list()
TP_list = list()
TN_list = list()
FP_list = list()
FN_list = list()

for i in range(n_iterations):

    imputer = KNNImputer(n_neighbors=7)
    Ximputer = imputer.fit_transform(covid_df)
    dataframe=pd.DataFrame(Ximputer, columns=covid_cols)

    outlier_detect = IsolationForest(n_estimators=150, max_samples=621, contamination=float(0.07), max_features=covid_df.shape[1])

    outlier_detect.fit(dataframe)
    outliers_predicted = outlier_detect.predict(dataframe)

    covid_check = dataframe[outlier_detect.predict(dataframe) == -1]

    dataframe = dataframe[outlier_detect.predict(dataframe) != -1]

    values = dataframe.values

    n_size = int(len(dataframe) * 0.80)
	# prepare train and test sets
    data_sample = resample(values, n_samples=n_size)

    dataframe = pd.DataFrame(data_sample, columns=covid_cols)
    # split into input and output elements
    y = dataframe.SARSCov # Target variable
    X = dataframe.drop(['SARSCov'], axis = 1) # Features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sm = SMOTE(k_neighbors=11)
    x_res, y_res = sm.fit_sample(X_train, y_train)

    S_train, S_test = stacking(models,
                               x_res, y_res, X_test,
                               regression=False,

                               mode='oof_pred',

                               needs_proba=False,

                               save_dir=None,

                               metric=metrics.accuracy_score,

                               n_folds=10,

                               stratified=True,

                               shuffle=True,

                               verbose=2)

    model = XGBClassifier(learning_rate=1.0,
                          n_estimators=300, max_depth=17)

    # fit model
    model = model.fit(S_train, y_res)

	# evaluate model
    y_pred = model.predict(S_test)
    score = metrics.accuracy_score(y_test, y_pred)

    accuracies_stats.append(score)

    probs = model.predict_proba(S_test)
    probs = probs[:, 1]
    AUC_stats.append(metrics.roc_auc_score(y_test, probs))

    pre_stats.append(metrics.precision_score(y_test, y_pred, average='binary'))
    recall_stats.append(metrics.recall_score(y_test, y_pred, average='binary'))
    f1_stats.append(metrics.f1_score(y_test, y_pred, average='binary'))

    confusion = metrics.confusion_matrix(y_test, y_pred)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # recall of the positive
    sensitivity = TP / float(FN + TP)

    # recall of the negative class
    specificity = TN / (TN + FP)

    sensitivity_stats.append(sensitivity)
    specificity_stats.append(specificity)

    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)

print("Average TP  = {}".format(np.round(np.array(TP_list).mean(), 4)))
print("Average TN  = {}".format(np.round(np.array(TN_list).mean(), 4)))
print("Average FP  = {}".format(np.round(np.array(FP_list).mean(), 4)))
print("Average FN  = {}".format(np.round(np.array(FN_list).mean(), 4)))

# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(accuracies_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(accuracies_stats, p))

print("Average accuracy  = {}".format(np.round(np.array(accuracies_stats).mean(), 4)))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(AUC_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(AUC_stats, p))

print("Average AUC       = {}".format(np.round(np.array(AUC_stats).mean(), 4)))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(pre_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(pre_stats, p))
print("Average precision = {}".format(np.round(np.array(pre_stats).mean(), 4)))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(recall_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(recall_stats, p))
print("Average recall    = {}".format(np.round(np.array(recall_stats).mean(), 4)))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(f1_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(f1_stats, p))
print("Average f1 score = {}".format(np.round(np.array(f1_stats).mean(), 4)))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(sensitivity_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(sensitivity_stats, p))
print("Sensitivity score = {}".format(np.round(np.array(sensitivity_stats).mean(), 4)))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(specificity_stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(specificity_stats, p))
print("Specificity score = {}".format(np.round(np.array(specificity_stats).mean(), 4)))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
