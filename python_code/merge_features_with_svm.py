import pandas as pd
import numpy as np
import sys
import os
import getopt
import math
import easy_excel
from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory
from sklearn import svm
from sklearn.model_selection import *
import sklearn.ensemble
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
mem = Memory("./mycache")
@mem.cache
def get_data(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]

def performance(labelArr, predictArr):
    #labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.; TN = 0.; FP = 0.; FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
    SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    GM=math.sqrt(recall*SP)
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return precision,recall,SN,SP,GM,TP,TN,FP,FN
def arff2svm(arff_files):
    svm_files = []
    for arff_file in arff_files:
        name = arff_file[0: arff_file.rindex('.')]
        tpe = arff_file[arff_file.rindex('.')+1:]
        svm_file = name+".libsvm"
        svm_files.append(svm_file)
        if tpe == "arff":
            if os.path.exists(svm_file):
                pass
            else:
                f = open(arff_file)
                w = open(svm_file, 'w')
                flag = False
                for line in f.readlines():
                    if flag:
                        if line.strip() == '':
                            continue
                        temp = line.strip('\n').split(',')
                        w.write(temp[len(temp)-1])
                        for i in range(len(temp)-1):
                            w.write(' '+str(i+1)+':'+str(temp[i]))
                        w.write('\n')
                    else:
                        line = line.upper()
                        if line.startswith('@DATA') or flag:
                            flag = True
                f.close()
                w.close()
        elif tpe == "libsvm":
            continue
        else:
            print "File format error! Arff and libsvm are passed."
            sys.exit()
    return svm_files

opts, args = getopt.getopt(sys.argv[1:], "hi:c:t:o:s:m:", )
for op, value in opts:
    if op == "-i":
        input_files = str(value)
        input_files = input_files.replace(" ", "").split(',')
        for input_file in input_files:
            if input_file == "":
                print "Warning: please insure no blank in your input files !"
                sys.exit()
    elif op == "-c":
        cv = int(value)
    elif op == "-t":
        split_rate = float(value)
    elif op == "-o":
        excel_name = str(value)
    elif op == "-s":
        if str(value) != "0":
            isSearch = True
    elif op == "-m":
        if str(value) == "0":
            isMultipleThread = False
    elif op == "-h":
        print 'Cross-Validate: python easy_classify.py -i {input_file.libsvm} -c {int: cross validate folds}'
        print 'Train-Test: python easy_classify.py -i {input_file.libsvm} -t {float: test size rate of file}'
        print 'More information: https://github.com/ShixiangWan/Easy-Classify'
        sys.exit()
whole_result=[]
whole_dimension=[]
if __name__=="__main__":
    for i in xrange(len(input_files)):
        for j in xrange(i+1,len(input_files)):
            first_name=input_files[i].strip("").split('.')[0]
            second_name=input_files[j].strip("").split('.')[0]
            output_name=first_name+'_'+second_name+'.csv'
            first_file = pd.read_csv(input_files[i], header=None, index_col=None)  # sys.argv[1])
            end = len(first_file.values[0])
            first_file = first_file.values[:, 0:end - 1]
            first_file = pd.DataFrame(first_file).astype(float)
            second_file = pd.read_csv(input_files[j], header=None, index_col=None)
            end= len(second_file.values[0])
            second_file=second_file.values[:,0:end-1]
            second_file = pd.DataFrame(second_file).astype(float)
            print "first_file_num:", len(first_file)
            print "first_file_length:", len(first_file.values[0])
            print "second_file_num:", len(second_file)
            print "second_file_length:", len(second_file.values[0])
            output_file = pd.concat([first_file, second_file], axis=1)
            print "output_file_num:", len(output_file)
            print "output_file_len:", len(output_file.values[0])
            scaler = MinMaxScaler()
            output_file = scaler.fit_transform(np.array(output_file))
            print "normalization"
            pd.DataFrame(output_file).to_csv(output_name, header=None, index=False)
            print output_file
            train_data = pd.DataFrame(output_file)
            X = np.array(train_data)
            Y = list(map(lambda x: 1, xrange(len(train_data) / 2)))
            Y2 = list(map(lambda x: 0, xrange(len(train_data) / 2)))
            Y.extend(Y2)
            Y = np.array(Y)
            svc = svm.SVC()
            # parameters = {'kernel':['rbf'], 'C': np.linspace(32750,32799,100), 'gamma': np.linspace(0.00001,0.00009,20)}
            parameters = {'kernel': ['rbf'], 'C': [math.pow(2,e) for e in range(-5,15,2)], 'gamma': [math.pow(2,e) for e in range(-15, -5, 2)]}
            clf = GridSearchCV(svc, parameters, cv=10, n_jobs=-1, scoring='roc_auc')
            clf.fit(X, Y)
            C = clf.best_params_['C']
            print clf.best_score_
            gamma = clf.best_params_['gamma']
            y_predict = cross_val_predict(svm.SVC(kernel='rbf', C=C, gamma=gamma), X, Y, cv=10, n_jobs=-1)
            ROC_AUC_area = "%0.4f" % cross_val_score(svm.SVC(kernel='rbf', C=C, gamma=gamma), X, Y, cv=10, n_jobs=-1).mean()
            ACC = metrics.accuracy_score(Y, y_predict)
            precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
            F1_Score = metrics.f1_score(Y, y_predict)
            F_measure = F1_Score
            MCC = metrics.matthews_corrcoef(Y, y_predict)
            pos = TP + FN
            neg = FP + TN

            whole_result.append([['svm' + "C:" + str(C) + "gamma:" + str(gamma), ACC, precision, recall, SN, SP, GM, F_measure,
                          F1_Score, MCC, ROC_AUC_area, TP, FN, FP, TN, pos, neg]])
            whole_dimension.append(str(X.shape[1]))
            print whole_result
    easy_excel.save("svm_crossvalidation", whole_dimension, whole_result,'svm.xls')
# print RFH_PseDNC
# RFH_=pd.read_csv(sys.argv[1],header=None,index_col=None)
# RFH_=pd.DataFrame(RFH_).astype(float)
# PseDNC_=pd.read_csv(sys.argv[2],header=None,index_col=None)
# print len(PseDNC_.values[0])
# RFH_PseDNC=pd.concat([RFH_,PseDNC_],axis=1)
# pd.DataFrame(RFH_PseDNC).to_csv(sys.argv[3],header=None,index=False)
# print RFH_PseDNC

