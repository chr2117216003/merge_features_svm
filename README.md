# construct a python code to merge the features of csv file,and use svm classifier to do a binaru classification
#用来融合特征并且通过svm分类器来训练的特征融合简单程序
****
## 程序依赖包

## 程序执行命令
用于特征融合跑svm的程序，会将输入的csv文件进行融合跑程序，并且会输出融合的文件，最后得到总得分析表svm.xls，目前程序初步开始，仅仅拥有svm分类器，默认的参数取值范围为
**C:2^-5~2^15,gamma:2^-15~2^-5**
例子：`python merge_features_with_svm.py -i firstFile.csv,secondFile.csv,thirdFile.csv`
## 程序参考
