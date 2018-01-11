Download train.gz from https://www.kaggle.com/c/avazu-ctr-prediction/data ,extract csv from it and rename to train_1.csv.
Place all the python scripts and this trainset under one directory.
Next steps will be executed from this particular directory.

Steps of execution:
1. For the purpose of visualization, run python Visualization.py. This will return all the visualized data.(Not a mandatory step)
2. For data cleaning and preprocessing, run python preprocess.py. This takes the train_1.csv as input and writes useful data to train.csv file.
   This train.csv will be used as input to execute any of the below algorithms.(Mandatory step)

3. For Naive Bayes, run python basian.py.
4. For SVM, run python svm.py.
5. For C4.5, run C45.py
6. For Random Forest, run python randomforest.py

All these algorithms on their respective run will return accuracy, true positive, true negative, false positive, false negative, precision, recall and f-measure.