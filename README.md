# StrokePrediction
StrokePrediction Using dataset

## WHY?
Since stroke is a disease that can suddenly occur as one of the major cause of death in Korea
and the disability lasts for a considerable period of time, 
the goal of our project is to predict stroke comprehensively with various factor (gender, age, diseases, smoking, etc.).

## Used Dataset (Kaggle)
Stroke Prediction Dataset
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset


## Used Function
### Specification details of function click term_project.py
+ statistical_info(df)
>> Show visually statistical information of dataSet
+ stroke_distribution(df)
>> show visually distribution of stroke with several features
+ label_encoder(df)
>> encode categorical features in dataSet using LabelEncoder
+ tune_knn_hyper_parameter(X, y, start, end, method='gscv')
>> tune hyper parameter of K-Nearest Neighbors (k)
+ train_and_evaluate_knn(X, y, k=5, test_size=0.2, param_tuning=False, method='gscv')
>> train knn model and evaluate its model
+ knn_KFold(X, y, classifier, split=5, param_tuning=False, k=5)
>> evaluate K-Nearest Neighbors model using K-Fold cross validation method
+ show_confusion_matrix(y_test, y_predict, title)
>> show heatmap of confusion matrix for actual and predicted
+ show_feature_importances(X, y)
>> plot importance of each feature using ExtraTreesClassifier
+ decisionTree(x_train, y_train, x_test, y_test)
>> A decision tree is created through train data set and predictions are made through test data set
+ gridSearch()
>> Find the optimal depth among the depth of the tree from 1 to 10
+ drawTree()
>> Visualize the decision tree
+ kFordEv(n)
>> Cross-validation is performed by separating into n sets of folds
+ crossVal()
>> It performs learning and predictive evaluation internally at once
