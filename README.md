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

## Conclusion
In the case of K-Nearest Neighbors, the result of combination with robust scaling was the highest accuracy (k=16, accuracy=0.951272, with stratify and shuffle options)
And in the case of Decision Tree, the same accuracy result was obtained regardless of which scaling method was used (max_depth=1, accuracy=0.9513211, with stratify and shuffle options).
In conclusion, when a Decision Tree Classifier with max_depth of 1 was used, the accuracy was the highest (95.13211%).

Stroke is the No. 1 disease in mortality, and it is rapidly increasing over the age of 50, and once it occurs, it causes serious psychological, physical, and economic aftereffects, so a project was planned to predict and prevent it in advance.
But we think our project failed. we think we need more features because we're dealing with an important topic of life, and we need at least 100,000 samples that include what doctors think are features.
What doctors predicted in common with patients
1.	The higher the age, the higher the patient's probability of developing a stroke.
2.	Depending on the degree of systolic blood pressure (the higher the high blood pressure), the higher the probability of stroke.
3.	Heart-related diseases are higher.
4.	Diabetes is also highly associated with stroke.
5.	Bad lifestyle: smoking, obesity, and lack of exercise, eating habits, stress, and drinking too much.
6.	Chronic diseases that cause stroke: high blood pressure, diabetes, hyperlipidemia, heart disease, conventional stroke.
7.	Uncontrollable risk factors: age, gender, race, family history.
8.	Controllable risk factors: hypertension, diabetes, heart disease (atrial fibrillation, coronary disease, etc.), hyperlipidemia, smoking, obesity, carotid artery stenosis, etc.
※ It can be controlled by steady drug use and lifestyle correction, and the most dangerous factor is when they have experienced a stroke once before.

Since it is a life-related topic, the impact is great even if we predict it a little wrong.
In particular, it was predicted that it was not a stroke, but in the case of a stroke, it was predicted that it was a stroke, but the wavelength was greater than that of a non-stroke.
Stroke is also divided into ischemic and hemorrhagic strokes, and weights are given differently for each data feature
we think it failed because there was no such work.
The prediction accuracy result is 95.13211%, but it is judged that it is difficult to use for medical purposes because it is estimated to be lower in reality.

