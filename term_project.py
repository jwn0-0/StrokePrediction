import numpy as np
import pandas as pd
# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
# for data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# for feature importances
from sklearn.ensemble import ExtraTreesClassifier
# knn classifier
from sklearn.neighbors import KNeighborsClassifier
# decision tree classifier
from sklearn.tree import DecisionTreeClassifier
# for data evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

'''
show visually statistical information of dataSet
@:param:
    - df: entire dataSet
@:return
    None
'''
def statistical_info(df):
    # split categorical , numerical column
    cat_cols = df.select_dtypes(include='object').columns.to_list()
    num_cols = df.select_dtypes(exclude='object').columns.to_list()[1:]

    k = 1
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Numerical Categories Distribution")

    for col, k in zip(num_cols, np.arange(1, len(num_cols))):
        ax = fig.add_subplot(1, 5, k)
        sns.distplot(df[col], ax=ax, kde=True, kde_kws={"color": 'r', "alpha": 0.8, "linewidth": 2},
                     hist_kws={"linewidth": 3, "alpha": 1, "color": "g"})

    # distribution categorical feature and stroke
    k = 1
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle("Categorical Categories Distribution")

    for col, k in zip(cat_cols, np.arange(1, len(cat_cols) + 1)):
        ax = fig.add_subplot(1, 6, k)
        sns.countplot(df[col], ax=ax, linewidth=3, alpha=0.5, hue=df['stroke'])
        plt.xticks(rotation=90)
        fig.tight_layout()


'''
show visually distribution of stroke with several features
@:param:
    - df: entire dataSet
@:return
    None
'''
def stroke_distribution(df):
    # age
    plt.figure(figsize=(6, 6))
    sns.color_palette('colorblind')
    sns.kdeplot(df.query('stroke == 0')['age'], color='blue', shade=True, label='No_stroke')
    sns.kdeplot(df.query('stroke == 1')['age'], color='red', shade=True, label='Stroke')
    plt.title("Distribution of Age with Stroke")
    plt.legend()
    plt.show()
    # glucose level
    plt.figure(figsize=(6, 6))
    sns.color_palette('colorblind')
    sns.kdeplot(df.query('stroke == 0')['avg_glucose_level'], color='blue', shade=True, label='No_stroke')
    sns.kdeplot(df.query('stroke == 1')['avg_glucose_level'], color='red', shade=True, label='Stroke')
    plt.title("Distribution of Glucose level with stroke")
    plt.legend()
    plt.show()
    # bmi level
    plt.figure(figsize=(6, 6))
    sns.color_palette('colorblind')
    sns.kdeplot(df.query('stroke == 0')['bmi'], color='blue', shade=True, label='No_stroke')
    sns.kdeplot(df.query('stroke == 1')['bmi'], color='red', shade=True, label='Stroke')
    plt.title("Distribution of bmi level with stroke")
    plt.legend()
    plt.show()


'''
encode categorical features in dataSet using LabelEncoder 
@:param
    - df: dataFrame to encode
@:return
    encoded dataFrame
'''
def label_encoder(df):
    for i in cat_cols:
        le = LabelEncoder()
        df[i] = le.fit_transform(df[i])
    return df


'''
scale numerical features in independent variables using a various scaling method
@:param
    - X: dataSet for independent variable in separated target and independent
    - numeric_col_names: list of name of numerical features 
    - method(default=std): scaling method
        std - StandardScaling
        minmax - MinMaxScaling
        robust - RobustScaling
@:return
    scaled result
'''
def scaling_data(X, numeric_col_names, method='std'):
    if method == 'std':
        std = StandardScaler()
        return std.fit_transform(X[numeric_col_names])
    elif method == 'minmax':
        min_max = MinMaxScaler()
        return min_max.fit_transform(X[numeric_col_names])
    elif method == 'robust':
        robust = RobustScaler()
        return robust.fit_transform(X[numeric_col_names])
    else:
        return -1


'''
tune hyper parameter of K-Nearest Neighbors (k)
@:param 
    - X: independent variables
    - y: target variable
    - start: start point for getting best param in range
    - end: end point for getting best param in range
    - method(default='gscv'): tuning method
         gscv: GridSearch method
         rgscv: Randomized method
@:return
    the best hyper parameter integer k 
'''
def tune_knn_hyper_parameter(X, y, start, end, method='gscv'):
    # create new a base knn model
    classifier = KNeighborsClassifier()
    # create dictionary of all values we want to test for n_neighbors from 'start' to 'end'
    param_grid = {'n_neighbors': np.arange(start, end)}
    # if hyper parameter tuning method is Grid Search
    if method == 'gscv':
        # use GridSearchCV class
        knn_gscv = GridSearchCV(classifier, param_grid, cv=5)
        # fit model to data
        knn_gscv.fit(X, y)
        # return tuned best hyper parameter
        return knn_gscv.best_params_['n_neighbors']
    # if hyper parameter tuning method is Randomized search
    elif method == 'rgscv':
        # use RandomizedSearchCV class with 'n_iter' param(number of search)
        knn_rgscv = RandomizedSearchCV(classifier, param_grid, n_iter=40, cv=5)
        # fit model to data
        knn_rgscv.fit(X, y)
        # return tuned best hyper parameter
        return knn_rgscv.best_params_['n_neighbors']
    # else error
    else:
        return -1


'''
train knn model and evaluate its model
@:param 
    - X: independent variables
    - y: target variable
    - test_size(default=0.2): ratio of test set from 0 to 1 for split set
    - param_tuning(default=5): whether tuning hyper parameter or not
    - method(default='gscv'): tuning method
         gscv: GridSearch method
         rgscv: Randomized method
@:return
    dataFrames of result of holdout method and k-fold method 
'''
def train_and_evaluate_knn(X, y, k=5, test_size=0.2, param_tuning=False, method='gscv'):
    # split data set into train set and test set with stratify, shuffle options
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True,
                                                        stratify=y)
    if not param_tuning:  # if not use hyper parameter tuning
        # create new knn model with k
        knnClassifier = KNeighborsClassifier(n_neighbors=k)
        # fit model to data
        knnClassifier.fit(X_train, y_train)
        # predict target feature using test set
        y_predict = knnClassifier.predict(X_test)

        # show confusion matrix for the above result
        show_confusion_matrix(y_test, y_predict, 'K-Nearest Neighbors Confusion Matrix (default k=' + str(k) + ')')

        # get each scores of holdout method result
        holdout_result = {'k': k
            , 'Test Size Ratio': test_size
            , 'Accuracy': accuracy_score(y_test, y_predict)
            , 'Precision': precision_score(y_test, y_predict, zero_division=0)
            , 'Recall': recall_score(y_test, y_predict, zero_division=0)
            , 'F1 score': f1_score(y_test, y_predict, zero_division=0)}

        # get score of kfold method result
        kfold_result = knn_KFold(X, y, knnClassifier, split=5, param_tuning=param_tuning)

        # return holdout result and k-fold result
        return holdout_result, kfold_result
    else:  # if use hyper parameter tuning
        # get best hyper parameter according to tuning method
        tunedParameter = tune_knn_hyper_parameter(X, y, 1, 30, method=method)
        # create new knn model with hyper parameter tuned
        tunedParameterClassifier = KNeighborsClassifier(n_neighbors=tunedParameter)
        # fit model to data
        tunedParameterClassifier.fit(X_train, y_train)
        # predict target feature using test set
        y_tuned_predict = tunedParameterClassifier.predict(X_test)

        # show confusion matrix for the above result
        show_confusion_matrix(y_test, y_tuned_predict,
                              'K-Nearest Neighbors Confusion Matrix (best param k=' + str(tunedParameter) + ')')

        # get each scores of holdout method result
        holdout_result = {'k': tunedParameter
            , 'Test Size Ratio': test_size
            , 'Accuracy': accuracy_score(y_test, y_tuned_predict)
            , 'Precision': precision_score(y_test, y_tuned_predict, zero_division=0)
            , 'Recall': recall_score(y_test, y_tuned_predict, zero_division=0)
            , 'F1 score': f1_score(y_test, y_tuned_predict, zero_division=0)}

        # get score of kfold method result
        kfold_result = knn_KFold(X, y, tunedParameterClassifier, split=5, param_tuning=param_tuning, k=tunedParameter)
        # return holdout result and k-fold result
        return holdout_result, kfold_result


'''
evaluate K-Nearest Neighbors model using K-Fold cross validation method
@:param 
    - X: independent variables
    - y: target variable
    - split(default=5): number of folds
    - param_tuning(default=False): whether tuning hyper parameter or not 
    - k(default=5): hyper parameter k knn
@:return
    the dictionary for k and average of each cv accuracy
'''
def knn_KFold(X, y, classifier, split=5, param_tuning=False, k=5):
    # prepare cross validation
    cv = KFold(n_splits=split, shuffle=True, random_state=42)
    # if not use hyper parameter tuning
    if not param_tuning:
        # create new base knn model
        classifier = KNeighborsClassifier()
        # fit model to data
        classifier.fit(X, y)
        # train model cross validation
        accuracy1 = cross_val_score(classifier, X, y, scoring="accuracy", cv=cv)
        # return dictionary containing hyper parameter k and average of each cv accuracy
        return {'k': k, 'Mean Accuracy': accuracy1.mean()}
    # if use hyper parameter tuning
    else:
        # train model of knn used tuned hyper parameter
        accuracy2 = cross_val_score(classifier, X, y, scoring="accuracy", cv=cv)
        # return dictionary containing tuned hyper parameter k and average of each cv accuracy
        return {'k': k, 'Mean Accuracy': accuracy2.mean()}


'''
show heatmap of confusion matrix for actual and predicted
@:param 
    - X: the test set (actual data)
    - y: the predict set (predicted data)
    - title: title of confusion matrix heatmap
@:return
    None
'''
def show_confusion_matrix(y_test, y_predict, title):
    # create confusion matrix for y_test and y_predict
    confusionMatrix = confusion_matrix(y_test, y_predict)
    # plot confusion matrix in heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusionMatrix, annot=True, fmt='d', ax=ax, cmap='OrRd')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0.5, 1.5], ['no stroke(0)', 'stroke(1)'])
    plt.yticks([0.5, 1.5], ['no stroke(0)', 'stroke(1)'])
    plt.show()


'''
plot importance of each feature using ExtraTreesClassifier
@:param 
    - X: independent variables
    - y: target variable
@:return
    None
'''
def show_feature_importances(X, y):
    # create ExtraTreesClassifier instance
    model = ExtraTreesClassifier()
    # fit model to data
    model.fit(X, y)
    # show plot importance of each feature
    fi = pd.Series(model.feature_importances_, index=X.columns)
    fi.nlargest(10).plot(kind='barh')
    plt.title('Feature Importances')
    plt.show()


'''
function: decisionTree
A decision tree is created through train data set and predictions are made through test data set.
return the model's accuracy.
@:param
    - model: decision tree classifier
@:return
    - train accuracy and test_accuracy
'''
def decisionTree(model):
    model.fit(x_train, y_train)
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    return train_accuracy, test_accuracy


'''
function: gridSearch
Find the optimal depth among the depth of the tree from 1 to 10.
- refit=True: Re-learn with the best parameter settings.
@:return
    optimal hyperparameters, highest accuracy, and accuracy of the test data set.
'''
def gridSearch():
    m_depth = np.linspace(1, 10)
    m_depth = m_depth.astype(int)

    # Set parameter to dictionary form
    grid_params = {'max_depth': m_depth}
    # Set the test run by dividing the hyperparameters into 5 train, test set fold.
    grid_tree = GridSearchCV(dt_clf, param_grid=grid_params, cv=5, scoring='accuracy', refit=True,
                             return_train_score=True)
    # Learn/assess hyperparameters sequentially.
    grid_tree.fit(x_train, y_train)

    # Hyperparameter adjustment in the learning process returns the model with the best performance
    estimator = grid_tree.best_estimator_
    # Assess model performance in test data
    # gr_pred = estimator.predict(x_test)

    return grid_tree.best_params_, grid_tree.best_score_, estimator


'''
function: drawTree
Visualize the decision tree
- figsize(width, height): Set graph pattern size.
- max_depth: Set how much longer to extend except for the root node.
- filed: Decide whether to color the node for the class
- feature_names: passing the name of the attribute
- rounded: Determines whether to round the border of the node.
@:param
    - model: decision tree base model 
    - best_model: hyperparameter tuned decision tree
@:return
    None
'''
def drawTree(model, best_model):
    plt.figure(figsize=(45, 20))
    tree.plot_tree(model, filled=True, feature_names=feature_names, rounded=True)
    plt.title('Decision Tree')
    plt.show()
    # pruning
    tree.plot_tree(best_model, filled=True, feature_names=feature_names, rounded=True)
    plt.title('Decision Tree (best param)')
    plt.show()
    tree.plot_tree(dt_clf, max_depth=2, filled=True, feature_names=feature_names, rounded=True)
    plt.title('max depth = 2')
    plt.show()


'''
function: kFordEv
Cross-validation is performed by separating into n sets of folds.
- cv_accuracy : Create a list for accuracy per fold set.
- n_iter = repeat n times
@:param
    n: number of split in k-fold 
    model: decision tree model
@:return
    the average of the accuracy per fold set.
'''
def kFordEv(n, model):
    kfold = KFold(n_splits=n)

    cv_accuracy = []
    n_iter = 0

    for train_index, test_index in kfold.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # training
        model.fit(x_train, y_train)
        # predict
        pred = model.predict(x_test)
        n_iter += 1

        # Measure accuracy every iteration(Round to 5 decimal places)
        accuracy = np.round(accuracy_score(y_test, pred), 5)
        train_size = x_train.shape[0]
        test_size = x_test.shape[0]
        print('\n>{0}th cross-validation accuracy : {1},  training data size : {2},  validation data size : {3}'
              .format(n_iter, accuracy, train_size, test_size))
        print('>{0} validation set index : {1}'.format(n_iter, test_index))

        cv_accuracy.append(accuracy)

    return np.mean(cv_accuracy)


'''
function: crossVal
It performs learning and predictive evaluation internally at once.
- cv : Number of cross-validation sets
- scoring : Performance indicator, current accuracy is performance indicator.
@:param
    model: decision tree model
@:return
    verification-specific accuracy and average verification accuracy.
'''
def crossVal(model):
    scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=5)
    return scores, np.mean(scores)


'''
Data Inspection
'''

# read csv file
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

# print dataSet information
print(stroke.head())
print(stroke.describe())
print(stroke.info())
print(format(stroke.shape))

# plot data information
statistical_info(stroke)
stroke_distribution(stroke)

'''
Data Preprocessing
'''

stroke['smoking_status'] = stroke['smoking_status'].replace('Unknown', np.NaN)
# show missing value
miss_val = stroke.isnull().sum() / len(stroke) * 100
print("\nMissing values ratio ")
print(miss_val)

print("\nMissing values in variable bmi: {:.2f}%".format(miss_val['bmi']))
print("Missing values in variable smoking_status: {:.2f}%".format(miss_val['smoking_status']))
print("stroke shape: {}".format(stroke.shape))

# fillna with bmi mean value
stroke['bmi'] = stroke['bmi'].fillna(stroke['bmi'].mean())
# fillna with never_smoked
stroke['smoking_status'] = stroke['smoking_status'].fillna('never smoked')

clean_stroke = stroke[stroke['smoking_status'].notnull()]
# show there's no more missing values
miss_val = clean_stroke.isnull().sum() / len(clean_stroke) * 100

print("\nAfter cleaning missing values")
print(miss_val)
print("\nMissing values in variable 'bmi': {}".format(miss_val['bmi']))
print("Missing values in variable 'smoking_status': {}".format(miss_val['smoking_status']))
print("Shape of stroke without missing values: {}".format(clean_stroke.shape))

# Make encoding function
X = stroke.drop(['id', 'stroke'], axis=1)
y = stroke['stroke']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.to_list()
cat_cols = X.select_dtypes(include=['object']).columns.to_list()

# Encoding
X = label_encoder(X)

# Scaling
X[num_cols] = scaling_data(X, num_cols, method='robust')

print(X.head())

# Show correlation between each features
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.matshow(X.corr(), )
ax.set_xticks(np.arange(X.shape[1]))
ax.set_yticks(np.arange(X.shape[1]))
ax.set_xticklabels(X.columns, rotation=90)
ax.set_yticklabels(X.columns)
# create color Heatmap
colorbar = ax.figure.colorbar(im, ax=ax)
colorbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom", fontsize=10)
fig.tight_layout()
plt.show()

# show each feature's importances
show_feature_importances(X, y)

'''
Data Analysis & Evaluation
'''

'''============ K-Nearest Neighbors ============='''
# get evaluation result of K-NN classifier using holdout method and k-fold cross validation
baseModelHoldout, baseModelKFold = train_and_evaluate_knn(X, y, param_tuning=False, test_size=0.2, k=5)
parameterTunedHoldout, parameterTunedKFold = train_and_evaluate_knn(X, y, param_tuning=True, test_size=0.2)

# print Holdout method
print('※ Holdout Method: K-Nearest Neighbors')
predictResult = pd.DataFrame([baseModelHoldout, parameterTunedHoldout], index=['K-NN (default)', 'K-NN (best param)'])
print(predictResult)

# print K-Fold Cross Validation method (K=5)
print('※ 5-Fold Cross Validation: K-Nearest Neighbors')
kfold_cross_validation_Result = pd.DataFrame([baseModelKFold, parameterTunedKFold], columns=['k', 'Mean Accuracy'],
                                             index=['K-NN (default)', 'K-NN (best param)'])
print(kfold_cross_validation_Result)

'''============== Decision Tree ==============='''

# train, test(validation) split
'''
test_size = 0.2 : 20% data is used as test(validation)/ default 0.25
- x_train : feature value in train dataset
- x_test : feature value in test dataset
- y_train : objective variable in train dataset
- y_train : objective variable in train dataset
'''
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print(x_train.shape)
print(x_test.shape)

# store the decision tree library in dt_clf
dt_clf = DecisionTreeClassifier(random_state=1)
# Convert the nave of the train dataset column to a list and store it.
feature_names = X.columns.tolist()

train_accuracy, test_accuracy = decisionTree(dt_clf)
print('DecisionTree training data score:{:.4f}'.format(train_accuracy))
print('DecisionTree testing data score:{:.4f}'.format(test_accuracy))

best_param, best_score, estimator = gridSearch()
print('\nGridSearchCV hyper-parameter :', best_param)
print('GridSearchCV highest average accuracy : {:.4f}'.format(best_score))
best_param_train_accuracy, best_param_test_accuracy = decisionTree(estimator)
print('DecisionTree training data score(best param):{:.4f}'.format(best_param_train_accuracy))
print('DecisionTree testing data score(best param):{:.4f}'.format(best_param_test_accuracy))

drawTree(dt_clf, estimator)

print('\n# result 5-fold :')
kfold_mean_accuracy = kFordEv(5, dt_clf)
print('->> average validation accuracy :', kfold_mean_accuracy)
print('\n\n# result 5-fold(best param :')
kfold_mean_accuracy = kFordEv(5, estimator)
print('->> average validation accuracy(best param) :', kfold_mean_accuracy)

print('\n# result cross_validation :')
cv_score, cv_mean_score = crossVal(dt_clf)
print('>cross validation accuracy : {0}'.format(cv_score))
print('>average validation accuracy : ', np.mean(cv_mean_score))
print('\n# result cross_validation(best param) :')
cv_score, cv_mean_score = crossVal(estimator)
print('>cross validation accuracy : {0}'.format(cv_score))
print('>average validation accuracy : ', np.mean(cv_mean_score))