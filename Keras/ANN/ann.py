import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib
import keras


def build_classifier(optimizer):
    classifer = Sequential()

    # adding the input layer and the first hidden layer with dropout (random drop neurons for less overfit)
    classifer.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifer.add(Dropout(0.1))

    classifer.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifer.add(Dropout(0.1))

    classifer.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifer.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

    return classifer


if __name__ == "__main__":
    # Read the data set
    dataset = pd.read_csv("Churn_Modelling.csv")

    # Take the values you actually want to use (pandas)
    x = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # ---- DATA PREPROCESSING

    # The label encoder gives the variables a number instead of a string
    labelencoder_X1 = LabelEncoder()
    x[:, 1] = labelencoder_X1.fit_transform(x[:, 1])

    labelencoder_X2 = LabelEncoder()
    x[:, 2] = labelencoder_X1.fit_transform(x[:, 2])

    # The one hot encoder takes a column that has been encoded, and makes as many columns as you need to have a binary
    # true/false representing the presence of that label
    onehotencoder = OneHotEncoder(categorical_features = [1])
    x = onehotencoder.fit_transform(x).toarray()

    # We're stripping away the first variable here. I believe this is because 0,0 can represent it
    x = x[:, 1:]

    # We then need to output our test vs training data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Scaling brings everything into the same scale. This is important do the weights work correctly. I think it uses
    # standard deviations
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train) # Fit the scaling and apply
    x_test = sc.transform(x_test)   # Just apply

    # OLD CODE JUST FOR REFERENCE BELOW ----------------------------------------
    # ---- NEURAL NET STUFF

    # Init ANN
    # clas = build_classifier()
    # clas.fit(x_train, y_train, batch_size=10, nb_epoch=100)
    # #
    # # Prediction
    # y_pred = clas.predict(x_test)
    # y_pred = y_pred > 0.5

    # Single Prediction Instructions

    # TO make a new prediction, use classifer.predict() and
    # pass in a numpy array np.array()
    # it needs to be two dimensional np.array([[]]) cause the
    # predict methods expects an array of arrays
    # you also need to scale the data. sklearn.preprocessing above.
    # This means call transform() from the sc object

    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix((y_test, y_pred))  # how good were predictions, false positive, misses, etc.

    # K-Fold Cross Validation
    # This replaces some everything after Data Preprocessing

    # kfold_class = KerasClassifier( build_fn=build_classifier, batch_size=10, epochs=100)
    # accuracies = cross_val_score(estimator=kfold_class, X=x_train, y=y_train, cv=10, n_jobs=-1)
    # OLD CODE JUST FOR REFERENCE ABOVE ----------------------------------------

    # Grid Search (Search a grid of different types of configurations. Using k-fold cross validation)
    gridsearch_class = KerasClassifier(build_fn=build_classifier)

    parameters = {'batch_size': [25, 32],
                  'epochs': [100, 500],
                  'optimizer': ['adam', 'rmsprop']}

    # parameters = {'batch_size': [32],
    #               'epochs': [1],
    #               'optimizer': ['adam']}

    g_search = GridSearchCV(gridsearch_class, parameters, scoring='accuracy')

    g_search.fit(x_train, y=y_train)

    best_parameters = g_search.best_params_
    best_accuracy = g_search.best_score_

    # saving model
    print(g_search.best_estimator_.model.save('my_model.h5'))

    from keras.models import load_model

    x = load_model('my_model.h5')
