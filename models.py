import pandas as pd
import preprocessingfile as preprocess
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

def NN(original_data, original_X, original_Y, combined_training_data, x_train1, x_train2, x_train, x_test, x_val, y_train1, y_train2, y_train, y_test, y_val):
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and first hidden layer
    classifier.add(Dense(15, activation='relu', input_dim=len(original_X.columns)))
    classifier.add(Dense(8, activation='relu'))
    classifier.add(Dense(5, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))  # Output layer
    
    # Compile the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    classifier.fit(x_train, y_train, batch_size=10, epochs=100)
    
    return classifier

def random_forest(original_data, original_X, original_Y, combined_training_data, x_train1, x_train2, x_train, x_test, x_val, y_train1, y_train2, y_train, y_test, y_val):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(x_train, y_train)
    return clf

def svm(original_data, original_X, original_Y, combined_training_data, x_train1, x_train2, x_train, x_test, x_val, y_train1, y_train2, y_train, y_test, y_val):
    from sklearn.svm import SVC
    clf = SVC(gamma='auto')
    clf.fit(x_train, y_train)
    return clf

def cnn(original_data, original_X, original_Y, combined_training_data, x_train1, x_train2, x_train, x_test, x_val, y_train1, y_train2, y_train, y_test, y_val):
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten

    x_train_matrix = x_train.values
    x_val_matrix = x_val.values
    y_train_matrix = y_train.values
    y_val_matrix = y_val.values
    
    img_rows, img_cols = 1, len(original_X.columns)
    
    x_train1 = x_train_matrix.reshape(x_train_matrix.shape[0], img_rows, img_cols, 1)
    x_val1 = x_val_matrix.reshape(x_val_matrix.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=1, activation='relu'))
    model.add(Conv2D(16, kernel_size=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train1, y_train_matrix, epochs=40)
    return model
