import pandas as pd
import models
import preprocessingfile as preprocess
from sklearn.metrics import *

data = 'C:/DataBase/MKPL/project/Software-Defect-Prediction/Data/pc2.csv'
original_data, original_X, original_Y, combined_training_data, x_train1, x_train2, x_train, x_test, x_val, y_train1, y_train2, y_train, y_test, y_val = preprocess.my_sdp_preprocessor(data)
all_data = [original_data, original_X, original_Y, combined_training_data, x_train1, x_train2, x_train, x_test, x_val, y_train1, y_train2, y_train, y_test, y_val]

cnn_clf = models.cnn(*all_data) 
svm_clf = models.svm(*all_data)
rf_clf = models.random_forest(*all_data)
nn_clf = models.NN(*all_data)

def print_accuracy(model):
    if model == nn_clf:
        y_pred_on_val = model.predict(x_val) > 0.5
        y_pred_on_test = model.predict(x_test) > 0.5
    elif model == cnn_clf:
        x_val_matrix = x_val.values.reshape(x_val.shape[0], 1, len(x_val.columns), 1)
        y_pred_on_val = model.predict(x_val_matrix) > 0.5
        x_test_matrix = x_test.values.reshape(x_test.shape[0], 1, len(x_test.columns), 1)
        y_pred_on_test = model.predict(x_test_matrix) > 0.5
    else:
        y_pred_on_val = model.predict(x_val)
        y_pred_on_test = model.predict(x_test)

    print('******', str(model), '******')   
    print('||Validation Set||')
    print('Accuracy:', balanced_accuracy_score(y_val, y_pred_on_val))
    print('Avg Precision:', average_precision_score(y_val, y_pred_on_val))
    print('f1_score:', f1_score(y_val, y_pred_on_val))
    print('Precision:', precision_score(y_val, y_pred_on_val))
    print('Recall:', recall_score(y_val, y_pred_on_val))
    print('ROC_AUC:', roc_auc_score(y_val, y_pred_on_val))

    print('||Test Set||')
    print('Accuracy:', balanced_accuracy_score(y_test, y_pred_on_test))
    print('Avg Precision:', average_precision_score(y_test, y_pred_on_test))
    print('f1_score:', f1_score(y_test, y_pred_on_test))
    print('Precision:', precision_score(y_test, y_pred_on_test))
    print('Recall:', recall_score(y_test, y_pred_on_test))
    print('ROC_AUC:', roc_auc_score(y_test, y_pred_on_test))

    val_result = pd.concat([y_val['defects'].reset_index(drop=True), pd.DataFrame(y_pred_on_val, columns=['defects1'])], axis=1)
    test_result = pd.concat([y_test['defects'].reset_index(drop=True), pd.DataFrame(y_pred_on_test, columns=['defects1'])], axis=1)
    
    val_result = val_result.rename(columns={'defects': 'val_actual', 'defects1': 'val_predict'})
    test_result = test_result.rename(columns={'defects': 'test_actual', 'defects1': 'test_predict'})
    
    return val_result, test_result

svm_val_result, svm_test_result = print_accuracy(svm_clf)
rf_val_result, rf_test_result = print_accuracy(rf_clf)
nn_val_result, nn_test_result = print_accuracy(nn_clf)
cnn_val_result, cnn_test_result = print_accuracy(cnn_clf)

new_val_set_x = pd.concat([svm_val_result['val_predict'], rf_val_result['val_predict'], nn_val_result['val_predict'], cnn_val_result['val_predict']], axis=1)
new_test_set_x = pd.concat([svm_test_result['test_predict'], rf_test_result['test_predict'], nn_test_result['test_predict'], cnn_test_result['test_predict']], axis=1)
