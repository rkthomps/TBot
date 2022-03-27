# This module is apparently for evaluation although I forgot where
# I even use this.

# I don't believe is is currently important


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

class Evaluator:
    def __init__(self, prediction='classification'):
        self.prediction = prediction

    ## Returns the correct metrics depending on if the prediction is
    ## regression or classification
    def get_metrics(self, predicted, actual):
        if self.prediction == 'classification':
            return self.get_class_metrics(predicted, actual)
        return self.get_reg_metrics(predicted, actual)
        
    ## Returns a number of metrics given predicted and actual values
    def get_reg_metrics(self, predicted, actual):
        # MSE
        np_pred = np.array(predicted)
        np_act = np.array(actual)
        mse = ((np_pred - np_act) ** 2).mean()
        acc = (np.maximum(0, 1 - np.absolute(np_pred - np_act) / np_act)).mean()
        hl_acc = (((np_pred > 1) & (np_act > 1)) | ((np_pred < 1) & (np_act < 1))).mean()
        return {'Mean Squared': mse, 'Accuracy': acc, 'High Low Accuracy': hl_acc}
    
    ## Returns metrics given actual and predicted values
    def get_class_metrics(self, predicted, actual):
        met_dic = {}
        pred = np.full(predicted.shape, 0)
        pred[np.arange(predicted.shape[0]), np.argmax(predicted, axis=1)] = 1
        success = predicted * actual
        met_dic['Accuracy'] = (success).sum(axis=1).mean()
        tp = success.sum(axis=0)
        fp = (predicted * (1 - success)).sum(axis=0)
        fn = ((1 - predicted) *  success).sum(axis=0)
        prec = tp/(tp + fp)
        rec = tp/(tp + fn)
        for i in range(predicted.shape[1]):
            met_dic['Precision_' + str(i)] = prec[i]
            met_dic['Recall_' + str(i)] = rec[i]

        return met_dic


    ## Runs k-fold cross validation
    ## Assumes all variables other than the response are
    ## predictors
    def cross_validate(self, k, in_df, response_var, model_instantiate=None, model_pipeline=None):
        if model_instantiate == None and model_pipeline == None:
            raise ValueError('Need some way to determine the model')
        shuffled = in_df.sample(frac=1)
        test_ind = [int(i) for i in np.floor(np.linspace(0, len(shuffled), k + 1))]
        metrics = {}
        for i in range(k):
            train = pd.concat((in_df.iloc[0: test_ind[i]], in_df.iloc[test_ind[i + 1]:]))
            test = in_df.iloc[test_ind[i]:test_ind[i + 1]]
            train_y = train[response_var]
            train_x = train.drop(columns=response_var, axis=1).copy()
            test_y = test[response_var]
            test_x = test.drop(columns=response_var, axis=1).copy()
            
            if model_instantiate:
                model = eval(model_instantiate)
            else:
                model = model_pipeline

            model.fit(train_x, train_y)
            tr_pred = model.predict(train_x)
            te_pred = model.predict(test_x)
            
            metric_dic = self.get_metrics(tr_pred, train_y.values)
            for key, value in metric_dic.items():
                if key not in metrics:
                    metrics[key] = [0, 0]
                metrics[key][0] += value
                
            metric_dic = self.get_metrics(te_pred, test_y.values)
            for key, value in metric_dic.items():
                metrics[key][1] += value
        df = pd.DataFrame(metrics, index=pd.Series(['Train', 'Test'], name='Phase'))
        df = df.transform(lambda x: x / k)
        return df     
