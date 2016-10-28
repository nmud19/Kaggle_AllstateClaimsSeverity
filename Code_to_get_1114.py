
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import mean_absolute_error

train = pd.read_csv('D:\\Kaggle\\AllState\\Data\\train.csv')
test = pd.read_csv('D:\\Kaggle\\AllState\\Data\\test.csv')

test['loss'] = np.nan
joined = pd.concat([train, test])


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))

if __name__ == '__main__':
    for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)
            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x

            joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
            
        joined[column] = pd.factorize(joined[column].values, sort=True)[0]

    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]
    
    shift = 200
    y = np.log(train['loss'] + shift)
    ids = test['id']
    
    #Build model off top 20 important features
    #imp_feat_names = ['cont14', 'cont7', 'cont6', 'cat100', 'cont1', 'cont8', 'cat112', 'cont4', 'cont13', 'cont11', 'cont2', 'cont5', 'cont3', 'cont9', 'cont10', 'cont12', 'cat113', 'cat110', 'cat116', 'cat101']
    train['cont14'] =  np.sqrt(train['cont14'])/np.log(train['cont14']) 
    test['cont14'] =  np.sqrt(test['cont14'])/np.log(test['cont14']) 
    
    X = train.drop(['loss', 'id'], 1)
    X_test = test.drop(['loss', 'id'], 1)
    
    RANDOM_STATE = 2016
    params = {
        'min_child_weight': 1,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 12,
        'subsample': 0.8,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': False,
        'seed': RANDOM_STATE
    }

    xgtrain = xgb.DMatrix(X, label=y)
    xgtest = xgb.DMatrix(X_test)

    model = xgb.train(params, xgtrain, int(2012 / 0.9), feval=evalerror)

    prediction = np.exp(model.predict(xgtest)) - shift

    submission = pd.DataFrame()
    submission['id'] = ids
    submission['loss'] = prediction
    submission.to_csv('D:\\Kaggle\\AllState\\Output\\sub_1114LB_ver2.csv', index=False)



