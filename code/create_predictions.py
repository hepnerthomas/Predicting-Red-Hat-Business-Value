# Thomas Hepner
# 9/3/2016
# Predicting Red Hat Business Value

# Import libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.metrics import roc_auc_score 
import xgboost as xgb
from sklearn.cross_validation import LabelKFold
from itertools import product

# Set random seed
random.seed(0)

def check_memory_usage(data):
    """ Takes DataFrame as input and returns memory usage statistics. 
    """
    print ''
    print 'Checking memory usage statistics... '
    print(data.info(memory_usage=True))      

def load_data():
    """ Load train, test, and people data. 
    """   
    # Set work directory
    os.chdir('D:/OneDrive/Documents/Kaggle/Predicting Red Hat Business Value/data')        
    
    # Load data
    train = pd.read_csv("act_train.csv", sep = ",") 
    test = pd.read_csv("act_test.csv", sep = ",") 
    people = pd.read_csv("people.csv", sep = ",") 
    
    # Return data
    return train, test, people

def split_data(train, test): 
    """ Split train into inputs and outputs. 
    """
    y_train = train['outcome']
    X_train = train.drop('outcome', axis = 1)    
    X_test = test
    return X_train, X_test, y_train

def clean_data(X_train, X_test):
    """ Remove unnecessary strings from features and convert to integer values. 
    """
    # convert categorical/string variables to numerical
    counter = 0
    for data in (X_train, X_test):
        columns = list(data.columns)
        columns.remove('date')
        columns.remove('people_id')
        columns.remove('activity_id')
        
        # remove non-integer characters from people_id, activity_id
        id_columns = ['people_id']
        for column in id_columns: 
            # remove characters before '_'
            data[column] = [float(x.split('_', 1)[-1]) for x in data[column]]
            
            # convert to integer
            data[column] = data[column].astype(np.int)   
                        
        # remove non-integer characters from other columns except date
        for column in columns: 
            # convert nan to 'Type 0'
            data.ix[data[column].isnull(), column] = 'type 0'
            
            # remove string characters
            data[column] = [x.split('type ', 1)[-1] for x in data[column]]

            # convert to integer
            data[column] = data[column].astype(np.uint16)  
        
        # override data
        if counter == 0:
            X_train = data
        else: 
            X_test = data
        counter += 1
            
    return X_train, X_test

def clean_people(people): 
    """ I know this is an odd function name. Converts string variables in people data set
        to integer data types.
    """
    columns = list(people.columns)
    columns.remove('people_id')
    columns.remove('date')
    
    # remove non-integer characters from people_id, activity_id
    id_columns = ['people_id']
    for column in id_columns: 
        # remove characters before '_'
        people[column] = [float(x.split('_', 1)[-1]) for x in people[column]]
        
        # convert to integer
        people[column] = people[column].astype(np.int)   
    
    # convert strings to numeric/bool
    for column in columns:
        # convert boolean types to integer
        if(people[column].dtype == 'bool'):
            people[column] = people[column].astype(int)
        
        # remove non-integer characters from string type variables and convert to integer
        if(people[column].dtype == 'object'):
            people[column] = [float(x.split(' ', 1)[-1]) for x in people[column]]
            people[column] = people[column].astype(np.uint32)

    return people

def merge_data(X_train, X_test, people):
    """ Merges features from people DataFrame into train and test data sets. 
    """
    
    X_train = X_train.merge(people, how = 'left', on = 'people_id')
    X_test = X_test.merge(people, how = 'left', on = 'people_id')
    
    return X_train, X_test
    
def merge_people_modified(X_train, X_test, people):
    """ All group_1 values that appear in the people data set are grouped into the same 
        value. The feature is then merged with the train and test data.
    """
     # Reduce cardinality of group_1 feature
    single_groups = people['group_1'].value_counts()
    single_index = single_groups[single_groups == 1].index
    people.ix[people['group_1'].isin(single_index), 'group_1'] = 99999  
    
    # Drop old group_1 variable
    X_train = X_train.drop('group_1', axis = 1)
    X_test = X_test.drop('group_1', axis = 1)
    
    # Merge refined 'group_1' into data sets
    people = people.ix[: , ['people_id', 'group_1']]
    X_train = X_train.merge(people, how = 'left', on = 'people_id')
    X_test = X_test.merge(people, how = 'left', on = 'people_id')    
    
    return X_train, X_test

def mean_outcomes_by_categorical_variables(X_train, X_test):
    """ Takes X_train and X_test as inputs. Creates new features which are mean outcomes by categorical feature ids.
    """
    # Add outcome to X_train dataframe
    X_train['outcome'] = y_train

    categorical_columns = ['activity_category', 'char_1_x', 'char_2_x', 'char_3_x'
                           , 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x'
                           , 'char_1_y', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y'
                           , 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y'
                           ]
    
    # Means
    for column in categorical_columns: 
        mean_column = pd.DataFrame({column + '_mean_outcome' : X_train.groupby(column)['outcome'].mean()}).reset_index()          
        X_train = X_train.merge(mean_column, on = column, how = 'left')    
        X_test = X_test.merge(mean_column, on = column, how = 'left')  
        
    X_train = X_train.drop('outcome', axis = 1)

    return X_train, X_test

def mean_outcome_by_group_date(X_train, X_test):
    """ Predict test outcomes using (group_1, date_x) mean outcome where they exist. 
        This takes advantage of a leak in the competition data. It has 100% classification 
        accuracy for test data which has corresponding group_1 and date_x values in the train data.
    """
    
    # Add outcome to X_train dataframe
    X_train['outcome'] = y_train  # Yes, this is redundant code
    X_test['outcome'] = np.nan
    
    # Convert date variables to datetime data type
    X_train['date_x'] = pd.to_datetime(X_train['date_x'])   
    X_test['date_x'] = pd.to_datetime(X_test['date_x']) 
         
    # date_x and group_1; 100% classification accuracy on train, only classifies 59% of test data
    columns = ['group_1', 'date_x']
    mean_outcomes_date = pd.DataFrame({'mean_outcomes_date' : X_train.groupby(columns)['outcome'].mean(), 'n_outcomes_date' : X_train.groupby(columns)['outcome'].count()}).reset_index()               
    mean_outcomes_date = mean_outcomes_date.sort_values(by = columns)
    mean_outcomes_date['mean_outcomes_date'] = mean_outcomes_date['mean_outcomes_date'].astype(float)

    # create dataframe for all (group_1, date_x) combinations
    columns = ['people_id', 'group_1', 'date_x', 'outcome']
    group_date_grid_df = pd.concat([X_train.ix[:, columns], X_test.ix[:, columns]])

    mindatetime = group_date_grid_df['date_x'].min()
    maxdatetime = group_date_grid_df['date_x'].max()
    days = [maxdatetime - datetime.timedelta(days=x) for x in range(0, (maxdatetime - mindatetime).days+1)][::-1]
    
    testset = people[people['people_id'].isin(group_date_grid_df[group_date_grid_df['outcome'].isnull()]['people_id'])].index
    grid = set(group_date_grid_df[~group_date_grid_df['people_id'].isin(people.iloc[testset]['people_id'])]['group_1'])
    df = pd.DataFrame.from_records(product(grid, days))
    df.columns = ['group_1', 'date_x']
    df.sort_values(['group_1', 'date_x'], inplace = True)

    # Merge all df of all (group_1, date_x) combinations with mean_outcomes_date
    columns = ['group_1', 'date_x']
    df = df.merge(mean_outcomes_date, on = columns, how = 'left')    

    # Create 'filled' outcome column in df using weighted average of forward and backwark fill by group
    df['filled'] = np.nan
    df = df.groupby('group_1').apply(interpolate_outcomes_weighted_means)
    
    # Merge with X_train and X_test
    X_train = X_train.merge(df, on = columns, how = 'left')
    X_test = X_test.merge(df, on = columns, how = 'left')         
    
    # Fill mean_outcomes_date column with (group_1, date_x) predictions - DO NOT FILL NULL values
    X_test['mean_outcomes_date'] = X_test.ix[X_test['people_id'].isin(people['people_id'][testset]), 'filled']
    null_test_indices = X_test.ix[X_test['mean_outcomes_date'].isnull(), :].index       
    X_test.ix[null_test_indices, :] = 0.5
    
    # drop date and activity_id variables
    X_train = X_train.drop(['date_x', 'date_y', 'activity_id', 'outcome', 'mean_outcomes_date', 'n_outcomes_date', 'filled'], axis = 1) 
    X_test = X_test.drop(['date_x', 'date_y', 'activity_id', 'outcome', 'n_outcomes_date', 'filled'], axis = 1) 
        
    return X_train, X_test, null_test_indices

def interpolate_outcomes_weighted_means(x):
    """ Interpolate values between dates with missing outcome data for a particular group.
    """
    x = x.reset_index(drop=True)
    g = x['mean_outcomes_date'].copy()
    if g.shape[0] < 3: 
        x['filled'] = g ## Will be replaced later.
        return x
            
    # Find all non-NA indexes, combine them with outside borders at beginning and end of array
    borders = np.append([0], g[g.isnull() == False].index, axis=0)
    borders = np.append(borders, [len(g)], axis=0)
    
    # establish forward and backward - looking indexes
    forward_border = borders[1:len(borders)]
    backward_border = borders[0:(len(borders) - 1)]
  
    # prepare vectors for filling
    forward_border_g = list(g[forward_border])
    backward_border_g = list(g[backward_border])
    
    forward_border_n = np.asarray(x.ix[forward_border, 'n_outcomes_date'])
    backward_border_n = np.asarray(x.ix[backward_border, 'n_outcomes_date']) 
    
    forward_border_n[np.isnan(forward_border_n)] = 1
    backward_border_n[np.isnan(backward_border_n)] = 1
     
    forward_border_g[len(forward_border_g)-1] = abs(forward_border_g[len(forward_border_g) - 2] - 0.2)
    backward_border_g[0] = abs(forward_border_g[0] - 0.1) 
    
    # generate fill vectors    
    times = forward_border - backward_border
    forward_g_fill = np.repeat([forward_border_g], times)
    backward_g_fill = np.repeat([backward_border_g], times)
    
    forward_n_fill = np.repeat([forward_border_n], times)
    backward_n_fill = np.repeat([backward_border_n], times)
      
    # linear interpolation
    vec = (forward_n_fill * forward_g_fill + backward_n_fill * backward_g_fill) / (forward_n_fill + backward_n_fill)
    vec = pd.Series(vec)
    
    g[g.isnull()] = vec.ix[g.isnull()]
    x['filled'] = g
    
    # Return filled outcomes
    return x  

def create_folds(X_train, people, n_folds):
    """ Creates 5-folds from train data for cross validation purposes.
    """

    # Select people ids for each fold
    labels = people['people_id'].unique()
    label_kfold = LabelKFold(labels, n_folds = n_folds)
    cv_folds = []
    for train_index, cv_index in label_kfold: 
        cv_folds.append([people.ix[train_index, 'people_id'], people.ix[cv_index, 'people_id']])

    # Identify idencies for people_ids in folds
    train_indices = []
    cv_indices = []
    for fold in cv_folds: 
        fold[0] = fold[0].astype(np.int64)
        fold[1] = fold[1].astype(np.int64)
        train_fold = X_train[X_train['people_id'].isin(fold[0])].index
        cv_fold = X_train[X_train['people_id'].isin(fold[1])].index
        train_indices.append(train_fold)
        cv_indices.append(cv_fold)

    return cv_folds, train_indices, cv_indices

def create_sparse_data(X_train, X_test):
    """ Convert train and test data into sparse Dataframes converting categorical features into OneHotEncoded
        dummy variables.
    """

    # Convert DataFrame to Sparse Matrix
    from sklearn.preprocessing import OneHotEncoder
    from scipy.sparse import csr_matrix
    categorical_columns = ['activity_category', 'group_1', 'char_1_x', 'char_2_x', 'char_3_x'
                           , 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x'
                           , 'char_1_y', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y'
                           , 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y'
                           ]
    all_columns = list(X_train.columns)
    indices = []
    for column in categorical_columns: 
        indices.append(all_columns.index(column))
    enc = OneHotEncoder(categorical_features = indices)

    combined = pd.concat([X_train, X_test]).reset_index(drop = True)
    enc.fit(combined)
    combined = csr_matrix(enc.transform(combined))
    
    train_rows = np.asarray(range(0, X_train.shape[0]), dtype = np.int64)
    test_rows = np.asarray(range(X_train.shape[0], X_test.shape[0] + X_train.shape[0]), dtype = np.int64)
    
    X_train_sparse = combined[train_rows, :]
    X_test_sparse = combined[test_rows, :] 
    
    return X_train_sparse, X_test_sparse
        
def feature_importance_scores_xgb(model_xgb): 
    """ Plot and save xgboost feature importance scores.
    """
    fscore = model_xgb.booster().get_fscore()
    feature_importances = []
    for ft, score in fscore.iteritems(): 
        feature_importances.append({'Feature': ft, 'Importance': score})
    feature_importances = pd.DataFrame(feature_importances)
    feature_importances = feature_importances.sort_values(by = 'Importance', ascending = False).reset_index(drop=True)
    feature_importances['Importance'] = 100.0 * feature_importances['Importance'] / feature_importances['Importance'].sum()
    
    feature_names = feature_importances['Feature']
    importances = feature_importances['Importance']
    y_pos = np.asarray(range(len(feature_names), 0, -1))
    
    plt.barh(y_pos[0:20], importances[0:20], align = 'center', alpha = 1.0)
    plt.yticks(y_pos[0:20], feature_names[0:20]) 
    plt.xlabel('Scores')    
    plt.title('Feature Importance Scores')  
    plt.show()
    
    return feature_importances  

def plot_learning_curves(model_xgb):
    """ Plot train and validation auc over iterations.
    """
    # retrieve model performance
    results = model_xgb.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plot auc
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    plt.ylabel('Area Under the ROC Curve')
    plt.title('XGBoost AUC Performance')
    plt.show()
    
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()
        
def print_summary_statistics(X_train, y_train, X_validation, y_validation, model, columns):
    """ Takes predictive model, data, and selected columns as inputs. 
        Plots model learning curve. 
        Prints train and validation CV scores.
    """
    plot_learning_curves(model)
    model_train = generate_predictions(X_train.ix[:, columns], model)
    model_validation = generate_predictions(X_validation.ix[:, columns], model)
    print 'Train AUC Score: ' + str(round(roc_auc_score(y_train, model_train), 4))   
    print 'Validation AUC Score: ' + str(round(roc_auc_score(y_validation, model_validation), 4))            
        
def generate_predictions(X_test, model):
    """ Takes test data and model as inputs. Outputs predictions for test data.
    """
    predictions = [x[1] for x in model.predict_proba(X_test)]
    return predictions
    
def write_predictions(test, predictions, model_name):
    """ Generates CSV file from test data predictions. 
    """
    os.chdir('D:/OneDrive/Documents/Kaggle/Predicting Red Hat Business Value/submissions')  
    df = pd.DataFrame()
    df['activity_id'] = test['activity_id']
    df['outcome'] = predictions
    df.to_csv(model_name + str('.csv'), index = False, header = True)
    os.chdir('D:/OneDrive/Documents/Kaggle/Predicting Red Hat Business Value/submissions')   
    
def build_cv_model_xgb(model, X_train, y_train, X_test, y_test, train_indices, cv_indices):
    """ Fits an XGBOOST model for each fold in the train_indices and cv_indices variables.
        Selects best model based on cross validation AUC scores. Prints model learning curves 
        and feature importances. Returns the best model and its cross validation AUC score on the train data.
    """
    # Store cv predictions (will use for model stacking)
    predictions = np.asarray(np.repeat([0.0], X_train.shape[0]), dtype=np.float64)
    models = []
    cv_scores = []
    
    # Train model with cv folds
    for i in range(len(train_indices)):
        
        # Select indices
        print 'CV Fold ' + str(i) + ':'
        train_index = train_indices[i]
        cv_index = cv_indices[i]
        
        # Fit model
        model.fit(X_train.ix[train_index], y_train[train_index]
                , eval_set = [(X_train.ix[train_index], y_train[train_index]), (X_train.ix[cv_index], y_train[cv_index])]
                , eval_metric = ["error", "auc"]
                , early_stopping_rounds = 10)    
        
        # Generate predictions for cv indices in train data
        predictions[cv_index] = generate_predictions(X_train.ix[cv_index], model)
        
        # Save model and scores
        models.append(model)
        cv_scores.append(model_xgb.evals_result()['validation_0']['auc'][-1])
        print ''        
        
    # Find best CV model
    max_score = max(cv_scores)
    max_index = cv_scores.index(max_score)
    best_model = models[max_index]  
        
    # Print learning curves
    columns = X_train.columns
    print_summary_statistics(X_train.ix[train_indices[max_index]], y_train[train_indices[max_index]]
                                , X_train.ix[cv_indices[max_index]], y_train[cv_indices[max_index]], best_model, columns)  
                                
    # Print feature importances
    feature_importances = feature_importance_scores_xgb(best_model)
    print ''
    print 'Feature Importance Scores: '
    print feature_importances[0:10]  
    
    # Print cv score for entire train set
    train_cv_score = roc_auc_score(y_train, predictions)
    print 'Train CV Score: ' + str(round(train_cv_score, 4))

    return best_model, train_cv_score    

### Execute code! ###
if __name__ == '__main__':
    """ Executes code and generates predictions for test data.
    """
    
    print "1. Loading data..."
    train, test, people = load_data()        
        
    print "2. Format data for prediction purposes ... " 
    X_train, X_test, y_train = split_data(train, test)
    
    print "3. Clean features in data ... "
    X_train, X_test = clean_data(X_train, X_test)    
    people = clean_people(people)   
        
    print "4. Merge data ... " 
    X_train, X_test = merge_data(X_train, X_test, people)
    
    print "5. Mean outcome by (group_1, date_x)..."
    X_train, X_test, null_test_indices = mean_outcome_by_group_date(X_train, X_test)        

    print "6. Mean outcome by categorical variables..."
    X_train, X_test = mean_outcomes_by_categorical_variables(X_train, X_test)    
    
    print "7. Modify group_1 variable in people DataFrame and merge into train/test data..."
    X_train, X_test = merge_people_modified(X_train, X_test, people)
    y_test = X_test['mean_outcomes_date']
    X_test = X_test.drop('mean_outcomes_date', axis = 1)
    
    print "8. Create CV folds..."
    cv_folds, train_indices, cv_indices = create_folds(X_train, people, n_folds = 5) 
    X_train = X_train.drop('people_id', axis = 1)
    X_test = X_test.drop('people_id', axis = 1)
    
    print "9. Building models and generating predictions..."   
    # XGBOOST Model
    params_xgb = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 8, 'nthread': -1, 'seed': 0}    
    model_xgb = xgb.XGBClassifier().set_params(**params_xgb)         
    
    # Find best XGBOOST model with cross-validation
    columns = X_train.columns
    best_model, cv_score = build_cv_model_xgb(model_xgb, X_train.ix[:, columns], y_train, X_test.ix[:, columns]
                                              , y_test, train_indices, cv_indices
                                             )  
                                              
    # Use best model from CV to generate predictions
    best_model.fit(X_train, y_train)
    y_test[null_test_indices] = generate_predictions(X_test.ix[null_test_indices, :], best_model)    
    write_predictions(test, y_test, 'submission_xgb_depth8_traincv_lb_')        
    
    print "10. Complete!"    
                                              