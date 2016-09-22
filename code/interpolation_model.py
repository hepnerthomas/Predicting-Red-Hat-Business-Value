# Thomas Hepner
# 9/3/2016
# Predicting Red Hat Business Value

# Import libraries
import os
import random
import numpy as np
import pandas as pd
import datetime
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

def mean_outcome_by_group_date(X_train, X_test):
    """ Predict test outcomes using (group_1, date_x) mean outcome where they exist. 
        This takes advantage of a leak in the competition data. It has 100% classification 
        accuracy for test data which has corresponding group_1 and date_x values in the train data.
    """
    # Add outcome to X_train dataframe
    X_train['outcome'] = y_train  # Yes, I know this is redundant code
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

    # Create 'filled' outcome column in df using average of forward and backwark fill by group
    df['filled'] = np.nan
    df = df.groupby('group_1').apply(interpolate_outcomes_weighted_means)    
    
    # Merge with X_train and X_test
    X_train = X_train.merge(df, on = columns, how = 'left')
    X_test = X_test.merge(df, on = columns, how = 'left')         
    
    # Fill mean_outcomes_date column with (group_1, date_x) predictions - DO NOT FILL NULL values
    X_test['mean_outcomes_date'] = X_test.ix[X_test['people_id'].isin(people['people_id'][testset]), 'filled']
    null_test_indices = X_test.ix[X_test['mean_outcomes_date'].isnull(), :].index       
    X_test.ix[X_test['mean_outcomes_date'].isnull(), :] = 0.5
       
    # drop date and activity_id variables
    X_train = X_train.drop(['date_x', 'date_y', 'activity_id', 'outcome', 'mean_outcomes_date', 'filled'], axis = 1) 
    X_test = X_test.drop(['date_x', 'date_y', 'activity_id', 'outcome', 'filled'], axis = 1) 
        
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

def write_predictions(test, predictions, model_name):
    """ Generates CSV file from predictions for test data. 
    """
    os.chdir('D:/OneDrive/Documents/Kaggle/Predicting Red Hat Business Value/submissions')  
    df = pd.DataFrame()
    df['activity_id'] = test['activity_id']
    df['outcome'] = predictions
    df.to_csv(model_name + str('.csv'), index = False, header = True)
    os.chdir('D:/OneDrive/Documents/Kaggle/Predicting Red Hat Business Value/submissions')      

### Execute code! ###
if __name__ == '__main__':
    """ Executes code generating predictions for test data.
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
    
    print "6. Writing predictions to CSV..."
    model_name = "interpolation_model_lb_"
    write_predictions(test, X_test['mean_outcomes_date'], model_name)

