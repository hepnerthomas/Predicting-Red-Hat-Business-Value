Thomas Hepner, September 2016:

Project code for the Red Hat Predicting Business Value Kaggle competiton that I developed to place 186th out of 2,271 
competitors. The final submission was based on a straight average of 4 models: 2 models I generated, and 2 popular
public scripts in the forums.

The project code is broken up into 2 files: 

1. interpolation_model.py: 
	- Loads train, test, and people competition data sets.
	- Converts categorical features to numeric in all data sets.
	- Merges people data with train and test data respectively.
	- Generates predictions for test data based on mean outcome by (group_1, date) features.
	- Writes predictions to CSV for submission.

2. create_predictions.py:
	- Loads train, test, and people competition data sets.
	- Converts categorical features to numeric in all data sets.
	- Merges people data with train and test data respectively.
	- Generates predictions for test data based on mean outcome by (group_1, date) features.
	- Creates new features based on mean outcome by categorical feature ids. 
	- Modifies the group_1 variable in people data by grouping all unique groups into a single group. 
	  Merges new variable into train and test data.
	- Trains XGBOOST on train data. 
	- Generates predictions for cases in test data that did not have a matching (group_1, date) 
	  with XGBOOST model. 
	- Writes predictions to CSV for submission.
