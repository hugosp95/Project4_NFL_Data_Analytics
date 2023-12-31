#!/usr/bin/env python
# coding: utf-8


# import all necessary dependencies
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
from sqlalchemy.types import String, Float, Integer
import sklearn as skl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import numpy as np


# read in the four raw data files
season_2015_df = pd.read_excel('wr_season_2015.xlsx')
season_2016_df = pd.read_excel('wr_season_2016.xlsx')
season_2017_df = pd.read_excel('wr_season_2017.xlsx')
season_2018_df = pd.read_excel('wr_season_2018.xlsx')
season_2019_df = pd.read_excel('wr_season_2019.xlsx')
season_2020_df = pd.read_excel('wr_season_2020.xlsx')
season_2021_df = pd.read_excel('wr_season_2021.xlsx')


## Data Cleaning

# preview new dataframe
season_2015_df.head()


# drop rank column from all dataframes since we don't need this value for the model
season_2015_df = season_2015_df.drop(columns=['Rk'])
season_2016_df = season_2016_df.drop(columns=['Rk'])
season_2017_df = season_2017_df.drop(columns=['Rank'])
season_2018_df = season_2018_df.drop(columns=['Rank'])
season_2019_df = season_2019_df.drop(columns=['Rank'])
season_2020_df = season_2020_df.drop(columns=['Rank'])
season_2021_df = season_2021_df.drop(columns=['Rank'])


# rename columns to be clear terminology
season_2015_df = season_2015_df.rename(columns={'Pos': 'Position', 'GMS': 'Games', 'REC': 'Receptions', 'TGTS': 'Targets', 'PCT': 'Percentage',
                               'YDS': 'Yards', 'TD': 'Touchdowns', 'LNG': 'Long', 'Y/T': 'Yards_per_target', 'Y/R': 'Yards_per_reception', 'ATT': 'Attempts',
                               'TD.1': 'Rushing_touchdown', 'YDS.1': 'Rushing_yards', 'AVG': 'Average_rushing_yards', 'FUM': 'Fumbles', 'LST': 'Lost_yards',
                               'FPTS/G': 'Fantasy_points_per_game', 'FPTS': 'Fantasy_points'})



# rename columns to be clear terminology
season_2016_df = season_2016_df.rename(columns={'Pos': 'Position', 'GMS': 'Games', 'REC': 'Receptions', 'TGTS': 'Targets', 'PCT': 'Percentage',
                               'YDS': 'Yards', 'TD': 'Touchdowns', 'LNG': 'Long', 'Y/T': 'Yards_per_target', 'Y/R': 'Yards_per_reception', 'ATT': 'Attempts',
                               'TD.1': 'Rushing_touchdown', 'YDS.1': 'Rushing_yards', 'AVG': 'Average_rushing_yards', 'FUM': 'Fumbles', 'LST': 'Lost_yards',
                               'FPTS/G': 'Fantasy_points_per_game', 'FPTS': 'Fantasy_points'})



# rename columns to be clear terminology
season_2017_df = season_2017_df.rename(columns={'Pos': 'Position', 'GMS': 'Games', 'REC': 'Receptions', 'TGTS': 'Targets', 'PCT': 'Percentage',
                               'YDS': 'Yards', 'TD': 'Touchdowns', 'LNG': 'Long', 'Y/T': 'Yards_per_target', 'Y/R': 'Yards_per_reception', 'ATT': 'Attempts',
                               'TD.1': 'Rushing_touchdown', 'YDS.1': 'Rushing_yards', 'AVG': 'Average_rushing_yards', 'FUM': 'Fumbles', 'LST': 'Lost_yards',
                               'FPTS/G': 'Fantasy_points_per_game', 'FPTS': 'Fantasy_points'})



# rename columns to be clear terminology
season_2018_df = season_2018_df.rename(columns={'Pos': 'Position', 'GMS': 'Games', 'REC': 'Receptions', 'TGTS': 'Targets', 'PCT': 'Percentage',
                               'YDS': 'Yards', 'TD': 'Touchdowns', 'LNG': 'Long', 'Y/T': 'Yards_per_target', 'Y/R': 'Yards_per_reception', 'ATT': 'Attempts',
                               'TD.1': 'Rushing_touchdown', 'YDS.1': 'Rushing_yards', 'AVG': 'Average_rushing_yards', 'FUM': 'Fumbles', 'LST': 'Lost_yards',
                               'FPTS/G': 'Fantasy_points_per_game', 'FPTS': 'Fantasy_points'})



# rename columns to be clear terminology
season_2019_df = season_2019_df.rename(columns={'Pos': 'Position', 'GMS': 'Games', 'REC': 'Receptions', 'TGTS': 'Targets', 'PCT': 'Percentage',
                               'YDS': 'Yards', 'TD': 'Touchdowns', 'LNG': 'Long', 'Y/T': 'Yards_per_target', 'Y/R': 'Yards_per_reception', 'ATT': 'Attempts',
                               'TD.1': 'Rushing_touchdown', 'YDS.1': 'Rushing_yards', 'AVG': 'Average_rushing_yards', 'FUM': 'Fumbles', 'LST': 'Lost_yards',
                               'FPTS/G': 'Fantasy_points_per_game', 'FPTS': 'Fantasy_points'})



# rename columns to be clear terminology
season_2020_df = season_2020_df.rename(columns={'Pos': 'Position', 'GMS': 'Games', 'TGTS': 'Targets', 'REC': 'Receptions', 'PCT': 'Percentage',
                               'YDS': 'Yards', 'TD': 'Touchdowns', 'LNG': 'Long', 'Y/T': 'Yards_per_target', 'Y/R': 'Yards_per_reception', 'ATT': 'Attempts',
                               'TD.1': 'Rushing_touchdown', 'YDS.1': 'Rushing_yards', 'AVG': 'Average_rushing_yards', 'FUM': 'Fumbles', 'LST': 'Lost_yards',
                               'FPTS/G': 'Fantasy_points_per_game', 'FPTS': 'Fantasy_points'})



# rename columns to be clear terminology
season_2021_df = season_2021_df.rename(columns={'Pos': 'Position', 'GMS': 'Games', 'TGTS': 'Targets', 'REC': 'Receptions', 'PCT': 'Percentage',
                               'YDS': 'Yards', 'TD': 'Touchdowns', 'LNG': 'Long', 'Y/T': 'Yards_per_target', 'Y/R': 'Yards_per_reception', 'ATT': 'Attempts',
                               'TD.1': 'Rushing_touchdown', 'YDS.1': 'Rushing_yards', 'AVG': 'Average_rushing_yards', 'FUM': 'Fumbles', 'LST': 'Lost_yards',
                               'FPTS/G': 'Fantasy_points_per_game', 'FPTS': 'Fantasy_points'})



# add column to each dataframe that contains the year of play
season_2015_df['Year'] = '2015'
season_2016_df['Year'] = '2016'
season_2017_df['Year'] = '2017'
season_2018_df['Year'] = '2018'
season_2019_df['Year'] = '2019'
season_2020_df['Year'] = '2020'
season_2021_df['Year'] = '2021'


# combine all dataframes for seasons that we want to evaluate in the model
# all_dfs = [season_2015_df, season_2016_df, season_2017_df, season_2018_df, season_2019_df, season_2020_df, season_2021_df]
# results = pd.concat(all_dfs)
all_dfs = [season_2018_df, season_2019_df, season_2020_df, season_2021_df]
results = pd.concat(all_dfs)


# write the clean dataframes to csv for future use
season_2015_df.to_csv('wr_season_2015_clean.csv', index=False)


# write the clean dataframes to csv for future use
season_2016_df.to_csv('wr_season_2016_clean.csv', index=False)


# write the clean dataframes to csv for future use
season_2017_df.to_csv('wr_season_2017_clean.csv', index=False)


# write the clean dataframes to csv for future use
season_2018_df.to_csv('wr_season_2018_clean.csv', index=False)


# write the clean dataframes to csv for future use
season_2019_df.to_csv('wr_season_2019_clean.csv', index=False)


# write the clean dataframes to csv for future use
season_2020_df.to_csv('wr_season_2020_clean.csv', index=False)


# write the clean dataframes to csv for future use
season_2021_df.to_csv('wr_season_2021_clean.csv', index=False)


# check datatypes for creating sql table
season_2019_df.dtypes


# create engine to connect to postgresql database
engine = create_engine('postgresql://postgres:postgres@localhost:5432/nfl_db')

# write season_2019_df to a postgres table
season_2019_df.to_sql('season_2019', engine, index= False, if_exists='replace', chunksize = 500,
                 dtype = {'Name': String,
                  'Team': String,
                  'Position': String,
                  'Games': Integer,
                  'Targets': Integer,
                  'Receptions': Integer,
                  'Percentage': Float,
                  'Yards': Integer,
                  'Touchdowns': Integer,
                  'Long': Integer,
                  'Yards_per_target': Float,
                  'Yards_per_recption': Float,
                  'Attempts': Integer,
                  'Rushing_yards': Integer,
                  'Average_rushing_yards': Float,
                  'Rushing_touchdown': Integer,
                  'Fumbles': Integer,
                  'Lost_yards': Integer,
                  'Fantasy_points_per_game': Float,
                  'Fantasy_points': Float,
                  'Year': String})


# write season_2020_df to a postgres table
season_2020_df.to_sql('season_2020', engine, index= False, if_exists='replace', chunksize = 500,
                 dtype = {'Name': String,
                  'Team': String,
                  'Position': String,
                  'Games': Integer,
                  'Targets': Integer,
                  'Receptions': Integer,
                  'Percentage': Float,
                  'Yards': Integer,
                  'Touchdowns': Integer,
                  'Long': Integer,
                  'Yards_per_target': Float,
                  'Yards_per_recption': Float,
                  'Attempts': Integer,
                  'Rushing_yards': Integer,
                  'Average_rushing_yards': Float,
                  'Rushing_touchdown': Integer,
                  'Fumbles': Integer,
                  'Lost_yards': Integer,
                  'Fantasy_points_per_game': Float,
                  'Fantasy_points': Float,
                  'Year': String})


# write season_2021_df to a postgres table
season_2021_df.to_sql('season_2021', engine, index= False, if_exists='replace', chunksize = 500,
                 dtype = {'Name': String,
                  'Team': String,
                  'Position': String,
                  'Games': Integer,
                  'Targets': Integer,
                  'Receptions': Integer,
                  'Percentage': Float,
                  'Yards': Integer,
                  'Touchdowns': Integer,
                  'Long': Integer,
                  'Yards_per_target': Float,
                  'Yards_per_recption': Float,
                  'Attempts': Integer,
                  'Rushing_yards': Integer,
                  'Average_rushing_yards': Float,
                  'Rushing_touchdown': Integer,
                  'Fumbles': Integer,
                  'Lost_yards': Integer,
                  'Fantasy_points_per_game': Float,
                  'Fantasy_points': Float,
                  'Year': String})


# In[33]:


# with engine.connect() as con:
#     con.execute(text("ALTER TABLE season_2019 ADD PRIMARY KEY (Name);"))
#     con.execute(text('Select * from season_2019 limit(10)')).fetchall()


# In[15]:


# insp = inspect(engine)
# print(insp.get_table_names())


# In[16]:


# columns = insp.get_columns('season_2019')
# for column in columns:
#     print(column['name'], column['type'])


# In[17]:


# query all records from the season_2019 table
# with engine.connect() as con:
#     season_2019_data = pd.read_sql("SELECT * FROM season_2019", con)


## Data Preprocessing

# create a list of columns that will not be looped through for model optimization
NO_LOOP = ['Touchdowns', 'Name', 'Team', 'Position', 'Year']

# drop the columns from the dataframe that we do not want to loop through
columns_to_loop_through = results.drop(columns=NO_LOOP)

# create a list of columns to drop when training the model
DROP_COLUMNS = ['Team', 'Position', 'Year']

# drop the columns from the dataframe that we don't want to include whe training the model
all_columns_df = results.drop(columns=DROP_COLUMNS)

# create bins and labels for touchdown column
touchdown_bins = [-1, 3, 6, 9, 12, 15, 100]
bin_labels = [1,2,3,4,5,6]

# set starting variables for tracking highest accuracy values
highest_accuracy = 0
highest_column_dropped = None
highest_neural_num = 0
third_layer_used = False

# set Name columns as index to retain this information but not include in in model optimization
all_columns_df = all_columns_df.set_index('Name')

# create list of options for having or not having a third hidden layer in the model
THIRD_LAYER = [True, False]

# create list of tuples for possible neuron combinations to try in hidden layers 1 and 2
num_of_neurons = [(100, 70), (70, 40), (40, 10)]

# start by loopoing through each column as identified in list above
for COLUMN in columns_to_loop_through:
    # in each column loop, try with and without a third hidden layer
    for add_extra_layers in THIRD_LAYER:
        # in each column loop and with both third hidden layer combinations, try different combinations of neurons for hidden layers 1 and 2
        for neuron_nums in num_of_neurons:
            loop_results = all_columns_df
            loop_results.drop(columns=COLUMN)
    
    
            # y is the target and x is the features
            # for this case we're training on touchdown performance first
            loop_results['Touchdown_bins'] = pd.cut(loop_results['Touchdowns'], bins = touchdown_bins, labels = bin_labels)
            y = loop_results['Touchdown_bins']
            X = loop_results.drop(columns=['Touchdown_bins', 'Touchdowns'])


            # create the training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=.20)


            # create a StandardScaler instance
            scaler = StandardScaler()


            # fit the StandardScaler
            X_scaler = scaler.fit(X_train)


            # scale the data
            X_train_scaled = X_scaler.transform(X_train)
            X_test_scaled = X_scaler.transform(X_test)


            ## Model Creation

            # define the model
            nn = Sequential()


            # first hidden layer
            nn.add(Dense(units=neuron_nums[0], activation = 'relu', input_dim = X_test.shape[1]))


            # second hidden layer
            nn.add(Dense(units=neuron_nums[1], activation='relu'))


            if add_extra_layers:
                # third hidden layer
                nn.add(Dense(units=20, activation='relu'))


            # output layer
            nn.add(Dense(units=1, activation='relu'))


            # check the structure of the model
            nn.summary()


            # compile the model - use mae or mse for loss function
            nn.compile(loss="mse", optimizer="adam", metrics=["accuracy"])


            # train the model
            fit_model = nn.fit(X_train_scaled, y_train, epochs=60)


            # evaluate the model using the test data
            model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
            print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

            if model_accuracy > highest_accuracy:
                highest_accuracy = model_accuracy
                highest_column_dropped = COLUMN
                highest_neural_num = neuron_nums
                third_layer_used = add_extra_layers


print("HIGHEST OVERALL")
print(f"ACCURACY: {highest_accuracy}, COLUMN_DROPPED {highest_column_dropped}, NEURAL NUM {highest_neural_num}, THIRD LAYER {third_layer_used}")

## ACCURACY 0.7525
## COLUMN_DROPPED Games
## NEURAL NUM 70, 40
## THIRD LAYER True
