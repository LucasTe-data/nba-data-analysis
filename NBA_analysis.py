# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:25:44 2020

@author: Lucas
"""

#%% Load Data
import pandas as pd

PATH = 'C:/Users/Lucas/Documents/Python/Data_Analysis/Basketball_player_stats/players_stats_by_season_full_details.csv'
players_df = pd.read_csv(PATH)

#%% Look at shape, data and missing values
print(F'''
Shape: {players_df.shape}
      
Columns:
{players_df.columns.tolist()}''')

print(F"""
Missing:
{players_df.isnull().mean().mul(100).sort_values(ascending = False).round(2).astype(str) + '%'}    
""")

print(players_df.describe())


print(players_df['League'].unique())

#%% Extract Data about NBA Players
nba_df = players_df.loc[players_df['League'] == 'NBA'].copy()

#%% Add Year Column and List
nba_df['Year'] = nba_df['Season'].str[:4]
nba_df['Year'] = nba_df['Year'].astype('int')
print(nba_df.head(10))
print(nba_df.groupby('Year').count())

#%% Plot 3 Point Attempts and Field Goal Attempts against Points per season
import matplotlib.pyplot as plt

nba_sample = nba_df.sample(300)

plt.scatter(nba_sample['PTS'],nba_sample['3PA'], color = 'green', edgecolors = 'black', label = '3-Point Attempts')
plt.scatter(nba_sample['PTS'],nba_sample['FGA'], color = 'red', edgecolors = 'black', label = 'Field Goal Attempts')
plt.ylabel('Points scored/season')
plt.suptitle('Points/Season in relation to 3-Point Attempts and Field Goal Attempts')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#%% Plot heigth and weight against Points per season
nba_sample = nba_df.sample(300)

plt.scatter(nba_sample['height_cm'],nba_sample['BLK'], color = 'blue', edgecolors = 'black', label = 'Blocks')
plt.title('Scatter plot (Height)')
plt.xlabel('Height in cm')
plt.ylabel('Blocks/season')
plt.show()

plt.scatter(nba_sample['weight_kg'],nba_sample['BLK'], color = 'blue', edgecolors = 'black', label = 'Blocks')
plt.title('Scatter plot (Weight)')
plt.xlabel('Weight in kg')
plt.ylabel('Blocks/season')
plt.show()

#%% Linear Regression for heigth, weight and height+weight
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

BLK_reg = LinearRegression()

variable = ['height_cm']
X_train, X_test, y_train, y_test = train_test_split(nba_df[variable], nba_df['BLK'], test_size=0.33, random_state=42)

BLK_reg = LinearRegression()
BLK_reg.fit(X_train, y_train)

print('Linear Regression (Height):')
score_train = round((BLK_reg.score(X_train, y_train)*100),2)
print(F'Model Score (Train): {score_train}%')

score = round((BLK_reg.score(X_test, y_test)*100),2)
print(F'Model Score (Test): {score}%')

equation = F'BLK = {round(BLK_reg.intercept_,3)} + {round(BLK_reg.coef_[0],3)} * height (in cm) + \u03B5'

print(F'''
Regression line equation:
{equation}''')
print()
print()

height_min = nba_df['height_cm'].min()
height_max = nba_df['height_cm'].max()
BLK_predicted = BLK_reg.predict([[height_min],[height_max]])
height_borders = [height_min,height_max]

count = 0
while BLK_predicted[0] < 0:
    count += 1
    height_pred = nba_df['height_cm'].min() + count
    BLK_predicted = BLK_reg.predict([[height_pred],[height_max]])
    height_borders = [height_pred,height_max]

plt.scatter(nba_sample['height_cm'],nba_sample['BLK'], color = 'blue', edgecolors = 'black', label = 'Blocks')
plt.plot(height_borders, BLK_predicted, color = 'red', label = 'Regression')
plt.title('Regression plot (Height)')
plt.xlabel('Height in cm')
plt.ylabel('Blocks/season')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#Same Regression but for weight
BLK_reg2 = LinearRegression()

variable = ['weight_kg']
X_train, X_test, y_train, y_test = train_test_split(nba_df[variable], nba_df['BLK'], test_size=0.33, random_state=42)

BLK_reg2 = LinearRegression()
BLK_reg2.fit(X_train, y_train)

print('Linear Regression (Weight):')
score_train = round((BLK_reg2.score(X_train, y_train)*100),2)
print(F'Model Score (Train): {score_train}%')

score = round((BLK_reg2.score(X_test, y_test)*100),2)
print(F'Model Score (Test): {score}%')

equation = F'BLK = {round(BLK_reg2.intercept_,3)} + {round(BLK_reg2.coef_[0],3)} * weight (in kg) + \u03B5'

print(F'''
Regression line equation:
{equation}''')
print()
print()

weight_min = nba_df['weight_kg'].min()
weight_max = nba_df['weight_kg'].max()
BLK_predicted = BLK_reg2.predict([[weight_min],[weight_max]])
weight_borders = [weight_min,weight_max]

count = 0
while BLK_predicted[0] < 0:
    count += 1
    weight_pred = nba_df['weight_kg'].min() + count
    BLK_predicted = BLK_reg2.predict([[weight_pred],[weight_max]])
    weight_borders = [weight_pred,weight_max]

plt.scatter(nba_sample['weight_kg'],nba_sample['BLK'], color = 'blue', edgecolors = 'black', label = 'Blocks')
plt.plot(weight_borders, BLK_predicted, color = 'red', label = 'Regression')
plt.title('Regression plot (Weight)')
plt.xlabel('Weight in kg')
plt.ylabel('Blocks/season')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#Multivariat Regression (height and weight)
BLK_reg3 = LinearRegression()

variable = ['height_cm','weight_kg']
X_train, X_test, y_train, y_test = train_test_split(nba_df[variable], nba_df['BLK'], test_size=0.33, random_state=42)

BLK_reg3 = LinearRegression()
BLK_reg3.fit(X_train, y_train)

print('Linear Regression (Height and Weight):')
score_train = round((BLK_reg3.score(X_train, y_train)*100),2)
print(F'Model Score (Train): {score_train}%')

score = round((BLK_reg3.score(X_test, y_test)*100),2)
print(F'Model Score (Test): {score}%')

from scipy.stats import pearsonr
print(F'Correlation between height and weight in the NBA: {round(pearsonr(nba_df["height_cm"],nba_df["weight_kg"])[0],4)}')  



#%% Is there a change in professinal basketball regarding the 3 Point Shot
nba_df_2010 = nba_df.loc[nba_df['Year'] < 2010].copy()
nba_df_2020 = nba_df.loc[nba_df['Year'] >= 2010].copy()


from matplotlib.ticker import PercentFormatter

fig, (ax1, ax2) = plt.subplots(2,1, sharey = True)
ax1.hist(nba_df_2010['3PA'], weights = [1/len(nba_df_2010['3PA'])] * len(nba_df_2010['3PA']), ec='black', label = '1999 - 2009')
ax2.hist(nba_df_2020['3PA'], color = 'red', weights = [1/len(nba_df_2020['3PA'])] * len(nba_df_2020['3PA']), ec='black', label = '2010 - 2019')
ax1.set_title('3 Point Attempts')
ax1.legend()
ax2.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()


year_ticks = range(1999,2020,2)
variable_list = ['3PA', '3PM']
print(nba_df.groupby('Year')[variable_list].sum())
nba_df.groupby('Year')[variable_list].sum().plot(kind = 'line')
plt.title('Evolution of the 3 Pointer')
plt.xticks(year_ticks)
plt.axvline(x = 2010, color = 'red')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


from scipy.stats import ttest_ind, levene

if levene(nba_df_2010['3PA'],nba_df_2020['3PA']).pvalue > 0.05:
    stat_3P, pval_3P = ttest_ind(nba_df_2010['3PA'],nba_df_2020['3PA'])
    print(F'''
T-Test for 3 Point attempts (Between 1999-2009 and 2010-2019) - Equal-Variance
Mean (1999-2009): {round(nba_df_2010['3PA'].mean(),2)}
Mean (2010-2020): {round(nba_df_2020['3PA'].mean(),2)}
p-value = {round(pval_3P, 4)}
Statistic = {round(stat_3P, 4)}''')
else:
    stat_3P, pval_3P = ttest_ind(nba_df_2010['FGA'],nba_df_2020['3PA'], equal_var = False)
    print(F'''
T-Test for 3 Point attempts (Between 1999-2009 and 2010-2019) - Non-Equal-Variance
Mean (1999-2009): {round(nba_df_2010['3PA'].mean(),2)}
Mean (2010-2020): {round(nba_df_2020['3PA'].mean(),2)}
p-value = {round(pval_3P, 4)}
Statistic = {round(stat_3P, 4)}''')

#%% Extracting Euroleague Data
EL_df = players_df.loc[players_df['League'] == 'Euroleague'].copy()
EL_df['Year'] = EL_df['Season'].str[:4]
EL_df['Year'] = EL_df['Year'].astype('int')
print(EL_df.head(10))

#%%
print(F"""
Missing:
{EL_df.isnull().mean().mul(100).sort_values(ascending = False).round(2).astype(str) + '%'}    
""")

#%% Add missing weight data using a Linear Regression (strong linear assumption between height and weight)
missing_df = EL_df[(EL_df['weight_kg'].isnull()) & (EL_df['height_cm'].notna())].copy()
non_missing_df = EL_df[(EL_df['weight_kg'].notna()) & (EL_df['height_cm'].notna())].copy()

weight_reg = LinearRegression()
weight_reg.fit(non_missing_df[['height_cm']],non_missing_df[['weight_kg']])

print(F'Score of Linear Regression (Height/Weight): {round(weight_reg.score(non_missing_df[["height_cm"]], non_missing_df[["weight_kg"]]),4)}')

pred_weight = weight_reg.predict(missing_df[['height_cm']]).round(0).astype('int')
missing_df['weight_kg'] = pred_weight

EL_df.update(missing_df)
#%% Testing if there is a difference between NBA and Euroleague in height
mean_nba = nba_df['height_cm'].mean()
mean_EL = EL_df['height_cm'].mean()
var_nba = nba_df['height_cm'].var()
var_EL = EL_df['height_cm'].var()

fig, (ax1, ax2) = plt.subplots(2,1, sharey = True)
ax1.hist(nba_df['height_cm'], weights = [1/len(nba_df['height_cm'])] * len(nba_df['height_cm']), ec='black', label = 'NBA')
ax2.hist(EL_df['height_cm'], color = 'red', weights = [1/len(EL_df['height_cm'])] * len(EL_df['height_cm']), ec='black', label = 'Euroleague')
ax1.axvline(x = mean_nba, color = 'black', label = 'Mean')
ax2.axvline(x = mean_EL, color = 'black', label = 'Mean')
ax1.set_title('Height in cm')
ax1.legend()
ax2.legend()
plt.show()


if levene(nba_df['height_cm'],EL_df['height_cm']).pvalue > 0.05:
    stat_h, pval_h = ttest_ind(nba_df['height_cm'],EL_df['height_cm'], nan_policy = 'omit')
    print(F'''
T-Test for height in cm (Between NBA and Euroleague) - Equal-Variance
Mean (NBA): {round(mean_nba,2)}
Variance (NBA) : {round(var_nba,2)}
Mean (Euroleague): {round(mean_EL,2)}
Variance (Euroleague) : {round(var_EL,2)}
p-value = {round(pval_h, 4)}
Statistic = {round(stat_h, 4)}''')
else:
    stat_h, pval_h = ttest_ind(nba_df['height_cm'],EL_df['height_cm'], equal_var = False, nan_policy = 'omit')
    print(F'''
T-Test for height in cm (Between NBA and Euroleague) - Non-Equal-Variance
Mean (NBA): {round(mean_nba,2)}
Variance (NBA) : {round(var_nba,2)}
Mean (Euroleague): {round(mean_EL,2)}
Variance (Euroleague) : {round(var_EL,2)}
p-value = {round(pval_h, 4)}
Statistic = {round(stat_h, 4)}''')


#%% Testing if there is a difference between NBA and Euroleague in weight
mean_nba = nba_df['weight_kg'].mean()
mean_EL = EL_df['weight_kg'].mean()
var_nba = nba_df['weight_kg'].var()
var_EL = EL_df['weight_kg'].var()


fig, (ax1, ax2) = plt.subplots(2,1, sharey = True)
ax1.hist(nba_df['weight_kg'], weights = [1/len(nba_df['weight_kg'])] * len(nba_df['weight_kg']), ec='black', label = 'NBA')
ax2.hist(EL_df['weight_kg'], color = 'red', weights = [1/len(EL_df['weight_kg'])] * len(EL_df['weight_kg']), ec='black', label = 'Euroleague')
ax1.axvline(x = mean_nba, color = 'black', label = 'Mean')
ax2.axvline(x = mean_EL, color = 'black', label = 'Mean')
ax1.set_title('Weight in kg')
ax1.legend()
ax2.legend()
plt.show()

if levene(nba_df['weight_kg'],EL_df['weight_kg']).pvalue > 0.05:
    stat_w, pval_w = ttest_ind(nba_df['weight_kg'],EL_df['weight_kg'], nan_policy = 'omit')
    print(F'''
T-Test for weight in cm (Between NBA and Euroleague) - Equal-Variance
Mean (NBA): {round(mean_nba,2)}
Variance (NBA) : {round(var_nba,2)}
Mean (Euroleague): {round(mean_EL,2)}
Variance (Euroleague) : {round(var_EL,2)}
p-value = {round(pval_w, 4)}
Statistic = {round(stat_w, 4)}''')
else:
    stat_w, pval_w = ttest_ind(nba_df['weight_kg'],EL_df['weight_kg'], equal_var = False, nan_policy = 'omit')
    print(F'''
T-Test for weight in cm (Between NBA and Euroleague) - Non-Equal-Variance
Mean (NBA): {round(mean_nba,2)}
Variance (NBA) : {round(var_nba,2)}
Mean (Euroleague): {round(mean_EL,2)}
Variance (Euroleague) : {round(var_EL,2)}
p-value = {round(pval_w, 4)}
Statistic = {round(stat_w, 4)}''')

print()

#%% Add Position Column using the Sportsreference API
# from sportsreference.nba.teams import Teams
# import time
# import datetime
# import numpy as np

# years = np.flip(nba_df['Year'].unique())
# #pos_dict = {}
# start_time = time.time()

# while True:
#     try:
#         for year in years:
#             print(year)
#             teams = Teams(year = year)
            
#             for team in teams:
#                 print(F'{team.name} ({year})')
#                 roster = team.roster
#                 for player in roster.players:
#                     if player.name not in pos_dict:
#                         try:
#                             name = player.name
#                             pos = player.position
#                             pos_dict[name] = pos
#                             print(player.name)
#                         except:
#                             print('ERROR')
#                             continue
                        
#                 runtime = datetime.timedelta(seconds = round((time.time() - start_time),2))
#                 print(F'Runtime: {runtime}')
#                 print()
#         break
#     except:
#         print('Waiiiiit')
#         time.sleep(300)
# print(len(pos_dict))


#%% Add Position Column using the Sportsreference API (2)
# nba_df['position'] = nba_df['Player'].map(pos_dict)
# print(nba_df.head())

# print(F"""
# Missing:
# {nba_df.isnull().mean().mul(100).sort_values(ascending = False).round(2).astype(str) + '%'}    
# """)
#%% Save new Dataframe as csv
#nba_df.to_csv('C:/Users/Lucas/Documents/Python/Data_Analysis/Basketball_player_stats/nba_players_position.csv')
nba_df = pd.read_csv('C:/Users/Lucas/Documents/Python/Data_Analysis/Basketball_player_stats/nba_players_position.csv')

#%% Prepare dataset
def clean_position(position):
    if position[0] == 'C':
        return position[0]
    else:
        return position[0:2]


nba_df_clean = nba_df[nba_df['position'].notna()].copy()
nba_df_clean['position'] = nba_df_clean['position'].apply(lambda x: F'{clean_position(x)}')

#%% EDA for Position
import seaborn as sns
from scipy import stats
from statsmodels.sandbox.stats.multicomp import MultiComparison
 
positions = ['PG', 'SG', 'SF', 'PF', 'C']

sns.countplot(nba_df_clean['position'], order = positions, ec='black')
plt.ylim(0, 1400)
plt.title('Number of Players by Position')
plt.xlabel('Position')
plt.ylabel('No. of Players')
plt.show()

position_counts = nba_df_clean['position'].value_counts()
print(position_counts)
print()

test_stat = stats.chisquare(position_counts)
print(F'''Chi Square Value of Chi Square Test: {round(test_stat.statistic,4)}
P-Value of Chi Square Test: {round(test_stat.pvalue,4)}
''')

#Trying to see if SG Position is an Outlier
position_counts = position_counts[0:4]
print(position_counts)
print()

test_stat = stats.chisquare(position_counts)
print(F'''Chi Square Value of Chi Square Test: {round(test_stat.statistic,4)}
P-Value of Chi Square Test: {round(test_stat.pvalue,4)}
''')

heights = []
print('Mean height per position:')
for position in positions:
    mean_height = round(nba_df_clean[nba_df_clean['position'] == position]['height_cm'].mean(),2)
    print(F"{position} : {mean_height} cm")
    heights.append(mean_height)
sns.barplot(positions, heights, ec = 'black')
plt.title('Average height by Position')
plt.xlabel('Position')
plt.ylabel('Height')
plt.show()
print()

weights = []
print('Mean weight per position:')
for position in positions:
    mean_weight = round(nba_df_clean[nba_df_clean['position'] == position]['weight_kg'].mean(),2)
    print(F"{position} : {mean_weight} kg")
    weights.append(mean_weight)
sns.barplot(positions, weights, ec = 'black')
plt.title('Average weight by Position')
plt.xlabel('Position')
plt.ylabel('Weight')
plt.show()
print()

PG = nba_df_clean[nba_df_clean['position'] == 'PG']
SG = nba_df_clean[nba_df_clean['position'] == 'SG']
PF = nba_df_clean[nba_df_clean['position'] == 'PF']
SF = nba_df_clean[nba_df_clean['position'] == 'SF']
C = nba_df_clean[nba_df_clean['position'] == 'C']

print(stats.f_oneway(PG['height_cm'], SG['height_cm'], PF['height_cm'], SF['height_cm'], C['height_cm']))
MultiComp = MultiComparison(nba_df_clean['height_cm'], nba_df_clean['position'])
print(MultiComp.tukeyhsd().summary())
print()

print(stats.f_oneway(PG['weight_kg'], SG['weight_kg'], PF['weight_kg'], SF['weight_kg'], C['weight_kg']))
MultiComp = MultiComparison(nba_df_clean['weight_kg'], nba_df_clean['position'])
print(MultiComp.tukeyhsd().summary())
print()

#%% Classify Playerposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

variables = ['weight_kg', 'height_cm']
X_train, X_test, y_train, y_test = train_test_split(nba_df_clean[variables], nba_df_clean['position'], test_size=0.33, random_state=42)

scaler = StandardScaler().fit(nba_df_clean[variables])
X_train_transformed = scaler.transform(X_train, copy = True)
X_test_transformed = scaler.transform(X_test, copy = True)

position_tree = DecisionTreeClassifier()
position_tree.fit(X_train, y_train)

score_train = round((position_tree.score(X_train, y_train)*100),2)
score = round((position_tree.score(X_test, y_test)*100),2)
print('Descision Tree Classification of Player Position (Height and Weight):')
print(F'Model Score (Train): {score_train}% (Depth = {position_tree.get_depth()})')
print(F'Model Score (Test): {score}%')

importance = position_tree.feature_importances_
print('Feature Importance:')    
for item in range(0,len(variables)):
    print(F'{variables[item]}: {round(importance[item], 2)*100}%')
print()


position_clf = RandomForestClassifier()
position_clf.fit(X_train, y_train)

score_train = round((position_clf.score(X_train, y_train)*100),2)
score = round((position_clf.score(X_test, y_test)*100),2)
print('Random Forest Classification of Player Position (Height and Weight):')
print(F'Model Score (Train): {score_train}%')
print(F'Model Score (Test): {score}%')

importance = position_clf.feature_importances_
print('Feature Importance:')
for item in range(0,len(variables)):
    print(F'{variables[item]}: {round(importance[item]*100, 2)}%')
print()


scaler = StandardScaler().fit(nba_df_clean[variables])
X_train_transformed = scaler.transform(X_train, copy = True)
X_test_transformed = scaler.transform(X_test, copy = True)

position_log = LogisticRegression()
position_log.fit(X_train_transformed, y_train)

score_train = round((position_log.score(X_train_transformed, y_train)*100),2)
score = round((position_log.score(X_test_transformed, y_test)*100),2)
print('Logistic Regression of Player Position (Height and Weight):')
print(F'Model Score (Train): {score_train}%')
print(F'Model Score (Test): {score}%')
print()


#%% Compare models to find best optionx
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score
import numpy as np

y_true = nba_df_clean['position']
y_tree = position_tree.predict(nba_df_clean[variables])
y_clf = position_clf.predict(nba_df_clean[variables])
y_log = position_log.predict(scaler.transform(nba_df_clean[variables], copy = True))

y_list = [y_tree, y_clf, y_log, y_true]
model_dict = {0 : 'Decision Tree', 1 : 'Random Forest', 2 : 'Logistic Regression', 3 : 'True Values'}

for prediction in range(0,len(y_list) - 1):
    sns.countplot(y_list[prediction], order = positions, ec='black')
    plt.ylim(0, 1400)
    plt.title(F'Predicited Number of Players by Position ({model_dict[prediction]})')
    plt.xlabel('Position')
    plt.ylabel('No. of Players')
    plt.show()

print('Reliability Test with Cohens Kappa')
for i in range(0,len(y_list)):
    for x in range(i+1, len(y_list)):
        ck = cohen_kappa_score(y_list[i], y_list[x])
        print(F'Cohens Kappa: {round(ck*100,2)}% ({model_dict[i]}/{model_dict[x]})')
print()

kf = KFold(n_splits = 5, shuffle = True)

print('Cross Validation:')

scores_tree = cross_val_score(DecisionTreeClassifier(), nba_df_clean[variables], nba_df_clean['position'], cv = kf)
print(F'Decision Tree mean score: {round(np.mean(scores_tree)*100, 2)}%')

scores_rand_f = cross_val_score(RandomForestClassifier(), nba_df_clean[variables], nba_df_clean['position'], cv = kf)
print(F'Decision Tree mean score: {round(np.mean(scores_rand_f)*100, 2)}%')

scores_log = cross_val_score(LogisticRegression(), scaler.transform(nba_df_clean[variables], copy = True), nba_df_clean['position'], cv = kf)
print(F'Logistic Regression mean score: {round(np.mean(scores_log)*100, 2)}%')
print()

#No real difference, both tree models give almost always the same answer, we are gonna use the decision tree because it is simpler
#Log Regression doesn't really work

#%% Save the Model to work with it outside of this script
import pickle

filename = 'position_model.pkl'
pickle.dump(position_clf, open(filename, 'wb'))
print('Model Saved')
