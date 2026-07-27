
# ===== CELL 0 (markdown) =====
#  TRAIN TEST SPLIT (80/ 20)  > Feature Extraction > SMOTE > Feature selection (RF, scoring = 'roc_auc') >  No Normalization > Custom Ensemble (acc: 94, f1: 73)

# ===== CELL 1 (markdown) =====
## Imports

# ===== CELL 2 (code) =====
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ===== CELL 3 (markdown) =====
## Data Preprocessing


# ===== CELL 4 (code) =====
data = pd.io.stata.read_stata('SHILD2012_cens_eng_16-64.dta')
data.to_csv('my_stata_file.csv', index= False)

# ===== CELL 5 (code) =====
df = pd.read_csv('my_stata_file.csv')

print(df)

# ===== CELL 6 (code) =====
df.head(10)

# ===== CELL 7 (code) =====
import seaborn as sns
plt.figure(figsize = (10, 5))
sns.countplot(x = df['A1'])
plt.show()



# ===== CELL 8 (code) =====
A1 = []
counter = 0
for row in df['A1']:
    if row == 'Discontent':
        A1.append(1)
    elif row == 'Content':
        A1.append(0)
    elif row == 'Very content':
        A1.append(0)
    elif row == 'Very discontent':
        A1.append(1)
    else:
        A1.append(-1)

    ++counter
df['A1'] = A1

# ===== CELL 9 (code) =====
# Removing all instances where A1 == 0. A1 = 0 are "don't know" answers in content vs discontent with life column
df = df[df.A1 > -1]

# ===== CELL 10 (code) =====

# imports and initialization
import numpy as np
import os
# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# ===== CELL 11 (code) =====
# Distribution of content vs discontent

import seaborn as sns
plt.figure(figsize = (5, 5))
fig = sns.countplot(x = df['A1'])

save_fig("imbalance-1.png")
plt.show()



# ===== CELL 12 (code) =====

print(df.shape)
df.head()

# ===== CELL 13 (code) =====
# Separate the independent (X) and dependent (y) features

y_df = df['A1']
df = df.drop(columns = ['A1', 'respnb', 'method', 'pervgt', 'G8', 'H1', 'H2', 'H3', 'K5', 'R10', 'R12_a'], axis=1)



# ===== CELL 14 (code) =====
missing_values = data.isnull().sum()


# ===== CELL 15 (code) =====
#missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# ===== CELL 16 (code) =====
# Set the figure size
plt.figure(figsize=(12, 6))

# Create a bar plot using seaborn
ax = sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")

# Remove x-axis labels
ax.set_xticklabels([])

# Customize the plot
plt.xlabel('Features')
plt.ylabel('Missing Values Count')
plt.title('Missing Values Count per Feature')

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig("missing-before.png")


# ===== CELL 17 (code) =====
import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(df)

save_fig("missing-1")
# Save the figure as an image file
plt.savefig("missing-1.png")

# ===== CELL 18 (markdown) =====
###after

# ===== CELL 19 (code) =====
#dropping columns with 18% or more null values
percentage = 18
min = int(((100-percentage)/100)*df.shape[0])

df.dropna(axis=1, thresh = min, inplace= True)

df.shape

# ===== CELL 20 (code) =====
# remove duplicate entries, if any
df.drop_duplicates(inplace = True)

df.shape

# ===== CELL 21 (code) =====
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame with some features dropped
# Calculate the missing values count for each column
missing_values = df.isnull().sum()

# Set the figure size
plt.figure(figsize=(12, 6))

# Create a bar plot using seaborn
ax = sns.barplot(x=missing_values.index, y=missing_values.values, palette="magma")

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Remove x-axis labels
ax.set_xticklabels([])

# Customize the plot
plt.xlabel('Features')
plt.ylabel('Missing Values Count')
plt.title('Missing Values Count per Feature (After Dropping)')

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig("missing-after.png")

# ===== CELL 22 (code) =====
import missingno as msno
msno.matrix(df)
# Save the figure as an image file
plt.savefig("missing-2.png")

# ===== CELL 23 (code) =====
#print unique values of features

for cols in df.columns:
    unique_vals = df[cols].unique()
    if df[cols].dtype == object:
        print("Feature ", cols, " has ", len(unique_vals), " unique values ----> ", unique_vals , " ----> Datatype: ", df[cols].dtypes)

# ===== CELL 24 (markdown) =====
### Demographic Info

# ===== CELL 25 (code) =====
m_count = df['gender'].value_counts()['Man']
print(f"Male: {m_count}")
f_count = df['gender'].value_counts()['Woman']
print(f"Female: {f_count}")

# ===== CELL 26 (code) =====
m_count = df['job'].value_counts()['Holds an ordinary or supported job']
print(f"Has Job: {m_count}")
f_count = df['job'].value_counts()['Doesn\'t hold an ordinary or supported job']
print(f"No Job: {f_count}")


# ===== CELL 27 (code) =====
m_count = df['C1'].value_counts()['Yes']
print(f"Suffers from suffer a long-term physical health problem or disability: {m_count}")
f_count = df['C1'].value_counts()['No']
print(f"Does not: {f_count}")

# ===== CELL 28 (code) =====
df['age'].value_counts(bins=10, sort=False)

# ===== CELL 29 (code) =====
m_count = df['G1'].value_counts()['Yes']
print(f"Has spouse: {m_count}")
f_count = df['G1'].value_counts()['No']
print(f"Does not: {f_count}")

# ===== CELL 30 (markdown) =====
###Categorical encoding

# ===== CELL 31 (code) =====
# Convert Catergorical values to numeric representation

df['gender'].replace({'Woman':0, 'Man':1}, inplace = True)

df['int_result'].replace({'1. Answer':0}, inplace = True)

df['A2'].replace({'Average':1, 'Well':2, 'Very well':3, 'Poor':1, 'Very poor':0, "Don't know":-1}, inplace = True)

#B1-B13 'Without difficulty':2, 'With some difficulty':0, 'With much difficulty':2,'Not at all':3, "Don't know":4, 'Refuse to answer':5
#--

df['B1'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3}, inplace = True)

df['B2'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3}, inplace = True)

df['B3'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3}, inplace = True)

df['B4'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3}, inplace = True)

df['B5'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3}, inplace = True)

df['B6'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3}, inplace = True)

df['B7'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3}, inplace = True)

df['B8'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3, "Don't know":-1}, inplace = True)

df['B9'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3, "Don't know":-1}, inplace = True)

df['B10'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['B11'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3, "Don't know":-1}, inplace = True)

df['B12'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3, "Don't know":-1}, inplace = True)

df['B13'].replace({'With some difficulty':1, 'Without difficulty':2, 'With much difficulty':0,'Not at all':3}, inplace = True)
#--

df['B14'].replace({'No':0, 'Yes':1}, inplace = True)

df['B17_a'].replace({'Yes':1, 'No':0}, inplace = True)

df['C1'].replace({'No':0, 'Yes':1, "Don't know":-1}, inplace = True)

df['C4'].replace({'No':2, 'Yes, one':1, "Don't know":-1, 'Yes, more':0}, inplace = True)


#D1-E3 often, always, sometimes, rarely, never, don't know 0<1<5
#--
df['D1'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4}, inplace = True)

df['D2'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D3'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4}, inplace = True)

df['D4'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4}, inplace = True)

df['D5'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4}, inplace = True)

df['D6'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4}, inplace = True)

df['D7'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D8'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4}, inplace = True)

df['D9'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D10'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D11'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D12'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D13'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D14'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D15'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D16'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['D17'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4, 'Refuse to answer':-1.5}, inplace = True)

df['E3'].replace({'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, "Don't know":-1, 'Always':4}, inplace = True)
# --

df['E5_a'].replace({'No':0, 'Yes':1, "Don't know":-1}, inplace = True)

df['E6'].replace({'Yes':1, 'No':0, 'Refuse to answer':-1}, inplace = True)

df['E12'].replace({'Once a week':6, '2-3 times a week':7,
 'Less than once a month, but several times a year':4, 'Never':0,
 'Almost every day':8, 'Less frequently':1, 'Once a month':3, 'Every day':9,
 'Once every fortnight':5, "Don't know":-1}, inplace = True)

df['E17'].replace({'Friends/colleagues':0, 'Partner/spouse/boy-/girlfriend':1,
 'I don\x92t share this with anyone':2, 'Siblings':3, 'Parents':4, 'Others':5,
 "Don't know":-1, 'Children':7, 'Staff':8, 'Other family':9, 'Refuse to answer':-1}, inplace = True)

df['education'].replace({'Completed secondary school or more (eksamensskole)':0,
 'Completed compulsory school (folkeskole, 9 years)':1}, inplace = True)

df['job'].replace({'Holds an ordinary or supported job':0,
 "Doesn't hold an ordinary or supported job":1}, inplace = True)


df['F10'].replace({'No':0, 'Yes':1, "Don't know":-1}, inplace = True)

df['F11'].replace({'No':0, 'Yes':1, "Don't know":-1}, inplace = True)

df['F15'].replace({'9':9, '8':8, '6':6, '7':7 , '2':2, '5':5, '1':1, '4':4, '0':0, '3':3, "Don't know":-1, '0 very low':0, 'Refuse to answer':-1.5}, inplace = True)


df['G1'].replace({'No':0, 'Yes':1, 'Refuse to answer':-1}, inplace = True)

df['G6'].replace({'Yes, my mom is alive':1, 'Yes, my dad is alive':2, 'No':0, 'Yes, both are alive':4,
 "Don't Know":-1}, inplace = True)

df['G7'].replace({'No, none of my parents':0, 'Yes, one parent':1, 'Yes, both parents':2,
 "Don't Know":-1}, inplace = True)


df['G10_a'].replace({'No':0, 'Yes':1, "Don't know":-1}, inplace = True)

df['G11_a'].replace({'Yes':1, 'No':0, "Don't know":-1}, inplace = True)

#J1- J20
df['J1'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2, "Don't Know":-1}, inplace = True)

df['J2'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2, 'Refuse to answer':-1, '99':-1}, inplace = True)

df['J3'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2, 'Refuse to answer':-1, '99':-1}, inplace = True)

df['J4'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2, 'Refuse to answer':-1, '99':-1}, inplace = True)



df['J18'].replace({ 'No':0, 'Yes':1, "Don't know":-1}, inplace = True)

df['J18a'].replace({'No':0, 'Yes':1, "Don't know":-1}, inplace = True)
df['J8a'].replace({'Yes':1, 'No':0, "Don't know":-1}, inplace = True)

df['J9'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2}, inplace = True)

df['J10'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2,"Don't Know":-1}, inplace = True)

df['J11'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2,"Don't Know":-1}, inplace = True)

df['J12'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2,"Don't Know":-1}, inplace = True)

df['J13'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2,"Don't Know":-1}, inplace = True)

df['J14'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2,"Don't Know":-1}, inplace = True)

df['J15'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2,"Don't Know":-1}, inplace = True)

df['J16_a'].replace({'No':0, 'Yes':1, "Don't know":-1}, inplace = True)

df['J20'].replace({'Yes':1, 'No':0, "Don't know":-1}, inplace = True)

df['K1_a'].replace({'Yes':1, 'No':0}, inplace = True)

#K1-K5

df['K1'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2}, inplace = True)

df['K2'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2}, inplace = True)

df['K3'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2}, inplace = True)

df['K4'].replace({'Less frequently':1, 'Daily':6, 'Several times a month':4, 'Several times a week':5,
 'Once a week':3, 'Never':0, 'Once a month':2}, inplace = True)

df['L8_1'].replace({'Withdraw the money immediately DKK 100,000 (EUR 13,407)':0,
 'Withdraw the money in 12 months DKK 102,000 (EUR 13,675)':1, "Don't know":2,
 'Refuse to answer':3}, inplace = True)

df['M2'].replace({'Salary, fee income':0, 'Early retirement/retirement pension':1,
 'Social security':2, 'Other':3, 'Pension schemes':4, 'Self-employment income':5,
 'Unemployment benefits':6, 'Incapacity benefit':7, 'Other welfare':8,
 "Don't Know":9, 'Trading bonds, shares and real estate':10, 'Interest income':11,
 'Inheritance':12, 'Black money':13}, inplace = True)

df['M8'].replace({'Good':3, 'Average':2, 'Bad':1, 'Very good':4, 'Very bad':0, "Don't Know":-1,
 'Refuse to answer':-1.5}, inplace = True)

df['N1'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['N2'].replace({'Yes':1, 'No':0, "Don't know":-1}, inplace = True)

df['N5'].replace({'Yes':1, 'No':0, "Don't know":-1}, inplace = True)

df['N8'].replace({'Yes':1, 'No':0, "Don't know":-1}, inplace = True)

df['N9'].replace({'Yes':1, 'No':0, "Don't know":-1}, inplace = True)

df['N12'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['N13'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['N16'].replace({'Yes':1, 'No':0, "Don't know":-1}, inplace = True)

df['Q1'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['Q1_a'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['Q1_b'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['Q1_e'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['Q1_f'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['Q1_g'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['Q3_a'].replace({'No':0,'Yes':1, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['Q4_a'].replace({'Yes':1, 'No':0, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

df['Q5'].replace({'I would consider it for a while, but probably say yes':0,
 'I would say yes, without hesitation':1, 'I would be very much in doubt':2,
 'I would say no, without hesitation':3, "Don't Know":4,
 'I would consider it for a while, but probably say no':5, 'Refuse to answer':6}, inplace = True)

df['Q6'].replace({'I would consider it for a while, but probably say yes':0,
 'I would say yes, without hesitation':1,
 'I would say no, without hesitation':2, 'I would be very much in doubt':3,
 'I would consider it for a while, but probably say no':4, "Don't Know":5,
 'Refuse to answer':6}, inplace = True)

df['R1'].replace({'None':0, '6-10 times':1, '1-2 times':2, 'More than 10 times':3, "Don't Know":4,
 '3-5 times':5, 'Refuse to answer':6}, inplace = True)

#df['R10'].replace({'Yes':1, 'No':0, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)

#df['R12_a'].replace({'Yes':1, 'No':0, "Don't know":-1, 'Refuse to answer':-1.5}, inplace = True)




# ===== CELL 32 (code) =====
y_df.describe()

# ===== CELL 33 (markdown) =====
## TRAIN TEST SPLIT

# ===== CELL 34 (code) =====
X = df
Y = y_df

# ===== CELL 35 (code) =====
# Splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y_df, test_size = 0.2, random_state = 20)
X_train.head(5)

# ===== CELL 36 (markdown) =====
## FEATURE EXTRACTION

# ===== CELL 37 (markdown) =====
### ZERO VARIANCE

# ===== CELL 38 (code) =====
# ZERO VARIANCE

from sklearn.feature_selection import VarianceThreshold

zeroVar = VarianceThreshold(threshold=0)               #variance=0
zeroVar.fit_transform(df)
#zeroVar.fit_transform(X_test)

zeroVar.get_support()        # columns having non-zero variance = True
sum(zeroVar.get_support())   # number of columns having non-zero variance

# ===== CELL 39 (code) =====
	#Dropping constant columns

constant_columns = [column for column in df.columns
                  if column not in df.columns[zeroVar.get_support()]]

	print(len(constant_columns))


X_train.drop(constant_columns,axis=1, inplace=True)
X_test.drop(constant_columns,axis=1, inplace=True)
X.drop(constant_columns,axis=1, inplace=True)

# ===== CELL 40 (code) =====

X_train.shape

# ===== CELL 41 (markdown) =====
### Correlation

# ===== CELL 42 (code) =====
# Pearson's Correlation Coefficient
import seaborn as sns
import numpy as np

corr = X_train.corr()
mask1 = np.triu(np.ones_like(corr, dtype=bool))
mask2 = np.tril(np.ones_like(corr, dtype=bool))
plt.figure(figsize = (106,106))
sns.heatmap(corr, annot = True, cmap = "seismic", mask = mask1);

# ===== CELL 43 (code) =====
# with the following function we can select highly correlated features
# it will remove the first feature that is highly correlated with another feature

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of the redundant columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if(abs(corr_matrix.iloc[i, j])) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

# ===== CELL 44 (code) =====
corr_features = correlation(df, 0.8) # 80% is a good value of the threshold
print(len(corr_features))
print(corr_features)

# ===== CELL 45 (code) =====
X_train.drop(corr_features,axis=1, inplace=True)
X_test.drop(corr_features,axis=1, inplace=True)
X.drop(corr_features,axis=1, inplace=True)
X_train.shape

# ===== CELL 46 (markdown) =====
### Outliers

# ===== CELL 47 (code) =====
import seaborn as sns
ax = sns.boxplot(data = X_train[['D2', 'D6', 'D8']])

ax.set(xlabel='Features', ylabel='Values')
# Save the figure as an image file
plt.savefig("outliers.png")


# ===== CELL 48 (code) =====
#replacing outliers that lie more than 2 standard deviations away from the mean


for col in X_train.columns:
  mean = X_train[col].mean()
  sd = X_train[col].std()
  median = X_train[col].median()
  X_train[col].mask(X_train[col] > mean+(2*sd), median, inplace=True)
  X_train[col].mask(X_train[col] < mean-(2*sd), median, inplace=True)

# ===== CELL 49 (code) =====
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.boxplot(data=X_train[['D2', 'D6', 'D8']])
ax.set(xlabel='Features', ylabel='Values')

# Save the figure as an image file
plt.savefig("outliers-2.png")



# ===== CELL 50 (code) =====
# Imputing null values

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df_cols = X_train.columns


imp = IterativeImputer(max_iter=10, random_state=21)
imp.fit(X_train)
X_train = imp.transform(X_train)
X_train = pd.DataFrame(X_train, columns = df_cols)

X_test = imp.transform(X_test)
X_test = pd.DataFrame(X_test, columns = df_cols)

X = imp.transform(X)
X = pd.DataFrame(X, columns = df_cols)

# ===== CELL 51 (code) =====
import missingno as msno
msno.matrix(X_train)


# ===== CELL 52 (code) =====
pd.set_option('display.max_columns', None)
X_train.describe()

# ===== CELL 53 (markdown) =====
### BALANCING

# ===== CELL 54 (code) =====
# Fixing imbalance with over & undersampling

import imblearn
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

oversample = SMOTETomek(sampling_strategy=.4, random_state=21, n_jobs=-1)
Xb, y_b = oversample.fit_resample(X_train, y_train)


# Checking if data has been balanced

from collections import Counter
#print(Counter(y))
print(Counter(y_b))

# ===== CELL 55 (code) =====
import seaborn as sns
plt.figure(figsize = (10, 5))
sns.countplot(x = y_b)
plt.show()

y_b.value_counts()

# Save the figure as an image file
plt.savefig("balance-2.png")

# ===== CELL 56 (code) =====
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy=1)


#Xdf=np.array(df)
#Xb = X_train
#y_b = y_train
y = y_train
Xb, y_b = undersample.fit_resample(Xb, y_b)

from collections import Counter
print(Counter(y))
print(Counter(y_b))

# ===== CELL 57 (code) =====
import seaborn as sns
plt.figure(figsize = (10, 5))
sns.countplot(x = y_b)
plt.show()
# Save the figure as an image file
plt.savefig("balance-3.png")
#y_b.value_counts()


# ===== CELL 58 (code) =====
columns = X_train.columns
Xb = pd.DataFrame(Xb, columns = columns)
X_train = Xb
y_train = y_b

# ===== CELL 59 (markdown) =====
## Feature Selection

# ===== CELL 60 (code) =====
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

rfc = RandomForestClassifier(random_state=21, n_jobs=-1)
#rfc = GradientBoostingClassifier()
#rfc = DecisionTreeClassifier()

rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
#rfecv.fit(Xb, y_b)
rfecv.fit(X_train, y_train)

# ===== CELL 61 (code) =====
print('Optimal number of features: {}'.format(rfecv.n_features_))

# ===== CELL 62 (code) =====
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], color='#303F9F', linewidth=3)

#plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
plt.show()

# Save the figure as an image file
plt.savefig("RFECV.png")

# ===== CELL 63 (code) =====
print(np.where(rfecv.support_ == False)[0])

# ===== CELL 64 (code) =====
print(np.where(rfecv.estimator_.feature_importances_ == 0)[0])

# ===== CELL 65 (code) =====
selected_features = X_train.drop(X_train.columns[np.where(rfecv.support_ == False)[0]], axis=1)

# ===== CELL 66 (code) =====
rfecv.estimator_.feature_importances_

# ===== CELL 67 (code) =====
dset = pd.DataFrame()
dset['attr'] = selected_features.columns
dset['importance'] = rfecv.estimator_.feature_importances_
dset = dset.sort_values(by='importance', ascending=False)

plt.figure(figsize=(16, 40))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()
# Save the figure as an image file
plt.savefig("RFECV_Feature_imp.png")

# ===== CELL 68 (code) =====
dset.tail(74)

# ===== CELL 69 (code) =====
"""import scikitplot as skplt
rf = RandomForestClassifier()
rf.fit(X, y)
skplt.estimators.plot_feature_importances(
         rf, feature_names=['petal length', 'petal width',
                        'sepal length', 'sepal width'])
matplotlib.axes._subplots.AxesSubplot
plt.show() """

# ===== CELL 70 (code) =====
X_train.shape

# ===== CELL 71 (code) =====
X_train.drop(X_train.columns[np.where(rfecv.support_ == False)[0]], axis=1 ,inplace = True)
#df.drop(df.columns[np.where(rfecv.estimator_.feature_importances_ < 0.009454)[0]], axis=1 ,inplace = True)
X_train.drop(X_train.columns[np.where(rfecv.estimator_.feature_importances_ < 0.0094)[0]], axis=1 ,inplace = True)

# ===== CELL 72 (code) =====
X_test.drop(X_test.columns[np.where(rfecv.support_ == False)[0]], axis=1 ,inplace = True)
#df.drop(df.columns[np.where(rfecv.estimator_.feature_importances_ < 0.009454)[0]], axis=1 ,inplace = True)
X_test.drop(X_test.columns[np.where(rfecv.estimator_.feature_importances_ < 0.0094)[0]], axis=1 ,inplace = True)

# ===== CELL 73 (code) =====
X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1 ,inplace = True)
#df.drop(df.columns[np.where(rfecv.estimator_.feature_importances_ < 0.009454)[0]], axis=1 ,inplace = True)
X.drop(X.columns[np.where(rfecv.estimator_.feature_importances_ < 0.0094)[0]], axis=1 ,inplace = True)

# ===== CELL 74 (code) =====
X_train.shape

# ===== CELL 75 (code) =====
pd.set_option('display.max_columns', None)
X_train.describe()

# ===== CELL 76 (markdown) =====
##Learning Curve
shows the effect of adding more samples during the training process. It is depicted by checking the statistical performance of the model in terms of training score and testing score.

# ===== CELL 77 (code) =====
!pip install scikit-plot

# ===== CELL 78 (code) =====
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import scikitplot as skplt

# load digits dataset
#X, y = load_digits(return_X_y=True)
# Assuming the target variable is stored in a column named 'target'

# define classifier

clf = LogisticRegression(solver = 'liblinear', penalty = 'l2')

# generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# plot the learning curve
skplt.estimators.plot_learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1)

plt.show()
plt.savefig("learning curve_lr.png")

# ===== CELL 79 (code) =====
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import scikitplot as skplt
import xgboost

# load digits dataset
#X, y = load_digits(return_X_y=True)
# Assuming the target variable is stored in a column named 'target'

# define classifier

clf = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.3, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.1, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=12, max_leaves=0, min_child_weight=1,
               monotone_constraints='()', n_estimators=600,
              n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=21,
              reg_alpha=0, reg_lambda=1)

# generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# plot the learning curve
skplt.estimators.plot_learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1)

plt.show()
plt.savefig("learning curve_xgb.png")

# ===== CELL 80 (code) =====
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import scikitplot as skplt
import lightgbm as lgb

# load digits dataset
#X, y = load_digits(return_X_y=True)
# Assuming the target variable is stored in a column named 'target'

# define classifier
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

clf = lgb.LGBMClassifier(**params)

# generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# plot the learning curve
skplt.estimators.plot_learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1)

plt.show()
plt.savefig("learning curve_LGB.png")

# ===== CELL 81 (markdown) =====
## FOREST PLOT

# ===== CELL 82 (code) =====
import statsmodels.api as sm

# build the model and fit the data
model = sm.Logit(y_train, X_train).fit()

# ===== CELL 83 (code) =====
print(model.summary())

# ===== CELL 84 (code) =====
import numpy as np
params = model.params
conf = model.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
# convert log odds to ORs
odds = pd.DataFrame(np.exp(conf))
# check if pvalues are significant
odds['pvalues'] = model.pvalues
odds['significant?'] = ['significant' if pval <= 0.05 else 'not significant' for pval in model.pvalues]
odds

# ===== CELL 85 (code) =====
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4), dpi=150)
ci = [odds.iloc[::-1]['Odds Ratio'] - odds.iloc[::-1]['2.5%'].values, odds.iloc[::-1]['97.5%'].values - odds.iloc[::-1]['Odds Ratio']]
plt.errorbar(x=odds.iloc[::-1]['Odds Ratio'], y=odds.iloc[::-1].index.values, xerr=ci,
            color='black',  capsize=3, linestyle='None', linewidth=1,
            marker="o", markersize=5, mfc="black", mec="black")
plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.xlabel('Odds Ratio and 95% Confidence Interval', fontsize=8)
plt.tight_layout()
plt.savefig('forest_plot.png', dpi=300)
plt.show()

# ===== CELL 86 (markdown) =====
## MODEL TRAIN + VALIDATION

# ===== CELL 87 (markdown) =====
###SVC

# ===== CELL 88 (code) =====
from sklearn.svm import SVC
from sklearn import metrics

model = SVC(class_weight={0: 1, 1: 1.4})
m = model.fit(X_train, y_train)
y_pred_svc = m.predict(X_test)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred_svc))

# ===== CELL 89 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_SVC = metrics.confusion_matrix(y_test, y_pred_svc)
print(cm_SVC)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred_svc, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_svc))

# ===== CELL 90 (markdown) =====
###LGB

# ===== CELL 91 (code) =====
import lightgbm as lgb

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

lgb = lgb.LGBMClassifier(**params)
y_pred = lgb.fit(X_train, y_train).predict(X_test)

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


# ===== CELL 92 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_LGB = metrics.confusion_matrix(y_test, y_pred)
print(cm_LGB)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 93 (markdown) =====
### NAIVE BAYES

# ===== CELL 94 (code) =====
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))



# ===== CELL 95 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_NB = metrics.confusion_matrix(y_test, y_pred)
print(cm_NB)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 96 (markdown) =====
### DECISION TREE

# ===== CELL 97 (code) =====
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=21)
clf = decision_tree.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# ===== CELL 98 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_DT = metrics.confusion_matrix(y_test, y_pred)
print(cm_DT)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 99 (markdown) =====
###Random Forest with oob evaluation

# ===== CELL 100 (code) =====
from sklearn.ensemble import RandomForestClassifier

param = {'n_estimators': 600,
 'min_samples_split': 2,
 'min_samples_leaf': 1,
 'max_features': 'log2',
 'max_depth': 780,
 'criterion': 'gini'}

# creating a RF classifier
clf = RandomForestClassifier(oob_score=True)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred_rf = clf.predict(X_test)
# metrics are used to find accuracy or error
from sklearn import metrics
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred_rf))

# ===== CELL 101 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_RF = metrics.confusion_matrix(y_test, y_pred_rf)
print(cm_RF)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred_rf, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_rf))

# ===== CELL 102 (code) =====
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
skfold = StratifiedKFold(n_splits=20)
clf = DecisionTreeClassifier()
results = cross_val_score(clf, X_train, y_train, cv = skfold, scoring = 'f1_macro')
print(results)
print()
print ("F1_Macro = ", np.mean(results), "+/-", np.std(results))

results = cross_val_score(clf, X_train, y_train, cv = skfold, scoring = 'accuracy')
print("\n" , results)
print()
print ("Accuracy = ", np.mean(results), "+/-", np.std(results))

# ===== CELL 103 (markdown) =====
### RANDOM FOREST

# ===== CELL 104 (code) =====
from sklearn.ensemble import RandomForestClassifier

param = {'n_estimators': 600,
 'min_samples_split': 2,
 'min_samples_leaf': 1,
 'max_features': 'log2',
 'max_depth': 780,
 'criterion': 'gini'}

# creating a RF classifier
clf = RandomForestClassifier(class_weight={0: 10, 1: .1})

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)
# metrics are used to find accuracy or error
from sklearn import metrics
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# ===== CELL 105 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
print(metrics.confusion_matrix(y_test, y_pred))
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 106 (code) =====
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
skfold = StratifiedKFold(n_splits=20)
clf = DecisionTreeClassifier()
results = cross_val_score(clf, X_train, y_train, cv = skfold, scoring = 'f1_macro')
print(results)
print()
print ("F1_Macro = ", np.mean(results), "+/-", np.std(results))

results = cross_val_score(clf, X_train, y_train, cv = skfold, scoring = 'accuracy')
print("\n" , results)
print()
print ("Accuracy = ", np.mean(results), "+/-", np.std(results))

# ===== CELL 107 (markdown) =====
### GRADIENT BOOSTING

# ===== CELL 108 (code) =====
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1,
                                 max_depth=1, random_state=21).fit(X_train, y_train)

y_pred = clf.predict(X_test)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# ===== CELL 109 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_GB = metrics.confusion_matrix(y_test, y_pred)
print(cm_GB)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 110 (markdown) =====
### ADA BOOST

# ===== CELL 111 (code) =====

#clf = RandomForestClassifier(n_estimators = 600, max_features = 'log2', max_depth= 780,random_state=20, criterion = 'gini')
clf = DecisionTreeClassifier(max_depth=1)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(base_estimator=clf, n_estimators=600, random_state=21, learning_rate=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# ===== CELL 112 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_AB = metrics.confusion_matrix(y_test, y_pred)
print(cm_AB)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 113 (markdown) =====
### Logistic Regression

# ===== CELL 114 (code) =====
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver = 'liblinear', penalty = 'l2')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


# ===== CELL 115 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_LR = metrics.confusion_matrix(y_test, y_pred)
print(cm_LR)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 116 (markdown) =====
### XG BOOST

# ===== CELL 117 (code) =====
## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
 "booster" : ['gbtree', 'gblinear' ,'dart']

}

# ===== CELL 118 (code) =====
## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost

# ===== CELL 119 (code) =====
classifier=xgboost.XGBClassifier()

# ===== CELL 120 (code) =====

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# ===== CELL 121 (code) =====
classifier=xgboost.XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

# ===== CELL 122 (code) =====
from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_train,y_train)
timer(start_time) # timing ends here for "start_time" variable

# ===== CELL 123 (code) =====

random_search.best_estimator_

# ===== CELL 124 (code) =====
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.3, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.1, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=12, max_leaves=0, min_child_weight=1,
               monotone_constraints='()', n_estimators=600,
              n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=21,
              reg_alpha=0, reg_lambda=1)

# ===== CELL 125 (code) =====
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# ===== CELL 126 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_XGB = metrics.confusion_matrix(y_test, y_pred)
print(cm_XGB)
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 127 (markdown) =====
###SVC standard scalar

# ===== CELL 128 (code) =====
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# ===== CELL 129 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
print(metrics.confusion_matrix(y_test, y_pred))
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 130 (markdown) =====
###Kneighbors

# ===== CELL 131 (code) =====
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# ===== CELL 132 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
print(metrics.confusion_matrix(y_test, y_pred))
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 133 (markdown) =====
###Bagging Classifier


# ===== CELL 134 (code) =====
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
clf = BaggingClassifier(base_estimator=SVC(),
                         n_estimators=10, random_state=21, n_jobs = -1).fit(X_train, y_train)
clf.predict(X_test)
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# ===== CELL 135 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
print(metrics.confusion_matrix(y_test, y_pred))
print()
print("Classification Report")
print(metrics.classification_report(y_test, y_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))

# ===== CELL 136 (markdown) =====
## TRAINSET PERFORMANCE

# ===== CELL 137 (code) =====
y_train_pred = clf.predict(X_train)

# ===== CELL 138 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
print(metrics.confusion_matrix(y_train, y_train_pred))
print()

print("Classification Report")
print(metrics.classification_report(y_train, y_train_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train, y_train_pred))

# ===== CELL 139 (markdown) =====
## CUSTOM ENSEMBLE

# ===== CELL 140 (markdown) =====
#### TESTSET PERFORMANCE

# ===== CELL 141 (code) =====
## Ensemble of Models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

estimator = []
#estimator.append(('GNB', GaussianNB()))
estimator.append(('RF', RandomForestClassifier(class_weight={0: 5, 1: .1}, n_estimators = 400, random_state=21) ))
#estimator.append(('RF', RandomForestClassifier(oob_score=True)  ))
estimator.append(('gb', GradientBoostingClassifier()))
estimator.append(('lgb', lgb))

#estimator.append(('dt', DecisionTreeClassifier(random_state=0, max_depth=2)) )
#estimator.append(('cnb', ComplementNB()) )
#estimator.append(('xgb', xgboost.XGBClassifier()) )
#estimator.append(('AB', AdaBoostClassifier()) )
#estimator.append(('lr', LogisticRegression(solver = 'liblinear')) )

hard_voting = VotingClassifier(estimators = estimator, voting ='soft', n_jobs = -1)
x = hard_voting.fit(X_train, y_train)
y_pred_ensemble = hard_voting.predict(X_test)

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred_ensemble))


# ===== CELL 142 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
cm_ensemble = metrics.confusion_matrix(y_test, y_pred_ensemble)
print(cm_ensemble)
print()

print("Classification Report")
print(metrics.classification_report(y_test, y_pred_ensemble, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_ensemble))

# ===== CELL 143 (markdown) =====
## Stack Bar Plot for Error Analysis

# ===== CELL 144 (code) =====
import matplotlib.pyplot as plt
import seaborn as sns

# Load the confusion matrices
confusion_matrices = [
    cm_SVC,
    cm_LGB,
    cm_NB,
    cm_DT,
    cm_RF,
    cm_GB,
    cm_AB,
    cm_LR,
    cm_LGB,
    cm_ensemble
]

# Create a list of the model names
model_names = [
    "SVC",
    "LGB",
    "NB",
    "DT",
    "RF",
    "GB",
    "AB",
    "LR",
    "XGB",
    "Ensemble",
]

# Assuming you have stored the confusion matrices in variables cm1, cm2, ..., cm12

# Extract false positives and false negatives from each confusion matrix
fp = [cm[0][1] for i, cm in enumerate(confusion_matrices)]
fn = [cm[1][0] for i, cm in enumerate(confusion_matrices)]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot stacked bars for false positives and false negatives
ax.bar(model_names, fp, label='False Positives', color='lightblue')
ax.bar(model_names, fn, bottom=fp, label='False Negatives', color='lightcoral')

# Rotate x-axis tick labels for better visibility (optional)
plt.xticks(rotation=45, ha='right')

# Add labels, title, and legend
ax.set_xlabel('Model')
ax.set_ylabel('Count')
ax.legend()

# Adjust the layout and save the plot
plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300)

# Show the plot (optional)
plt.show()

# ===== CELL 145 (markdown) =====
### Significance Testing

# ===== CELL 146 (code) =====
from scipy.stats import wilcoxon

# Assuming you have predictions for two models: model1_predictions and model2_predictions

# Calculate the differences between the predicted values and the actual values
model1_diff = y_pred_svc - y_test
model2_diff = y_pred_rf - y_test
model3_diff = y_pred_ensemble - y_test

# Perform the Wilcoxon signed-rank test
statistic1, p_value1 = wilcoxon(model1_diff, model2_diff)

# Perform the Wilcoxon signed-rank test
statistic2, p_value2 = wilcoxon(model2_diff, model3_diff)

# Print the test statistic and p-value
print("Wilcoxon signed-rank test results:")
print(f"Test statistic: {statistic1}")
print(f"P-value: {p_value1}")

# Print the test statistic and p-value
print("Wilcoxon signed-rank test results:")
print(f"Test statistic: {statistic2}")
print(f"P-value: {p_value2}")

# Compare the p-value to the significance level
alpha = 0.00000005  # Chosen significance level
if p_value1 < alpha:
    print("There is a significant difference between the two models.")
else:
    print("There is no significant difference between the two models.")

if p_value2 < alpha:
    print("There is a significant difference between the two models.")
else:
    print("There is no significant difference between the two models.")


# ===== CELL 147 (markdown) =====
##Realiability diagram / Calibration Curve
is a graphical representation of the relationship between predicted probabilities and observed outcomes in a binary classification problem
- The calibration curve plots the mean predicted probability for each bin against the true fraction of positive class instances in that bin. The ideal calibration curve is a diagonal line, indicating perfect calibration where the predicted probabilities are well-calibrated and correspond closely to the actual probabilities.
-it helps to assess the reliability of the classifier's probability estimates and can be useful in applications where accurate probability estimation is important

# ===== CELL 148 (code) =====
import matplotlib.pyplot as plt
import scikitplot as skplt

# List of classifiers and their names
classifiers = [
    (GradientBoostingClassifier(), 'Gradient Boost'),
    (AdaBoostClassifier(), 'Ada Boost'),
    (LogisticRegression(), 'Logistic Regression'),
    (RandomForestClassifier(), 'Random Forest'),
    (xgboost.XGBClassifier(), 'XGBoost'),
    (DecisionTreeClassifier(), 'Decision Tree')
    #(VotingClassifier(estimators = estimator), 'Ensemble')
]

# Prepare empty lists for probabilities and classifier names
probas_list = []
clf_names = []

# Loop over classifiers and compute probabilities
for clf, clf_name in classifiers:
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    probas_list.append(proba)
    clf_names.append(clf_name)

# Plot calibration curves for each classifier
skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, n_bins=10, figsize=(6, 6))

plt.tight_layout()  # Adjust subplot spacing
plt.show()
plt.savefig("calibration_curve.png")

# ===== CELL 149 (markdown) =====
###Ensemble (calibration & learning curve)

# ===== CELL 150 (code) =====
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import scikitplot as skplt

estimator = []
#estimator.append(('GNB', GaussianNB()))
estimator.append(('RF', RandomForestClassifier(class_weight={0: 5, 1: .1}, n_estimators = 400, random_state=20) ))
#estimator.append(('RF', RandomForestClassifier(oob_score=True)  ))
estimator.append(('gb', GradientBoostingClassifier()))
estimator.append(('lgb', lgb))

#estimator.append(('dt', DecisionTreeClassifier(random_state=0, max_depth=2)) )
#estimator.append(('cnb', ComplementNB()) )
#estimator.append(('xgb', xgboost.XGBClassifier()) )
#estimator.append(('AB', AdaBoostClassifier()) )
#estimator.append(('lr', LogisticRegression(solver = 'liblinear')) )

# define classifier
clf = VotingClassifier(estimators = estimator, voting='soft')

# generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# plot the learning curve
skplt.estimators.plot_learning_curve(clf, X_train, y_train, cv=5, n_jobs=-1)

plt.show()
plt.savefig("learning_curve_ensemble.png")

# ===== CELL 151 (code) =====
import matplotlib.pyplot as plt
import scikitplot as skplt

# List of classifiers and their names
classifiers = [
    (GradientBoostingClassifier(), 'Gradient Boost'),
    (AdaBoostClassifier(), 'Ada Boost'),
    (LogisticRegression(), 'Logistic Regression'),
    (RandomForestClassifier(), 'Random Forest'),
    (xgboost.XGBClassifier(), 'XGBoost'),
    (DecisionTreeClassifier(), 'Decision Tree'),
    (VotingClassifier(estimators = estimator, voting='soft'), 'Ensemble')
]

# Prepare empty lists for probabilities and classifier names
probas_list = []
clf_names = []

# Loop over classifiers and compute probabilities
for clf, clf_name in classifiers:
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    probas_list.append(proba)
    clf_names.append(clf_name)

# Plot calibration curves for each classifier
skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, n_bins=10, figsize=(6, 6))

plt.tight_layout()  # Adjust subplot spacing
plt.show()
plt.savefig("calibration_curve_ensemble.png")

# ===== CELL 152 (markdown) =====
TRAINSET PERFORMANCE

# ===== CELL 153 (code) =====
y_train_pred = hard_voting.predict(X_train)

# ===== CELL 154 (code) =====
from sklearn import metrics
print("Confusion Matrix for the Test Case")
print(metrics.confusion_matrix(y_train, y_train_pred))
print()

print("Classification Report")
print(metrics.classification_report(y_train, y_train_pred, digits = 3))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train, y_train_pred))

# ===== CELL 155 (markdown) =====
# EX AI

# ===== CELL 156 (code) =====
!pip install lime

# ===== CELL 157 (code) =====
import lime

#we're importing lime_tabular to work with tabular data. Lime also supports other data like images etc.
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Content', 'Discontent'],
    mode='classification'
)

# ===== CELL 158 (code) =====
y_test.head(50)

# ===== CELL 159 (markdown) =====
### Case -1

# ===== CELL 160 (code) =====
X_test.iloc[25], y_test.iloc[25]

# ===== CELL 161 (code) =====
#why does this instance in 2nd row belong to Discontent class?

exp = explainer.explain_instance(
    data_row=X_test.iloc[25],
    predict_fn=hard_voting.predict_proba, num_features = 20
)

exp.show_in_notebook(show_table=True)

# ===== CELL 162 (code) =====
plt = exp.as_pyplot_figure()
plt.set_figwidth(10)
plt.set_figheight(6)
plt.tight_layout()
plt.savefig("case-1.png")
#save_fig("case-1.png")

# ===== CELL 163 (markdown) =====
### Case-2

# ===== CELL 164 (code) =====
X_test.iloc[-50], y_test.iloc[-50]

# ===== CELL 165 (code) =====
#why does this instance in 10th row belong to Content class?

exp = explainer.explain_instance(
    data_row=X_test.iloc[-50],
    predict_fn=hard_voting.predict_proba, num_features = 29
)

exp.show_in_notebook(show_table=True)

# ===== CELL 166 (code) =====
plt = exp.as_pyplot_figure()
plt.set_figwidth(10)
plt.set_figheight(6)
plt.tight_layout()
plt.savefig("case-2.png")

# ===== CELL 167 (markdown) =====
## Save Best Model

# ===== CELL 168 (code) =====
!pip install joblib
import joblib

# ===== CELL 169 (code) =====
# Save the model to a file using joblib
joblib.dump(lgb, 'best_model.pkl')

# ===== CELL 170 (code) =====
from google.colab import files

# Download the model file
files.download('best_model.pkl')

# ===== CELL 171 (markdown) =====
## Deploy

# ===== CELL 172 (code) =====
!pip install joblib
import joblib

# ===== CELL 173 (code) =====
loaded_model = joblib.load('best_model.pkl')

# ===== CELL 174 (code) =====
loaded_model

# ===== CELL 175 (code) =====
!pip install gradio

# ===== CELL 176 (code) =====
import gradio as gr
import pandas as pd

# Define the questionnaire questions and specify the type of input for each question.
questions = [
    "What is your age?",
    "How would you rate your health generally? (0-10)",
    "Do you suffer from a long-term physical health problem or disability? (0-1)",
    "How often are you depressed? (0-10)",
    "Do you see yourself as a person who is relaxed and handles stress well? (0-1)",
    "Do you see yourself as a person who can be tensed? (0-1)",
    "Do you see yourself as a person who worries a lot? (0-1)",
    "Do you see yourself as a person who is emotionally stable, and not easily excited? (0-1)",
    "Do you see yourself as a person who does not give up until the task is completed? (0-1)",
    "Do you see yourself as a person who prepares plans and implements them? (0-1)",
    "Do you see yourself as a person who gets nervous easily? (0-1)",
    "Do you see yourself as a person who is easily distracted? (0-1)",
    "How tall are you? (number of centimeters)",
    "How much do you weigh? (kilos)",
    "In the past year, have you consulted other practitioners or therapists? (0-1)",
    "Who do you primarily talk to about personal and serious problems?(0-9)",
    "Do you hold a job? (0-1)",
    "On a scale from 0-10, how content are you generally with your job?",
    "Do you have a spouse/partner? (0-1)",
    "In the past year, how often have you spent time with other relatives? (0-10)",
    "In the past year, how often have you spent time with acquaintances? (0-10)",
    "In the past year, how often have you been to the cinema, a concert, or the theater? (0-10)",
    "Not counting online newspapers, how often have you read a newspaper in the past year? (0-10)",
    "In the past year, how often have you been abroad on holiday or family visit? (0-10)",
    "What is your primary source of income, when considering all your sources of income?(0-1)"
    "In the past year, how much have you spent on medicine, nutritional supplements?",
    "How would you rate your current finances? (0-10)"
]

def pred(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27):

    # Create a dictionary from the provided parameters
    data = {
        'age': [a1],
        'A2': [a2],
        'C1': [a3],
        'D2': [a4],
        'D4': [a5],
        'D6': [a6],
        'D8': [a7],
        'D10': [a8],
        'D11': [a9],
        'D15': [a10],
        'D16': [a11],
        'D17': [a12],
        'E1': [a13],
        'E2': [a14],
        'E5_a': [a15],
        'E17': [a16],
        'job': [a17],
        'F15': [a18],
        'G1': [a19],
        'J2': [a20],
        'J4': [a21],
        'J9': [a22],
        'J14': [a23],
        'J17': [a24],
        'M2': [a25],
        'M6': [a26],
        'M8': [a27]
    }

    df = pd.DataFrame(data)

    prediction = loaded_model.predict(df)
    return "Content" if prediction[0] == 0 else "Discontent"


demo = gr.Interface(
    fn=pred,
    inputs = [gr.Number(label="Age"),gr.Slider(0, 10, label="How would you rate your health generally? Please rate it on a scale from 0 (Very Poor) to 10 (Very Well).", step=1),gr.Slider(0, 1, label="Do you suffer from a long-term physical health problem or disability? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 10, label="How often are you depressed? Please rate it on a scale from 0 (Never) to 10(Always).", step=1),gr.Slider(0, 1, label="Do you see yourself as a person who is relaxed and handles stress well? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 1, label="Do you see yourself as a person who can be tensed? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 1, label="Do you see yourself as a person who worries a lot? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 1, label="Do you see yourself as a person who is emotionally stable, and not easily excited? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 1, label="Do you see yourself as a person who does not give up until the task is completed? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 1, label="Do you see yourself as a person who prepares plans and implements them? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 1, label="Do you see yourself as a person who gets nervous easily? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 1, label="Do you see yourself as a person who is easily distracted? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Number(label="How tall are you? (number of centimeters)"),gr.Number(label="How much do you weigh? (kilos)"),gr.Slider(0, 1, label="In the past year, have you consulted other practitioners or therapists? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 9, label="Who do you primarily talk to about personal and serious problems? Please rate your choice on a scale from 0 to 9, with 0 : 'Friends/colleagues,' 1 : 'Partner/spouse/boy-/girlfriend,' 2 : 'I don't share this with anyone,' 3 : 'Siblings,' 4 : 'Parents,' 5 : 'Others,' 7 : 'Children,' 8 : 'Staff,' and 9 : 'Other family.",step=1),gr.Slider(0, 1, label="Do you hold a job? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 10, label="On a scale from 0-10, how content are you generally with your job?", step=1),gr.Slider(0, 1, label="Do you have a spouse/partner? Please select 'No' (0) or 'Yes' (1).", step=1),gr.Slider(0, 6, label="In the past year, how often have you spent time with other relatives? Please rate your answer on a scale from 0 to 6, with 0 : 'Never,' 1 : 'Less frequently,' 2 : 'Once a month,' 3 : 'Once a week,' 4 : 'Several times a month,' 5 : 'Several times a week,' and 6: 'Daily.", step=1),gr.Slider(0, 6, label="In the past year, how often have you spent time with acquaintances? Please rate your answer on a scale from 0 to 6, with 0 for 'Never,' 1 for 'Less frequently,' 2 for 'Once a month,' 3 for 'Once a week,' 4 for 'Several times a month,' 5 for 'Several times a week,' and 6 for 'Daily.", step=1),gr.Slider(0, 6, label="In the past year, how often have you been to the cinema, a concert, or the theater? Please rate your answer on a scale from 0 to 6, with 0 for 'Never,' 1 for 'Less frequently,' 2 for 'Once a month,' 3 for 'Once a week,' 4 for 'Several times a month,' 5 for 'Several times a week,' and 6 for 'Daily.", step=1),gr.Slider(0, 6, label="Not counting online newspapers, how often have you read a newspaper in the past year? Please rate your answer on a scale from 0 to 6, with 0 for 'Never,' 1 for 'Less frequently,' 2 for 'Once a month,' 3 for 'Once a week,' 4 for 'Several times a month,' 5 for 'Several times a week,' and 6 for 'Daily.", step=1),gr.Number(0, 30, label="In the past year, how often have you been abroad on holiday or family visit?", step=1),gr.Slider(0, 13, label="What is your primary source of income, when considering all your sources of income? Please select from the following options: 0 for 'Salary, fee income,' 1 for 'Early retirement/retirement pension,' 2 for 'Social security,' 3 for 'Other,' 4 for 'Pension schemes,' 5 for 'Self-employment income,' 6 for 'Unemployment benefits,' 7 for 'Incapacity benefit,' 8 for 'Other welfare,' 9 for 'Don't Know,' 10 for 'Trading bonds, shares, and real estate,' 11 for 'Interest income,' 12 for 'Inheritance,' and 13 for 'Black money.",step=1),gr.Number(label="In the past year, how much have you spent on medicine, nutritional supplements?", step=1),gr.Slider(0, 4, label="How would you rate your current finances? Please rate your answer on a scale from 0 to 4, with 0 representing 'Very bad,' 1 for 'Bad,' 2 for 'Average,' 3 for 'Good,' and 4 for 'Very good.", step=1)],
    outputs=[gr.Label()],
)

demo.launch(show_error=True)
