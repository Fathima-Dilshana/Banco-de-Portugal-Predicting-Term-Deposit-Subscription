# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

"""# Load Dataset"""

from google.colab import drive
drive.mount('/content/drive')

bank = pd.read_csv('/content/drive/MyDrive/Bank.csv', delimiter=';')
bank.head()

"""# Data Cleaning"""

bank.describe().T

bank.info()

bank.columns

print(bank.isnull().sum())
print(bank.isna().sum())

# Check for duplicates
print(f'Number of duplicates: {bank.duplicated().sum()}')

# Remove duplicates
bank.drop_duplicates(inplace=True)

# Verify duplicates are removed
print(f'Number of duplicates after removal: {bank.duplicated().sum()}')

# Check for missing values
print(bank.isnull().sum())

# Fill missing values with mode for categorical columns
for column in bank.select_dtypes(include=['object']).columns:
    bank[column].fillna(bank[column].mode()[0], inplace=True)

# Fill missing values with median for numerical columns
for column in bank.select_dtypes(include=['number']).columns:
    bank[column].fillna(bank[column].median(), inplace=True)

# Verify missing values are filled
print(bank.isnull().sum())

# Download the cleaned CSV file
from google.colab import files
bank.to_csv('cleaned_bank.csv', encoding = 'utf-8-sig')
files.download('cleaned_bank.csv')

"""# The Dataset is imbalanced, thus accuracy is not the suitable evaluation metric. we build the best model and evaluate it with precision, recall or F1 score"""

scoring = 'accuracy'
bank['y'].value_counts(normalize=True)

poutcomes = bank[bank.poutcome != 'nonexistent'].poutcome.apply(lambda x: 1 if x == 'success' else 0)
coutcomes = bank.y.apply(lambda x: 1 if x == 'yes' else 0)

print('Number of records:', len(bank))
print('Success Rate (Current Campaign):', coutcomes.sum() / len(bank))
print('Success Rate (Previous Campaign):', poutcomes.sum() / len(poutcomes))

dtypes = pd.DataFrame(bank.dtypes.rename('type')).reset_index().astype('str')

"""# Excluding duration"""

dtypes = dtypes.query('index != "duration"')
numeric = dtypes[(dtypes.type.isin(['int64', 'float64'])) & (dtypes['index'] != 'duration')]['index'].values
categorical = dtypes[~(dtypes['index'].isin(numeric)) & (dtypes['index'] != 'y')]['index'].values

print('Numeric:\n', numeric,end='\n\n')
print('Categorical:\n', categorical)

"""# Converting dependent variable categorical to dummy"""

y = pd.get_dummies(bank['y'], columns = ['y'], prefix = ['y'], drop_first = True)
bank.head()

"""# Bank client data Analysis and Categorical Treatment"""

bank_client = bank.iloc[: , 0:7]
bank_client.head()

"""## Knowing the categorical variables"""

print('Jobs:\n', bank_client['job'].unique())

print('Marital:\n', bank_client['marital'].unique())

print('Education:\n', bank_client['education'].unique())

print('Default:\n', bank_client['default'].unique())
print('Housing:\n', bank_client['housing'].unique())
print('Loan:\n', bank_client['loan'].unique())

"""##  Age"""

#Trying to find some strange values or null values
print('Min age: ', bank_client['age'].max())
print('Max age: ', bank_client['age'].min())
print('Null Values: ', bank_client['age'].isnull().any())

"""# Visualization"""

fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'age', data = bank_client)
ax.set_xlabel('Age', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Count Distribution', fontsize=15)
sns.despine()

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'age', data = bank_client, orient = 'v', ax = ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.distplot(bank_client['age'], ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Age', fontsize=15)
ax2.set_ylabel('Occurence', fontsize=15)
ax2.set_title('Age x Ocucurence', fontsize=15)
ax2.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()

"""# Calculate the outliers using IQR Method:   """

print('Ages above: ', bank_client['age'].quantile(q = 0.75) +
                      1.5*(bank_client['age'].quantile(q = 0.75) - bank_client['age'].quantile(q = 0.25)), 'are outliers')

print('Numerber of outliers: ', bank_client[bank_client['age'] > 69.6]['age'].count())
print('Number of clients: ', len(bank_client))
print('Outliers are:', round(bank_client[bank_client['age'] > 69.6]['age'].count()*100/len(bank_client),2), '%')

# Calculating some values to evaluete this independent variable
print('MEAN:', round(bank_client['age'].mean(), 1))
# A low standard deviation indicates that the data points tend to be close to the mean or expected value
# A high standard deviation indicates that the data points are scattered
print('STD :', round(bank_client['age'].std(), 1))
# I thing the best way to give a precisly insight abou dispersion is using the CV (coefficient variation) (STD/MEAN)*100
#    cv < 15%, low dispersion
#    cv > 30%, high dispersion
print('CV  :',round(bank_client['age'].std()*100/bank_client['age'].mean(), 1), ', High middle dispersion')

"""Conclusion about AGE, due to almost high dispersion and just looking at this this graph we cannot conclude if age have a high effect to our variable y, need to keep searching for some pattern. high middle dispersion means we have people with all ages and maybe all of them can subscript a term deposit, or not.
The outliers was calculated, so we go with fitting the model with and without them

##  JOBS
"""

# What kind of jobs clients this bank have, if you cross jobs with default, loan or housing, there is no relation
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'job', data = bank_client)
ax.set_xlabel('Job', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)
sns.despine()

"""## MARITAL"""

# What kind of 'marital clients' this bank have, if you cross marital with default, loan or housing, there is no relation
fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
sns.countplot(x = 'marital', data = bank_client)
ax.set_xlabel('Marital', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)
sns.despine()

"""## EDUCATION"""

# What kind of 'education clients this bank have, if you cross education with default, loan or housing, there is no relation
fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
sns.countplot(x = 'education', data = bank_client)
ax.set_xlabel('Education', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Education Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)
sns.despine()

"""##  DEFAULT, HOUSING, LOAN"""

# Default, has credit in default ?
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
sns.countplot(x = 'default', data = bank_client, ax = ax1, order = ['no', 'unknown', 'yes'])
ax1.set_title('Default', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.tick_params(labelsize=15)

# Housing, has housing loan ?
sns.countplot(x = 'housing', data = bank_client, ax = ax2, order = ['no', 'unknown', 'yes'])
ax2.set_title('Housing', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)

# Loan, has personal loan ?
sns.countplot(x = 'loan', data = bank_client, ax = ax3, order = ['no', 'unknown', 'yes'])
ax3.set_title('Loan', fontsize=15)
ax3.set_xlabel('')
ax3.set_ylabel('Count', fontsize=15)
ax3.tick_params(labelsize=15)

plt.subplots_adjust(wspace=0.25)

print('Default:\n No credit in default:'     , bank_client[bank_client['default'] == 'no']     ['age'].count(),
              '\n Unknown credit in default:', bank_client[bank_client['default'] == 'unknown']['age'].count(),
              '\n Yes to credit in default:' , bank_client[bank_client['default'] == 'yes']    ['age'].count())

print('Housing:\n No housing in loan:'     , bank_client[bank_client['housing'] == 'no']     ['age'].count(),
              '\n Unknown housing in loan:', bank_client[bank_client['housing'] == 'unknown']['age'].count(),
              '\n Yes to housing in loan:' , bank_client[bank_client['housing'] == 'yes']    ['age'].count())

print('Housing:\n No to personal loan:'     , bank_client[bank_client['loan'] == 'no']     ['age'].count(),
              '\n Unknown to personal loan:', bank_client[bank_client['loan'] == 'unknown']['age'].count(),
              '\n Yes to personal loan:'    , bank_client[bank_client['loan'] == 'yes']    ['age'].count())

"""### BANK CLIENTS CONCLUSION
Jobs, Marital and Education i think the best analysis is just the count of each variable, if we relate with the other ones its is not conclusive, all this kind of  variables has yes, unknown and no for loan, default and housing.

Default, loan and housing, its just to see the distribution of people.

##  Bank Client Categorical Treatment
- Jobs, Marital, Education, Default, Housing, Loan. Converting to continuous due the feature scaling will be apllyed later
"""

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
bank_client['job']      = labelencoder_X.fit_transform(bank_client['job'])
bank_client['marital']  = labelencoder_X.fit_transform(bank_client['marital'])
bank_client['education']= labelencoder_X.fit_transform(bank_client['education'])
bank_client['default']  = labelencoder_X.fit_transform(bank_client['default'])
bank_client['housing']  = labelencoder_X.fit_transform(bank_client['housing'])
bank_client['loan']     = labelencoder_X.fit_transform(bank_client['loan'])

#function to creat group of ages, this helps because we have 78 differente values here
def age(dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4

    return dataframe

age(bank_client);

bank_client.head()

print(bank_client.shape)
bank_client.head()

"""# Related with the last contact of the current campaign"""

bank_related = bank.iloc[: , 7:11]
bank_related.head()

bank_related.isnull().any()

print("Kind of Contact: \n", bank_related['contact'].unique())
print("\nWhich monthis this campaing work: \n", bank_related['month'].unique())
print("\nWhich days of week this campaing work: \n", bank_related['day_of_week'].unique())

"""## Duration"""

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'duration', data = bank_related, orient = 'v', ax = ax1)
ax1.set_xlabel('Calls', fontsize=10)
ax1.set_ylabel('Duration', fontsize=10)
ax1.set_title('Calls Distribution', fontsize=10)
ax1.tick_params(labelsize=10)

sns.distplot(bank_related['duration'], ax = ax2)
sns.despine(ax = ax2)
ax2.set_xlabel('Duration Calls', fontsize=10)
ax2.set_ylabel('Occurence', fontsize=10)
ax2.set_title('Duration x Ocucurence', fontsize=10)
ax2.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout()

"""##### PLease note: duration is different from age, Age has 78  values and Duration has 1544 different values"""

print("Max duration  call in minutes:  ", round((bank_related['duration'].max()/60),1))
print("Min duration  call in minutes:   ", round((bank_related['duration'].min()/60),1))
print("Mean duration call in minutes:   ", round((bank_related['duration'].mean()/60),1))
print("STD duration  call in minutes:   ", round((bank_related['duration'].std()/60),1))
# Std close to the mean means that the data values are close to the mean

"""# Calculation of outlier for duration"""

print('Duration calls above: ', bank_related['duration'].quantile(q = 0.75) +
                      1.5*(bank_related['duration'].quantile(q = 0.75) - bank_related['duration'].quantile(q = 0.25)), 'are outliers')

print('Numerber of outliers: ', bank_related[bank_related['duration'] > 644.5]['duration'].count())
print('Number of clients: ', len(bank_related))
#Outliers in %
print('Outliers are:', round(bank_related[bank_related['duration'] > 644.5]['duration'].count()*100/len(bank_related),2), '%')

"""### if the call duration is iqual to 0, then is obviously that this person didn't subscribed."""

bank[(bank['duration'] == 0)]

"""##  Contact, Month, Day of Week"""

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,6))
sns.countplot(bank_related['contact'], ax = ax1)
ax1.set_xlabel('Contact', fontsize = 10)
ax1.set_ylabel('Count', fontsize = 10)
ax1.set_title('Contact Counts')
ax1.tick_params(labelsize=10)

sns.countplot(bank_related['month'], ax = ax2, order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
ax2.set_xlabel('Months', fontsize = 10)
ax2.set_ylabel('')
ax2.set_title('Months Counts')
ax2.tick_params(labelsize=10)

sns.countplot(bank_related['day_of_week'], ax = ax3)
ax3.set_xlabel('Day of Week', fontsize = 10)
ax3.set_ylabel('')
ax3.set_title('Day of Week Counts')
ax3.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.25)

print('Ages above: ', bank_related['duration'].quantile(q = 0.75) +
                      1.5*(bank_related['duration'].quantile(q = 0.75) - bank_related['duration'].quantile(q = 0.25)), 'are outliers')

bank_related[bank_related['duration'] > 640].count()

"""## Contact, Month, Day of Week treatment"""

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
bank_related['contact']     = labelencoder_X.fit_transform(bank_related['contact'])
bank_related['month']       = labelencoder_X.fit_transform(bank_related['month'])
bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week'])

bank_related.head()

"""## Converting the Duration into minutes"""

def duration(data):

    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']   = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration']   = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data
duration(bank_related);

bank_related.head()

"""# Social and economic context attributes"""

bank_se = bank.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
bank_se.head()

"""# Other attributes"""

bank_o = bank.loc[: , ['campaign', 'pdays','previous', 'poutcome']]
bank_o.head()

bank_o['poutcome'].unique()

bank_o['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)

"""# Model"""

bank_final= pd.concat([bank_client, bank_related, bank_se, bank_o], axis = 1)
bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx',
                     'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]
bank_final.shape

from sklearn.ensemble import RandomForestClassifier

"""### Feature Importance using Random Forest"""

pd.DataFrame(data = RandomForestClassifier().fit(bank_final,y).feature_importances_,index=bank_final.columns
             ,columns=['Feature Importance']).plot.barh();

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.2, random_state = 101)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

X_train.head()

from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.metrics import cohen_kappa_score

"""# LogisticRegression"""

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
logpred = logmodel.predict(X_test)
print('The Confusion Matrix \n',+confusion_matrix(y_test, logpred),end='\n\n')
print('Accuracy : ',+round(accuracy_score(y_test, logpred),2)*100)
print('Cohen kappa : ',+round(cohen_kappa_score(y_test, logpred),2))
LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=k_fold, n_jobs=1, scoring = scoring)).mean()

"""# KNeighborsClassifier"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knnpred = knn.predict(X_test)

print(confusion_matrix(y_test, knnpred))
print(round(accuracy_score(y_test, knnpred),2)*100)
print('Cohen kappa : ',+round(cohen_kappa_score(y_test, knnpred),2))
KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring = scoring).mean())

"""# SVC"""

from sklearn.svm import SVC
svc= SVC(kernel = 'sigmoid')
svc.fit(X_train, y_train)
svcpred = svc.predict(X_test)
print(confusion_matrix(y_test, svcpred))
print(round(accuracy_score(y_test, svcpred),2)*100)
print('Cohen kappa : ',+round(cohen_kappa_score(y_test, svcpred),2))
SVCCV = (cross_val_score(svc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = scoring).mean())

"""# DecisionTreeClassifier"""

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train, y_train)
dtreepred = dtree.predict(X_test)

print(confusion_matrix(y_test, dtreepred))
print(round(accuracy_score(y_test, dtreepred),2)*100)
print('Cohen kappa : ',+round(cohen_kappa_score(y_test, dtreepred),2))
DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = scoring).mean())

"""# RandomForestClassifier"""

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
rfcpred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfcpred ))
print(round(accuracy_score(y_test, rfcpred),2)*100)
print('Cohen kappa : ',+round(cohen_kappa_score(y_test, rfcpred),2))
RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = scoring).mean())

"""# GaussianNB"""

from sklearn.naive_bayes import GaussianNB
gaussiannb= GaussianNB()
gaussiannb.fit(X_train, y_train)
gaussiannbpred = gaussiannb.predict(X_test)
probs = gaussiannb.predict(X_test)

print(confusion_matrix(y_test, gaussiannbpred ))
print(round(accuracy_score(y_test, gaussiannbpred),2)*100)
print('Cohen kappa : ',+round(cohen_kappa_score(y_test, gaussiannbpred),2))
GAUSIAN = (cross_val_score(gaussiannb, X_train, y_train, cv=k_fold, n_jobs=1, scoring = scoring).mean())

"""# XGBClassifier"""

from xgboost import XGBClassifier
xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)

print(confusion_matrix(y_test, xgbprd ))
print(round(accuracy_score(y_test, xgbprd),2)*100)
print('Cohen kappa : ',+round(cohen_kappa_score(y_test, xgbprd),2))
XGB = (cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10).mean())

"""# GradientBoostingClassifier"""

from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
gbkpred = gbk.predict(X_test)
print(confusion_matrix(y_test, gbkpred ))
print(round(accuracy_score(y_test, gbkpred),2)*100)
print('Cohen kappa : ',+round(cohen_kappa_score(y_test, gbkpred),2))
GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring = scoring).mean())

models = pd.DataFrame({
                'Models': ['Random Forest Classifier', 'Decision Tree Classifier', 'Support Vector Machine',
                           'K-Near Neighbors', 'Logistic Model', 'Gausian NB', 'XGBoost', 'Gradient Boosting'],
                'Score':  [RFCCV, DTREECV, SVCCV, KNNCV, LOGCV, GAUSIAN, XGB, GBKCV]})

models.sort_values(by='Score', ascending=False)

"""# XGBOOST ROC/ AUC , BEST MODEL"""

from sklearn import metrics
fig, (ax, ax1) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
probs = xgb.predict_proba(X_test)
preds = probs[:,1]
fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(y_test, preds)
roc_aucxgb = metrics.auc(fprxgb, tprxgb)

ax.plot(fprxgb, tprxgb, 'b', label = 'AUC = %0.2f' % roc_aucxgb)
ax.plot([0, 1], [0, 1],'r--')
ax.set_title('Receiver Operating Characteristic XGBOOST ',fontsize=10)
ax.set_ylabel('True Positive Rate',fontsize=20)
ax.set_xlabel('False Positive Rate',fontsize=15)
ax.legend(loc = 'lower right', prop={'size': 16})

#Gradient
probs = gbk.predict_proba(X_test)
preds = probs[:,1]
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, preds)
roc_aucgbk = metrics.auc(fprgbk, tprgbk)

ax1.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)
ax1.plot([0, 1], [0, 1],'r--')
ax1.set_title('Receiver Operating Characteristic GRADIENT BOOST ',fontsize=10)
ax1.set_ylabel('True Positive Rate',fontsize=20)
ax1.set_xlabel('False Positive Rate',fontsize=15)
ax1.legend(loc = 'lower right', prop={'size': 16})

plt.subplots_adjust(wspace=1)

"""# Logistic ,Random Forest ,KNN,GaussianDB, Decision Tree"""

fig, ax_arr = plt.subplots(nrows = 2, ncols = 3, figsize = (25,20))

#-------------------- LOGMODEL --------------------
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fprlog, tprlog, thresholdlog = metrics.roc_curve(y_test, preds)
roc_auclog = metrics.auc(fprlog, tprlog)

ax_arr[0,0].plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)
ax_arr[0,0].plot([0, 1], [0, 1],'r--')
ax_arr[0,0].set_title('Receiver Operating Characteristic Logistic ',fontsize=20)
ax_arr[0,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,0].legend(loc = 'lower right', prop={'size': 16})

#-------------------- RANDOM FOREST --------------------
probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

ax_arr[0,1].plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
ax_arr[0,1].plot([0, 1], [0, 1],'r--')
ax_arr[0,1].set_title('Receiver Operating Characteristic Random Forest ',fontsize=20)
ax_arr[0,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,1].legend(loc = 'lower right', prop={'size': 16})

#-------------------- KNN----------------------
probs = knn.predict_proba(X_test)
preds = probs[:,1]
fprknn, tprknn, thresholdknn = metrics.roc_curve(y_test, preds)
roc_aucknn = metrics.auc(fprknn, tprknn)

ax_arr[0,2].plot(fprknn, tprknn, 'b', label = 'AUC = %0.2f' % roc_aucknn)
ax_arr[0,2].plot([0, 1], [0, 1],'r--')
ax_arr[0,2].set_title('Receiver Operating Characteristic KNN ',fontsize=20)
ax_arr[0,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,2].legend(loc = 'lower right', prop={'size': 16})

#-------------------- DECISION TREE ---------------------
probs = dtree.predict_proba(X_test)
preds = probs[:,1]
fprdtree, tprdtree, thresholddtree = metrics.roc_curve(y_test, preds)
roc_aucdtree = metrics.auc(fprdtree, tprdtree)

ax_arr[1,0].plot(fprdtree, tprdtree, 'b', label = 'AUC = %0.2f' % roc_aucdtree)
ax_arr[1,0].plot([0, 1], [0, 1],'r--')
ax_arr[1,0].set_title('Receiver Operating Characteristic Decision Tree ',fontsize=20)
ax_arr[1,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,0].legend(loc = 'lower right', prop={'size': 16})

#-------------------- GAUSSIAN ---------------------
probs = gaussiannb.predict_proba(X_test)
preds = probs[:,1]
fprgau, tprgau, thresholdgau = metrics.roc_curve(y_test, preds)
roc_aucgau = metrics.auc(fprgau, tprgau)

ax_arr[1,1].plot(fprgau, tprgau, 'b', label = 'AUC = %0.2f' % roc_aucgau)
ax_arr[1,1].plot([0, 1], [0, 1],'r--')
ax_arr[1,1].set_title('Receiver Operating Characteristic Gaussian ',fontsize=20)
ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,1].legend(loc = 'lower right', prop={'size': 16})

#-------------------- ALL PLOTS ----------------------------------
ax_arr[1,2].plot(fprgau, tprgau, 'b', label = 'Gaussian', color='black')
ax_arr[1,2].plot(fprdtree, tprdtree, 'b', label = 'Decision Tree', color='blue')
ax_arr[1,2].plot(fprknn, tprknn, 'b', label = 'Knn', color='brown')
ax_arr[1,2].plot(fprrfc, tprrfc, 'b', label = 'Random Forest', color='green')
ax_arr[1,2].plot(fprlog, tprlog, 'b', label = 'Logistic', color='grey')
ax_arr[1,2].set_title('Receiver Operating Comparison ',fontsize=20)
ax_arr[1,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,2].legend(loc = 'lower right', prop={'size': 16})

plt.subplots_adjust(wspace=0.2)
plt.tight_layout()

"""# ANALYZING THE RESULTS

**we have to decide which one is the best model, and we have two types of wrong values:**
- False Positive, means the client do NOT SUBSCRIBED to term deposit, but the model thinks he did.
- False Negative, means the client SUBSCRIBED to term deposit, but the model said he dont.
- The first one its most harmful, because we think that we already have that client but we dont and maybe we lost him in other future campaings
- The second its not good but its ok, we have that client and in the future we'll identify that in truth he's already our client

### our objective here, is to find the best model by confusion matrix with the lowest False Positive as possible.
"""

from sklearn.metrics import classification_report
from sklearn.metrics import recall_score

"""# Confusion_Matrix And Classification_Report of KNN"""

print('KNN Confusion Matrix\n', confusion_matrix(y_test, knnpred),end='\n\n')
print('KNN Reports\n',classification_report(y_test, knnpred))

"""# Confusion_Matrix And Classification_Report of SVC"""

print('SVC Confusion Matrix\n', confusion_matrix(y_test, svcpred),end='\n\n')
print('SVC Reports\n',classification_report(y_test, svcpred))

"""# Confusion_Matrix And Classification_Report of Decision Tree"""

print('Decision Tree Confusion Matrix\n', confusion_matrix(y_test, dtreepred),end='\n\n')
print('Decision Tree Reports\n',classification_report(y_test, dtreepred))

"""# Confusion_Matrix And Classification_Report of Random Forest"""

print('Random Forest Confusion Matrix\n', confusion_matrix(y_test, rfcpred),end='\n\n')
print('Random Forest Reports\n',classification_report(y_test, rfcpred))

"""# Confusion_Matrix And Classification_Report of Gaussian"""

print('Gaussian Confusion Matrix\n', confusion_matrix(y_test, gaussiannbpred),end='\n\n')
print('Gaussian Forest Reports\n',classification_report(y_test, gaussiannbpred))

"""# Confusion_Matrix And Classification_Report of XgBoost"""

print('Xgboost Confusion Matrix\n', confusion_matrix(y_test, xgbprd),end='\n\n')
print('Xgboost Forest Reports\n',classification_report(y_test, xgbprd))

"""# Confusion_Matrix And Classification_Report of Gradient Boost"""

print('Gradient Boosting Confusion Matrix\n', confusion_matrix(y_test, gbkpred),end='\n\n')
print('Gradient Boosting Reports\n',classification_report(y_test, gbkpred))

"""# Conclusion :
After the analysis we see that our interest is over decreasing the False Negative means the client SUBSCRIBED to term deposit, but the model said he dont which indicates RECALL. So, we conclude that the model with high RECALL would be best suited for the problem statement.
"""

pd.DataFrame(data = [recall_score(y_test,logpred, average='weighted'),
recall_score(y_test,knnpred, average='weighted'),
recall_score(y_test,svcpred, average='weighted'),
recall_score(y_test,dtreepred, average='weighted'),
recall_score(y_test,rfcpred, average='weighted'),
recall_score(y_test,gaussiannbpred, average='weighted'),
recall_score(y_test,xgbprd, average='weighted'),
recall_score(y_test,gbkpred, average='weighted')],index=['Logistic','KNN','SVC','DT','RF','NB','XG','GB'],
            columns=['Recall Score']).sort_values(by='Recall Score',ascending=False)

"""# Prediction on the test data¶
In the below task, we have performed a prediction on the test data. We have used Gradient Boost for this prediction.

We have to perform the same preprocessing operations on the test data that we have performed on the train data.

We then make a prediction on the preprocessed test data using the Gradient Boost model with the best parameter values we've got.
"""

# Preprocessed Test File
test = pd.read_csv('/content/drive/MyDrive/Bank.csv',delimiter=";")
test.head()

# Assuming 'test' is your DataFrame
test.insert(0, 'Serial Number', range(1, 1 + len(test)))
test.head()

# Initialize Smote and predict GB and print serial Number and  term deposit

from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score

# Assuming X_train, y_train, X_test, y_test, and gbk are already defined

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the Gradient Boosting model with SMOTE data
gbk_smote = GradientBoostingClassifier()
gbk_smote.fit(X_train_smote, y_train_smote)

# Predict using the Gradient Boosting model with SMOTE data
gbk_smote_pred = gbk_smote.predict(X_test)

# Print serial number and term deposit prediction
print("Serial Number | Term Deposit Prediction")
# Use the length of gbk_smote_pred to iterate
for i in range(len(gbk_smote_pred)):
  # Access the correct row in test using iloc
  print(f"{test.iloc[i]['Serial Number']} | {gbk_smote_pred[i]}")

# Calculate and print the recall score
recall_smote = recall_score(y_test, gbk_smote_pred, average='weighted')
print(f"Recall Score (SMOTE): {recall_smote}")

# Print serial number and term deposit prediction for top 5
print("Serial Number | Term Deposit Prediction")
for i in range(20):
  print(f"{test.iloc[i]['Serial Number']} | {gbk_smote_pred[i]}")

# Initialize SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the Gradient Boosting model with SMOTE data
gbk_smote = GradientBoostingClassifier()
gbk_smote.fit(X_train_smote, y_train_smote)

# Predict using the Gradient Boosting model with SMOTE data
gbk_smote_pred = gbk_smote.predict(X_test)

# Get serial number input from the user
serial_number = int(input("Enter the serial number: "))

# Find the corresponding prediction
try:
  prediction = gbk_smote_pred[test[test['Serial Number'] == serial_number].index[0]]
  print(f"Term Deposit Prediction for Serial Number {serial_number}: {prediction}")
except IndexError:
  print(f"Serial Number {serial_number} not found in the dataset.")
