# %% [markdown]
# ## Case Study 1 

# %% [markdown]
# Below is a data set that represents thousands of loans made through the Lending Club platform, which is
# a platform that allows individuals to lend to other individuals.
# We would like you to perform the following using the language of your choice:
# 
# - Describe the dataset and any issues with it.
# - Generate a minimum of 5 unique visualizations using the data and write a brief description of
# your observations. Additionally, all attempts should be made to make the visualizations visually
# appealing
# - Create a feature set and create a model which predicts interest_rate using at least 2 algorithms.
# Describe any data cleansing that must be performed and analysis when examining the data.
# - Visualize the test results and propose enhancements to the model, what would you do if you
# had more time. Also describe assumptions you made and your approach.

# %% [markdown]
# Disclaimer: Someone who is a essentially a sure bet to pay back a loan will have an easier time getting a loan with a low interest rate than someone who appears to be riskier. And for people who are very risky? They may not even get a loan offer, or they may not have accepted the loan offer due to a high interest rate. It is important to keep that last part in mind, since this data set only represents loans actually made, i.e. do not mistake this data for loan applications!

# %% [markdown]
# ### Import the required libraries

# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import utils

# %%
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# %% [markdown]
# ### Loading the data

# %%
loans_full_schema = pd.read_csv('loans_full_schema.csv')

# %%
loans_full_schema.shape

# %%
loans_full_schema.info()

# %% [markdown]
# The dataset contains 10000 observations (rows) and 55 variables (columns). Most of the columns are numeric but we also have categorical variables(both ordinal and nominal) and datetime variables.

# %% [markdown]
# ### Adjust datetime `issue_month` column

# %%
loans_full_schema['issue_month'] = pd.to_datetime(loans_full_schema['issue_month'])

# %%
loans_full_schema.issue_month.info()

# %% [markdown]
# ### Checking for missing values 

# %%
loans_full_schema.isnull().sum()

# %%
missing_val_ind = loans_full_schema.isnull().sum().to_numpy().nonzero()
missing_val_ind

# %%
missing_values_count = loans_full_schema.isnull().sum().iloc[missing_val_ind]
missing_values_count.keys()

# %% [markdown]
# The number of the missing values for each feature:

# %%
missing_values_count

# %% [markdown]
# There are a high percentage of values that is missing for some features: 
# ```
# FEATURE                     % MISSING
# 'annual_income_joint'       85%,
# 'verification_income_joint' 85%, 
# 'debt_to_income_joint'      85%, 
# 'months_since_last_delinq'  57%, 
# 'months_since_90d_late'     77%'
# ```

# %% [markdown]
# About these features:
# - annual_income_joint = If this is a joint application, then the annual income of the two parties applying 
# - verification_income_joint = Type of verification of the joint income, IF joint (40 joint observations without inforamtion about the verification)
# - debt_to_income_joint = Debt-to-income ratio for the two parties, IF joint 
# - months_since_last_delinq = Months since the last delinquency (behind on payments)

# %%
loans_full_schema.application_type.value_counts()[1] / len(loans_full_schema.application_type)

# %% [markdown]
# 
# They are mostly about the observations with `application_type = joint`, which is about 15% of the total observations. So it is expected to have a lot `nan` values.

# %% [markdown]
# ### Check for duplicate rows 

# %%
print(f'Search for duplicate rows in dataset: {loans_full_schema.duplicated().sum()}')

# %% [markdown]
# There are no duplicate rows! 

# %% [markdown]
# ### And now some more details about some important features

# %%
loans_full_schema.annual_income.value_counts()

# %% [markdown]
# Range of years of earliest credit line

# %%
loans_full_schema.earliest_credit_line.min(), loans_full_schema.earliest_credit_line.max()

# %% [markdown]
# Number of current accounts that are 120 and 30 days past due: only 1 for 30 days

# %%
loans_full_schema[['num_accounts_120d_past_due', 'num_accounts_30d_past_due']].sum()

# %% [markdown]
# There are 12 different loan purposes, with the most popoular: 
# ```
# debt_consolidation    5144
# credit_card           2249
# other                  914
# ```

# %%
loans_full_schema.loan_purpose.value_counts()

# %% [markdown]
# #### Basic statistics for :
# 

# %% [markdown]
# - the annual income of the observations

# %%
loans_full_schema.annual_income.describe().round(2)

# %% [markdown]
# - the interest rates of the loans

# %%
loans_full_schema.interest_rate.describe().round(2)

# %% [markdown]
# - the amount of the loans

# %%
loans_full_schema.loan_amount.describe().round(2)

# %% [markdown]
# - debt-to-income ratio

# %%
loans_full_schema.debt_to_income.describe()

# %% [markdown]
# ### 5 Visualizations

# %% [markdown]
# It would be interesting to see the interest rate against the loan amount, annual income, grade, history of delayed payment 

# %%
plt.figure(figsize=[10,8])

binsize = 1000
bins = np.arange(loans_full_schema['loan_amount'].min(), loans_full_schema['loan_amount'].max()+binsize, binsize)
plt.hist(data = loans_full_schema, x = 'loan_amount', bins=bins, histtype='barstacked', color='skyblue', ec='blue')
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount (in thousands $)')
plt.ylabel('# of Observations')
plt.xticks([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], ['5','10', '15', '20', '25', '30', '35', '40'])
plt.show()


# %% [markdown]
# What we can say about this histogram is that the amounts that are being chosen are amounts that are round numbers.
# 
# Most common amount is 10,000 $, approximately 10% of the loans.

# %% [markdown]
# Rescale the Loan Amount data, for a more clear distribution

# %%
plt.figure(figsize=[10,8])
log_binsize = 0.025
bins = 10 ** np.arange(3, np.log10(loans_full_schema['loan_amount'].max()) + log_binsize, log_binsize)
# plt.subplot(221)
plt.hist(data = loans_full_schema, x = 'loan_amount', bins=bins, histtype='barstacked', color='skyblue', ec='blue')
plt.title('Log Scaled Loan Amount Distribution')
plt.xscale('log')
plt.xlabel('Loan Amount (in thousands $)')
plt.ylabel('# of Observations')
plt.xticks([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '15', '20', '25', '30', '35', '40'])
plt.show()


# %% [markdown]
# The highest value for a loan amount is between 10K and 20K, while high number we can see between 20K and 30K. 

# %%
plt.figure(figsize=[10,8])
plt.barh(loans_full_schema.interest_rate, loans_full_schema.annual_income, color = 'skyblue')
plt.title('Income VS Interest Rate')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Interest Rate')
plt.xticks([0, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000, 225000, 2500000],['0', '0.5','0.75', '1', '1.25', '1.5', '1.75', '2', '2.25', '2.5'])
plt.show() 


# %% [markdown]
# The lower the annual income the higher the interest rate.

# %%
plt.figure(figsize=[10,8])
# plt.subplot(222)
grades = loans_full_schema.grade.value_counts().sort_index()

plt.bar(loans_full_schema.grade, loans_full_schema.interest_rate, color = 'skyblue')
plt.title('Loan Grade VS Interest Rate')
plt.xlabel('Loan Grade')
plt.ylabel('Interest Rate')
plt.show()
 


# %% [markdown]
# Lowest Interest Rate for the best Grade A. The rate increases as the grade is getting lower.

# %%
plt.figure(figsize=[10,8])
grades = loans_full_schema.grade.value_counts().sort_index()
ticks = loans_full_schema.grade.value_counts().sort_index()
plt.bar(grades.keys(), grades)
plt.title('Loan Grade Counts')
plt.xlabel('Loan Grade')
plt.ylabel('# of Observations')
plt.xticks(['A', 'B', 'C', 'D', 'E', 'F', 'G'], ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
plt.yticks(ticks=ticks)
plt.show()


# %% [markdown]
# Most of the loans made have grade B, then C and A. This is expected since in our data we only have loans that were actually made (hence approved).

# %% [markdown]
# Some information about the borrowrers

# %%
loans_full_schema.homeownership.value_counts()

# %%
loans_full_schema.verified_income.value_counts()

# %%
states = loans_full_schema.state.value_counts().sort_values(ascending=False)
states[0:10]
others = pd.Series({'Others': states[10:].sum()})
states_topten = pd.concat([states[0:10],others])

homeownership = loans_full_schema.homeownership.value_counts()
verified = loans_full_schema.verified_income.value_counts()

plt.figure(figsize = [15,15])
plt.subplot(1,3,1)
plt.pie(states_topten, labels = states_topten.keys(), autopct='%1.1f%%' , shadow=True, startangle=90)
plt.title('# of Loan Observations in each state (Top 10)')

plt.subplot(1,3,2)
plt.pie(homeownership, labels=homeownership.keys(), autopct='%1.1f%%',startangle=90, shadow=True)
plt.title('Types of Home ownership')

plt.subplot(1,3,3)
plt.pie(verified, labels=verified.keys(), autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Types of verification of income')

plt.show()

#  THIS COULD BE A PIE CHART WITH THE STATES 
#  or a bar chart 

# %% [markdown]
# - Top 5 states that have made loans: 
#     ```
#     CA        1330 
#     TX         806
#     NY         793
#     FL         732
#     IL         382
#     ```
# - Almost 50% have a mortgage for their home, and 40% live on rent 
# - A bit more than 1/3 of observations' incomes in not verified 

# %% [markdown]
# #### Below are some general descriptions of features that I think play some important role to the outcome (approval / disapproval) of the loan. 

# %%
loans_full_schema.total_debit_limit.describe().round(2)

# %%
loans_full_schema.account_never_delinq_percent.describe().round(2)

# %%
loans_full_schema.term.value_counts()

# %% [markdown]
# The length of the loan is either 36 months (3 years) or 60 months (5 years)

# %%
loans_full_schema.loan_status.value_counts()

# %%
status = loans_full_schema.loan_status.value_counts()
status  = status.drop('Current')

plt.figure(figsize=[8,8])
plt.bar(status.keys(), status, width=5*[0.5])
plt.xticks(rotation = 30)
plt.title('Loan Status')
plt.show()

# %% [markdown]
# But with the "Current" bar being way higher than the others, we can focus on the other 4 status

# %%
status = np.array(loans_full_schema.loan_status.value_counts().keys())
status[0] = status[0] + '\n (up to 9375)'
plt.figure(figsize=[8,8])
plt.bar(status, loans_full_schema.loan_status.value_counts(), width=6*[0.5])
plt.ylim(top = 500)
plt.xticks(status, rotation = 30)
plt.title('Loan Status')
plt.show()

# %% [markdown]
# - Fully Paid only 4% 
# - Late (16-30 days) 4%
# - Late (31-120 days) 7%
# - Charged Off 0,1% 
# 

# %%
loans_full_schema.initial_listing_status.value_counts()

# %% [markdown]
# Amounts paid for 4 important categories: amount of loan paid, amount left, intest paid, late fees paid

# %%
loans_full_schema[['paid_total','paid_principal', 'paid_interest', 'paid_late_fees']].head(10)

# %% [markdown]
# ### Predicting the Interest Rate

# %% [markdown]
# #### I am going to get rid of 45 features that aren't useful for predicting the interest rate
# 

# %% [markdown]
# The features that I keep: 
# - 'annual_income',
# - 'debt_to_income',
# - 'delinq_2y',
# - 'num_historical_failed_to_pay',
# - 'loan_amount',
# - 'interest_rate',
# - 'installment',
# - 'grade', 
# - 'sub_grade',
# - 'balance',

# %%
prediction_features = loans_full_schema.drop(['emp_title', 'emp_length', 'state', 'homeownership', 'verified_income', 'annual_income_joint',
       'verification_income_joint', 'debt_to_income_joint', 'months_since_last_delinq', 'earliest_credit_line',
       'inquiries_last_12m', 'total_credit_lines', 'open_credit_lines',
       'total_credit_limit', 'total_credit_utilized',
       'num_collections_last_12m',
       'months_since_90d_late', 'current_accounts_delinq',
       'total_collection_amount_ever', 'current_installment_accounts',
       'accounts_opened_24m', 'months_since_last_credit_inquiry',
       'num_satisfactory_accounts', 'num_accounts_120d_past_due',
       'num_accounts_30d_past_due', 'num_active_debit_accounts',
       'total_debit_limit', 'num_total_cc_accounts', 'num_open_cc_accounts',
       'num_cc_carrying_balance', 'num_mort_accounts',
       'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt',
       'loan_purpose', 'application_type',  'term',
        'issue_month', 'loan_status', 'initial_listing_status', 'disbursement_method',
        'paid_total', 'paid_principal', 'paid_interest',
       'paid_late_fees'], axis=1)

prediction_features.info()

# %% [markdown]
# Grade and Sub-grade are not numeric data

# %%
loans_full_schema.grade.value_counts().sort_index()

# %%
loans_full_schema.sub_grade.value_counts().sort_index()

# %% [markdown]
# for now I am dropping the grade & subgrade columns 

# %%
prediction_features.drop([ 'grade', 'sub_grade'], axis = 1, inplace=True)


# %%
prediction_features

# %%
652.53/18.01

# %%
652.53/ 7500

# %%
prediction_features[prediction_features['debt_to_income'].isnull()]


# %%
prediction_features[prediction_features.annual_income == 0]


# %% [markdown]
# There are 24 rows missing the debt_to_income input, so I will drop these rows

# %%
prediction_features.dropna(inplace=True)

# %%
prediction_features.shape

# %% [markdown]
# Now we have 9976 observations and 8 features 

# %% [markdown]
# #### Find a model to predict interest rate

# %%
x_train = prediction_features.drop('interest_rate',axis=1)
x_train

# %% [markdown]
# Split the data needed for predictions in 60% training, 20% cross validation, 20% test

# %%
# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(prediction_features.drop('interest_rate', axis=1), prediction_features.interest_rate, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# %%
del x_, y_
print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

# %%
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# %% [markdown]
# Normalize the training data

# %%
scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(x_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(x_norm,axis=0)}")

# %% [markdown]
# Create Regression Model

# %%
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# %%
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# %%
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(x_norm)
# make a prediction using w,b. 
y_pred = np.dot(x_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# %%
x_features = prediction_features.columns
x_features

# %%
# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,7,figsize=(20,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_norm[:100,i], y_train[:100], label = 'target',)
    ax[i].set_xlabel(x_features[i])
    ax[i].scatter(x_norm[:100,i],y_pred[:100],color='red', label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# %% [markdown]
# This is not a good model to predict the interest rate. If I had more time I would try: 
# - feature engineering and polynomial regression
# - cross validation tests to see what I can modify to get a better model
# - getting rid of features that do not help with predicting the target value (interest rate)


