# %% [markdown]
# ## Case Study 2 

# %% [markdown]
# There is 1 dataset(csv) with 3 years worth of customer orders. There are 4 columns in the csv dataset:
# index, CUSTOMER_EMAIL(unique identifier as hash), Net_Revenue, and Year.
# 
# For each year we need the following information:
# - Total revenue for the current year
# - New Customer Revenue e.g. new customers not present in previous year only
# - Existing Customer Growth. To calculate this, use the Revenue of existing customers for current
# year –(minus) Revenue of existing customers from the previous year
# - Revenue lost from attrition
# - Existing Customer Revenue Current Year
# - Existing Customer Revenue Prior Year
# - Total Customers Current Year
# - Total Customers Previous Year
# - New Customers
# - Lost Customers
# 
# Additionally, generate a few unique plots highlighting some information from the dataset. Are there any
# interesting observations?

# %% [markdown]
# ### Import the required libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as mcolors

# %% [markdown]
# ### Loading the data

# %%
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# %%
orders = pd.read_csv('casestudy.csv')

# %%
orders.shape

# %% [markdown]
# There are 685,927 orders in total and 4 columns: `customer_email, net_revenue, year`

# %%
orders.head()

# %% [markdown]
# We don't need to have the index two times, so we drop the first column 

# %%
orders.drop('Unnamed: 0', axis=1, inplace=True)

# %%
orders.head()

# %% [markdown]
# We have information for 3 years: 2015, 2016, 2017

# %%
orders.net_revenue.describe().round(2)

# %% [markdown]
# ### Checking for missing values 

# %%
orders.isnull().sum()

# %% [markdown]
# There are no missing values

# %% [markdown]
# ### For each year:

# %% [markdown]
# 
# #### - Total revenue:

# %%
total_revenue = orders.groupby("year", as_index=False).sum()
total_revenue

# %% [markdown]
# #### - New Customers 
# 

# %%
def new_customer(prev, next):
    oldnew = orders[orders.year.isin({prev, next})] # customers with orders in the years prev and/or next 
    once = oldnew.drop_duplicates('customer_email',keep=False) # customers that made orders only in one of the prev/next year
    new = once[once.year == next]
    return new


# %%
new_16 = new_customer(2015,2015)
new_16.head()


# %%
new_17 = new_customer(2016,2017)
new_17.head()


# %% [markdown]
# In the `new_16` dataframe, we have all the customers that ordered only in 2016 and not in 2015. There are `145062` new customers.
# 
# In the `new_17` dataframe, we have all the customers that ordered only in 2015 and not in 2016. There are `229028` new customers.
# 

# %% [markdown]
# #### - Lost Customers

# %%
def lost_customer(prev, next):
    oldnew = orders[orders.year.isin({prev, next})] # customers with orders in the years prev and/or next 
    once = oldnew.drop_duplicates('customer_email',keep=False) # customers that made orders only in one of the prev/next year
    lost = once[once.year == prev]
    return lost


# %%

lost_15 = lost_customer(2015,2016)
lost_15


# %%
lost_16 = lost_customer(2016,2017)
lost_16


# %% [markdown]
# In the `lost_15` dataframe, we have all the customers that ordered in 2015 but did not in 2016. There are `171710 ` lost customers.
# 
# In the `lost_16` dataframe, we have all the customers that ordered in 2016 but did not in 2017. There are `183687` new customers.

# %% [markdown]
# #### - New Customer Revenue e.g. new customers not present in previous year only
# 

# %%
new_revenue_16 = new_16.net_revenue.sum().round(2)
new_revenue_17 = new_17.net_revenue.sum().round(2)

# %%
new_revenue_16, new_revenue_17

# %% [markdown]
# In the `new_revenue_16` variable, we have the total revenue of the customers that ordered only in 2016 and not in 2015. There is `29036749.19` revenue due to new customers.
# 
# In the `new_revenue_17` variable, we have the total revenue of the customers that ordered only in 2017 and not in 2016. There is `28776235.04` revenue due to new customers.

# %% [markdown]
# #### - Existing customer growth. 
# 
# revenue of existing customers for current year - revenue of existing customers for previous year

# %%
def growth(prev, next):
    oldnew = orders[orders.year.isin({prev, next})] # customers with orders in the prev or next years 
    # oldnew.shape
    duplicate =  oldnew[oldnew.duplicated('customer_email',keep=False)] # customers that have made orders in both prev and next years
    # once = oldnew.drop_duplicates('customer_email',keep=False)
    duplicate_prev = duplicate[duplicate.year == prev].set_index(duplicate[duplicate.year == prev].customer_email).sort_index(axis=1)
    duplicate_next = duplicate[duplicate.year == next].set_index(duplicate[duplicate.year == next].customer_email).sort_index(axis=1)
    growth = duplicate_next.net_revenue.subtract(duplicate_prev.net_revenue)
    return growth

# %%
growth_16 = growth(2015,2016)
growth_16.name = 'growth_for_16'
growth_17 = growth(2016,2017)
growth_17.name = 'growth_for_17'
growths = pd.concat([growth_16, growth_17], axis=1, copy=False)

# %%
merged = orders.merge(growths, how='outer',on='customer_email')

# %%
total_growth = merged[['growth_for_16', 'growth_for_17']].sum()
total_growth

# %%
growths.sum()

# %% [markdown]
# In the `growth_16` we have the difference in revenue for customers from 2015 to 2016.
# 
# In the `growth_17` we have the difference in revenue for customers from 2016 to 2017.
# 
# `merged` is the new dataframe containing two new columns `growth 16, growth 17` with the growth information for period `2015-2016` and `2016-2017` respectively

# %% [markdown]
# #### - Revenue lost from attrition

# %% [markdown]
# We focus on the customers that were lost from one year to the other

# %%
def attrition_loss(prev, next):
    prev_rev = orders[orders.year == prev].net_revenue.sum()
    oldnew = orders[orders.year.isin({prev, next})] # customers with orders in the prev or next years 
    duplicate =  oldnew[oldnew.duplicated('customer_email',keep=False)] # customers that have made orders in both prev and next years
    stable_rev = duplicate[duplicate.year == next].net_revenue.sum()
    loss = (prev_rev - stable_rev).round(2)
    return loss 


# %%
loss_16 = attrition_loss(2015,2016)
loss_16

# %%
loss_17 = attrition_loss(2016,2017)
loss_17

# %% [markdown]
# `loss_16` is the calculated revenue loss due to attrition  (customers lost) from 2015 to 2016
# 
# `loss_17` is the calculated revenue loss due to attrition  (customers lost) from 2016 to 2017
# 

# %% [markdown]
# #### - Existing customer revenue current year 

# %%
def rev_current_year(year):
    revenue = orders[orders.year == year].net_revenue.sum()
    return revenue

# %%
orders[orders.year == 2015].net_revenue

# %%
rev_15 = rev_current_year(2015)
rev_15

# %% [markdown]
# #### - Total Customers for each year

# %%
total_customers = orders.groupby("year").customer_email.count()
total_customers

# %% [markdown]
# ### Visualizing some information

# %%
total_revenue

# %%
# y = np.linspace(0,35000000,)
plt.figure(figsize = (10,8)) 
plt.bar([2015, 2016, 2017], total_revenue.net_revenue, width=0.5)
# plt.xticks(['15', '16', '17'] ,['15', '16', '17'])
plt.yticks([5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000], ['5','10', '15', '20', '25', '30', '35'])
plt.xlabel('Years')
plt.ylabel('Total Revenue x 1,000,000')
plt.title('Total Revenue for all years')
plt.show()

# %%
growths_info = growths.describe()
growths_info['mean_info'] = growths_info.apply(lambda row: row.mean(), axis=1)

# %%
mu = growths_info.mean_info[1].round(2)
sigma = growths_info.mean_info[2].round(2)
min = growths_info.mean_info[3].round(2)
max = growths_info.mean_info[7].round(2)
x_axis = np.arange(min, max, 1)

plt.figure(figsize=(12,12))
plt.hist([growths.growth_for_16.round(),growths.growth_for_17.round()], bins=50, density=True, label=['2016','2017'], alpha=0.75, color=['mediumspringgreen', 'royalblue'])
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma), c='r', label='$N(0.66, 101.39)$')
plt.xlabel('Growth')
plt.ylabel('Distribution')
plt.title('Histogram of Growth')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# The distribution of growth for both year periods is normal, with `mean = 0`, and `std = 101`


