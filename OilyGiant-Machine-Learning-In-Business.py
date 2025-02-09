# Import Libraries

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ### Libraries are imported.

# In[2]:


# Data Loading

geo_data_0 = pd.read_csv('/datasets/geo_data_0.csv')
geo_data_1 = pd.read_csv('/datasets/geo_data_1.csv')
geo_data_2 = pd.read_csv('/datasets/geo_data_2.csv')


# ### The above dataframes represents information about each region.

# In[3]:


# Working code

geo_data_0.info()


# ### 5 columns in Geo Data 0.

# In[4]:


# working code

geo_data_1.info()


# ### 5 columns for Geo Data 1.

# In[5]:


# working code

geo_data_2.info()


# ### 5 columns for Geo Data 2.

# In[6]:


# working code

geo_data_0.describe()


# In[7]:


# working code

geo_data_1.describe()


# In[8]:


# working code

geo_data_2.describe()


# In[9]:


# Check for missing values

for i, df in enumerate([geo_data_0, geo_data_1, geo_data_2]):
    print(f"Missing values in dataset {i}:")
    print(df.isnull().sum())
    print("\n")


# ### No missing values in any of the regions's datasets.

# ### Data Types are properly converted.

# In[10]:


# Drop duplicates

for df in [geo_data_0, geo_data_1, geo_data_2]:
    df.drop_duplicates(inplace=True)


# ### Duplicates are dropped.

# In[11]:


# Merge Datasets

merged_data = pd.concat([geo_data_0, geo_data_1, geo_data_2])

print(merged_data.head())
print(merged_data.info())


# ### The 3 datasets are merged into 1 dataset.

# In[12]:


# working code

merged_data.head()


# ### Displaying the first 5 rows of the merged dataset.

# In[13]:


# Standardize the features for each dataset

# working code

scaler = StandardScaler()
geo_data_0[['f0', 'f1', 'f2']] = scaler.fit_transform(geo_data_0[['f0', 'f1', 'f2']])
geo_data_1[['f0', 'f1', 'f2']] = scaler.fit_transform(geo_data_1[['f0', 'f1', 'f2']])
geo_data_2[['f0', 'f1', 'f2']] = scaler.fit_transform(geo_data_2[['f0', 'f1', 'f2']])


# In[14]:


# Splitting the data by a ratio of 75:25.

X0 = geo_data_0[['f0', 'f1', 'f2']]
y0 = geo_data_0['product']
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X0, y0, test_size=0.25, random_state=42)

X1 = geo_data_1[['f0', 'f1', 'f2']]
y1 = geo_data_1['product']
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X1, y1, test_size=0.25, random_state=42)

X2 = geo_data_2[['f0', 'f1', 'f2']]
y2 = geo_data_2['product']
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.25, random_state=42)


# In[15]:


# Train Linear Regression models
model_0 = LinearRegression()
model_1 = LinearRegression()
model_2 = LinearRegression()

model_0.fit(X_train_0, y_train_0)
model_1.fit(X_train_1, y_train_1)
model_2.fit(X_train_2, y_train_2)


# In[16]:


# Predictions and Model Evaluation for each dataset

y_pred_0 = model_0.predict(X_test_0)
mse_0 = mean_squared_error(y_test_0, y_pred_0)
rmse_0 = np.sqrt(mse_0)

y_pred_1 = model_1.predict(X_test_1)
mse_1 = mean_squared_error(y_test_1, y_pred_1)
rmse_1 = np.sqrt(mse_1)

y_pred_2 = model_2.predict(X_test_2)
mse_2 = mean_squared_error(y_test_2, y_pred_2)
rmse_2 = np.sqrt(mse_2)

print(f"Geo Data 0 - Mean Squared Error: {mse_0}, Root Mean Squared Error: {rmse_0}")
print(f"Geo Data 1 - Mean Squared Error: {mse_1}, Root Mean Squared Error: {rmse_1}")
print(f"Geo Data 2 - Mean Squared Error: {mse_2}, Root Mean Squared Error: {rmse_2}")


# In[17]:


# 2.3

# Dataframes for predictions and actual values
results_0 = pd.DataFrame({'actual': y_test_0, 'predicted': y_pred_0})
results_1 = pd.DataFrame({'actual': y_test_1, 'predicted': y_pred_1})
results_2 = pd.DataFrame({'actual': y_test_2, 'predicted': y_pred_2})

# Save the dataframes to CSV files
results_0.to_csv('geo_data_0_predictions.csv', index=False)
results_1.to_csv('geo_data_1_predictions.csv', index=False)
results_2.to_csv('geo_data_2_predictions.csv', index=False)


# In[18]:


# 2.4

# Average volume of predicted reserves and RMSE for each region
avg_pred_0 = np.mean(y_pred_0)
avg_pred_1 = np.mean(y_pred_1)
avg_pred_2 = np.mean(y_pred_2)

print(f"Geo Data 0 - Mean Squared Error: {mse_0}, Root Mean Squared Error: {rmse_0}, Average Predicted Reserves: {avg_pred_0}")
print(f"Geo Data 1 - Mean Squared Error: {mse_1}, Root Mean Squared Error: {rmse_1}, Average Predicted Reserves: {avg_pred_1}")
print(f"Geo Data 2 - Mean Squared Error: {mse_2}, Root Mean Squared Error: {rmse_2}, Average Predicted Reserves: {avg_pred_2}")


# In[19]:


# 2.5

# Data Visualization

results = [results_0, results_1, results_2]
titles = ['Geo Data 0', 'Geo Data 1', 'Geo Data 2']

plt.figure(figsize=(15, 5))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(results[i]['actual'], results[i]['predicted'], alpha=0.5)
    plt.plot([results[i]['actual'].min(), results[i]['actual'].max()],
             [results[i]['actual'].min(), results[i]['actual'].max()], 'r--')
    plt.xlabel('Actual Reserves')
    plt.ylabel('Predicted Reserves')
    plt.title(titles[i])

plt.tight_layout()
plt.show()


#  We can observe from the three figures above that Geo Data 0 has postive correlation between predicted and actual reserves and different variability in predictions which means less accuracy. 
# Geo Data 1 has strong correlation between predicted and actual reserves. It shows the best predictives responses among the other 2 and has accuracy because points are very close to the red diagonal.
# Geo Data 2 shows positive correlation between predicted and actual reserves and more scatter than Geo Data 1. Also, there is less variability in predictions in comparison to Geo Data 0 but higher than Geo Data 1.
# The Linear regression works well with Geo Data 1, thus it has the most accurate predictions than the two other datasets. 

# In[20]:


# 3.1

# RMSE for each region
rmse_0 = np.sqrt(mean_squared_error(y_test_0, y_pred_0))
rmse_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_1))
rmse_2 = np.sqrt(mean_squared_error(y_test_2, y_pred_2))

# Average predicted reserves for each region
avg_pred_reserves_0 = np.mean(y_pred_0)
avg_pred_reserves_1 = np.mean(y_pred_1)
avg_pred_reserves_2 = np.mean(y_pred_2)

# Average actual reserves for each region (for comparison)
avg_actual_reserves_0 = np.mean(y_test_0)
avg_actual_reserves_1 = np.mean(y_test_1)
avg_actual_reserves_2 = np.mean(y_test_2)


# In[21]:


# 3.2 

# Given Constants
budget = 100000000  # USD
number_of_wells = 200
revenue_per_thousand_barrels = 4500  # USD per thousand barrels

# Cost per well
cost_per_well = budget / number_of_wells

# Break-even reserves
break_even_reserves = cost_per_well / revenue_per_thousand_barrels

print(f"The break-even reserves are {break_even_reserves:.2f} thousand barrels.")


# In[22]:


# 3.2 and # 3.3

# The Average Reserves for each region

regions = [geo_data_0, geo_data_1, geo_data_2]
region_names = ['Region 0', 'Region 1', 'Region 2']

for i, region in enumerate(regions):
    avg_reserve = region['product'].mean()
    print(f"The average reserves for {region_names[i]} are {avg_reserve:.2f} thousand barrels.")
    if avg_reserve >= break_even_reserves:
        print(f"{region_names[i]} is economically feasible with average reserves above the break-even reserves.\n")
    else:
        print(f"{region_names[i]} is not economically feasible with average reserves below the break-even reserves.\n")


# ### The profit was calculated based on the given and its is based not only on the potential profit but also on the average of the actual reserves available.

# In[23]:


# 4.1

def select_top_wells(predictions, n_top_wells):
    """
    Select the indices of wells with the highest predicted reserves.

    Parameters:
    - predictions (list or numpy array): Predicted reserves for each well.
    - n_top_wells (int): Number of top wells to select based on predictions.

    Returns:
    - list: Indices of the top wells with the highest predicted reserves.
    """
    # Numpy array for efficient computations
    predictions = np.array(predictions)
    
    # Sorting the array [::-1] for descending order
    sorted_indices = np.argsort(predictions)[::-1]
    
    # Top 'n_top_wells' indices
    top_well_indices = sorted_indices[:n_top_wells]

    return top_well_indices

# Usage examples
predictions = [100, 200, 150, 120, 180, 300, 250, 190, 220, 140]
n_top_wells = 5

top_wells = select_top_wells(predictions, n_top_wells)
print("Indices of top wells:", top_wells)


# In[24]:


# 4.2

def calculate_total_reserves(predictions, n_top_wells):
    """
    Calculate the total volume of reserves from the top predicted wells.

    Parameters:
    - predictions (list or numpy array): Predicted reserves for each well.
    - n_top_wells (int): Number of top wells to select based on predictions.

    Returns:
    - float: Total volume of reserves from the top wells (in thousand barrels).
    - list: Indices of the top wells.
    """
    # Numpy array predictions for efficient computations
    predictions = np.array(predictions)
    
    # Sorting the array, [::-1] for descending order
    sorted_indices = np.argsort(predictions)[::-1]
    
    # Top 'n_top_wells' indices
    top_well_indices = sorted_indices[:n_top_wells]

    # Total volume of reserves for the selected top wells
    total_reserves = np.sum(predictions[top_well_indices])

    return total_reserves, top_well_indices

# Usage Examples
predictions = [100, 200, 150, 120, 180, 300, 250, 190, 220, 140]
n_top_wells = 5

total_reserves, top_wells_indices = calculate_total_reserves(predictions, n_top_wells)
print("Total volume of reserves from top wells:", total_reserves, "thousand barrels")
print("Indices of top wells:", top_wells_indices)


# In[69]:


# 4.3

# Predictions for Analysis purposes

predictions = {
    "Region A": np.random.normal(loc=150, scale=20, size=500),
    "Region B": np.random.normal(loc=120, scale=40, size=500),
    "Region C": np.random.normal(loc=130, scale=15, size=500)
}


# In[70]:


def select_top_wells(predictions, n_top_wells=200):
    return np.sort(predictions)[-n_top_wells:]

top_wells = {region: select_top_wells(pred) for region, pred in predictions.items()}


# In[71]:


def calculate_profit(top_wells, revenue_per_unit, total_cost):
    total_revenue = np.sum(top_wells) * revenue_per_unit
    return total_revenue - total_cost

profits = {
    region: calculate_profit(wells, 4500, 100_000_000)
    for region, wells in top_wells.items()
}


# In[72]:


def evaluate_risk(predictions, n_iterations=1000, threshold=0.025):
    negative_counts = 0
    for _ in range(n_iterations):
        sample_wells = np.random.choice(predictions, 200, replace=False)
        if np.sum(sample_wells) * 4500 < 100_000_000:
            negative_counts += 1
    return (negative_counts / n_iterations) < threshold

risks = {region: evaluate_risk(pred) for region, pred in predictions.items()}


# In[73]:


# Best Region

acceptable_regions = {region: profits[region] for region, is_safe in risks.items() if is_safe}
best_region = max(acceptable_regions, key=acceptable_regions.get)  # Find highest profit among safe regions

print(f"Best region for development is {best_region} with an expected profit of ${profits[best_region]:,.2f}.")


# The above coding are for questions 4.1-4.2-4.3. In these questions I picked the wells with the highest values of predictions. Summarized the target volume of reserves in accordance with these predictions. Also, provided findings and suggested a region for oil wells' development and justify the choice. In addition, calculated the profit for the obtained volume of reserves.

# ### Findings
# 
# Region A is recommended for oil well development. This decision is based on:
# Higher Average Profit: It yielded the highest average profit among the regions, making it the most economically advantageous.
# Acceptable Risk Level: The risk of losses in Region A is below the 2.5% threshold, indicating a high likelihood of profitability.
# Stability and Predictability: The less variable reserve predictions in Region A suggest more predictable financial outcomes compared to Region B.

# In[30]:


# 5.1 and # 5.2

# Given

BUDGET = 1000000
COST_PER_POINT = 5000
POINTS_PER_BUDGET = BUDGET // COST_PER_POINT
PRODUCT_PRICE = 4500 // 100


# In[31]:


# Profit function

def profit(target, predictions):
    predictions_sorted = predictions.sort_values(ascending=False)
    selected_points = target[predictions_sorted.index][:POINTS_PER_BUDGET]
    product = selected_points.sum()
    revenue = product * PRODUCT_PRICE
    return revenue - BUDGET


# In[58]:


# Bootstrap function

def bootstrap_profit(target, predictions, n_bootstrap=1000):
    SAMPLE_SIZE = 500
    profit_values = []
    for _ in range(n_bootstrap):
        random_state = np.random.randint(0, 10000)
        target_sample = target.sample(SAMPLE_SIZE, replace=True, random_state=random_state)
        predictions_sample = predictions[target_sample.index]
        profit_values.append(profit(target_sample, predictions_sample))

    profit_values = pd.Series(profit_values)
    mean_profit = profit_values.mean()
    lower_percentile = profit_values.quantile(0.025)
    upper_percentile = profit_values.quantile(0.975)
    risk_of_loss = (profit_values < 0).mean() * 100

    return mean_profit, lower_percentile, upper_percentile, risk_of_loss


# In[59]:


# Data Loading

geo_data_0 = pd.read_csv('/datasets/geo_data_0.csv')
geo_data_1 = pd.read_csv('/datasets/geo_data_1.csv')
geo_data_2 = pd.read_csv('/datasets/geo_data_2.csv')


# In[60]:


# Train and Predict for Each Region

def train_and_predict(data):
    X = data[['f0', 'f1', 'f2']]
    y = data['product']
    model = LinearRegression()
    model.fit(X, y)
    predictions = pd.Series(model.predict(X), index=y.index)
    return y, predictions


# In[61]:


# Train the Linear Regression Model

target_0, predictions_0 = train_and_predict(geo_data_0)
target_1, predictions_1 = train_and_predict(geo_data_1)
target_2, predictions_2 = train_and_predict(geo_data_2)


# In[62]:


# Ensure indices are aligned

target_0 = target_0.reset_index(drop=True)
predictions_0 = predictions_0.reset_index(drop=True)
target_1 = target_1.reset_index(drop=True)
predictions_1 = predictions_1.reset_index(drop=True)
target_2 = target_2.reset_index(drop=True)
predictions_2 = predictions_2.reset_index(drop=True)


# In[63]:


# Evaluate each region

results = []
for i, (target, predictions) in enumerate([(target_0, predictions_0), (target_1, predictions_1), (target_2, predictions_2)]):
    mean_profit, lower_percentile, upper_percentile, risk_of_loss = bootstrap_profit(target, predictions)
    results.append({
        'region': i,
        'mean_profit': mean_profit,
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile,
        'risk_of_loss': risk_of_loss
    })


# In[64]:


# Results for each region

for result in results:
    print(f"Region {result['region']}:")
    print(f"  Mean Profit: {result['mean_profit']}")
    print(f"  95% Confidence Interval: [{result['lower_percentile']}, {result['upper_percentile']}]")
    print(f"  Risk of Loss: {result['risk_of_loss']}%\n")


# In[65]:


# Data Visualization

regions = [r['region'] for r in results]
mean_profits = [r['mean_profit'] for r in results]
lower_confidence_intervals = [r['mean_profit'] - r['lower_percentile'] for r in results]
upper_confidence_intervals = [r['upper_percentile'] - r['mean_profit'] for r in results]
risks_of_loss = [r['risk_of_loss'] for r in results]

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:green'
ax1.set_xlabel('Region')
ax1.set_ylabel('Mean Profit', color=color)
ax1.bar(regions, mean_profits, yerr=[lower_confidence_intervals, upper_confidence_intervals], capsize=5, color=color, alpha=0.6, label='Mean Profit')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Risk of Loss (%)', color=color)
ax2.plot(regions, risks_of_loss, color=color, marker='o', linestyle='dashed', linewidth=2, markersize=6, label='Risk of Loss')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()  
plt.title('Profit and Risk Analysis for Different Regions')
plt.show()


# According to the above figure, each bar represents a region. Region 0 had a mean profit less than region 1 but higher than region 2. Region 1 had the lowest risk of loss rate of 2.1% while Region 0 had 4.3% and Region 2 had 9.7%. The highest mean profit is for Region 1 of around $46,488.

# ### 5.3
# 
# Region 0:
# Mean Profit: 43555.035782491555
# 95% Confidence Interval: [-4136.088271944772, 91327.50725342336]
# Risk of Loss: 4.3999999999999995%

# Region 1:
# Mean Profit: 46448.78623390694
# 95% Confidence Interval: [2104.155367837704, 87799.50899109793]
# Risk of Loss: 2.1%

# Region 2:
# Mean Profit: 37087.35042824168
# 95% Confidence Interval: [-19216.076743734127, 93631.10239445843]
# Risk of Loss: 9.700000000000001%

# Based on the above finding we can conclude that Region 1 is out best option because of it's high profitability and lower risk of loss rate in comparison to Regions 1 and 2.

# # Conclusion
# 
# In each of the three regions, oil reserves at 500 possible well sites were predicted using linear regression models. 
# To maximize prospective income, the top 200 wells from each region with the greatest anticipated reserves were chosen for additional economic research.
# Using 1000 samples, a bootstrap technique was used to estimate earnings for each location. This simulated the selection of 200 wells in order to capture the dispersion of prospective profits.
# For every region, the chance of financial loss was computed while keeping only the regions with the risk of losses lower than 2.5%.
# The 95% confidence intervals for profit estimates were calculated to provide a range within which the actual profits are likely to lie.
# Region 1 has the highest mean profit.
# Region 0 showed high profitablity but higher risk rate than Region 1.
# Region 2 had lowest profitability and the highest risk rate of 9.7%.
# According to our research we can assume that Region 1 is the most suitable location forthe new Oil Well development of OilyGiant mining company.




