import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Import the Dataset
df = pd.read_csv("ai_jobs_market_2025_2026.csv")

# Inspect the Data
# Quick Overview
print("First five rows:")
print(df.head())

# Understanding data types and summary statistics
print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())

# Check for missing values and duplicates
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

# Basic Filtering and Grouping
# Explore important categorical variables
print("\nCountries in the dataset:")
print(df["country"].unique())

print("\nRemote work options:")
print(df["remote_work"].unique())

print("\nJob categories:")
print(df["job_category"].unique())

# Filtering: jobs in the USA
usa_jobs = df[df["country"] == "USA"]

print("\nNumber of USA jobs:")
print(len(usa_jobs))

# Filtering: fully remote jobs
remote_jobs = df[df["remote_work"] == "Fully Remote"]

print("\nNumber of fully remote jobs:")
print(len(remote_jobs))

# Filtering: fully remote jobs in the USA
usa_remote_jobs = df[(df["country"] == "USA") & (df["remote_work"] == "Fully Remote")]

print("\nNumber of fully remote USA jobs:")
print(len(usa_remote_jobs))

# Filtering: high salary jobs
high_salary_jobs = df[df["annual_salary_usd"] > 200000]

print("\nNumber of high salary jobs:")
print(len(high_salary_jobs))

# Grouping: average salary by job category
salary_by_category = (
    df.groupby("job_category")["annual_salary_usd"].mean().sort_values(ascending=False)
)

print("\nAverage salary by job category:")
print(salary_by_category)

# Grouping: number of jobs by job category
job_count_by_category = df.groupby("job_category")["job_id"].count()

print("\nNumber of jobs by job category:")
print(job_count_by_category)


# Visualization
# Average Salary by Job Category
salary_by_category.sort_values().plot(kind="barh", figsize=(9, 6))

plt.title("Average Annual Salary by AI Job Category")
plt.xlabel("Average Annual Salary (USD)")
plt.ylabel("Job Category")
plt.tight_layout()

plt.savefig("salary_by_category.png")
plt.show()

# Distribution of annual salaries
plt.figure(figsize=(8, 5))

plt.hist(df["annual_salary_usd"], bins=20)

plt.title("Distribution of Annual Salaries")
plt.xlabel("Annual Salary (USD)")
plt.ylabel("Number of Jobs")
plt.tight_layout()

plt.savefig("salary_distribution.png")
plt.show()

# Salary by experience level
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="experience_level", y="annual_salary_usd")

plt.title("Annual Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Annual Salary (USD)")
plt.tight_layout()

plt.savefig("salary_by_experience_level.png")
plt.show()

# Years of experience vs annual salary
plt.figure(figsize=(8, 5))

sns.regplot(
    data=df, x="years_of_experience", y="annual_salary_usd", scatter_kws={"alpha": 0.4}
)

plt.title("Years of Experience vs Annual Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Annual Salary (USD)")
plt.tight_layout()

plt.savefig("experience_vs_salary.png")
plt.show()


# Explore a Machine Learning Algorithm
# Can years of experience help predict annual salary in the AI job market?
# Predict salary using years of experience
X = df[["years_of_experience"]]
y = df["annual_salary_usd"]

# Split the data into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict salaries for the testing data
y_pred = model.predict(X_test)

# Display model information
print("\nLinear Regression Model:")

print("Coefficient:")
print(model.coef_[0])

print("Intercept:")
print(model.intercept_)

print("\nFirst five predicted salaries:")
print(y_pred[:5])

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:")
print(mae)
print("R-squared:")
print(r2)
