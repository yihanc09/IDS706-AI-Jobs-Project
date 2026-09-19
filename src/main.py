import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Load the AI jobs dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df


def inspect_data(df):
    """Print basic information about the dataset."""
    print("First five rows:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nSummary statistics:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicates:")
    print(df.duplicated().sum())


# Filtering
def filter_usa_jobs(df):
    """Return jobs located in the USA."""
    return df[df["country"] == "USA"]


def filter_remote_jobs(df):
    """Return fully remote jobs."""
    return df[df["remote_work"] == "Fully Remote"]


def filter_usa_remote_jobs(df):
    """Return fully remote jobs located in the USA."""
    return df[(df["country"] == "USA") & (df["remote_work"] == "Fully Remote")]


def filter_high_salary_jobs(df, salary_threshold=200000):
    """Return jobs with salaries above the selected threshold."""
    return df[df["annual_salary_usd"] > salary_threshold]


def calculate_salary_by_category(df):
    """Calculate average salary for each job category."""
    return (
        df.groupby("job_category")["annual_salary_usd"]
        .mean()
        .sort_values(ascending=False)
    )


# Grouping
def calculate_job_count_by_category(df):
    """Calculate number of jobs in each job category."""
    return df.groupby("job_category")["job_id"].count().sort_values(ascending=False)


def calculate_salary_by_remote_work(df):
    """Calculate average salary by remote work type."""
    return (
        df.groupby("remote_work")["annual_salary_usd"]
        .mean()
        .sort_values(ascending=False)
    )


def calculate_salary_by_llm_role(df):
    """Calculate average salary for LLM and non-LLM roles."""
    return (
        df.groupby("is_llm_role")["annual_salary_usd"]
        .mean()
        .sort_values(ascending=False)
    )


def calculate_demand_by_category(df):
    """Calculate average demand score by job category."""
    return (
        df.groupby("job_category")["demand_score"].mean().sort_values(ascending=False)
    )


# Visualization
def create_salary_by_category_plot(df):
    salary_by_category = calculate_salary_by_category(df)

    salary_by_category.sort_values().plot(kind="barh", figsize=(9, 6))

    plt.title("Average Annual Salary by AI Job Category")
    plt.xlabel("Average Annual Salary (USD)")
    plt.ylabel("Job Category")
    plt.tight_layout()

    plt.savefig("figures/salary_by_category.png")
    plt.close()


def create_salary_distribution_plot(df):
    plt.figure(figsize=(8, 5))

    plt.hist(df["annual_salary_usd"], bins=20)

    plt.title("Distribution of Annual Salaries")
    plt.xlabel("Annual Salary (USD)")
    plt.ylabel("Number of Jobs")
    plt.tight_layout()

    plt.savefig("figures/salary_distribution.png")
    plt.close()


def create_salary_by_experience_plot(df):
    plt.figure(figsize=(8, 5))

    sns.boxplot(data=df, x="experience_level", y="annual_salary_usd")

    plt.title("Annual Salary by Experience Level")
    plt.xlabel("Experience Level")
    plt.ylabel("Annual Salary (USD)")
    plt.tight_layout()

    plt.savefig("figures/salary_by_experience_level.png")
    plt.close()


def create_experience_salary_plot(df):
    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=df,
        x="years_of_experience",
        y="annual_salary_usd",
        scatter_kws={"alpha": 0.4},
    )

    plt.title("Years of Experience vs Annual Salary")
    plt.xlabel("Years of Experience")
    plt.ylabel("Annual Salary (USD)")
    plt.tight_layout()

    plt.savefig("figures/experience_vs_salary.png")
    plt.close()


def create_remote_salary_plot(df):
    plt.figure(figsize=(8, 5))

    sns.boxplot(data=df, x="remote_work", y="annual_salary_usd")

    plt.title("Annual Salary by Remote Work Type")
    plt.xlabel("Remote Work Type")
    plt.ylabel("Annual Salary (USD)")
    plt.tight_layout()

    plt.savefig("figures/salary_by_remote_work.png")
    plt.close()


def create_llm_salary_plot(df):
    plt.figure(figsize=(7, 5))

    sns.boxplot(data=df, x="is_llm_role", y="annual_salary_usd")

    plt.title("Annual Salary: LLM vs Non-LLM Roles")
    plt.xlabel("Role Type")
    plt.ylabel("Annual Salary (USD)")
    plt.xticks([0, 1], ["Non-LLM Role", "LLM Role"])
    plt.tight_layout()

    plt.savefig("figures/salary_by_llm_role.png")
    plt.close()


def create_demand_by_category_plot(df):
    demand_by_category = calculate_demand_by_category(df)

    demand_by_category.sort_values().plot(kind="barh", figsize=(9, 6))

    plt.title("Average Demand Score by AI Job Category")
    plt.xlabel("Average Demand Score")
    plt.ylabel("Job Category")
    plt.tight_layout()

    plt.savefig("figures/demand_by_category.png")
    plt.close()


def create_demand_salary_plot(df):
    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=df, x="demand_score", y="annual_salary_usd", scatter_kws={"alpha": 0.4}
    )

    plt.title("Demand Score vs Annual Salary")
    plt.xlabel("Demand Score")
    plt.ylabel("Annual Salary (USD)")
    plt.tight_layout()

    plt.savefig("figures/demand_vs_salary.png")
    plt.close()


# Machine Learning
def train_salary_model(df):
    """Train a linear regression model to predict annual salary."""

    features = [
        "years_of_experience",
        "demand_score",
        "ai_salary_premium_pct",
        "benefits_score_10",
        "is_senior",
        "is_remote_friendly",
        "is_llm_role",
    ]

    X = df[features]
    y = df["annual_salary_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2


def main():
    os.makedirs("figures", exist_ok=True)

    # Load data
    df = load_data("ai_jobs_market_2025_2026.csv")

    # Inspect data
    inspect_data(df)

    # Filtering
    usa_jobs = filter_usa_jobs(df)
    remote_jobs = filter_remote_jobs(df)
    usa_remote_jobs = filter_usa_remote_jobs(df)
    high_salary_jobs = filter_high_salary_jobs(df)

    print("\nNumber of USA jobs:")
    print(len(usa_jobs))

    print("\nNumber of fully remote jobs:")
    print(len(remote_jobs))

    print("\nNumber of fully remote USA jobs:")
    print(len(usa_remote_jobs))

    print("\nNumber of jobs with salary above $200,000:")
    print(len(high_salary_jobs))

    # Grouping
    print("\nAverage salary by job category:")
    print(calculate_salary_by_category(df))

    print("\nNumber of jobs by job category:")
    print(calculate_job_count_by_category(df))

    print("\nAverage salary by remote work type:")
    print(calculate_salary_by_remote_work(df))

    print("\nAverage salary by LLM role:")
    print(calculate_salary_by_llm_role(df))

    print("\nAverage demand score by job category:")
    print(calculate_demand_by_category(df))

    # Visualization
    create_salary_by_category_plot(df)
    create_salary_distribution_plot(df)
    create_salary_by_experience_plot(df)
    create_experience_salary_plot(df)
    create_remote_salary_plot(df)
    create_llm_salary_plot(df)
    create_demand_by_category_plot(df)
    create_demand_salary_plot(df)

    # Machine learning
    model, mae, r2 = train_salary_model(df)

    print("\nMultiple Linear Regression Model")

    print("\nCoefficients:")
    print(model.coef_)

    print("\nIntercept:")
    print(model.intercept_)

    print("\nModel Evaluation:")

    print("Mean Absolute Error:")
    print(mae)

    print("R-squared:")
    print(r2)


if __name__ == "__main__":
    main()
