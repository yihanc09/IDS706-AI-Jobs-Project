import pandas as pd
import polars as pl
import time

# Load the dataset
DATA_PATH = "ai_jobs_market_2025_2026.csv"

df_pd = pd.read_csv(DATA_PATH)
df_pl = pl.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Number of rows:", len(df_pd))
print()

repetitions = 100

# Filtering
start = time.perf_counter()

for i in range(repetitions):
    pandas_filter = df_pd[df_pd["annual_salary_usd"] > 100000]

pandas_filter_time = (time.perf_counter() - start) / repetitions


start = time.perf_counter()

for i in range(repetitions):
    polars_filter = df_pl.filter(pl.col("annual_salary_usd") > 100000)

polars_filter_time = (time.perf_counter() - start) / repetitions

# Grouping and average salary
start = time.perf_counter()

for i in range(repetitions):
    pandas_group = df_pd.groupby("job_category")["annual_salary_usd"].mean()

pandas_group_time = (time.perf_counter() - start) / repetitions


start = time.perf_counter()

for i in range(repetitions):
    polars_group = df_pl.group_by("job_category").agg(
        pl.col("annual_salary_usd").mean()
    )

polars_group_time = (time.perf_counter() - start) / repetitions


# Sorting by salary
start = time.perf_counter()

for i in range(repetitions):
    pandas_sort = df_pd.sort_values("annual_salary_usd", ascending=False)

pandas_sort_time = (time.perf_counter() - start) / repetitions


start = time.perf_counter()

for i in range(repetitions):
    polars_sort = df_pl.sort("annual_salary_usd", descending=True)

polars_sort_time = (time.perf_counter() - start) / repetitions


print("Pandas vs. Polars Performance Comparison")

print("Filtering:")
print("Pandas:", pandas_filter_time, "seconds")
print("Polars:", polars_filter_time, "seconds")
print()

print("Grouping and aggregation:")
print("Pandas:", pandas_group_time, "seconds")
print("Polars:", polars_group_time, "seconds")
print()

print("Sorting:")
print("Pandas:", pandas_sort_time, "seconds")
print("Polars:", polars_sort_time, "seconds")
print()

print("Example Pandas group result:")
print(pandas_group)
print()

print("Example Polars group result:")
print(polars_group)
