import pandas as pd
import polars as pl
import time

# Load the same dataset with Pandas and Polars
DATA_PATH = "ai_jobs_market_2025_2026.csv"

df_pd = pd.read_csv(DATA_PATH)
df_pl = pl.read_csv(DATA_PATH)

print("Pandas vs. Polars Performance Comparison")
print("Number of rows:", len(df_pd))
print()

repetitions = 100

# Filtering: jobs in the USA
start = time.perf_counter()
for i in range(repetitions):
    pandas_usa_jobs = df_pd[df_pd["country"] == "USA"]
pandas_usa_time = (time.perf_counter() - start) / repetitions

start = time.perf_counter()
for i in range(repetitions):
    polars_usa_jobs = df_pl.filter(pl.col("country") == "USA")
polars_usa_time = (time.perf_counter() - start) / repetitions

# Filtering: fully remote jobs
start = time.perf_counter()
for i in range(repetitions):
    pandas_remote_jobs = df_pd[df_pd["remote_work"] == "Fully Remote"]
pandas_remote_time = (time.perf_counter() - start) / repetitions

start = time.perf_counter()
for i in range(repetitions):
    polars_remote_jobs = df_pl.filter(pl.col("remote_work") == "Fully Remote")
polars_remote_time = (time.perf_counter() - start) / repetitions

# Filtering: fully remote jobs in the USA
start = time.perf_counter()
for i in range(repetitions):
    pandas_usa_remote_jobs = df_pd[
        (df_pd["country"] == "USA") & (df_pd["remote_work"] == "Fully Remote")
    ]
pandas_usa_remote_time = (time.perf_counter() - start) / repetitions

start = time.perf_counter()
for i in range(repetitions):
    polars_usa_remote_jobs = df_pl.filter(
        (pl.col("country") == "USA") & (pl.col("remote_work") == "Fully Remote")
    )
polars_usa_remote_time = (time.perf_counter() - start) / repetitions

# Filtering: high salary jobs
start = time.perf_counter()
for i in range(repetitions):
    pandas_high_salary_jobs = df_pd[df_pd["annual_salary_usd"] > 200000]
pandas_high_salary_time = (time.perf_counter() - start) / repetitions

start = time.perf_counter()
for i in range(repetitions):
    polars_high_salary_jobs = df_pl.filter(pl.col("annual_salary_usd") > 200000)
polars_high_salary_time = (time.perf_counter() - start) / repetitions

# Grouping
start = time.perf_counter()
for i in range(repetitions):
    pandas_salary_by_category = (
        df_pd.groupby("job_category")["annual_salary_usd"]
        .mean()
        .sort_values(ascending=False)
    )
pandas_salary_group_time = (time.perf_counter() - start) / repetitions

start = time.perf_counter()
for i in range(repetitions):
    polars_salary_by_category = (
        df_pl.group_by("job_category")
        .agg(pl.col("annual_salary_usd").mean())
        .sort("annual_salary_usd", descending=True)
    )
polars_salary_group_time = (time.perf_counter() - start) / repetitions

# Grouping
start = time.perf_counter()
for i in range(repetitions):
    pandas_job_count_by_category = df_pd.groupby("job_category")["job_id"].count()
pandas_count_group_time = (time.perf_counter() - start) / repetitions

start = time.perf_counter()
for i in range(repetitions):
    polars_job_count_by_category = df_pl.group_by("job_category").agg(
        pl.col("job_id").count()
    )
polars_count_group_time = (time.perf_counter() - start) / repetitions


print("Analysis Results Check")
print("USA jobs - Pandas:", len(pandas_usa_jobs))
print("USA jobs - Polars:", polars_usa_jobs.height)
print()

print("Fully remote jobs - Pandas:", len(pandas_remote_jobs))
print("Fully remote jobs - Polars:", polars_remote_jobs.height)
print()

print("Fully remote USA jobs - Pandas:", len(pandas_usa_remote_jobs))
print("Fully remote USA jobs - Polars:", polars_usa_remote_jobs.height)
print()

print("High salary jobs - Pandas:", len(pandas_high_salary_jobs))
print("High salary jobs - Polars:", polars_high_salary_jobs.height)
print()


print("Average Time per Operation (seconds)")
print()

print("Filter: USA jobs")
print("Pandas:", pandas_usa_time)
print("Polars:", polars_usa_time)
print()

print("Filter: fully remote jobs")
print("Pandas:", pandas_remote_time)
print("Polars:", polars_remote_time)
print()

print("Filter: fully remote USA jobs")
print("Pandas:", pandas_usa_remote_time)
print("Polars:", polars_usa_remote_time)
print()

print("Filter: salary above $200,000")
print("Pandas:", pandas_high_salary_time)
print("Polars:", polars_high_salary_time)
print()

print("Group by job category: average salary")
print("Pandas:", pandas_salary_group_time)
print("Polars:", polars_salary_group_time)
print()

print("Group by job category: job count")
print("Pandas:", pandas_count_group_time)
print("Polars:", polars_count_group_time)
print()


print("Pandas average salary by category:")
print(pandas_salary_by_category)
print()

print("Polars average salary by category:")
print(polars_salary_by_category)
print()

print("Pandas job count by category:")
print(pandas_job_count_by_category)
print()

print("Polars job count by category:")
print(polars_job_count_by_category)
