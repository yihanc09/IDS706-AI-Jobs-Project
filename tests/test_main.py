import os

import pandas as pd

from src.main import (
    load_data,
    filter_usa_jobs,
    filter_usa_remote_jobs,
    filter_high_salary_jobs,
    calculate_salary_by_category,
    train_salary_model,
    main,
)

DATA_FILE = "ai_jobs_market_2025_2026.csv"


def test_load_data():
    df = load_data(DATA_FILE)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "job_id" in df.columns
    assert "annual_salary_usd" in df.columns
    assert "years_of_experience" in df.columns


def test_filter_usa_jobs():
    df = load_data(DATA_FILE)

    usa_jobs = filter_usa_jobs(df)

    assert isinstance(usa_jobs, pd.DataFrame)
    assert len(usa_jobs) > 0
    assert (usa_jobs["country"] == "USA").all()


def test_filter_usa_remote_jobs():
    df = load_data(DATA_FILE)

    usa_remote_jobs = filter_usa_remote_jobs(df)

    assert isinstance(usa_remote_jobs, pd.DataFrame)
    assert len(usa_remote_jobs) > 0
    assert (usa_remote_jobs["country"] == "USA").all()
    assert (usa_remote_jobs["remote_work"] == "Fully Remote").all()


def test_filter_high_salary_jobs():
    df = load_data(DATA_FILE)

    high_salary_jobs = filter_high_salary_jobs(
        df,
        salary_threshold=200000,
    )

    assert isinstance(high_salary_jobs, pd.DataFrame)
    assert len(high_salary_jobs) > 0
    assert (high_salary_jobs["annual_salary_usd"] > 200000).all()


def test_filter_high_salary_jobs_edge_case():
    df = load_data(DATA_FILE)

    high_salary_jobs = filter_high_salary_jobs(
        df,
        salary_threshold=10000000,
    )

    assert isinstance(high_salary_jobs, pd.DataFrame)
    assert len(high_salary_jobs) == 0


def test_salary_by_category():
    df = load_data(DATA_FILE)

    salary_by_category = calculate_salary_by_category(df)

    assert isinstance(salary_by_category, pd.Series)
    assert len(salary_by_category) > 0
    assert salary_by_category.notnull().all()
    assert (salary_by_category > 0).all()


def test_train_salary_model():
    df = load_data(DATA_FILE)

    model, mae, r2 = train_salary_model(df)

    assert model is not None
    assert len(model.coef_) == 7
    assert mae >= 0
    assert isinstance(r2, float)
    assert r2 <= 1


def test_main():
    main()

    expected_figures = [
        "salary_by_category.png",
        "salary_distribution.png",
        "salary_by_experience_level.png",
        "experience_vs_salary.png",
        "salary_by_remote_work.png",
        "salary_by_llm_role.png",
        "demand_by_category.png",
        "demand_vs_salary.png",
    ]

    for figure in expected_figures:
        figure_path = os.path.join("figures", figure)

        assert os.path.exists(figure_path)
        assert os.path.getsize(figure_path) > 0
