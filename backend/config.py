"""
config.py
----------
Centralised app settings, loaded from environment variables (via a .env
file). Keeping all configuration in one place - instead of scattering
os.getenv() calls across the codebase - makes it obvious what needs to be
set up before the app will run, and keeps secrets (like DB passwords) out
of the source code.

Copy `.env.example` to `.env` and fill in your own values before running
the backend.
"""

import os

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if one exists.
load_dotenv()


class Settings:
    # --- MySQL connection settings ---
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "telco_churn_db")

    # --- Path to the trained model pipeline produced by train_model.py ---
    MODEL_PATH = os.getenv("MODEL_PATH", "../model/churn_pipeline.pkl")


settings = Settings()
