# AutoTabular

AutoTabular is a web-based AutoML tool that enables users to upload CSV files, select a target column, choose a task (classification or regression), and train machine learning models (e.g., LightGBM, XGBoost, Random Forest). It offers a no-code interface for data processing, model training, evaluation, and inference on new data.

## Project Goal

The goal of AutoTabular is to provide an efficient, user-friendly platform for automated machine learning, allowing users to:
- Upload structured CSV files for data processing (handling missing values, outliers).
- Preview data tables and visualize distributions.
- Train and validate tree-based models with customizable parameters.
- View evaluation metrics (e.g., accuracy, precision, F1 score) and confusion matrices.
- Perform inference on new CSV files with the same column structure.

## Features (Planned)

- **Frontend**: Interactive interface for CSV upload, task selection, model training, and result visualization.
- **Backend**: FastAPI-based APIs for data processing, model training, validation, and inference.
- **Database**: PostgreSQL to store CSV file metadata and model information.
- **Models**: Support for LightGBM, XGBoost, and Random Forest.

## Current Progress

- **Database Setup**: PostgreSQL database (`autotabular`) created and accessible via DBeaver.
- **Backend Configuration**:
  - Environment variables configured in `.env` and loaded via `app/core/config.py`.
  - SQLAlchemy base class defined in `app/db/base.py`.
  - Database connection established in `app/db/session.py`.

## Prerequisites

- Python 3.10
- PostgreSQL 14.x (installed via Homebrew on macOS)
- DBeaver (optional, for database visualization)

## Create a Conda Environment

- conda create --name AutoTabular python=3.10

## Install Dependencies

- pip install -r requirements.txt
