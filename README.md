# FIFA World Cup Winner Predictor Pipeline

An automated machine learning classification pipeline specifically designed to predict FIFA World Cup Winners based on team historical statistics.

**Deployed Architecture:** Features a Vercel-ready FastAPI serverless backend with a beautiful HTML/JS frontend to interact with the trained models.

## Features
- **Automated Pipeline**: End-to-end data processing for the Kaggle FIFA World Cup dataset using `kagglehub`.
- **Feature Selection**: Hand-picked optimal features (FIFA Rank, Market Value, Wins, Titles, Host status) for fast inference.
- **Model Training**: Hyperparameter tuning via GridSearchCV (Logistic Regression, Random Forest, SVM, Decision Tree, Gradient Boosting, KNN).
- **Evaluation**: Auto-generates metrics and confusion matrices.
- **Vercel Serverless API**: A deployable FastAPI application.
- **Dynamic Frontend UI**: A vibrant frontend to test the model.

## Installation & Local Usage

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the ML pipeline to generate the models:
   ```bash
   python main.py
   ```

4. Run the API Locally (for testing):
   ```bash
   uvicorn api.index:app --reload
   ```

## Vercel Deployment Instructions

This repository is already configured to be deployed on Vercel as a Python Serverless Function with static frontend assets.

1. Go to [Vercel](https://vercel.com/) and create an account if you don't have one.
2. Click **Add New Project** and import this GitHub repository.
3. Vercel will automatically detect the configuration from `vercel.json`.
4. Leave the Framework Preset as `Other` or `Vite` (it will use the custom `vercel.json`).
5. Click **Deploy**. Vercel will build your API backend and serve your static `index.html`.

Enjoy your Advanced ML Pipeline!
