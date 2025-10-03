@echo off
REM MLOps Capstone Project - Retail Analytics
REM Quick Start Script for Windows

echo 🏪 Starting Retail Analytics MLOps Pipeline...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "logs" mkdir logs

REM Copy environment file
if not exist ".env" (
    echo 🔧 Setting up environment variables...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your specific configuration
)

REM Check if data file exists
if not exist "data\raw\customer_shopping_data.csv" (
    echo 📊 Sample data not found. The pipeline will create sample data automatically.
    echo 💡 To use real data, download from: https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset
    echo    and place it in data\raw\customer_shopping_data.csv
)

REM Run data processing pipeline
echo 🔄 Running data processing pipeline...
python src\data_processing\pipeline.py

REM Start the services
echo 🚀 Starting services...

REM Start FastAPI server
echo 🌐 Starting FastAPI server...
start "FastAPI Server" cmd /k "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for API to start
timeout /t 5 /nobreak >nul

REM Start Streamlit dashboard
echo 📊 Starting Streamlit dashboard...
start "Streamlit Dashboard" cmd /k "streamlit run dashboards\streamlit_dashboard.py --server.port 8501"

echo.
echo 🎉 Services started successfully!
echo.
echo 📍 Access points:
echo    🌐 FastAPI Server: http://localhost:8000
echo    📖 API Documentation: http://localhost:8000/docs  
echo    📊 Streamlit Dashboard: http://localhost:8501
echo.
echo 📚 Next steps:
echo    1. Visit the dashboard to explore retail insights
echo    2. Check API documentation for available endpoints
echo    3. Run tests: pytest tests/
echo    4. View logs in the logs/ directory
echo.

pause
