#!/bin/bash

# MLOps Capstone Project - Retail Analytics
# Quick Start Script

echo "🏪 Starting Retail Analytics MLOps Pipeline..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ Python and pip found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed models logs

# Copy environment file
if [ ! -f ".env" ]; then
    echo "🔧 Setting up environment variables..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your specific configuration"
fi

# Check if data file exists
if [ ! -f "data/raw/customer_shopping_data.csv" ]; then
    echo "📊 Sample data not found. The pipeline will create sample data automatically."
    echo "💡 To use real data, download from: https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset"
    echo "   and place it in data/raw/customer_shopping_data.csv"
fi

# Run data processing pipeline
echo "🔄 Running data processing pipeline..."
python src/data_processing/pipeline.py

# Start the services
echo "🚀 Starting services..."

# Start FastAPI server in background
echo "🌐 Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to start
sleep 5

# Start Streamlit dashboard in background
echo "📊 Starting Streamlit dashboard..."
streamlit run dashboards/streamlit_dashboard.py --server.port 8501 &
DASHBOARD_PID=$!

echo ""
echo "🎉 Services started successfully!"
echo ""
echo "📍 Access points:"
echo "   🌐 FastAPI Server: http://localhost:8000"
echo "   📖 API Documentation: http://localhost:8000/docs"
echo "   📊 Streamlit Dashboard: http://localhost:8501"
echo ""
echo "🛠️  To stop services:"
echo "   Press Ctrl+C or run: kill $API_PID $DASHBOARD_PID"
echo ""
echo "📚 Next steps:"
echo "   1. Visit the dashboard to explore retail insights"
echo "   2. Check API documentation for available endpoints"
echo "   3. Run tests: pytest tests/"
echo "   4. View logs in the logs/ directory"
echo ""

# Keep script running
wait
