#!/bin/bash
# FPL Manager v3 - Startup Script

echo "🚀 Starting FPL Manager v3..."
echo "================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Copying .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before continuing."
    echo "   Required: FPL_TEAM_ID"
    echo "   Optional: API keys for weather, news, and AI features"
    read -p "Press Enter to continue or Ctrl+C to exit and configure..."
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p models/
mkdir -p logs/

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check Node.js installation
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend
npm install
cd ..

# Start backend in background
echo "🔧 Starting backend server..."
python3 backend/api/app.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:5000/api/health > /dev/null; then
    echo "✅ Backend server is running"
else
    echo "❌ Backend server failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo "🎨 Starting frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "🎉 FPL Manager v3 is starting up!"
echo "================================="
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo ""; echo "🛑 Stopping FPL Manager v3..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo "✅ All services stopped"; exit 0' INT

# Keep script running
wait