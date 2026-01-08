echo "🚀 Starting Agentic Assistant..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "❌ Please edit .env and add your API keys, then run this script again."
    exit 1
fi

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if Tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract OCR not found. Please install it:"
    echo "   Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "   macOS: brew install tesseract"
    exit 1
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Please install it:"
    echo "   Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "   macOS: brew install ffmpeg"
    exit 1
fi

# Run the application
echo "✅ Starting server..."
python -m app.main