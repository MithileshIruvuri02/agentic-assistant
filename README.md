# 🤖 Agentic Assistant

An intelligent multi-agent system that processes text, images, PDFs, and audio files, understands user intent, and autonomously performs the correct task.

## 🌟 Features

- **Multi-Input Support**: Text, Images (OCR), PDFs, Audio (Speech-to-Text), YouTube URLs
- **Intelligent Intent Recognition**: Automatically detects what users want to do
- **Smart Clarification**: Asks follow-up questions when intent is unclear
- **Multi-Agent Architecture** (Bonus): Separate planner and executor agents
- **Cost Estimation** (Bonus): Predicts API costs before execution
- **Multiple Task Types**:
  - Text Extraction (with OCR confidence)
  - YouTube Transcript Fetching
  - Conversational Q&A
  - Summarization (1-line + 3 bullets + 5 sentences)
  - Sentiment Analysis (label + confidence + justification)
  - Code Explanation (language + explanation + bugs + complexity)
  - Audio Transcription + Summary

## 🏗️ Architecture

The system uses a **multi-agent architecture** with:

1. **Planner Agent**: Analyzes input, understands intent, creates execution plans
2. **Executor Agent**: Executes tasks based on plans
3. **Cost Estimator**: Calculates estimated API costs before execution

### Architecture Diagram

```
User Input → Input Processor → Planner Agent → Cost Estimator → Executor Agent → Result
                   ↓                ↓                                    ↓
            (OCR/PDF/Audio)   (Intent + Plan)                    (Task Services)
```

## 📁 Project Structure

```
agentic-assistant/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── agents/              # Multi-agent system
│   │   ├── planner_agent.py
│   │   ├── executor_agent.py
│   │   └── cost_estimator.py
│   ├── services/            # Processing services
│   │   ├── input_processor.py
│   │   ├── ocr_service.py
│   │   ├── pdf_service.py
│   │   ├── audio_service.py
│   │   ├── youtube_service.py
│   │   ├── summarizer.py
│   │   ├── sentiment_analyzer.py
│   │   └── code_explainer.py
│   ├── models/              # Data models
│   │   └── schemas.py
│   └── api/
│       └── routes.py
├── frontend/                # Chat UI
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── tests/                   # Test suite
│   ├── test_api.py
│   ├── test_services.py
│   └── test_agents.py
├── requirements.txt
└── README.md
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9+
- Tesseract OCR installed
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd agentic-assistant
```

2. **Install Tesseract OCR**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

3. **Install FFmpeg**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from: https://ffmpeg.org/download.html
```

4. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

5. **Install dependencies**
```bash
pip install -r requirements.txt
```

6. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GROQ_API_KEY=your_groq_key_here

```

### Run the Application

```bash
python -m app.main
```

Or use uvicorn directly:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at: `http://localhost:8000`

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_api.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

## 📊 Test Cases

The system handles all required test cases:

### 1. Audio Lecture (5 min)
```
Upload: 5-minute audio file
Expected: Transcription + 1-line + 3 bullets + 5-sentence summary + duration
```

### 2. PDF with Meeting Notes
```
Upload: PDF with meeting notes
Text: "What are the action items?"
Expected: Extracted text + identified action items
```

### 3. Code Screenshot
```
Upload: Image with code snippet
Text: "Explain"
Expected: OCR text + language detection + explanation + bug warnings
```



## 📝 API Endpoints

### `GET /`
Serves the chat UI

### `GET /health`
Health check endpoint

### `POST /api/process`
Main processing endpoint

**Request:**
- `text` (optional): Text input
- `file` (optional): File upload
- `clarification_response` (optional): Response to clarification question
- `previous_request_id` (optional): Previous request ID for clarifications

**Response:**
```json
{
  "request_id": "uuid",
  "status": "completed|needs_clarification|failed",
  "input_type": "text|image|pdf|audio",
  "extracted_content": {...},
  "execution_plan": {...},
  "result": {...},
  "clarification_question": "...",
  "logs": ["..."],
  "total_cost": 0.0123
}
```

## 🎨 UI Features

- **Chat-like Interface**: Intuitive messaging interface
- **File Upload**: Drag & drop or click to upload
- **Real-time Processing**: Live updates as tasks execute
- **Cost Tracking**: Running total of API costs
- **Rich Formatting**: Code blocks, bullet points, metadata display

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **AI Models**: Groq Llama
- **OCR**: Tesseract, EasyOCR
- **PDF**: PyPDF2, pdfplumber
- **Testing**: pytest, pytest-asyncio
- **Frontend**: HTML, CSS, Vanilla JavaScript

## 📈 Performance

- Average processing time: 2-5 seconds (depending on task)
- Cost per request: $0.001 - $0.05 (varies by task and content length)
- Supports files up to 50MB
- Audio up to 30 minutes

## 🐛 Troubleshooting

### Tesseract not found
```bash
# Set TESSDATA_PREFIX environment variable
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

### Audio processing fails
```bash
# Ensure FFmpeg is installed and in PATH
ffmpeg -version
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```


