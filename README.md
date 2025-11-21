# Loop AI Hospital Network Assistant 🏥

A voice-enabled hospital search and network verification assistant built with Flask and RAG (Retrieval-Augmented Generation).

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python Flask |
| **Search/RAG** | FAISS (vector search) + BM25 (keyword search) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Voice Input** | Web Speech API (Speech Recognition) |
| **Voice Output** | Web Speech API (Speech Synthesis/TTS) |
| **Frontend** | HTML, CSS, Vanilla JavaScript |
| **Data** | CSV (GIPSA Hospital List) |

## Features

- 🎤 **Voice-to-Voice**: Speak queries and hear responses
- 🔍 **Smart Hospital Search**: Find hospitals by name, brand, or city
- ✅ **Network Verification**: Check if a hospital is in-network
- 💬 **Conversational**: Handles follow-ups and asks clarifying questions
- 🚫 **Out-of-Scope Detection**: Gracefully handles irrelevant queries

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/loop-ai-assistant.git
cd loop-ai-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Add your hospital CSV file
# Place "List of GIPSA Hospitals - Sheet1.csv" in the root directory

# Run the application
python app.py
```

## Usage

Open `http://localhost:5000` in your browser.

### Sample Queries:
- "Tell me 3 hospitals around Bangalore"
- "Find Apollo Hospital in Delhi"
- "Is Manipal Sarjapur in Bangalore in my network?"
- "How many Fortis hospitals are there in Mumbai?"

## Project Structure

```
loop-ai-assistant/
├── app.py                 # Flask backend with RAG search
├── requirements.txt       # Python dependencies
├── templates/
│   └── voice_assistant.html   # Frontend with voice UI
└── List of GIPSA Hospitals - Sheet1.csv  # Hospital data
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the voice assistant UI |
| `/api/conversation` | POST | Processes user queries |
| `/api/health` | GET | Health check endpoint |

## How It Works

1. **Voice Input**: Web Speech API converts speech to text
2. **Intent Extraction**: Regex-based NLU identifies intent (find hospitals, verify network, etc.)
3. **RAG Search**: Hybrid search using FAISS (semantic) + BM25 (keyword)
4. **Response Generation**: Context-aware responses with conversation memory
5. **Voice Output**: Text-to-Speech reads the response aloud

## License

MIT
