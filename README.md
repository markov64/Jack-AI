# JACK — Personal AI Assistant for Data-Driven Productivity

**JACK (Just A Chill Knowledge-buddy)** is a modular Python-based AI assistant developed by Georgy Markov.  
It integrates large language models, financial data APIs, symbolic computation, and voice-based interaction into a secure, local tool designed for high-performance productivity.

JACK is structured for use cases spanning financial analysis, natural language querying, media control, and intelligent automation — optimized for academic, personal, or prototype-level deployments.

---

## 🔍 Core Functionality

- **Natural Language Processing**: Interface with GPT (OpenAI, Groq-compatible)
- **Financial Analytics**: Integrates with Alpha Vantage and yFinance for market data retrieval and simulation
- **Symbolic Computation**: Supports equation solving and expression parsing via SymPy
- **Media Control**: Connects to Spotify using Spotipy (OAuth 2.0)
- **Voice Output**: Utilizes macOS `say()` for spoken responses (optional)

All services are securely configured via environment variables using `.env`.

---

## 🛠 Technology Stack

| Feature               | Tool/API Used              |
|-----------------------|----------------------------|
| Language Models       | OpenAI, Groq               |
| Market Data           | Alpha Vantage, yFinance    |
| Music API             | Spotify (via Spotipy)      |
| Math Engine           | SymPy                      |
| Scripting Language    | Python                     |
| Voice Support (macOS) | Native `say()` command     |

---

## 🔧 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/markov64/Jack-AI.git
   cd Jack-AI
