# VideoInsight
VideoInsight is an advanced AI-powered tool that quickly transcribes videos and enables intelligent question-answering about the content. Transform your video content into searchable, analyzable text and engage in dynamic Q&A sessions in minutes!

# Implemented Features
 1: Swift Video Processing: Transcribe videos efficiently, typically processing hour-long content in under 3 minutes.
 2: State-of-the-Art Speech Recognition: Utilizes Whisper Large V3 model for high-accuracy transcription.
 3: Intelligent Q&A System
 4: Powered by Google's Gemini Pro model: Provides comprehensive and context-aware answers.
 5: k-Similarity Search: Retrieves relevant context efficiently.
 6: Score-based RAG (Retrieval-Augmented Generation): Improves answer quality.
    Efficient Retrieval
 7: FAISS (Facebook AI Similarity Search): Enables fast and scalable similarity search.
 8: API Key Management: Multiple Groq API Keys Rotation: Prevents rate limiting and ensures uninterrupted service.
 # Tech Stack
Python 3.8+
Streamlit
OpenAI Whisper (via Groq API)
Google Gemini Pro
FAISS
Azure OpenAI
MoviePy
Pydub
# Prerequisites
Python 3.8 or higher
API keys:
Google AI (for Gemini Pro)
Azure OpenAI
Multiple Groq API keys
FFmpeg installed on your system
# Installation
Clone the repository:
git clone https://github.com/krshubhammm/VideoInsight.git
cd VideoInsight
Install required packages:

pip install -r requirements.txt
Set up your environment variables in a .env file:

GOOGLE_API_KEY=your_google_api_key
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
GROQ_API_KEYS=your_groq_api_key1,your_groq_api_key2,your_groq_api_key3
# Usage
Run the Streamlit app:
streamlit run app.py
Upload a video file through the web interface.

Wait for the transcription process to complete.

Start asking questions about the video content in the chat interface.

This README.md provides a clear and structured overview of your project, making it easy for others to understand and contribute.
