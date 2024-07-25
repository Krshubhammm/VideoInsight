# VideoInsight
AI-powered Video Transcription and Intelligent Q&A System

## Description
VideoInsight is a cutting-edge AI tool that rapidly transcribes videos and enables intelligent question-answering based on the video content. Built with Streamlit and leveraging state-of-the-art language models, this application demonstrates the practical implementation of Retrieval-Augmented Generation (RAG) for video content analysis.

## Features

- Rapid Video Processing: Transcribes hour-long videos in under 3 minutes.
- Advanced Speech Recognition: Utilizes Whisper Large V3 model for high-accuracy transcription.
- Intelligent Chunking: Splits transcribed text into manageable chunks for efficient processing.
- Vector Embedding: Creates and stores vector embeddings of text chunks using Azure OpenAI for efficient retrieval.
- Conversational AI: Enables users to ask questions about the video content and receive contextually relevant answers.
- RAG Implementation: Utilizes score-based Retrieval-Augmented Generation to provide accurate and context-aware responses.
- K-Similarity Search: Implements FAISS for fast and efficient similarity search of relevant context.
- API Key Management: Rotates between multiple Groq API keys to prevent rate limiting and ensure uninterrupted service.

## Technology Stack

- Streamlit: For creating an intuitive and interactive user interface.
- OpenAI Whisper (via Groq API): For state-of-the-art speech recognition and transcription.
- Google Gemini Pro: Leverages advanced language understanding and generation capabilities.
- Azure OpenAI: Provides powerful embedding models for text representation.
- LangChain: Facilitates the creation of the conversational retrieval chain.
- FAISS: Enables efficient similarity search and clustering of dense vectors.
- MoviePy: For video processing and audio extraction.
- Pydub: Handles audio processing tasks.
- Python-dotenv: Management of environment variables.

## Installation

- Clone this repository:
git clone https://github.com/krshubhammm/VideoInsight.git
cd VideoInsight

- Install the required dependencies:
  pip install -r requirements.txt
  
- Set up your API keys in a .env file:
- GOOGLE_API_KEY=your_google_api_key_here
- AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
- AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
- GROQ_API_KEYS=your_groq_api_key1,your_groq_api_key2,your_groq_api_key3


## Usage

Start the Streamlit app:
- streamlit run app.py
- Access the application in your web browser at http://localhost:8501
- Upload a video file using the provided interface.
- Wait for the transcription process to complete.
- Start asking questions about the video content in the chat interface.
