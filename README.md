# VideoInsight
Transform hours of video into searchable, analyzable text and engage in dynamic Q&amp;A sessions in less than 3 minutes!
VideoInsight is a cutting-edge AI tool that transcribes videos and enables intelligent question-answering about the content. Transform hours of video into searchable, analyzable text and engage in dynamic Q&A sessions in minutes!
🚀 Features

Rapid Video Processing: Transcribe hour-long videos in under 3 minutes
Advanced Speech Recognition: Utilizes Whisper Large V3 for 95%+ accuracy
Speaker Diarization: Automatically identifies and labels different speakers
Multilingual Support: Handles 100+ languages with ease
Intelligent Q&A: Powered by Google's Gemini Pro model for comprehensive answers
Efficient Retrieval: Uses FAISS and Azure OpenAI embeddings for fast, accurate context retrieval
User-Friendly Interface: Built with Streamlit for easy interaction
Flexible API Management: Rotates between multiple API keys to prevent rate limiting

🛠️ Tech Stack

Python 3.8+
Streamlit
OpenAI Whisper
Google Gemini Pro
FAISS
Azure OpenAI
MoviePy
Pydub

📋 Prerequisites

Python 3.8 or higher
API keys for Google AI, Azure OpenAI, and Groq
FFmpeg installed on your system

🔧 Installation

Clone the repository:
Copygit clone https://github.com/yourusername/VideoInsight.git
cd VideoInsight

Install required packages:
Copypip install -r requirements.txt

Set up your environment variables in a .env file:
CopyGOOGLE_API_KEY=your_google_api_key
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
GROQ_API_KEYS=your_groq_api_key1,your_groq_api_key2


🚀 Usage

Run the Streamlit app:
Copystreamlit run app.py

Upload a video file through the web interface.
Wait for the transcription process to complete (typically under 3 minutes for hour-long videos).
Start asking questions about the video content in the chat interface.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check issues page.
📝 License
This project is MIT licensed.
🙏 Acknowledgements

OpenAI for the Whisper model
Google for the Gemini Pro model
Microsoft for Azure OpenAI services
The Streamlit team for their amazing framework
