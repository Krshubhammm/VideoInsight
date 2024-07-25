import streamlit as st
import os
import tempfile
import re
import time
import json
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Ensure API keys are set
if not GOOGLE_API_KEY or not AZURE_API_KEY or not AZURE_ENDPOINT:
    st.error("Please set GOOGLE_API_KEY, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_ENDPOINT in your .env file")
    st.stop()

class APIKeyManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.usage = {key: 0 for key in self.api_keys}

    def _load_api_keys(self):
        keys = os.environ.get("GROQ_API_KEYS", "").split(",")
        return [key.strip() for key in keys if key.strip()]

    def get_current_key(self):
        return self.api_keys[self.current_key_index]

    def rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    def update_usage(self, key, amount):
        self.usage[key] += amount
        if self.usage[key] >= 90:  # 90% usage threshold
            self.rotate_key()

api_manager = APIKeyManager()

def get_groq_client():
    return OpenAI(
        api_key=api_manager.get_current_key(),
        base_url="https://api.groq.com/openai/v1"
    )

@st.cache_data
def convert_video_to_audio(video_file):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    audio_path = video_path.replace('.mp4', '.mp3')
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()
    
    os.unlink(video_path)
    return audio_path

@st.cache_data
def chunk_audio(audio_path, chunk_length_ms=600000):
    audio = AudioSegment.from_mp3(audio_path)
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    os.unlink(audio_path)
    return chunks
def transcribe_audio_chunk(audio_chunk):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        audio_chunk.export(temp_file.name, format="mp3")
        with open(temp_file.name, "rb") as audio_file:
            while True:
                try:
                    groq = get_groq_client()
                    transcript = groq.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio_file,
                        response_format="json"
                    )
                    api_manager.update_usage(api_manager.get_current_key(), 10)  # Assume 10% usage per request
                    break
                except Exception as e:
                    if "API key" in str(e):
                        st.warning(f"API key error: {e}. Rotating to next key.")
                        api_manager.rotate_key()
                    else:
                        st.warning(f"Error: {e}. Retrying in 5 seconds...")
                        time.sleep(5)
    os.unlink(temp_file.name)
    return json.loads(transcript.model_dump_json())

def json_to_vtt(json_transcript, time_offset=0):
    vtt = "WEBVTT\n\n"
    if 'text' in json_transcript:
        start = ms_to_time(time_offset)
        end = ms_to_time(time_offset + 10000)
        text = json_transcript['text'].strip()
        vtt += f"{start} --> {end}\n{text}\n\n"
    return vtt

def ms_to_time(ms):
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def process_video(video_file):
    audio_path = convert_video_to_audio(video_file)
    audio_chunks = chunk_audio(audio_path)
    vtt_contents = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(audio_chunks):
        json_transcript = transcribe_audio_chunk(chunk)
        vtt_content = json_to_vtt(json_transcript, i * 600000)
        vtt_contents.append(vtt_content)
        progress_bar.progress((i + 1) / len(audio_chunks))
    combined_vtt = "".join(vtt_contents)
    return combined_vtt

def preprocess_vtt(vtt_text):
    vtt_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', '', vtt_text)
    vtt_text = re.sub(r'WEBVTT\n\n', '', vtt_text)
    vtt_text = re.sub(r'Kind:.*\n', '', vtt_text)
    vtt_text = re.sub(r'Language:.*\n', '', vtt_text)
    vtt_text = '\n'.join(line.strip() for line in vtt_text.split('\n') if line.strip())
    return vtt_text

def extract_content(vtt_text):
    pattern = r'<v (.*?)>(.*?)</v>'
    matches = re.findall(pattern, vtt_text)
    
    if matches:
        content_dict = {}
        for match in matches:
            participant, text = match
            if participant not in content_dict:
                content_dict[participant] = []
            content_dict[participant].append(text.strip())
        return content_dict, True
    else:
        sentences = re.split(r'(?<=[.!?])\s+', vtt_text)
        return {"transcript": sentences}, False

def get_text_chunks(content_dict, has_participants, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    
    if has_participants:
        for participant, texts in content_dict.items():
            participant_text = f"Participant {participant}:\n" + "\n".join(texts)
            chunks = text_splitter.split_text(participant_text)
            all_chunks.extend(chunks)
    else:
        full_text = " ".join(content_dict["transcript"])
        all_chunks = text_splitter.split_text(full_text)
    
    return all_chunks

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        deployment="text-embedding-ada-002",
        api_version="2024-02-01"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are an expert meeting analyzer with a deep understanding of conversation dynamics and content analysis. Your task is to provide comprehensive, accurate, and relevant answers based on the given context from a meeting transcript.

    Here are your instructions:

    1. Carefully analyze the entire provided context before formulating your response.
    2. If participant information is available, pay close attention to which participant said what. Always attribute statements to the correct participant in your response.
    3. If multiple participants discussed a topic, summarize each participant's perspective or contribution.
    4. If the transcript doesn't have participant information, focus on the content and flow of the discussion without attributing statements to specific individuals.
    5. Ensure your answer is complete and addresses all aspects of the question.
    6. Provide specific examples or direct quotes when relevant, attributing them to participants if that information is available.
    7. If asked about numbers or statistics, include the precise figures and who mentioned them (if known).
    8. If the information to fully answer the question is not available in the context, clearly state this and provide the best possible answer with the available information.
    9. Maintain the tone and style of a professional analyst throughout your response.
    10. If the question is ambiguous, address all possible interpretations.
    11. For questions about specific participants (if applicable), provide a comprehensive summary of their contributions to the discussion.
    12. Summarize key points at the end of longer responses for clarity, noting which participants contributed to each point if that information is available.
    13. If the transcript lacks participant information, focus on the overall themes, key points, and progression of ideas in the meeting.

    Context:
    {context}

    Question: {question}

    Comprehensive Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_transcript(vtt_content):
    preprocessed_text = preprocess_vtt(vtt_content)
    content_dict, has_participants = extract_content(preprocessed_text)
    text_chunks = get_text_chunks(content_dict, has_participants)
    vector_store = get_vector_store(text_chunks)
    return vector_store, content_dict, has_participants, text_chunks

def get_content_info(content_dict, has_participants):
    if has_participants:
        participants = list(content_dict.keys())
        content_info = "Participants in the meeting:\n"
        for i, participant in enumerate(participants, 1):
            content_info += f"{i}. {participant}\n"
    else:
        content_info = "This transcript does not contain participant information. The analysis will focus on the overall content and themes of the discussion."
    return content_info

def answer_question(vector_store, content_dict, has_participants, question):
    content_info = get_content_info(content_dict, has_participants)
    docs = vector_store.similarity_search(question, k=5)
    context = content_info + "\n" + "\n".join([doc.page_content for doc in docs])
    
    context_doc = Document(page_content=context)
    
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": [context_doc], "question": question})
    return response["output_text"]

def main():
    st.title("Video Transcription and Q&A System")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if 'vtt_content' not in st.session_state:
            with st.spinner("Processing video and generating VTT..."):
                vtt_content = process_video(uploaded_file)
                st.session_state.vtt_content = vtt_content

            st.success("VTT file generated successfully!")
            st.download_button(
                label="Download VTT file",
                data=vtt_content,
                file_name="transcript.vtt",
                mime="text/vtt"
            )

        if 'vector_store' not in st.session_state:
            with st.spinner("Processing transcript for Q&A..."):
                vector_store, content_dict, has_participants, chunks = process_transcript(st.session_state.vtt_content)
                st.session_state.vector_store = vector_store
                st.session_state.content_dict = content_dict
                st.session_state.has_participants = has_participants

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message(f"user"):
                st.write(question)
            with st.chat_message(f"assistant"):
                st.write(answer)

        # User input for questions
        user_question = st.chat_input("Ask a question about the video content:")
        if user_question:
            try:
                with st.spinner("Generating answer..."):
                    answer = answer_question(st.session_state.vector_store, st.session_state.content_dict, st.session_state.has_participants, user_question)
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, answer))

                # Display the new message
                with st.chat_message("user"):
                    st.write(user_question)
                with st.chat_message("assistant"):
                    st.write(answer)

            except Exception as e:
                st.error(f"An error occurred while answering the question: {e}")
                st.warning("Please try rephrasing your question or ask something else.")

if __name__ == "__main__":
    main()