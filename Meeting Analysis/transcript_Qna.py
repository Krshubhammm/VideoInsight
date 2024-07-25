import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Ensure API keys are set
if not GOOGLE_API_KEY or not AZURE_API_KEY or not AZURE_ENDPOINT:
    raise ValueError("Please set GOOGLE_API_KEY, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_ENDPOINT in your .env file")

def get_vtt_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_vtt(vtt_text):
    # Remove timestamps
    vtt_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', '', vtt_text)
    
    # Remove WEBVTT header and other metadata
    vtt_text = re.sub(r'WEBVTT\n\n', '', vtt_text)
    vtt_text = re.sub(r'Kind:.*\n', '', vtt_text)
    vtt_text = re.sub(r'Language:.*\n', '', vtt_text)
    
    # Remove empty lines and leading/trailing whitespace
    vtt_text = '\n'.join(line.strip() for line in vtt_text.split('\n') if line.strip())
    
    return vtt_text

def extract_content(vtt_text):
    # Try to extract participant-based content
    pattern = r'<v (.*?)>(.*?)</v>'
    matches = re.findall(pattern, vtt_text)
    
    if matches:
        # If participants are found, create a dictionary
        content_dict = {}
        for match in matches:
            participant, text = match
            if participant not in content_dict:
                content_dict[participant] = []
            content_dict[participant].append(text.strip())
        return content_dict, True
    else:
        # If no participants are found, split the text into sentences
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
        # If there are no participants, join sentences and then split
        full_text = " ".join(content_dict["transcript"])
        all_chunks = text_splitter.split_text(full_text)
    
    return all_chunks

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

def process_transcript(file_path):
    raw_text = get_vtt_text(file_path)
    preprocessed_text = preprocess_vtt(raw_text)
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
    
    # Create a Document object with the context
    context_doc = Document(page_content=context)
    
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": [context_doc], "question": question})
    return response["output_text"]

def main():
    print("Enhanced Meeting Transcript Q&A System")
    
    transcript_file = "transcript/Suddath.vtt"
    try:
        vector_store, content_dict, has_participants, chunks = process_transcript(transcript_file)
        
        # Display chunks
        print("\nText Chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:")
            print(chunk)
            print("-" * 50)
        
        while True:
            user_question = input("\nAsk a question about the meeting (or type 'exit' to quit): ")
            if user_question.lower() == 'exit':
                break
            
            try:
                answer = answer_question(vector_store, content_dict, has_participants, user_question)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"An error occurred while answering the question: {e}")
                print("Please try rephrasing your question or ask something else.")
    except Exception as e:
        print(f"An error occurred while processing the transcript: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()