import streamlit as st
import os
import base64
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

# Configure API Keys securely
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY")

# Constants
LOGO_PATH = "mitan_logo.png"
PDF_FILE = "MITAN_Energy_Company_Profile.pdf"

# Function to encode image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Embed logo and header in HTML with improved layout
logo_base64 = get_base64_image(LOGO_PATH)
st.markdown(
    f"""
    <style>
        .header-container {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            margin-bottom: 20px;
        }}
        .header-container h1 {{
            margin: 0;
            font-size: 32px;
            font-weight: bold;
        }}
        .chat-container {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .chat-message {{
            padding: 10px;
            border-radius: 8px;
            background-color: #f1f1f1;
            max-width: 80%;
            margin-bottom: 8px;
        }}
        .chat-user {{
            background-color: #4CAF50;
            color: white;
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .chat-assistant {{
            background-color: #2196F3;
            color: white;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .chat-icon {{
            width: 20px;
            height: 20px;
        }}
    </style>
    <div class="header-container">
        <img src="data:image/png;base64,{logo_base64}" style="width: 50px; margin-right: 15px;">
        <h1>Mitan Energy AI Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("Powered by **Mitan AI**! Ask anything about Mitan Energy.")

# Generate random unique ID for the session
@st.cache_resource
def generate_unique_id():
    return str(uuid.uuid4())

unique_session_id = generate_unique_id()

# Initialize Language Model
@st.cache_resource
def init_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

llm = init_llm()

# Initialize embeddings and vector store
@st.cache_resource
def init_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return InMemoryVectorStore(embeddings)

vector_store = init_vector_store()

# Load and split the document
@st.cache_resource
def load_and_split_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(docs)

all_splits = load_and_split_documents(PDF_FILE)
vector_store.add_documents(documents=all_splits)

# Create retriever tool
retriever = vector_store.as_retriever(search_type="mmr")
retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_company_information",
    description="Search and return information about Mitan Energy Company.",
)

# Create memory and agent
memory = MemorySaver()
config = {"configurable": {"thread_id": unique_session_id}}

SYSTEM_MESSAGE = '''
You are a helpful AI assistant for Mitan Energy Company Ltd.
Your name is Mitan AI. Only answer questions relating to Mitan Energy.
Use three sentences maximum and keep the answer as concise as possible.
Always use the retriever tool to get any information about the company. 
Call tool.
'''

@st.cache_resource
def init_agent():
    return create_react_agent(llm, [retriever_tool], state_modifier=SYSTEM_MESSAGE, checkpointer=memory)

langgraph_agent_executor = init_agent()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
st.markdown("### Chat History:")
for msg in st.session_state.messages:
    message_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
    icon = "👤" if msg["role"] == "user" else "🤖"
    st.markdown(f'<div class="chat-message {message_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

# User input and AI response
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-message chat-user">👤 {prompt}</div>', unsafe_allow_html=True)
    
    try:
        response = langgraph_agent_executor.invoke(
            input={"messages": st.session_state.messages},
            config=config
        )
        ai_response = response["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.markdown(f'<div class="chat-message chat-assistant">🤖 {ai_response}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")