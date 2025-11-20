import streamlit as st
import database
import bcrypt
import pickle
import time
import os

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq

# Load secrets from Streamlit secrets
# Load secrets from Streamlit secrets and map them to your variable names
sec_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

st.set_page_config(layout="wide")

# Validate secrets
if not HUGGINGFACEHUB_API_TOKEN or not GROQ_API_KEY:
    st.error(
        "API keys are not configured.\n\n"
        "Please set `GROQ_API_KEY` and `HUGGINGFACEHUB_API_TOKEN` in Streamlit secrets."
    )
    st.stop()

# Make HF token available to HuggingFaceEmbeddings via env var
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# Optional: also expose Groq key via env if any integration needs it
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the HuggingFace model and ChatGroq for chatbot
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
)

qa_prompt_template = """
You are a professional financial analyst. Please provide clear, well-structured answers.

Format your response as follows:
- If the information asked is in bullet points always give response in bullet points
- Use bullet points for key information
- Provide clear section headers
- Keep paragraphs concise
- Use numbered lists when appropriate
- Summarize key findings at the end

Context: {summaries}

Question: {question}

Please provide a comprehensive, well-formatted answer:
"""

QA_PROMPT = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["summaries", "question"],
)

chat_llm = ChatGroq(
    temperature=0.4,
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
)

# Streamlit UI Setup
st.title("Equity Research Tool & Chatbot 🔍🤖")


# Login and Signup Logic
def login():
    username = st.text_input("Username", "")
    email = st.text_input("Email (must be a Gmail)", "")
    password = st.text_input("Password", "", type="password")
    login_button = st.button("Login")

    if login_button:
        if database.login_user(username, email, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.email = email
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username, email, or password.")


def signup():
    username = st.text_input("Create Username", "")
    email = st.text_input("Enter your Email (@gmail.com only)", "")
    password = st.text_input("Create Password", "", type="password")
    signup_button = st.button("Signup")

    if signup_button:
        if not email.endswith("@gmail.com"):
            st.error("Invalid email. Please use a valid '@gmail.com' email.")
        elif database.register_user(username, email, password):
            st.success(f"Account created for {username}. Please log in.")
        else:
            st.error("Username or email already exists.")


def forgot_password():
    email = st.text_input("Enter your Email to Reset Password", "")
    new_password = st.text_input("Enter New Password", type="password")
    reset_button = st.button("Reset Password")

    if reset_button:
        if database.reset_password(email, new_password):
            st.success("Password reset successfully! Please log in.")
        else:
            st.error("Email not found!")


# Check if user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
    signup()
    forgot_password()
else:
    st.sidebar.success(f"Welcome {st.session_state.username}!")

    # **Main Content - Only after login**
    # Left Side: Equity Research Tool
    with st.sidebar:
        st.header("Equity Research")
        st.subheader("News Articles URLs")
        urls = [st.text_input(f"URL {i + 1}") for i in range(3)]
        process_url_click = st.button("Process URLs")

    # Define the path for persistent data storage
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, "faiss_vectordb.pkl")

    main_placeholder = st.empty()

    if process_url_click:
        loader = WebBaseLoader(urls)
        main_placeholder.caption("Data loading started...")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=0,
        )
        main_placeholder.caption("Text splitting started...")
        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings()
        vectorstore_index = FAISS.from_documents(docs, embeddings)
        main_placeholder.caption("Embedding vector started...")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_index, f)

    query = main_placeholder.text_input("Ask a question about the articles:")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    chain_type_kwargs={"prompt": QA_PROMPT},
                )
                result = chain.invoke({"question": query})

                st.header("Answer")
                st.write(result["answer"])

                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources")
                    for source in sources.split("\n"):
                        st.write(source)

    # Right Side: Chatbot
    with st.sidebar:
        st.header("General Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = chat_llm.invoke(prompt)
            st.markdown(response.content)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.content}
            )
