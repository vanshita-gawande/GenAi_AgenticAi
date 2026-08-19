import os
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("AI Research Assistant")


# Split documents
@st.cache_data
def load_and_split_documents():

    documents = []

    data_path = "data"

    for file in os.listdir(data_path):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(
                os.path.join(data_path, file)
            )

            documents.extend(loader.load())

    # =========================================
    # DOCUMENT CLEANING STARTS HERE
    # =========================================

    cleaned_documents = []

    noisy_words = [
        "copyright",
        "all rights reserved",
        "trademark",
        "unauthorized use",
        "limit of liability",
        "disclaimer of warranty"
    ]

    for doc in documents:

        text = doc.page_content.lower()

        page = doc.metadata.get("page", 0)

        # Skip front matter pages
        if page <= 2:
            continue

        noise_score = 0

        for word in noisy_words:
            if word in text:
                noise_score += 1

            # Skip highly noisy chunks/pages
            if noise_score >= 2:
                continue

            # Skip very short useless pages
            if len(text.strip()) < 300:
                continue
        
            cleaned_documents.append(doc)

    documents = cleaned_documents

    # =========================================
    # DOCUMENT CLEANING ENDS HERE
    # =========================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    splits = text_splitter.split_documents(documents)

    return splits

splits = load_and_split_documents()

# Embeddings - for fast generation using this  cache technique
@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# Vector Store
@st.cache_resource
def load_vectorstore():

    if not os.path.exists("chroma_db"):

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

    else:

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

    return vectorstore

vectorstore = load_vectorstore()

# Retriever - MMR = Maximum Marginal Relevance - instead of similar give diverse relevant chunks
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "fetch_k": 40, "lambda_mult": 0.7}
)

# LLM
@st.cache_resource
def load_llm():

    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

llm = load_llm()

# Prompt

prompt = ChatPromptTemplate.from_template("""
You are an advanced AI research assistant.

Your job is to generate accurate, grounded, and natural responses.

Guidelines:
1. Use the retrieved document context as the primary source.
2. Integrate general knowledge ONLY when it improves clarity or fills gaps.
3. Do not explicitly separate the answer into:
   - "document information"
   - "general knowledge"
4. Blend information naturally into one coherent explanation.
5. Avoid repetition.
6. If the retrieved context strongly supports the answer, prioritize it heavily.
7. If the context is weak, mention that some parts are based on general AI knowledge.
8. Keep responses professional and concise.

Context:
{context}

Question:
{question}

Answer:
""")

# Chain
chain = prompt | llm | StrOutputParser()

# User Input
question = st.text_input("Ask a question from your documents")

if question:
    with st.spinner("Generating response..."):

        docs = retriever.invoke(question)
            
    # =========================================
    # POST RETRIEVAL FILTERING
    # =========================================

    filtered_docs = []

    bad_words = [
        "copyright",
        "trademark",
        "all rights reserved",
        "unauthorized use",
        "license agreement",
        "warranty"
    ]

    for doc in docs:

        text = doc.page_content.lower()

        bad_score = 0

        for word in bad_words:

            if word in text:
                bad_score += 1

        if bad_score < 2:
            filtered_docs.append(doc)

    docs = filtered_docs


    # Create context
    context = "\n\n".join([doc.page_content for doc in docs])

    # RETRIEVAL CONFIDENCE CHECK
    weak_retrieval = False

    # Condition 1: Very small context
    if len(context.strip()) < 200:
        weak_retrieval = True

    # RESPONSE GENERATION
    if weak_retrieval:

        fallback_prompt = f"""
        You are an AI and Data Engineering research assistant.

        The uploaded documents are related to:
        - Artificial Intelligence
        - Generative AI
        - LLMs
        - Data Engineering
        - Cloud Platforms
        - Machine Learning
        - Enterprise Technology

        The retrieved context was weak.

        Answer the question using relevant technical knowledge ONLY.

        If a term has multiple meanings, prefer the AI/Data/Technology meaning.

        Question:
        {question}

        Provide a concise and professional answer.
        """
        response = llm.invoke(fallback_prompt).content

        st.warning(
            "Relevant document context not found properly. "
            "Answer generated using LLM knowledge."
        )

    else:

        response = chain.invoke({
            "context": context,
            "question": question
        })

        st.success("Answer generated using document context.")

    # DISPLAY ANSWER
    st.subheader("Answer")
    st.markdown(response)

    # DISPLAY RETRIEVED CHUNKS
    st.subheader("Retrieved Chunks")

    for i, doc in enumerate(docs):

        page = doc.metadata.get("page", "Unknown")

        source = os.path.basename(
            doc.metadata.get("source", "Unknown")
        )

        st.write(
            f"Chunk {i+1} | File: {source} | Page: {page}"
        )
        
        st.write(doc.page_content[:500])

        st.divider()
