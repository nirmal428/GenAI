import os
import shutil
import hashlib
import gc

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "mainDB-Chroma"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

* {
    box-sizing: border-box;
}


/* ============================================================
   MAIN APPLICATION
   ============================================================ */

.stApp {

    background:
        radial-gradient(
            circle at 80% 10%,
            rgba(35, 95, 220, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 15% 80%,
            rgba(20, 80, 180, 0.08),
            transparent 28%
        ),
        #070b14;

    color: #e8eefc;
}


/* ============================================================
   REMOVE STREAMLIT DEFAULT ELEMENTS
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #080d18 0%,
            #060a12 100%
        );

    border-right:
        1px solid rgba(100, 150, 255, 0.12);
}

section[data-testid="stSidebar"] > div {

    padding-top: 1.5rem;
}


/* Sidebar brand */

.sidebar-brand {

    display: flex;
    align-items: center;

    gap: 8px;

    color: #eef4ff;

    font-size: 19px;

    font-weight: 700;

    letter-spacing: -0.4px;

    margin-bottom: 28px;
}

.sidebar-brand-icon {

    font-size: 20px;
}


/* Sidebar labels */

.sidebar-label {

    color: #697791;

    font-size: 10px;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 1.5px;

    margin-bottom: 9px;
}


/* ============================================================
   HERO
   ============================================================ */

.hero-wrapper {

    padding-top: 20px;

    padding-bottom: 20px;
}


.hero-title {

    font-size: clamp(
        44px,
        6vw,
        76px
    );

    line-height: 0.95;

    font-weight: 800;

    letter-spacing: -4px;

    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            #b8ccff 42%,
            #4c8dff 100%
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    background-clip: text;

    margin-bottom: 20px;
}


.hero-subtitle {

    color: #8794ad;

    font-size: 16px;

    line-height: 1.75;

    max-width: 650px;
}


.online-status {

    display: inline-flex;

    align-items: center;

    gap: 9px;

    margin-top: 20px;

    padding:
        8px 14px;

    border:
        1px solid
        rgba(90, 150, 255, 0.22);

    border-radius: 999px;

    background:
        rgba(20, 35, 65, 0.45);

    color: #7faeff;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.2px;
}


.status-dot {

    width: 7px;

    height: 7px;

    border-radius: 50%;

    background: #54a0ff;

    box-shadow:
        0 0 8px
        rgba(84, 160, 255, 0.9);
}


/* ============================================================
   EMPTY STATE
   ============================================================ */

.empty-state {

    min-height: 390px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    text-align: center;

    margin-top: 30px;

    padding:
        45px 25px;

    border:
        1px dashed
        rgba(92, 135, 210, 0.24);

    border-radius: 24px;

    background:
        linear-gradient(
            145deg,
            rgba(14, 25, 45, 0.55),
            rgba(7, 13, 25, 0.45)
        );

    box-shadow:
        inset
        0 1px 0
        rgba(255,255,255,0.02);
}


.empty-icon {

    width: 72px;

    height: 72px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 20px;

    background:
        rgba(66, 124, 255, 0.14);

    border:
        1px solid
        rgba(91, 144, 255, 0.20);

    font-size: 30px;

    margin-bottom: 22px;

    box-shadow:
        0 0 35px
        rgba(47, 113, 255, 0.08);
}


.empty-title {

    color: #edf3ff;

    font-size: 23px;

    font-weight: 700;

    margin-bottom: 10px;
}


.empty-text {

    max-width: 520px;

    color: #77849c;

    font-size: 14px;

    line-height: 1.8;
}


/* ============================================================
   INFO CARDS
   ============================================================ */

.info-card {

    padding:
        15px;

    border-radius: 13px;

    background:
        rgba(13, 24, 42, 0.58);

    border:
        1px solid
        rgba(83, 134, 228, 0.13);

    margin-top: 12px;
}


.info-title {

    color: #6f7e98;

    font-size: 9px;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 1.2px;

    margin-bottom: 6px;
}


.info-value {

    color: #cddcff;

    font-size: 13px;

    line-height: 1.6;
}


/* ============================================================
   DOCUMENT CARD
   ============================================================ */

.document-card {

    padding: 14px;

    margin-top: 12px;

    border-radius: 13px;

    background:
        rgba(20, 31, 53, 0.60);

    border:
        1px solid
        rgba(91, 137, 230, 0.16);
}


.document-name {

    color: #e7efff;

    font-size: 13px;

    font-weight: 600;

    overflow: hidden;

    text-overflow: ellipsis;

    white-space: nowrap;
}


.document-meta {

    color: #6f7d96;

    font-size: 11px;

    line-height: 1.7;

    margin-top: 5px;
}


/* ============================================================
   SOURCE CARD
   ============================================================ */

.source-card {

    padding:
        10px 12px;

    margin-top: 7px;

    border-radius: 10px;

    background:
        rgba(20, 32, 55, 0.50);

    border:
        1px solid
        rgba(100, 150, 255, 0.10);

    color: #8896b0;

    font-size: 12px;

    line-height: 1.7;
}


.source-title {

    color: #b8caff;

    font-weight: 600;
}


/* ============================================================
   CHAT
   ============================================================ */

[data-testid="stChatMessage"] {

    background:
        rgba(13, 22, 38, 0.45);

    border:
        1px solid
        rgba(93, 135, 220, 0.08);

    border-radius: 15px;

    margin-bottom: 12px;
}


[data-testid="stChatInput"] {
    border-radius: 16px !important;
}

/* ============================================================
   CHAT INPUT — HIGH CONTRAST / VISIBLE TYPING
   ============================================================ */

[data-testid="stChatInput"] > div {
    background: #101827 !important;
    border: 1px solid rgba(100, 160, 255, 0.35) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 0 1px rgba(60, 120, 255, 0.05),
                0 8px 30px rgba(0, 0, 0, 0.18) !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    color: #f5f8ff !important;
    -webkit-text-fill-color: #f5f8ff !important;
    caret-color: #70a7ff !important;
    background: transparent !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
    color: #aab7cc !important;
    -webkit-text-fill-color: #aab7cc !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] input:focus {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    outline: none !important;
}

[data-testid="stChatInput"] button {
    color: #dce9ff !important;
}

[data-testid="stChatInput"] button:hover {
    color: #ffffff !important;
}

/* ============================================================
   GLOBAL TEXT — NORMAL, CLEAR CONTRAST
   ============================================================ */

.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp div,
.stApp li,
.stApp td,
.stApp th {
    color: #e7edf8;
}

.stApp .stMarkdown,
.stApp .stMarkdown p,
.stApp .stMarkdown li {
    color: #e7edf8 !important;
}

[data-testid="stChatMessage"] {
    color: #eef3fc !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #eef3fc !important;
}

/* Keep secondary text readable without making it too bright */
.hero-subtitle,
.empty-text,
.document-meta,
.source-card,
.app-footer {
    color: #aebbd0 !important;
}

.sidebar-label,
.info-title {
    color: #9aa9c0 !important;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploaderDropzone"] {

    background:
        rgba(12, 20, 35, 0.55) !important;

    border:
        1px dashed
        rgba(90, 140, 230, 0.25) !important;

    border-radius: 14px !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {

    width: 100%;

    border-radius: 10px !important;

    border:
        1px solid
        rgba(93, 145, 255, 0.18) !important;

    background:
        rgba(20, 35, 60, 0.55) !important;

    color: #aebfe0 !important;

    transition:
        all 0.2s ease;
}


.stButton > button:hover {

    border-color:
        rgba(100, 160, 255, 0.45) !important;

    background:
        rgba(30, 55, 95, 0.65) !important;

    color: #ffffff !important;
}


/* ============================================================
   EXPANDER
   ============================================================ */

[data-testid="stExpander"] {

    border:
        1px solid
        rgba(91, 137, 230, 0.12) !important;

    border-radius: 12px !important;

    background:
        rgba(12, 20, 35, 0.45) !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.app-footer {

    text-align: center;

    margin-top: 45px;

    padding: 25px 0;

    color: #46536b;

    font-size: 11px;

    border-top:
        1px solid
        rgba(100, 140, 220, 0.08);
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .hero-wrapper {
        padding-top: 10px;
    }

    .hero-title {

        font-size: 45px;

        letter-spacing: -2px;
    }

    .hero-subtitle {

        font-size: 14px;
    }

    .empty-state {

        min-height: 330px;

        padding:
            35px 20px;
    }

    .empty-title {

        font-size: 20px;
    }

    .empty-text {

        font-size: 13px;
    }

}

</style>
""")


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "retriever": None,
    "document_name": None,
    "document_pages": 0,
    "document_size": 0,
    "chunk_count": 0,
    "document_ready": False,
    "processed_file_id": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# CACHED EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# CACHED GROQ MODEL
# ============================================================

@st.cache_resource
def get_llm():

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to your .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key,
    )


# ============================================================
# PROMPT
# ============================================================

def get_prompt():

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant that answers questions using the provided PDF context.

Rules:
1. Answer the user's question using only the information provided in the context.
2. Do not make up or assume information that is not present in the context.
3. If the answer cannot be found in the context, clearly say:
   "I could not find the answer in the provided PDF."
4. Give a clear and concise answer.
5. When useful, explain the answer step-by-step.
6. Preserve important technical terms, names, numbers, and formulas from the PDF.

Context:
{context}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )


# ============================================================
# FILE SIZE
# ============================================================

def format_file_size(size):

    if size < 1024:

        return f"{size} B"

    if size < 1024 * 1024:

        return f"{size / 1024:.1f} KB"

    if size < 1024 * 1024 * 1024:

        return f"{size / (1024 * 1024):.1f} MB"

    return f"{size / (1024 * 1024 * 1024):.1f} GB"


# ============================================================
# FILE HASH
# ============================================================

def get_file_id(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# REMOVE OLD VECTOR DATABASE
# ============================================================

def remove_old_database():

    # Release old retriever first.
    st.session_state.retriever = None

    gc.collect()

    if os.path.exists(CHROMA_DIR):

        shutil.rmtree(
            CHROMA_DIR,
            ignore_errors=True
        )


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    if uploaded_file is None:

        raise ValueError(
            "No PDF was uploaded."
        )


    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------

    file_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    with open(
        file_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    loader = PyPDFLoader(
        file_path
    )

    documents = loader.load()


    if not documents:

        raise ValueError(
            "The PDF is empty or no text could be extracted."
        )


    # --------------------------------------------------------
    # Clean metadata
    # --------------------------------------------------------

    for document in documents:

        document.metadata["source"] = (
            uploaded_file.name
        )

        original_page = document.metadata.get(
            "page",
            0
        )

        try:

            original_page = int(
                original_page
            )

        except:

            original_page = 0

        document.metadata["page"] = (
            original_page + 1
        )


    # --------------------------------------------------------
    # Split PDF
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


    chunks = splitter.split_documents(
        documents
    )


    if not chunks:

        raise ValueError(
            "No text chunks were created from the PDF."
        )


    # --------------------------------------------------------
    # Remove previous database
    # --------------------------------------------------------

    remove_old_database()


    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embedding_model = get_embedding_model()


    # --------------------------------------------------------
    # Create Chroma
    # --------------------------------------------------------

    vectorstore = Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        collection_name="pdf_rag_collection",

        persist_directory=CHROMA_DIR,
    )


    # --------------------------------------------------------
    # MMR Retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


    return {

        "retriever": retriever,

        "document_name":
            uploaded_file.name,

        "document_pages":
            len(documents),

        "document_size":
            uploaded_file.size,

        "chunk_count":
            len(chunks),
    }


# ============================================================
# CLEAR DOCUMENT
# ============================================================

def clear_document():

    remove_old_database()


    st.session_state.messages = []

    st.session_state.document_name = None

    st.session_state.document_pages = 0

    st.session_state.document_size = 0

    st.session_state.chunk_count = 0

    st.session_state.document_ready = False

    st.session_state.processed_file_id = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-brand">

        <span class="sidebar-brand-icon">
            📚
        </span>

        <span>
            PDF RAG AI
        </span>

    </div>
    """)


    st.html("""
    <div class="sidebar-label">
        Upload PDF
    </div>
    """)


    uploaded_file = st.file_uploader(

        "Upload PDF",

        type=["pdf"],

        label_visibility="collapsed",

        help="Upload a PDF to create your RAG knowledge base.",
    )


    # ========================================================
    # PROCESS NEW PDF
    # ========================================================

    if uploaded_file is not None:

        current_file_id = get_file_id(
            uploaded_file
        )


        if (
            st.session_state.processed_file_id
            != current_file_id
        ):

            try:

                with st.spinner(
                    "Processing PDF..."
                ):

                    result = process_pdf(
                        uploaded_file
                    )


                # --------------------------------------------
                # Save state
                # --------------------------------------------

                st.session_state.retriever = (
                    result["retriever"]
                )

                st.session_state.document_name = (
                    result["document_name"]
                )

                st.session_state.document_pages = (
                    result["document_pages"]
                )

                st.session_state.document_size = (
                    result["document_size"]
                )

                st.session_state.chunk_count = (
                    result["chunk_count"]
                )

                st.session_state.document_ready = True

                st.session_state.processed_file_id = (
                    current_file_id
                )

                # New PDF = new conversation

                st.session_state.messages = []


                st.success(
                    "PDF processed successfully!"
                )


            except Exception as error:

                st.session_state.document_ready = False

                st.session_state.retriever = None

                st.error(
                    "PDF processing failed."
                )

                st.exception(
                    error
                )


    # ========================================================
    # DOCUMENT INFORMATION
    # ========================================================

    if st.session_state.document_ready:

        st.html(f"""
        <div class="document-card">

            <div class="document-name">

                📄
                {st.session_state.document_name}

            </div>

            <div class="document-meta">

                {st.session_state.document_pages}
                pages

                &nbsp;•&nbsp;

                {format_file_size(
                    st.session_state.document_size
                )}

                <br>

                {st.session_state.chunk_count}
                chunks

            </div>

        </div>
        """)


    else:

        st.html("""
        <div class="info-card">

            <div class="info-title">
                Document Status
            </div>

            <div class="info-value">
                Waiting for PDF...
            </div>

        </div>
        """)


    # ========================================================
    # RETRIEVER SETTINGS
    # ========================================================

    st.write("")


    with st.expander(
        "⚙️ Retriever Settings"
    ):

        st.caption(
            "Maximum Marginal Relevance retrieval"
        )

        st.write(
            "Search Type: MMR"
        )

        st.write(
            "k: 4"
        )

        st.write(
            "fetch_k: 10"
        )

        st.write(
            "lambda_mult: 0.5"
        )


    # ========================================================
    # REMOVE DOCUMENT
    # ========================================================

    if st.session_state.document_ready:

        if st.button(
            "🗑️ Remove Document",
            use_container_width=True,
        ):

            clear_document()

            st.rerun()


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.session_state.messages:

        if st.button(
            "🧹 Clear Chat",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.rerun()


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero-wrapper">

    <div class="hero-title">
        PDF RAG AI
    </div>

    <div class="hero-subtitle">
        Ask questions. Explore your documents.
        <br>
        Get answers grounded in your PDFs.
    </div>

    <div class="online-status">

        <span class="status-dot"></span>

        RAG SYSTEM ONLINE

    </div>

</div>
""")


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.document_ready:

    st.html("""
    <div class="empty-state">

        <div class="empty-icon">
            📚
        </div>

        <div class="empty-title">
            Upload a PDF to start
        </div>

        <div class="empty-text">

            Upload a book, research paper, notes,
            documentation, or any PDF and ask
            questions about it.

        </div>

    </div>
    """)


    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    st.write("")

    st.html("""
    <div class="sidebar-label">
        How It Works
    </div>
    """)


    col1, col2, col3 = st.columns(3)


    with col1:

        st.html("""
        <div class="info-card">

            <div class="info-title">
                01 — Upload
            </div>

            <div class="info-value">
                Upload your PDF document.
            </div>

        </div>
        """)


    with col2:

        st.html("""
        <div class="info-card">

            <div class="info-title">
                02 — Retrieve
            </div>

            <div class="info-value">
                MMR retrieves the most relevant chunks.
            </div>

        </div>
        """)


    with col3:

        st.html("""
        <div class="info-card">

            <div class="info-title">
                03 — Ask
            </div>

            <div class="info-value">
                Ask questions grounded in your PDF.
            </div>

        </div>
        """)


# ============================================================
# DOCUMENT READY
# ============================================================

else:

    # --------------------------------------------------------
    # ACTIVE DOCUMENT
    # --------------------------------------------------------

    st.html(f"""
    <div class="info-card">

        <div class="info-title">
            Active Document
        </div>

        <div class="info-value">

            📄
            {st.session_state.document_name}

            &nbsp; • &nbsp;

            {st.session_state.document_pages}
            pages

            &nbsp; • &nbsp;

            {st.session_state.chunk_count}
            chunks

        </div>

    </div>
    """)


    st.write("")


    # ========================================================
    # DISPLAY CHAT HISTORY
    # ========================================================

    for message in st.session_state.messages:

        role = message["role"]

        content = message["content"]


        with st.chat_message(
            role
        ):

            st.markdown(
                content
            )


            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            if (
                role == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "📚 Retrieved Sources"
                ):

                    for index, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.html(f"""
                        <div class="source-card">

                            <span class="source-title">

                                Source {index}

                            </span>

                            <br>

                            📄
                            {source["source"]}

                            &nbsp; • &nbsp;

                            Page
                            {source["page"]}

                        </div>
                        """)


    # ========================================================
    # CHAT INPUT
    # ========================================================

    query = st.chat_input(
        "Ask something about your PDF..."
    )


    if query:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                query
            )


        # ----------------------------------------------------
        # ASSISTANT
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            try:

                retriever = (
                    st.session_state.retriever
                )


                if retriever is None:

                    raise ValueError(
                        "Retriever is not available. "
                        "Please upload the PDF again."
                    )


                # ============================================
                # RETRIEVAL
                # ============================================

                with st.spinner(
                    "Searching your PDF..."
                ):

                    docs = retriever.invoke(
                        query
                    )


                # ============================================
                # NO RESULTS
                # ============================================

                if not docs:

                    answer = (
                        "I could not find the answer "
                        "in the provided PDF."
                    )

                    st.markdown(
                        answer
                    )


                    st.session_state.messages.append(
                        {
                            "role": "assistant",

                            "content": answer,

                            "sources": [],
                        }
                    )


                else:

                    # ========================================
                    # CONTEXT
                    # ========================================

                    context = "\n\n".join(

                        doc.page_content

                        for doc in docs
                    )


                    # ========================================
                    # PROMPT
                    # ========================================

                    prompt = get_prompt()


                    final_prompt = prompt.invoke(
                        {
                            "context": context,

                            "question": query,
                        }
                    )


                    # ========================================
                    # LLM
                    # ========================================

                    with st.spinner(
                        "Generating answer..."
                    ):

                        llm = get_llm()

                        response = llm.invoke(
                            final_prompt
                        )


                    answer = response.content


                    # ========================================
                    # DISPLAY ANSWER
                    # ========================================

                    st.markdown(
                        answer
                    )


                    # ========================================
                    # COLLECT SOURCES
                    # ========================================

                    sources = []


                    for doc in docs:

                        metadata = doc.metadata


                        source_name = metadata.get(
                            "source",
                            st.session_state.document_name
                        )


                        page_number = metadata.get(
                            "page",
                            "Unknown"
                        )


                        sources.append(
                            {
                                "source": source_name,

                                "page": page_number,
                            }
                        )


                    # ========================================
                    # REMOVE DUPLICATES
                    # ========================================

                    unique_sources = []

                    seen = set()


                    for source in sources:

                        source_key = (
                            source["source"],
                            source["page"],
                        )


                        if source_key not in seen:

                            seen.add(
                                source_key
                            )

                            unique_sources.append(
                                source
                            )


                    # ========================================
                    # DISPLAY SOURCES
                    # ========================================

                    if unique_sources:

                        with st.expander(
                            "📚 Retrieved Sources"
                        ):

                            for index, source in enumerate(
                                unique_sources,
                                start=1
                            ):

                                st.html(f"""
                                <div class="source-card">

                                    <span class="source-title">

                                        Source {index}

                                    </span>

                                    <br>

                                    📄
                                    {source["source"]}

                                    &nbsp; • &nbsp;

                                    Page
                                    {source["page"]}

                                </div>
                                """)


                    # ========================================
                    # SAVE RESPONSE
                    # ========================================

                    st.session_state.messages.append(
                        {
                            "role": "assistant",

                            "content": answer,

                            "sources": unique_sources,
                        }
                    )


            except Exception as error:

                error_message = (
                    "Sorry, something went wrong "
                    "while processing your question."
                )


                st.error(
                    error_message
                )


                st.exception(
                    error
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",

                        "content": error_message,

                        "sources": [],
                    }
                )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="app-footer">

    PDF RAG AI
    &nbsp;•&nbsp;
    LangChain
    &nbsp;•&nbsp;
    HuggingFace
    &nbsp;•&nbsp;
    Chroma
    &nbsp;•&nbsp;
    Groq

</div>
""")