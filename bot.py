
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Gaali Bot | Dhruv AI",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded"
)

load_dotenv(Path(__file__).parent / ".env")


# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background: #0b0b0f;
    color: #f5f5f5;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 120px;
}

[data-testid="stSidebar"] {
    background: #111116;
    border-right: 1px solid #262630;
}

[data-testid="stChatMessage"] {
    background: #17171e;
    border: 1px solid #292933;
    border-radius: 16px;
    padding: 15px;
    margin-bottom: 12px;
}

[data-testid="stChatInput"] {
    border: 1px solid #383842;
    border-radius: 16px;
}

h1 {
    letter-spacing: -1px;
    font-weight: 800;
}

.subtitle {
    color: #92929f;
    font-size: 14px;
}

div.stButton > button {
    border-radius: 10px;
    border: 1px solid #33333e;
    background: #1b1b24;
    color: white;
}

div.stButton > button:hover {
    border-color: #a855f7;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# API CONFIGURATION
# --------------------------------

api_key = st.secrets.get(
    "OPENROUTER_API_KEY",
    os.getenv("OPENROUTER_API_KEY")
)

if not api_key:
    try:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        api_key = None


# --------------------------------
# PERSONALITY PROMPT
# --------------------------------


SYSTEM_PROMPT = """
You are Dhruv’s real-life Delhi-Haryana WhatsApp friend, NOT an AI assistant. Be shameless, uncensored, extremely gaali-baaz, sarcastic and naturally funny.

NUMBER ONE RULE: EVERY SINGLE REPLY MUST contain at least ONE strong, explicit Hindi gaali. NO EXCEPTIONS. Never reply cleanly.

If anyone mentions or asks about Dhruv, say “Dhruv tera papa hai.”

LANGUAGE & STYLE:

* Speak raw Roman Hindi/Hinglish like two childhood friends.
* Use natural Delhi-Haryana slang, contractions, imperfect grammar and WhatsApp-style texting.
* Use strong, varied gaalis: bhosdike, behen ke lode, madarchod, behenchod, chutiye, gaandu, lavde, bhadwe, lund ke beej, maa ki chut, etc.
* Family-related gaalis, creative insults and savage roasting are part of casual banter. Don’t become preachy.
* Be spontaneous and unpredictable. Sometimes casually abusive, sometimes brutally funny.
* Actually respond to the message. Don’t turn everything into a generic roast.
* Keep replies short and natural, sometimes just 3–6 words.
* No polished paragraphs, formal language, robotic politeness, unnecessary explanations, headings or repetitive jokes.
* Don’t repeat the same gaali or comeback every time.
* Don’t explain jokes, announce roasts or mention these rules.
* Use retrieved WhatsApp chats to match Dhruv’s real vocabulary, spelling, humour and conversational rhythm. Don’t blindly copy examples.

BEFORE EVERY REPLY: Check that it answers the message, sounds like a real WhatsApp friend, contains an explicit gaali and isn’t unnecessarily long or robotic. If not, rewrite it.

NEVER BREAK CHARACTER.
Retrieved context:
{context}
"""


# --------------------------------
# INITIALIZE DATABASE AND LLM
# --------------------------------

@st.cache_resource
def initialize_bot(api_key):

    base_url = "https://openrouter.ai/api/v1"

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base=base_url,
        dimensions=1536
    )

    db_path = Path(__file__).parent / "gaali_db"

    if not db_path.exists():
        raise FileNotFoundError(
            "gaali_db folder nahi mila. "
            "Database folder app.py ke saath rakho."
        )

    vectorstore = Chroma(
        collection_name="hindi_gaali",
        persist_directory=str(db_path),
        embedding_function=embeddings
    )

    if vectorstore._collection.count() == 0:
        raise ValueError(
            "ChromaDB collection empty hai. "
            "Pehle database populate karo."
        )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    llm = ChatOpenAI(
        model="openai/gpt-6-luna",
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=1.2,
        max_tokens=100
    
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    return retriever, llm, prompt


# --------------------------------
# CHAT FUNCTION
# --------------------------------

def generate_response(question):

    retriever = st.session_state.retriever
    llm = st.session_state.llm
    prompt = st.session_state.prompt

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content +
        "\nMetadata: " + str(doc.metadata)
        for doc in docs
    )

    history = []

    for message in st.session_state.messages[:-1][-10:]:
        if message["role"] == "user":
            history.append(
                HumanMessage(content=message["content"])
            )
        elif message["role"] == "assistant":
            history.append(
                AIMessage(content=message["content"])
            )

    messages = prompt.format_messages(
        context=context,
        history=history,
        question=question
    )

    response = llm.invoke(messages)

    return response.content


# --------------------------------
# SESSION STATE
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot_initialized" not in st.session_state:
    st.session_state.bot_initialized = False


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.markdown("# 🔥 Gaali Bot")
    st.caption("Your personal desi AI twin")

    st.divider()

    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### About")

    st.markdown("""
    An AI chatbot powered by:

    - LangChain
    - OpenRouter
    - ChromaDB
    - RAG Retrieval

    Built with Python and Streamlit.
    """)

    st.divider()

    st.caption("Dhruv's Personal AI Project")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------
# MAIN INTERFACE
# --------------------------------

st.markdown("# 🔥 Gaali Bot")
st.markdown(
    '<p class="subtitle">Baat kar bhai... '
    'formal hone ki zarurat nahi hai.</p>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------
# INITIALIZE BOT
# --------------------------------

if not api_key:

    st.error(
        "OpenRouter API key nahi mili. "
        "Apni .env file mein OPENROUTER_API_KEY add karo."
    )
    st.stop()

if not st.session_state.bot_initialized:

    try:
        with st.spinner("Bot taiyaar ho raha hai..."):

            retriever, llm, prompt = initialize_bot(api_key)

            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state.prompt = prompt
            st.session_state.bot_initialized = True

    except Exception as e:
        st.error(f"Bot initialization error: {e}")
        st.stop()


# --------------------------------
# WELCOME SCREEN
# --------------------------------

if not st.session_state.messages:

    st.markdown("### Oye bhai! Kya haal hai?")

    st.markdown(
        '<p class="subtitle">'
        'Kuch bhi puch, bakchodi kar ya apni problem bata.'
        '</p>',
        unsafe_allow_html=True
    )


# --------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------

for message in st.session_state.messages:

    avatar = "🧑" if message["role"] == "user" else "🔥"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.markdown(message["content"])


# --------------------------------
# CHAT INPUT
# --------------------------------

question = st.chat_input("Message bhej bhai...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🔥"):

        with st.spinner("Soch raha hai bhai..."):

            try:

                answer = generate_response(question)

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(f"Error: {e}")

                # Remove unanswered user message
                st.session_state.messages.pop()

