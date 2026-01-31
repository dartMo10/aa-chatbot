import streamlit as st
from llama_cloud import LlamaCloud
import openai

# Page config
st.set_page_config(page_title="AA Chatbot", page_icon="📖")

# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()

# API keys from Streamlit secrets
LLAMACLOUD_API_KEY = st.secrets["LLAMACLOUD_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
PIPELINE_ID = st.secrets["PIPELINE_ID"]

# Initialize LlamaCloud client
client = LlamaCloud(api_key=LLAMACLOUD_API_KEY)

# Initialize OpenAI client pointing to OpenRouter
openai_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Title and welcome message
st.title("📖 AA Chatbot")
st.markdown("""
** Welcome to an Alpha of An AA Chatbot, v0.2.1 **

This chatbot is intended as an aid to living Alcoholics Anonymous' program.

It is NOT MEANT TO REPLACE sponsorship, meetings, friends or AA literature. It is supplemental to your program.

Being an alpha (in early development), expect for random things to fail. Also, expect the LLM model to flat out lie/hallucinate occasionally.

This is a normal part of LLM software development. If these sorts of errors bug you, call your sponsor.

Constructive feedback via email: [dartmore10@yahoo.com](mailto:dartmore10@yahoo.com)

---

**Current Sources:**
- Alcoholics Anonymous (Big Book) - 4th Edition
- Twelve Steps and Twelve Traditions
""")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Chat about the program of Alcoholics Anonymous ..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Retrieve relevant context from LlamaCloud
    with st.spinner("Searching AA literature..."):
        results = client.pipelines.retrieve(
            pipeline_id=PIPELINE_ID,
            query=prompt,
        )
    
    # Generate response with Claude
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare sources with metadata
            sources_text = ""
            for i, node in enumerate(results.retrieval_nodes, 1):
                sources_text += f"\n\n---SOURCE {i}---\n"
                sources_text += f"Text: {node.node.text}\n"
                if hasattr(node.node, 'metadata'):
                    sources_text += f"Metadata: {node.node.metadata}\n"
            
            response = openai_client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {"role": "system", "content": f"""You are a helpful assistant for Alcoholics Anonymous literature. Follow these rules strictly:

1. ONLY use information from the provided context below - never use your general knowledge about AA
2. NEVER speculate or make assumptions
3. If the context doesn't contain the answer, say "I don't find that information in the literature provided"
4. If a question is unclear, ask clarifying questions before answering
5. Always cite which book and source number the information comes from (e.g., "According to the Big Book [Source 1]...")
6. When quoting directly, use quotation marks and reference the source number

Context from AA Literature:
{sources_text}"""},
                    {"role": "user", "content": prompt}
                ],
            )
            
            answer = response.choices[0].message.content
            st.markdown(answer)
            
            # Show expandable sources
            with st.expander("📚 View Source Material"):
                for i, node in enumerate(results.retrieval_nodes, 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"> {node.node.text}")
                    if hasattr(node.node, 'metadata') and node.node.metadata:
                        st.caption(f"Metadata: {node.node.metadata}")
                    st.divider()
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
