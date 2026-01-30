import streamlit as st
from llama_cloud import LlamaCloud
import openai

# Page config
st.set_page_config(page_title="AA Literature Chat", page_icon="📖")

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

# Title
st.title("📖 AA Literature Chatbot")
st.caption("Ask questions about the Big Book and 12&12")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about AA literature..."):
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
        
        # Extract text from results
        context = "\n\n".join([node.node.text for node in results.retrieval_nodes])
    
    # Generate response with Claude
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = openai_client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant knowledgeable about AA literature. Use the following context from the Big Book and Twelve Steps and Twelve Traditions to answer questions accurately. If the answer isn't in the context, say so.\n\nContext:\n{context}"},
                    {"role": "user", "content": prompt}
                ],
            )
            
            answer = response.choices[0].message.content
            st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
