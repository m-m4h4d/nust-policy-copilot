import streamlit as st
import time
from backend import NustCopilot

st.set_page_config(page_title="NUST Copilot", page_icon="🎓")

# Initialize Backend (Cached)
@st.cache_resource
def get_bot():
    return NustCopilot()

bot = get_bot()

# UI Layout
st.title("🎓 NUST Policies Copilot")
st.markdown("Ask about GPA, Attendance, Refunds, and Hostels.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    strict_mode = st.toggle("Strict Mode (Cite-Checker)", value=True)
    st.info("Strict Mode removes citations not found in the source text.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Write your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get Response
        start_time = time.time()
        answer, sources = bot.answer(prompt, strict_mode=strict_mode)
        latency = time.time() - start_time
        
        message_placeholder.markdown(answer)
        
        # Display Sources (The "Popovers" requirement)
        with st.expander(f"📚 View Sources ({len(sources)} retrieved) - {latency:.2f}s"):
            for doc in sources:
                st.markdown(f"**{doc.metadata['doc_id']}**")
                st.caption(doc.page_content[:300] + "...")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})