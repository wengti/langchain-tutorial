from typing import List, Literal
from pydantic import BaseModel, Field
from backend.core import chat_with_llm

import streamlit as st

# set page config (browser tab)
st.set_page_config(page_title="NextJS Doc Helper")

# Title and sidebar
st.title("NextJS Documentation Helper")
with st.sidebar:
    clicked = st.button("Clear Chat", width="stretch")
    if clicked:
        st.session_state.pop("conversation")


# Setting up conversations
class ConversationMessage(BaseModel):
    role: Literal["ai", "user"]
    message: str
    sources: List[str] = Field(default_factory=list)


def display_message(item: ConversationMessage):
    with st.chat_message(item.role):
        st.write(item.message)
        if len(message_sources := item.sources) > 0:
            print(message_sources)
            with st.expander("Sources"):
                for s in message_sources:
                    st.markdown(f"* {s}")
    return None


conversation = [
    ConversationMessage(
        role="ai", message="Hello, you can ask me anything about NextJS!"
    )
]

if "conversation" not in st.session_state:
    st.session_state["conversation"] = conversation


for item in st.session_state["conversation"]:
    display_message(item)

# User input
user_input = st.chat_input()
if user_input:
    new_entry = ConversationMessage(role="user", message=user_input)
    st.session_state["conversation"].append(new_entry)
    display_message(new_entry)

    with st.spinner("Thinking...", show_time=True):
        ai_answer, ai_sources = chat_with_llm(question=user_input)
        st.session_state["conversation"].append(
            ConversationMessage(role="ai", message=ai_answer, sources=ai_sources)
        )
        st.rerun()
