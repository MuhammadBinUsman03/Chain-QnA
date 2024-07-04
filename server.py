#!/usr/bin/env python

from operator import itemgetter
from typing import List
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.pydantic_v1 import BaseModel, Field
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_core.documents import Document
import os
from langchain_fireworks import ChatFireworks
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage, AIMessage, trim_messages


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def __init__(self):
        super().__init__()  # Initialize base classes if needed
        self.add_message(SystemMessage(content="You are a helpful assistant."))

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""

        message = self._message_from_BaseMessage(message)
        self.messages.append(message)
    def clear(self) -> None:
        self.messages = []
    
    def _message_from_BaseMessage(self, message: BaseMessage) -> BaseMessage:
        """Map the BaseMessage to appropriate message-type"""
        _type = message.type
        _content=message.content
        if _type == "human":
            return HumanMessage(content = _content)
        elif _type == "ai":
            return AIMessage(content = _content)
        elif _type == "system":
            return SystemMessage(content = _content)
        elif _type == "chat":
            return ChatMessage(content = _content)
        elif _type == "function":
            return FunctionMessage(content = _content)
        elif _type == "tool":
            return ToolMessage(content = _content)
        elif _type == "AIMessageChunk":
            return AIMessageChunk(content = _content)
        elif _type == "HumanMessageChunk":
            return HumanMessageChunk(content = _content)
        elif _type == "FunctionMessageChunk":
            return FunctionMessageChunk(content = _content)
        elif _type == "ToolMessageChunk":
            return ToolMessageChunk(content = _content)
        elif _type == "SystemMessageChunk":
            return SystemMessageChunk(content = _content)
        elif _type == "ChatMessageChunk":
            return ChatMessageChunk(content = _content)
        else:
            raise ValueError(f"Got unexpected message type: {_type}")


store = {}
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]

# history = get_session_history("1", "1")
# history.add_message(SystemMessage(content="You are a helpful assistant."))


os.environ["FIREWORKS_API_KEY"] = 'G0A4kdyDVZ8O5Mm2whtTYiRRM4tbBSwczyyGT17RcK4UHgWB'
model = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

def fake_retriever(query):
    # assert isinstance(query, str)
    return [
        Document(page_content="cats are the answer"),
        Document(page_content="CAT POWERS"),
    ]


fake_retriever = RunnableLambda(fake_retriever)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parser = StrOutputParser()
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

context = itemgetter("question") | fake_retriever | format_docs
first_step = RunnablePassthrough.assign(context= context)

## Without trimming
# chain = first_step | prompt | model | parser

## With trimming, no streaming
chain = first_step | RunnablePassthrough.assign(messages=itemgetter("history") | trimmer) | prompt | model | parser

## for streaming on client side
# chain = first_step | RunnablePassthrough.assign(messages=itemgetter("history") | trimmer) | prompt | model

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    with_message_history,
    path="/chain",
    enable_feedback_endpoint=True
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

