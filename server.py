#!/usr/bin/env python
from dotenv import load_dotenv
import os
from operator import itemgetter
from pydantic import BaseModel as BM
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
from langchain_fireworks import ChatFireworks
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage, AIMessage, trim_messages, AIMessageChunk
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub


load_dotenv()
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)
model = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")

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


# Store session history based on user_id and corresponding conversation_id
store = {}
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]



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
    return [
        Document(page_content="________"),
        Document(page_content="____"),
    ]


fake_retriever = RunnableLambda(fake_retriever)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

trimmer = trim_messages(
    max_tokens=2048,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# parser = StrOutputParser()
########## Q/A Chain ##############
context = itemgetter("question") | fake_retriever | format_docs
first_step = RunnablePassthrough.assign(context= context)

## Without trimming
# chain = first_step | prompt | model | parser

## With trimming, no streaming
# chain = first_step | RunnablePassthrough.assign(messages=itemgetter("history") | trimmer) | prompt | model | parser

## for streaming on client side
chain = first_step | RunnablePassthrough.assign(messages=itemgetter("history") | trimmer) | prompt | model

# Pack Q/A chain in the Message History Runnable
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
#######################################################33

# Doc chunking
class Item(BM):
    text: str

# Dummy retriever, globally modified upon post request
retriever = Chroma.from_documents(documents=[Document(page_content='___'), Document(page_content='___')], embedding=CohereEmbeddings()).as_retriever(search_type="similarity", search_kwargs={"k": 2})

@app.post("/chunk/")
async def getRetriever(item: Item):
    '''Recieves PDF document content, creates chunks, dumps in chromaDB and prepares a retriever for rag chain'''
    global retriever, RecursiveCharacterTextSplitter, Chroma
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
    )
    chunks = text_splitter.split_text(item.text)
    vectorstore = Chroma.from_texts(texts=chunks, embedding=CohereEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
 

############## RAG Chain ##########################
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
)
###################################################

#  Q/A chain route
add_routes(
    app,
    with_message_history,
    path="/chain",
    enable_feedback_endpoint=True
)

#  RAG chain route
add_routes(
    app,
    rag_chain,
    path="/rag_chain",
    enable_feedback_endpoint=True
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

