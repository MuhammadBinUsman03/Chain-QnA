# 🤗💬 Chain-Q/A

RAG-application for chat completion and question-answering over PDF document built with Langchain 🦜🔗 framework, deployed with LangServe 🦜️🏓 and Streamlit for frontend. Utilizes Chat-completion LLM [Mixtral MoE 8x7B Instruct](https://fireworks.ai/models/fireworks/mixtral-8x7b-instruct) from Fireworks AI and [Cohere Embeddings](https://cohere.com/embed) for text encoding.

![image](https://github.com/MuhammadBinUsman03/Chain-QnA/assets/58441901/b0f385d4-d140-46bd-8f57-5387429c47bd)

### Features ✅
- Entire application (all chains / runnables) deployed with Langserve as a single REST API.
- In-Memory session history to keep track of chat history between user and assistant.
- Streamed token generation
- Message trimming to fit in the context length of model. (for QnA chain only)
- Two chains: for generic QnA / interaction and for question-answering over PDF documents.


# Architecture 📐
![Architecture](https://github.com/MuhammadBinUsman03/Chain-QnA/assets/58441901/75270a5a-86a8-4dcf-85e9-f482fcfb50a1)
## LangServe Encapsulation🦜️🏓
The chains are served through FastAPI endpoints on the same server:
- QnA-Chain: `/chain` 
- RAG-chain: `/rag_chain`

### RAG-Chain 📑🔗
PDF-Document content is posted from client-side on `/chunk` endpoint, where it is recursively splitted and dumped into Chroma VectorDB for similarity retrieval. For a given user query, relevant documents are pulled by `retriever` and passed as context to the model to output response.

```python
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
)
```
### QnA-Chain 💬🔗
Each session is assigned a `user_id` and a `conversation_id` to maintain an In-Memory chat history. This chain is packed with `RunnableWithMessageHistory`.

```python
chain = first_step | RunnablePassthrough.assign(messages=itemgetter("history") | trimmer) | prompt | model
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
```
## Streamlit Client 🔺
- When PDF chat is disabled, all user queries are directed to QnA-chain.
```python
remote_chain_qa = RemoteRunnable("http://localhost:8000/chain/")
```
- For PDF chat, PDF-content is posted to `/chunk` endpoint.
```python
requests.post("http://localhost:8000/chunk/", json={"text":text})
```
Then all queries for PDF QnA are directed to RAG-chain.
```python
remote_chain_rag = RemoteRunnable("http://localhost:8000/rag_chain/")
```
- Chain responses are streamed for smoother UX.
```python
def stream_data(query, remote_chain):
    '''Steaming output generator'''
    config = {"user_id": "user_id", "conversation_id": "conversation_id"}
    for r in remote_chain.stream(
        query,
        config={
                "configurable": config
        },
    ):
        yield r.content + ""
```
## Setup 💻
Create a python virtual environment.Clone the repository and install dependencies
```bash
git clone https://github.com/MuhammadBinUsman03/Chain-QnA.git
cd Chain-QnA
pip install -r requirements.txt
```

Start the LangServe server.
```bash
python server.py
```
Start the Streamlit app in split terminal.
```bash
streamlit run app.py
```
