from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

docs = [
    Document(page_content='LangChain makes it easy to work with LLMs.'),
    Document(page_content='LangChain is used to build LLM based applications.'),
    Document(page_content='Chroma is used to store and search document embeddings.'),
    Document(page_content='Embeddings are vector representations of text.'),
    Document(page_content='MMR helps you get diverse results when doing similarity search.'),
    Document(page_content='LangChain supports Chroma, FAISS, Pinecone, and more.')
]

embedding_model = OpenAIEmbeddings()

vectore_store = FAISS.from_documents(
    embedding_model = embedding_model,
    document = docs
)

retriever = vectore_store.as_retriever(
    search_type='mmr',
    search_kwargs = {'k':3, 'lambda_mult':0.5}
)

query = 'what is langchain'

results = retriever.invoke(query)

for i, doc in enumerate:
    print(f"\n- Result {i+1} -")
    print(doc.page_content)
