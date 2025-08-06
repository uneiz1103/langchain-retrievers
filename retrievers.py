from langchain_community import WikipediaRetriever
from dotenv import load_dotenv

load_dotenv()

# Initialize the retriever
retriever = WikipediaRetriever(top_k_results = 2, lang = 'en')

query = 'the geopolitical history of india and pakistan from the prespective of a chinese'

docs = retriever.invoke(query)

docs

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")