from langchain_community.retrievers import ArxivRetriever

# Create retriever
retriever = ArxivRetriever(
    load_max_docs=3
)

# Search arXiv
docs = retriever.invoke("Machine learning")

# Print retrieved papers
for i, doc in enumerate(docs, start=1):
    print(f"\n{'=' * 60}")
    print(f"Paper {i}")
    print(f"{'=' * 60}")

    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Published:", doc.metadata.get("Published"))
    print("Entry ID:", doc.metadata.get("Entry ID"))

    print("\nContent:")
    print(doc.page_content[:500])