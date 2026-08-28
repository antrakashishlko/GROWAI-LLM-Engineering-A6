# ============================================================
# 1. DOCUMENT LOADING AND CHUNKING
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the document
with open("document.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Document loaded successfully!")
print("Total characters:", len(text))

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))

# Show the first 3 chunks
print("\n--- First 3 Chunks ---")

for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i + 1}:")
    print(chunk)

# ============================================================
# 2. EMBEDDINGS AND CHROMADB
# ============================================================

from sentence_transformers import SentenceTransformer
import chromadb

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nEmbedding model loaded successfully!")

# Create a persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the collection
collection = chroma_client.get_or_create_collection(
    name="ai_documents"
)

# Create embeddings for all chunks
embeddings = embedding_model.encode(chunks).tolist()

# Store chunks and embeddings in ChromaDB
# Using upsert prevents duplicate-ID errors
collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("Chunks stored in ChromaDB successfully!")
print("Total documents in ChromaDB:", collection.count())

# ============================================================
# 3. BM25 KEYWORD SEARCH
# ============================================================

from rank_bm25 import BM25Okapi

# Tokenize chunks for BM25
tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

# Create BM25 index
bm25 = BM25Okapi(tokenized_chunks)

print("\nBM25 index created successfully!")

# Test BM25 search
test_query = "Moravec paradox"

query_tokens = test_query.lower().split()

bm25_scores = bm25.get_scores(query_tokens)

# Get top 3 results
top_indices = bm25_scores.argsort()[-3:][::-1]

print("\n--- BM25 Test Results ---")

for rank, index in enumerate(top_indices, start=1):
    print(f"\nRank {rank}:")
    print(chunks[index])

# ============================================================
# 4. HYBRID SEARCH WITH RECIPROCAL RANK FUSION
# ============================================================

def hybrid_search(query, top_k=10):
    """
    Combines vector search and BM25 keyword search
    using Reciprocal Rank Fusion (RRF).
    """

    # --------------------------------------------------------
    # 1. Vector Search
    # --------------------------------------------------------

    query_embedding = embedding_model.encode([query]).tolist()

    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    vector_documents = vector_results["documents"][0]

    # --------------------------------------------------------
    # 2. BM25 Search
    # --------------------------------------------------------

    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = bm25_scores.argsort()[-top_k:][::-1]

    bm25_documents = [
        chunks[index]
        for index in bm25_indices
    ]

    # --------------------------------------------------------
    # 3. Reciprocal Rank Fusion
    # --------------------------------------------------------

    rrf_scores = {}

    # RRF constant
    k = 60

    # Add vector search scores
    for rank, document in enumerate(vector_documents):
        rrf_scores[document] = (
            rrf_scores.get(document, 0)
            + 1 / (k + rank + 1)
        )

    # Add BM25 scores
    for rank, document in enumerate(bm25_documents):
        rrf_scores[document] = (
            rrf_scores.get(document, 0)
            + 1 / (k + rank + 1)
        )

    # Sort by combined RRF score
    ranked_documents = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-k documents
    return [
        document
        for document, score in ranked_documents[:top_k]
    ]

# Test hybrid search
test_query = "What are the goals of artificial intelligence?"

hybrid_results = hybrid_search(
    test_query,
    top_k=10
)

print("\n" + "=" * 60)
print("HYBRID SEARCH RESULTS")
print("=" * 60)

for i, result in enumerate(hybrid_results, start=1):
    print(f"\n--- Result {i} ---")
    print(result)

# ============================================================
# 5. CROSS-ENCODER RERANKING
# ============================================================

from sentence_transformers import CrossEncoder

# Load the cross-encoder model
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("\nCross-encoder loaded successfully!")

def rerank_results(query, documents, top_k=3):
    """
    Re-ranks retrieved documents using a cross-encoder
    and returns the top results.
    """

    # Create query-document pairs
    pairs = [
        [query, document]
        for document in documents
    ]

    # Calculate relevance scores
    scores = reranker.predict(pairs)

    # Combine documents and scores
    ranked_results = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-k results
    return ranked_results[:top_k]

# Rerank hybrid search results
reranked_results = rerank_results(
    test_query,
    hybrid_results,
    top_k=3
)

print("\n" + "=" * 60)
print("CROSS-ENCODER TOP 3 RESULTS")
print("=" * 60)

for i, (document, score) in enumerate(
    reranked_results,
    start=1
):
    print(f"\n--- Result {i} | Score: {score:.4f} ---")
    print(document)

# ============================================================
# 6. LLM ANSWER GENERATION
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Create the Ollama LLM
llm = ChatOllama(
    model="qwen3:0.6b",
    base_url="http://localhost:11434"
)

# RAG system prompt
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a document question-answering assistant.

Answer using ONLY the context provided below.

If the answer is not found in the context, say exactly:
"Not in context."

Do not use outside knowledge.

Context:
{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)

# Create the RAG chain
rag_chain = rag_prompt | llm

def answer_question(question):
    """
    Retrieves relevant chunks, reranks them,
    and generates an answer using the top 3 chunks.
    """

    # Retrieve top 10 using hybrid search
    hybrid_results = hybrid_search(
        question,
        top_k=10
    )

    # Rerank the top 10 and select the best 3
    reranked_results = rerank_results(
        question,
        hybrid_results,
        top_k=3
    )

    # Combine the top 3 chunks into context
    context = "\n\n".join(
        document
        for document, score in reranked_results
    )

    # Generate the final answer
    response = rag_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response.content

# Test one question
question = "What are the goals of artificial intelligence?"

answer = answer_question(question)

print("\n" + "=" * 60)
print("QUESTION")
print("=" * 60)
print(question)

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(answer)

# ============================================================
# 7. FIVE TEST QUESTIONS
# ============================================================

test_questions = [
    # 1. Exact keyword matching - BM25 strength
    "What is Moravec's paradox?",

    # 2. Semantic understanding - Vector search strength
    "Why did symbolic AI struggle with tasks that humans find easy?",

    # 3. AI history
    "How did artificial intelligence develop as a field?",

    # 4. AI ethics
    "What ethical concerns are associated with artificial intelligence?",

    # 5. AI regulation
    "Why is regulation of artificial intelligence important?"
]

print("\n" + "=" * 60)
print("RAG TEST RESULTS")
print("=" * 60)

for i, question in enumerate(
    test_questions,
    start=1
):

    print("\n" + "=" * 60)
    print(f"QUESTION {i}")
    print("=" * 60)

    print(question)

    answer = answer_question(question)

    print("\nANSWER:")
    print(answer)