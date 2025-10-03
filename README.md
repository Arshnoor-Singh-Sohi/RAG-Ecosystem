# Building Production-Grade RAG Systems: A Complete Tutorial

## Overview

Welcome to this comprehensive guide on building Retrieval-Augmented Generation (RAG) systems from the ground up. This project takes you on a journey from implementing basic RAG to mastering advanced techniques used in production systems.

**What is RAG?** Retrieval-Augmented Generation is a technique that enhances Large Language Models (LLMs) by giving them access to external knowledge. Instead of relying solely on their training data, RAG systems first retrieve relevant information from a knowledge base, then use that information to generate accurate, grounded responses.

**Why does this matter?** LLMs can hallucinate or provide outdated information. RAG solves this by connecting models to current, domain-specific data, making them reliable for real-world applications like customer support, research assistance, and enterprise knowledge management.

**What you'll learn:**
- How to build a basic RAG pipeline (indexing, retrieval, generation)
- Advanced query optimization techniques that dramatically improve retrieval accuracy
- Intelligent routing to multiple data sources
- State-of-the-art indexing strategies for handling complex documents
- Self-correcting RAG architectures that verify their own outputs
- Comprehensive evaluation frameworks to measure system performance

This isn't just a code repository—it's a complete education in building RAG systems that actually work in production.

---

## Core Concepts: Understanding RAG Architecture

Before diving into code, let's understand what makes RAG tick. Think of RAG as a three-stage pipeline:

### Stage 1: Indexing (Building the Knowledge Base)

Imagine you're organizing a massive library. You can't just throw books on shelves randomly and expect to find anything later. Instead, you need a system:

1. **Load documents** - Gather all your source material (web pages, PDFs, databases)
2. **Chunk the content** - Break large documents into smaller, digestible pieces (typically 500-1000 characters). Why? Because LLMs have limited context windows, and smaller chunks are more precise for retrieval.
3. **Create embeddings** - Convert each chunk into a mathematical vector that captures its meaning. Similar concepts will have similar vectors.
4. **Store in a vector database** - Save these embeddings in a specialized database (like Chroma or Pinecone) that can quickly find similar vectors.

### Stage 2: Retrieval (Finding Relevant Information)

When a user asks a question, your system doesn't read every document. Instead:

1. **Embed the query** - Convert the user's question into the same vector format
2. **Semantic search** - Find the chunks whose embeddings are most similar to the query embedding
3. **Return top results** - Typically retrieve the 3-5 most relevant chunks

This is like having a librarian who instantly knows which books contain information relevant to your question.

### Stage 3: Generation (Creating the Answer)

Now that you have relevant context:

1. **Construct a prompt** - Combine the user's question with the retrieved chunks
2. **Send to LLM** - Ask the model to answer based *only* on the provided context
3. **Return response** - The model generates a response grounded in your retrieved information

**The challenge:** Basic RAG works, but it's not production-ready. Questions might be poorly phrased, retrieval can miss relevant documents, and generated answers might still hallucinate. This project teaches you how to solve these problems.

---

## Getting Started

### Prerequisites

You should have a basic understanding of:
- Python programming (intermediate level)
- API concepts
- Machine learning fundamentals (helpful but not required)

### Installation

**Step 1: Clone this repository**
```bash
git clone <your-repo-url>
cd rag-ecosystem
```

**Step 2: Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with these dependencies:
```text
langchain==0.1.0
langchain-community==0.0.20
langchain-openai==0.0.5
langchain-core==0.1.23
chromadb==0.4.22
beautifulsoup4==4.12.3
openai==1.12.0
cohere==4.47
ragatouille==0.0.7
deepeval==0.21.0
grouse-eval==0.1.0
ragas==0.1.5
python-dotenv==1.0.0
tiktoken==0.6.0
```

**Step 4: Set up environment variables**

Create a `.env` file in your project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # For re-ranking
```

Get your API keys:
- OpenAI: https://platform.openai.com/api-keys
- Cohere: https://dashboard.cohere.com/api-keys

**Step 5: Launch Jupyter**
```bash
jupyter notebook RAG.ipynb
```

---

## Project Structure

```
rag-ecosystem/
├── RAG.ipynb                 # Main tutorial notebook
├── .env                      # API keys (create this, not tracked in git)
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
├── architecture.png         # High-level system diagram
├── simplerag.png           # Basic RAG visualization
├── indexing.png            # Indexing process diagram
├── retrieval.png           # Retrieval workflow
├── generator.png           # Generation stage
└── README.md               # This file
```

**Key Files Explained:**
- **RAG.ipynb** - The heart of the project. Contains all code examples, explanations, and progressive complexity from basic to advanced RAG techniques.
- **.env** - Stores your secret API keys. Never commit this to version control!
- **requirements.txt** - Lists all Python packages needed to run the project.

---

## Code Walkthrough: From Basic to Advanced RAG

### Part 1: Building Your First RAG System (30 minutes)

Let's start with the absolute basics. This section teaches you the foundational pattern you'll use throughout.

**1.1 Loading and Preparing Documents**

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Load a blog post about AI agents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-header", "post-title")
        )
    ),
)
docs = loader.load()
```

**What's happening here?**
- We're using `WebBaseLoader` to scrape a web page
- The `bs_kwargs` parameter tells BeautifulSoup to only grab specific sections (post content, header, title) and ignore navigation, ads, etc.
- This gives us cleaner data to work with

**1.2 Chunking the Document**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Each chunk will be ~1000 characters
    chunk_overlap=200   # Overlap prevents splitting mid-concept
)
splits = text_splitter.split_documents(docs)
```

**Why these numbers?**
- **chunk_size=1000**: Small enough to be precise, large enough to maintain context
- **chunk_overlap=200**: If a sentence or concept spans the boundary, the overlap ensures we don't lose it

Think of this like creating overlapping snapshots of a document. Each chunk stands alone but shares some context with its neighbors.

**1.3 Creating Embeddings and Vector Store**

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()  # Uses OpenAI's text-embedding-ada-002
)
```

**Behind the scenes:**
- Each chunk gets converted to a 1536-dimensional vector (for OpenAI embeddings)
- These vectors are stored in Chroma, an open-source vector database
- Chroma builds an index for fast similarity search

**1.4 Creating a Retriever**

```python
retriever = vectorstore.as_retriever()

# Test it out
docs = retriever.get_relevant_documents("What is Task Decomposition?")
print(docs[0].page_content)
```

This simple interface hides the complexity: when you call `get_relevant_documents()`, it embeds your query, searches the vector space, and returns the most similar chunks.

**1.5 Building the Generation Chain**

```python
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load a pre-built prompt template
prompt = hub.pull("rlm/rag-prompt")

# Create the LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=1)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the complete RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use it!
response = rag_chain.invoke("What is Task Decomposition?")
print(response)
```

**Understanding the chain:**
- `{"context": retriever | format_docs, ...}` - Retrieve docs and format them as a string
- `| prompt` - Insert context and question into the prompt template
- `| llm` - Send to the language model
- `| StrOutputParser()` - Extract just the text from the response

This is LangChain's "expression language" (LCEL). The `|` operator chains components together, making the data flow obvious.

---

### Part 2: Advanced Query Transformations (45 minutes)

Basic RAG fails when questions are ambiguous or use different terminology than your documents. Here's how to fix that.

**2.1 Multi-Query Generation**

**The Problem:** A user asks "What is task decomposition for LLM agents?" But your documents might use phrases like "breaking down tasks," "subtask creation," or "hierarchical planning." A single query vector might miss relevant documents.

**The Solution:** Generate multiple versions of the question.

```python
from langchain.prompts import ChatPromptTemplate

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

question = "What is task decomposition for LLM agents?"
queries = generate_queries.invoke({"question": question})

# Queries will look like:
# 1. How do LLM agents utilize task decomposition in their operations?
# 2. Can you explain the concept of task decomposition as applied to LLM agents?
# 3. In what ways do LLM agents benefit from task decomposition?
# ... and so on
```

Now you retrieve documents for *all* these queries and combine the results:

```python
from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """Flatten all retrieved docs and remove duplicates"""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question": question})
```

**Why this works:** By searching from multiple angles, you cast a wider net. Documents that didn't match the original phrasing might match one of the variations.

**2.2 RAG-Fusion with Reciprocal Rank Fusion**

Multi-Query retrieves more documents, but how do you decide which are most relevant? RAG-Fusion uses a clever scoring algorithm called Reciprocal Rank Fusion (RRF).

**The Intuition:** If a document appears near the top of results for *multiple* queries, it's probably very relevant. RRF rewards this consistency.

```python
def reciprocal_rank_fusion(results: list[list], k=60):
    """
    For each document, sum up its reciprocal ranks across all queries.
    k is a smoothing parameter (typically 60).
    """
    fused_scores = {}
    
    for docs in results:  # For each query's results
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Add reciprocal rank: 1/(rank + k)
            fused_scores[doc_str] += 1 / (rank + k)
            
    # Sort by score, highest first
    reranked = [
        (loads(doc), score) 
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked
```

**Example:** Suppose Document A appears at position 2 in query 1, position 1 in query 2, and position 10 in query 3. Its RRF score would be:
```
1/(2+60) + 1/(1+60) + 1/(10+60) ≈ 0.048
```

Documents consistently ranked high across queries get high scores.

**2.3 Query Decomposition**

**The Problem:** Complex questions like "What are the main components of an LLM-powered autonomous agent system?" are actually multiple questions bundled together.

**The Solution:** Break it into sub-questions, answer each, then synthesize.

```python
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. 
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
Generate multiple search queries related to: {question}
Output (3 queries):"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries_decomposition = (
    prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

question = "What are the main components of an LLM-powered autonomous agent system?"
sub_questions = generate_queries_decomposition.invoke({"question": question})

# Sub-questions might be:
# 1. What are the core functionalities of an LLM in an autonomous agent system?
# 2. How does natural language understanding contribute to the efficiency of LLM-powered agents?
# 3. What role does reinforcement learning play in LLM-powered autonomous agents?
```

Now answer each sub-question independently:

```python
from langchain import hub

prompt_rag = hub.pull("rlm/rag-prompt")

rag_results = []
for sub_question in sub_questions:
    retrieved_docs = retriever.get_relevant_documents(sub_question)
    answer = (prompt_rag | llm | StrOutputParser()).invoke({
        "context": retrieved_docs, 
        "question": sub_question
    })
    rag_results.append(answer)
```

Finally, synthesize all sub-answers into a complete response:

```python
def format_qa_pairs(questions, answers):
    formatted = ""
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        formatted += f"Question {i}: {q}\nAnswer {i}: {a}\n\n"
    return formatted.strip()

context = format_qa_pairs(sub_questions, rag_results)

synthesis_template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the original question: {question}
"""

synthesis_prompt = ChatPromptTemplate.from_template(synthesis_template)
final_answer = (synthesis_prompt | llm | StrOutputParser()).invoke({
    "context": context, 
    "question": question
})
```

**Why this works:** Complex questions often span multiple topics. By decomposing, you ensure comprehensive coverage and detailed answers.

---

### Part 3: Intelligent Routing (30 minutes)

Not all queries should go to the same data source. Imagine you have separate document collections for Python docs, JavaScript docs, and Go docs. You want to route queries intelligently.

**3.1 Logical Routing with Structured Output**

```python
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field

class RouteQuery(BaseModel):
    """Data model for routing decision"""
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ..., 
        description="Given a user question, choose which datasource would be most relevant."
    )

# Create an LLM that returns structured output matching our model
structured_llm = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to the appropriate data source.
Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

router = prompt | structured_llm

# Test it
result = router.invoke({
    "question": "Why doesn't the following code work: from langchain_core.prompts import ChatPromptTemplate"
})
print(result.datasource)  # Output: "python_docs"
```

**The Magic:** Using `with_structured_output()`, the LLM is constrained to return a valid `RouteQuery` object. No parsing needed—you get a Python object with type safety.

Now wire it up to different retrieval chains:

```python
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return python_rag_chain
    elif "js_docs" in result.datasource.lower():
        return js_rag_chain
    else:
        return golang_rag_chain

full_chain = router | RunnableLambda(choose_route)
final_answer = full_chain.invoke({"question": question})
```

**3.2 Semantic Routing**

What if your routing decision isn't categorical but based on semantic similarity? For example, route physics questions to a serious academic tone and math questions to a step-by-step pedagogical tone.

```python
from langchain_core.prompts import PromptTemplate
from langchain.utils.math import cosine_similarity

physics_template = """You are a very smart physics professor. 
You are great at answering questions about physics in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions.
You are so good because you are able to break down hard problems into their component parts, 
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

def prompt_router(input):
    """Route based on semantic similarity to prompt templates"""
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar_index = similarity.argmax()
    chosen_prompt = prompt_templates[most_similar_index]
    return PromptTemplate.from_template(chosen_prompt)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI()
    | StrOutputParser()
)

# Try it with a physics question
response = chain.invoke("What's a black hole")
```

**How it works:** The query is embedded, then compared to embeddings of each prompt template. The most semantically similar template is chosen dynamically. This is incredibly flexible—you can have dozens of expert prompts and automatically route to the best one.

---

### Part 4: Advanced Indexing Strategies (60 minutes)

Basic chunking works, but it suffers from a fundamental trade-off: small chunks are precise but lack context; large chunks have context but dilute relevance. Advanced indexing strategies solve this.

**4.1 Multi-Representation Indexing**

**Core Idea:** Store small summaries in the vector database but retrieve full documents.

**Why this works:** Summaries are dense and focused, making them easier to search. But once you find the right summary, you retrieve its full parent document for generation, giving you all the context you need.

```python
import uuid
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Step 1: Load documents
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# Step 2: Generate summaries for each document
summary_chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)

summaries = summary_chain.batch(docs, {"max_concurrency": 5})

# Step 3: Set up the multi-vector retriever
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
store = InMemoryByteStore()  # Stores full documents
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# Step 4: Add data
doc_ids = [str(uuid.uuid4()) for _ in docs]

# Create summary documents with doc_id metadata
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Store summaries in vector store
retriever.vectorstore.add_documents(summary_docs)

# Store full docs in docstore, linked by ID
retriever.docstore.mset(list(zip(doc_ids, docs)))

# Step 5: Query
query = "Memory in agents"
retrieved_docs = retriever.get_relevant_documents(query)
```

**What happens during retrieval:**
1. Your query is embedded and compared against summary embeddings
2. The best matching summaries are identified
3. Their `doc_id` metadata is used to look up full documents from the docstore
4. Full documents are returned for generation

**4.2 ColBERT: Token-Level Precision**

Traditional embeddings create one vector per chunk. ColBERT creates one vector per *token*, enabling word-level precision.

```python
from ragatouille import RAGPretrainedModel

# Load the ColBERT model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index a document (it handles chunking and tokenization internally)
full_document = get_wikipedia_page("Hayao_Miyazaki")

RAG.index(
    collection=[full_document],
    index_name="Miyazaki-ColBERT",
    max_document_length=180,
    split_documents=True,
)

# Query with token-level matching
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
```

**How ColBERT works:**
1. Every word in your query gets its own context-aware embedding
2. Every word in each document gets its own embedding
3. For each query word, find its best matching document word (max similarity)
4. Sum these scores across all query words

This is like a semantic keyword search—it catches relevant documents even if they use different phrasing, but it's precise enough to match specific terms.

---

### Part 5: Re-Ranking and Self-Correction (45 minutes)

Retrieval alone isn't perfect. Re-ranking is a crucial second pass that uses a more powerful (and slower) model to reorder results.

**5.1 Dedicated Re-Ranking with Cohere**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Set up base retriever (retrieves 10 docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Add Cohere re-ranker
compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever
)

# Query
question = "What is task decomposition for LLM agents?"
reranked_docs = compression_retriever.get_relevant_documents(question)

for doc in reranked_docs:
    print(f"Score: {doc.metadata['relevance_score']:.4f}")
    print(f"Content: {doc.page_content[:150]}...\n")
```

**Why re-rank?**
- Initial retrieval (vector search) is fast but approximate
- Re-ranking uses cross-attention between query and document, which is much more accurate
- Re-ranking is slower, so you only apply it to top candidates

Think of it like a two-stage filter: first stage casts a wide net, second stage carefully examines the catch.

**5.2 Self-Correcting RAG (CRAG)**

What if retrieved documents are irrelevant? Instead of blindly generating, check quality first.

**Conceptual Flow:**
1. Retrieve documents
2. Grade relevance (using an LLM)
3. If relevant → generate answer
4. If irrelevant → trigger web search or alternative retrieval
5. Generate answer from better sources

This is typically implemented with LangGraph (a state machine framework from LangChain), which is beyond our scope here but represents the cutting edge of production RAG systems.

---

### Part 6: Evaluation—Measuring Success (60 minutes)

You can't improve what you don't measure. RAG evaluation has three key dimensions:

1. **Faithfulness** - Is the answer grounded in the retrieved context? (Prevents hallucination)
2. **Relevance** - Is the answer actually addressing the question?
3. **Correctness** - Is the answer factually accurate compared to ground truth?

**6.1 Manual Evaluation with Custom Chains**

```python
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

class ResultScore(BaseModel):
    score: float = Field(..., description="Score from 0 to 1")

llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# Faithfulness evaluator
faithfulness_prompt = PromptTemplate(
    input_variables=["question", "context", "generated_answer"],
    template="""
    Question: {question}
    Context: {context}
    Generated Answer: {generated_answer}

    Evaluate if the generated answer can be deduced from the context.
    Score of 0 or 1, where 1 is perfectly faithful and 0 otherwise.
    You don't care if the answer is correct; only if it's derived from the context.
    
    Score:
    """
)

faithfulness_chain = faithfulness_prompt | llm.with_structured_output(ResultScore)

def evaluate_faithfulness(question, context, generated_answer):
    result = faithfulness_chain.invoke({
        "question": question,
        "context": context,
        "generated_answer": generated_answer
    })
    return result.score

# Test it
question = "What is 3+3?"
context = "6"
generated_answer = "6"
score = evaluate_faithfulness(question, context, generated_answer)
print(f"Faithfulness Score: {score}")  # Should be 0 (correct but not derivable from context)
```

**Why faithfulness matters more than correctness:** A correct answer that isn't grounded in your documents is still a hallucination. In production, you must know your system is pulling from your data, not from the LLM's training.

**6.2 Framework-Based Evaluation with RAGAS**

RAGAS (Retrieval-Augmented Generation Assessment) provides comprehensive metrics:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_correctness
from datasets import Dataset

# Prepare test data
questions = [
    "What is the three-headed dog's name?",
    "Who gave Harry his first broomstick?",
]

generated_answers = [
    "The three-headed dog is named Fluffy.",
    "Professor McGonagall gave Harry his Nimbus 2000.",
]

ground_truth = [
    "Fluffy",
    "Professor McGonagall",
]

retrieved_docs = [
    ["Hagrid mentioned the dog's name was Fluffy."],
    ["Professor McGonagall made an exception for Harry."],
]

# Create dataset
data = {
    'question': questions,
    'answer': generated_answers,
    'contexts': retrieved_docs,
    'ground_truth': ground_truth
}
dataset = Dataset.from_dict(data)

# Evaluate
metrics = [faithfulness, answer_relevancy, context_recall, answer_correctness]
result = evaluate(dataset=dataset, metrics=metrics)

print(result.to_pandas())
```

**Understanding the metrics:**
- **Faithfulness**: Does the answer use only information from retrieved context?
- **Answer Relevancy**: Does the answer actually address the question?
- **Context Recall**: Did we retrieve all necessary information?
- **Answer Correctness**: How accurate is the answer vs. ground truth?

---

## Usage Tutorial

### Example 1: Building a Customer Support RAG Bot

**Scenario:** Your company has 500 support documents. You want to build a RAG system so customers can ask questions.

**Step 1: Load your documents**
```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader('./support_docs/', glob="**/*.txt")
docs = loader.load()
```

**Step 2: Chunk appropriately for FAQs**
```python
# For FAQs, use smaller chunks with minimal overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)
```

**Step 3: Create vector store with metadata**
```python
# Add metadata for filtering (e.g., product category)
for chunk in chunks:
    chunk.metadata['product'] = extract_product_name(chunk.page_content)

vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
```

**Step 4: Build a RAG chain with product routing**
```python
def get_product_specific_retriever(product):
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"product": product}
        }
    )

# In your application logic:
user_question = "How do I reset my printer?"
detected_product = detect_product(user_question)  # Your logic
retriever = get_product_specific_retriever(detected_product)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke(user_question)
```

### Example 2: Research Assistant with Multi-Query

**Scenario:** You're building a research tool that needs to find relevant papers across multiple phrasing styles.

```python
# Load papers
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader('./papers/')
papers = loader.load()

# Chunk with larger size for research context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(papers)

# Create vector store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# Use multi-query for comprehensive retrieval
template = """Generate 5 different search queries for: {question}"""
prompt = ChatPromptTemplate.from_template(template)

multi_query_chain = (
    prompt 
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

retriever = vectorstore.as_retriever()

def retrieve_with_multi_query(question):
    queries = multi_query_chain.invoke({"question": question})
    all_docs = []
    for q in queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)
    return get_unique_union([all_docs])

# Use it
question = "What are the latest advances in quantum error correction?"
relevant_papers = retrieve_with_multi_query(question)
```

### Example 3: Evaluating Your RAG System

```python
# Create test cases
test_cases = [
    {
        "question": "What is Python's GIL?",
        "expected_answer": "The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects.",
        "retrieved_context": ["The GIL is a mutex in CPython..."]
    },
    # Add more test cases
]

# Evaluate
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

results = []
for tc in test_cases:
    # Generate answer with your RAG system
    actual_answer = rag_chain.invoke(tc["question"])
    
    # Create test case
    test_case = LLMTestCase(
        input=tc["question"],
        expected_output=tc["expected_answer"],
        actual_output=actual_answer,
        retrieval_context=tc["retrieved_context"]
    )
    
    # Evaluate
    result = evaluate(
        test_cases=[test_case],
        metrics=[FaithfulnessMetric(), AnswerRelevancyMetric()]
    )
    results.append(result)

# Analyze results
faithfulness_scores = [r.metrics[0].score for r in results]
avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
print(f"Average Faithfulness: {avg_faithfulness:.2f}")
```

---

## Advanced Notes

### Edge Cases and Limitations

**1. Cold Start Problem**
When your vector database is new and sparse, retrieval quality suffers. Solutions:
- Pre-populate with synthetic questions
- Use hybrid search (keyword + semantic)
- Implement user feedback loops to improve over time

**2. Out-of-Domain Queries**
Users might ask questions your documents can't answer. Detect this early:
```python
def detect_low_confidence(retrieved_docs, threshold=0.5):
    top_score = retrieved_docs[0].metadata.get('relevance_score', 0)
    if top_score < threshold:
        return "I don't have enough information to answer that confidently."
```

**3. Context Length Limits**
Even with long-context models, you'll hit limits. Strategies:
- Use multi-representation indexing to be selective
- Implement iterative refinement (answer, then follow-up questions)
- Break long documents into logical sections with metadata

**4. Cost Management**
OpenAI API costs add up. Optimize:
- Cache embeddings (don't re-embed the same content)
- Use smaller models for routing/classification
- Batch process where possible
- Consider open-source alternatives (Llama, Mistral) for generation

### Extensions and Next Steps

**1. Hybrid Search**
Combine semantic search with traditional keyword search:
```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# BM25 for keyword matching
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Semantic search
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6]  # Favor semantic but include keywords
)
```

**2. Conversational RAG**
Add memory so users can ask follow-up questions:
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True,
    output_key="answer",
    input_key="question"
)

conversational_rag = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Now you can do:
response1 = conversational_rag({"question": "What is RAG?"})
response2 = conversational_rag({"question": "How does it work?"})  # "it" refers to RAG
```

**3. Multimodal RAG**
Extend to images, tables, and charts:
```python
from langchain.document_loaders import UnstructuredImageLoader

image_loader = UnstructuredImageLoader("./diagrams/")
images = image_loader.load()

# Use multimodal embeddings (e.g., CLIP)
from langchain.embeddings import OpenCLIPEmbeddings

clip_embeddings = OpenCLIPEmbeddings()
multimodal_vectorstore = Chroma.from_documents(
    documents=chunks + images,
    embedding=clip_embeddings
)
```

**4. Production Deployment**
For production, consider:
- **Vector Database**: Switch from Chroma to Pinecone/Weaviate/Qdrant for scale
- **Caching**: Use Redis to cache common queries
- **Monitoring**: Track retrieval quality, latency, cost
- **A/B Testing**: Compare RAG architectures with real users

---

## Exercises for the Reader

### Beginner Level

**Exercise 1: Personal Knowledge Base**
Build a RAG system over your own notes or bookmarks.
- Load your notes from a folder
- Experiment with different chunk sizes (250, 500, 1000, 2000)
- Compare retrieval quality qualitatively

**Exercise 2: Evaluation Practice**
Create 10 test questions for your knowledge base.
- Write expected answers
- Run your RAG system and compare outputs
- Calculate faithfulness manually using the prompt from Part 6

### Intermediate Level

**Exercise 3: Multi-Source RAG**
Combine two different data sources (e.g., blog posts + documentation).
- Implement logical routing to decide which source to use
- Add metadata to identify source
- Create a chain that can query both and synthesize answers

**Exercise 4: Improve a Bad Baseline**
Start with basic RAG (chunk_size=500, no query transformation).
- Measure performance with RAGAS
- Apply one technique from Part 2 (multi-query, decomposition, etc.)
- Measure again and quantify improvement

### Advanced Level

**Exercise 5: Build CRAG**
Implement a self-correcting RAG system.
- Create a relevance grader that scores retrieved docs
- If score < threshold, trigger web search (using DuckDuckGo tool)
- Compare answers with/without correction

**Exercise 6: Custom Re-Ranker**
Build your own re-ranking model.
- Fine-tune a cross-encoder on your domain data
- Integrate it as a custom compressor
- Benchmark against Cohere re-ranker

---

## Contributing

Contributions are welcome! Here's how you can help:

**Reporting Issues**
- Use GitHub Issues to report bugs
- Include Python version, error traceback, and minimal reproduction

**Adding Features**
- Fork the repository
- Create a feature branch (`git checkout -b feature/amazing-feature`)
- Write tests if applicable
- Submit a Pull Request with clear description

**Improving Documentation**
- Fix typos or unclear explanations
- Add more examples or use cases
- Improve code comments

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Additional Resources

**Essential Reading**
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Retrieval-Augmented Generation Paper](https://arxiv.org/abs/2005.11401)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)

**Community and Support**
- LangChain Discord: [discord.gg/langchain](https://discord.gg/langchain)
- LangChain GitHub: [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

**Blogs and Tutorials**
- Lilian Weng's Blog (source for our examples): [lilianweng.github.io](https://lilianweng.github.io/)
- LangChain Blog: [blog.langchain.dev](https://blog.langchain.dev/)

---

## Acknowledgments

This tutorial builds on techniques from cutting-edge research and production systems:
- RAG-Fusion by Adrian H. Raudaschl
- RAPTOR by Nelson F. Liu et al.
- ColBERT by Omar Khattab and Matei Zaharia
- Self-RAG by Akari Asai et al.

Special thanks to Lilian Weng for her excellent blog posts on AI agents, which served as our primary data source for examples throughout this tutorial.

---

**Ready to build production-grade RAG systems?** Start with the basics in the notebook, experiment with the exercises, and gradually work your way up to advanced techniques. Remember: the best RAG system is one that's constantly evaluated and improved. Happy building! 🚀
