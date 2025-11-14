[Answer Engine.drawio](https://github.com/user-attachments/files/23554562/Answer.Engine.drawio)
# GenAI — RAG + LangGraph Demo

Upload a **knowledge base (KB)** and a set of **input questions**, then generate answers with a LangGraph RAG pipeline backed by **Weaviate** (Docker) and **OpenAI** (embeddings + generation). Review the generated answers and the exact retrieved context.

## Time Spent
About 40 hours
## Workflow & Tools
LangChain, LangGraph, LangSmith, Weaviate database, FastAPI, React
## Answer Engine
<img width="781" height="286" alt="Answer Engine drawio" src="https://github.com/user-attachments/assets/f0b10c8b-e5ef-40be-8bcf-ad06f6320170" />



               
## How It Works (High Level)
- Modified [Multi-Representation Indexing](https://www.youtube.com/watch?v=gTCU9I6QqCE): KB questions are embedded using OpenAI and stored as "child" docs in Weaviate vectorDB, and the corresponding answers are stored as "parent" docs. The reason for this approach was to overcome the limitations of modern approaches where you retrieve document snippets. 

  Common approaches to RAG involves retrievals from a set of documents. To apply to a setting with knowledge base as QA-pair entries, fusing all QA-pairs into a single document neglects the relationship between each question and answer, purely making it work as semantic search on all fused information. This new approach links each question to the corresponding answer and retrieves the relevant         questions from the input query. Then, the corresponding answers to retrieved relevant questions can be used for the context that the LLM would use to generate answers. 
- Retrieval: Multiple query rewrites + RRF fusion to retrieve and re-rank answers for relevant questions.
- Generation: An LLM answers only from the retrieved context, or returns
- Run-time Graders: Graders for Retrieval, Answer-Quality, Groundedness in between to make sure that the step is repeated in case each criterion fails
- Post Grader Using Langsmith (Optional, not part of the solution): Shows metric with GroundTruth answers for assessing the answers and retrievals (similar to the run-time graders)
---

## Performance vs Latency & Cost Trade-offs
This project aims to balance, **latency**, **cost**, **answer-quality**, **groundedness**, **relevance**, and **retrieval** 
- iterative re-generation, query re-writes, or re-retrievals in case each step in RAG fails (increases latency but effective for accurate answers + minimal hallucinations )
- Ensures the retrievals are good in the first run using RAG-Fusion workflow to eliminate the need for re-generation, query re-writes, or re-retrievals.    

### Cost Planning (gpt-4o-mini for generation, gpt-4o for judges)
I used API calls to light weight and well-performing OpenAI models to balance cost and performance quality. 
At the time of writing, OpenAI lists usage-based pricing per tokens (input vs output).
(Might change in the future)
Typical list prices you can use for rough estimates:
- **gpt-4o**: **$5.00 / 1M input tokens**, **$15.00 / 1M output tokens** (≈ $0.005 / 1k in, $0.015 / 1k out).
- **gpt-4o-mini**: **$0.15 / 1M input tokens**, **$0.60 / 1M output tokens** (≈ $0.00015 / 1k in, $0.00060 / 1k out).

## Quick Start

### Prerequisites
- **Python** 3.10+
- **Node** 18+
- **Docker** (for Weaviate)
- **OpenAI API key**
- (Optional) **LangSmith API key**

## 1. Environment
(Refer to backend/.env_example file)
Create `backend/.env`:

```bash
OPENAI_API_KEY=sk-...
#Optional (LangSmith tracing; safe to omit)
LANGSMITH_API_KEY=ls_...
```

## 2. Start Weaviate (Docker)

Ensure that your docker desktop is running (testing from Windows)

```bash
cd backend
docker-compose up -d
```
## 3. Backend
In the "backend" folder, run:

```bash
#install requirements
pip install -r requirements.txt
#Run fastapi
uvicorn app:app --reload
```

## 4. Frontend
From the "my-react18-app" folder, run:

```bash
#in case you haven't installed
npm install
npm run dev
```

This will open http://localhost:5173

