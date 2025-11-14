[Answer Engine.drawio](https://github.com/user-attachments/files/23554562/Answer.Engine.drawio)
# GenAI — RAG + LangGraph Demo

Upload a **knowledge base (KB)** and a set of **input questions**, then generate answers with a LangGraph RAG pipeline backed by **Weaviate** (Docker) and **OpenAI** (embeddings + generation). Review the generated answers and the exact retrieved context.

## Time Spent
About 40 hours
## Workflow & Tools
LangChain, LangGraph, LangSmith, Weaviate database, FastAPI, React
## Answer Engine
[Uploading Answer Engi<mxfile host="app.diagrams.net" agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36" version="29.0.2">
  <diagram name="페이지-1" id="KDjq_hDgT34cGa39nqLl">
    <mxGraphModel dx="944" dy="865" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="1UuNuln7PY0rcrK-yt1q-4" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-1" target="1UuNuln7PY0rcrK-yt1q-2">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-1" value="Input&amp;nbsp;&lt;div&gt;Question&lt;/div&gt;" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="60" y="260" width="70" height="40" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-7" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-2" target="1UuNuln7PY0rcrK-yt1q-5">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-2" value="Rewrite" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;" vertex="1" parent="1">
          <mxGeometry x="170" y="256.25" width="47.5" height="47.5" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-30" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-5" target="1UuNuln7PY0rcrK-yt1q-12">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-5" value="Retrieve" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;" vertex="1" parent="1">
          <mxGeometry x="260" y="256.25" width="47.5" height="47.5" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-18" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-12" target="1UuNuln7PY0rcrK-yt1q-2">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="190.0000000000001" y="369.9999999999999" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-32" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-12" target="1UuNuln7PY0rcrK-yt1q-22">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-12" value="Docs&lt;div&gt;relevant?&lt;/div&gt;" style="rhombus;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="343" y="244.07" width="70" height="71.87" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-19" value="No" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="330" y="310" width="60" height="30" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-27" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-22" target="1UuNuln7PY0rcrK-yt1q-26">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-22" value="Generate" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontSize=10;" vertex="1" parent="1">
          <mxGeometry x="460" y="170" width="47.5" height="47.5" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-23" value="Yes" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="390" y="244.07" width="60" height="30" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-34" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-26" target="1UuNuln7PY0rcrK-yt1q-22">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-37" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-26" target="1UuNuln7PY0rcrK-yt1q-38">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="640" y="119.99999999999989" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-26" value="&lt;font style=&quot;font-size: 9px;&quot;&gt;Hallucinations?&lt;/font&gt;" style="rhombus;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="550" y="157.82" width="70" height="71.87" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-35" value="Yes" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="530" y="226.25" width="60" height="30" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-36" value="No" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="600" y="157.82" width="60" height="30" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-39" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-38" target="1UuNuln7PY0rcrK-yt1q-2">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="695" y="66" />
              <mxPoint x="194" y="66" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-47" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="1UuNuln7PY0rcrK-yt1q-38" target="1UuNuln7PY0rcrK-yt1q-44">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-38" value="&lt;font size=&quot;1&quot;&gt;Answers&lt;/font&gt;&lt;div&gt;&lt;font size=&quot;1&quot;&gt;Question?&lt;/font&gt;&lt;/div&gt;" style="rhombus;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="660" y="85.94999999999999" width="70" height="71.87" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-40" value="No" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="650" y="60" width="60" height="30" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-42" value="Yes" style="text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;rounded=0;" vertex="1" parent="1">
          <mxGeometry x="720" y="100" width="50" height="20" as="geometry" />
        </mxCell>
        <mxCell id="1UuNuln7PY0rcrK-yt1q-44" value="&lt;div&gt;Final&lt;/div&gt;Answer" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
          <mxGeometry x="770" y="101.89" width="70" height="40" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
ne.drawio…]()



               
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

