# 🛡️ Adaptive RAG: HIPAA Cybersecurity Assistant

A production-ready **Adaptive Retrieval-Augmented Generation (RAG)** application built with LangGraph, FAISS, Groq, and Streamlit. The app intelligently routes questions to either a HIPAA document vectorstore or live web search, and includes a live financial dashboard for the top 20 US health companies.

---

## 🚀 Features

- **Adaptive RAG Pipeline** — Automatically routes questions to vectorstore or web search based on topic
- **HIPAA Document Q&A** — Answers questions from the Federal Register HIPAA Security Rule PDF (2025)
- **Self-Correcting Agent** — Grades retrieved documents, checks for hallucinations, and rewrites queries if needed
- **Live Web Search** — Falls back to Tavily web search for non-HIPAA questions
- **Top 20 Health Companies Dashboard** — Live financial data (Market Cap, Revenue, Net Income) via Yahoo Finance
- **CSV Export** — Download health company data as a CSV file

---

## ✅ Advantages

### 🧠 1. Adaptive Routing Intelligence
Unlike basic RAG systems that always query the vectorstore, the app **automatically decides** whether to use the document database or live web search — giving accurate answers for both specific HIPAA questions and general knowledge queries.

### 📄 2. Domain-Specific Legal Knowledge
The app is grounded in the **actual Federal Register HIPAA PDF**, not just general LLM knowledge. Answers are sourced from real regulatory text, making it reliable for compliance-related questions.

### 🔍 3. Self-Correcting Pipeline
The app doesn't just generate and return — it:
- Grades retrieved documents for relevance
- Checks the answer for hallucinations
- Rewrites the query and retries if the answer is poor



## 🏥 HIPAA-Specific Advantages

### Instant Regulatory Compliance Guidance
Instead of manually searching through **200+ pages** of the Federal Register, users get precise answers in seconds.

### Reduces Legal Research Time
Healthcare compliance officers, lawyers, and administrators can query complex regulatory language in plain English instead of reading dense legal text.

### Cost Savings for Healthcare Organizations
Small clinics and hospitals that **cannot afford compliance consultants** can use this app as a first-line tool to understand their HIPAA obligations.

### Useful Across Multiple Healthcare Roles

| Role | How They Use It |
|---|---|
| Compliance Officer | Check regulatory requirements instantly |
| IT Administrator | Understand technical security standards |
| Hospital Legal Team | Research penalties and obligations |
| Healthcare Startup | Understand rules before product launch |
| Medical Billing Staff | Clarify data handling obligations |

### Covers All Key HIPAA Topics

- **Privacy Rule** — patient data rights
- **Security Rule** — ePHI cybersecurity standards
- **Breach Notification Rule** — reporting obligations
- **HITECH Act** — extended enforcement and penalties
- **Business Associate Agreements** — third-party obligations
- **Individually Identifiable Health Information** — what qualifies as protected data

### Scales Across Multiple Regulations
The same architecture can be extended to cover other healthcare regulations by simply adding more documents:
```python
urls = [
    "HIPAA Security Rule PDF",         # Current
    "HIPAA Privacy Rule PDF",          # Add
    "HITECH Act PDF",                  # Add
    "CMS Conditions of Participation"  # Add
]
```

## 🏗️ Architecture

```
User Question
      │
      ▼
 Route Question
  ┌───┴───┐
  │       │
  ▼       ▼
FAISS   Tavily
Vector  Web
Store   Search
  │       │
  ▼       │
Grade     │
Documents │
  │       │
  ▼       ▼
    Generate
       │
       ▼
  Grade for
 Hallucination
       │
       ▼
  Grade Answer
  ┌────┴────┐
  │         │
  ▼         ▼
 END    Rewrite Query
            │
            ▼
         Retrieve
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Orchestration | LangGraph |
| Vector Store | FAISS |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Web Search | Tavily |
| Document Loader | PyPDFLoader |
| Financial Data | Yahoo Finance (`yfinance`) |
| Frontend | Streamlit |

