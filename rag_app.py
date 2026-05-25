import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Literal
from typing_extensions import TypedDict
import yfinance as yf
import pandas as pd

# LangChain / LangGraph imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="venv2/.env")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GROQ_API_KEY"]   = os.getenv("GROQ_API_KEY", "")
os.environ["HF_API_KEY"]     = os.getenv("HF_API_KEY", "")

st.set_page_config(page_title="Adaptive RAG - HIPAA Cybersecurity", layout="wide")
st.title("🛡️ Adaptive RAG: HIPAA Cybersecurity Assistant")
st.subheader("Leveraging LangGraph, FAISS, Groq, and Tavily Search")

HEALTH_TICKERS = {
    "UNH":  "UnitedHealth Group",
    "JNJ":  "Johnson & Johnson",
    "CVS":  "CVS Health",
    "ABT":  "Abbott Laboratories",
    "MRK":  "Merck & Co",
    "ELV":  "Elevance Health",
    "HUM":  "Humana",
    "CI":   "Cigna Group",
    "MCK":  "McKesson Corp",
    "ABC":  "AmerisourceBergen",
    "BMY":  "Bristol-Myers Squibb",
    "AMGN": "Amgen",
    "GILD": "Gilead Sciences",
    "ISRG": "Intuitive Surgical",
    "SYK":  "Stryker Corp",
    "BDX":  "Becton Dickinson",
    "ZTS":  "Zoetis",
    "IDXX": "IDEXX Laboratories",
    "BSX":  "Boston Scientific",
    "PFE":  "Pfizer",
}

@st.cache_data(ttl=3600)
def fetch_health_companies():
    """Fetch live financial data for top 20 US health companies."""
    data = []
    for ticker, name in HEALTH_TICKERS.items():
        try:
            info = yf.Ticker(ticker).info
            data.append({
                "Rank":       len(data) + 1,
                "Company":    name,
                "Ticker":     ticker,
                "Market Cap": info.get("marketCap", 0),
                "Revenue":    info.get("totalRevenue", 0),
                "Net Income": info.get("netIncomeToCommon", 0),
                "Employees":  info.get("fullTimeEmployees", "N/A"),
                "Sector":     info.get("sector", "Healthcare"),
                "52W High":   info.get("fiftyTwoWeekHigh", "N/A"),
                "52W Low":    info.get("fiftyTwoWeekLow", "N/A"),
            })
        except Exception:
            continue

    df = pd.DataFrame(data)
    df["Market Cap"] = df["Market Cap"].apply(lambda x: f"${x/1e9:.1f}B" if x else "N/A")
    df["Revenue"]    = df["Revenue"].apply(lambda x: f"${x/1e9:.1f}B" if x else "N/A")
    df["Net Income"] = df["Net Income"].apply(lambda x: f"${x/1e9:.1f}B" if x else "N/A")
    df["Employees"]  = df["Employees"].apply(lambda x: f"{x:,}" if isinstance(x, int) else x)
    return df

@st.cache_resource
def initialize_rag_components():
    """Initializes the retriever, tools, models, and compiles the LangGraph."""

    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        st.warning(" Environment keys missing. Please check your .env file.")

    # A. Load and Split Documents
    embd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    urls = ["https://www.govinfo.gov/content/pkg/FR-2025-01-06/pdf/2024-30983.pdf"]

    with st.spinner("Processing Federal Register HIPAA Document..."):
        docs     = [PyPDFLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=50
        )
        doc_splits  = text_splitter.split_documents(docs_list)
        vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embd)
        retriever   = vectorstore.as_retriever()

    # B. Tools & Models
    web_search_tool = TavilySearchResults(k=3)

    class RouteQuery(BaseModel):
        datasource: Literal["vectorstore", "web_search"] = Field(
            ..., description="Choose to route it to web search or a vectorstore."
        )

    class GradeDocuments(BaseModel):
        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    class GradeHallucinations(BaseModel):
        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    class GradeAnswer(BaseModel):
        binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

    router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"])
    structured_llm_router = router_llm.with_structured_output(RouteQuery)

    grader_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"])
    structured_llm_grader         = grader_llm.with_structured_output(GradeDocuments)
    structured_llm_hallucination  = grader_llm.with_structured_output(GradeHallucinations)
    structured_llm_answer         = grader_llm.with_structured_output(GradeAnswer)

    # Prompts
    question_router = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents about federal regulations and HIPAA Security Rule.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""),
        ("human", "{question}")
    ]) | structured_llm_router

    retrieval_grader = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no'."""),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]) | structured_llm_grader

    rag_chain = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. "
                   "Use the retrieved context to answer concisely in three sentences max. "
                   "If you don't know, say so."),
        ("human", "Question: {question}\nContext: {context}\nAnswer:"),
    ]) | grader_llm | StrOutputParser()

    hallucination_grader = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an LLM generation is grounded in retrieved facts. "
                   "Give a binary score 'yes' or 'no'."),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]) | structured_llm_hallucination

    answer_grader = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an answer resolves a question. "
                   "Give a binary score 'yes' or 'no'."),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ]) | structured_llm_answer

    question_rewriter = ChatPromptTemplate.from_messages([
        ("system", "You are a question re-writer that improves questions for vectorstore retrieval."),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
    ]) | grader_llm | StrOutputParser()

    # C. Graph Nodes
    def retrieve(state):
        documents = retriever.invoke(state["question"])
        return {"documents": documents, "question": state["question"]}

    def generate(state):
        generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
        return {"documents": state["documents"], "question": state["question"], "generation": generation}

    def grade_documents(state):
        filtered = []
        for d in state["documents"]:
            score = retrieval_grader.invoke({"question": state["question"], "document": d.page_content})
            if score.binary_score == "yes":
                filtered.append(d)
        return {"documents": filtered, "question": state["question"]}

    def transform_query(state):
        better_question = question_rewriter.invoke({"question": state["question"]})
        return {"documents": state["documents"], "question": better_question}

    def web_search(state):
        docs        = web_search_tool.invoke({"query": state["question"]})
        web_results = Document(page_content="\n".join([d["content"] for d in docs]))
        return {"documents": [web_results], "question": state["question"]}

    # Conditional edges
    def route_question(state):
        source = question_router.invoke({"question": state["question"]})
        return "web_search" if source.datasource == "web_search" else "vectorstore"

    def decide_to_generate(state):
        return "transform_query" if not state["documents"] else "generate"

    def grade_generation_v_documents_and_question(state):
        h_score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
        if h_score.binary_score == "yes":
            a_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
            return "useful" if a_score.binary_score == "yes" else "not useful"
        return "not supported"

    # D. Build Graph
    class GraphState(TypedDict):
        question:   str
        generation: str
        documents:  List[Document]

    workflow = StateGraph(GraphState)
    workflow.add_node("web_search",       web_search)
    workflow.add_node("retrieve",         retrieve)
    workflow.add_node("grade_documents",  grade_documents)
    workflow.add_node("generate",         generate)
    workflow.add_node("transform_query",  transform_query)

    workflow.add_conditional_edges(START, route_question, {"web_search": "web_search", "vectorstore": "retrieve"})
    workflow.add_edge("web_search",      "generate")
    workflow.add_edge("retrieve",        "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"})
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question,
        {"not supported": "generate", "useful": END, "not useful": "transform_query"})

    return workflow.compile()


# Compile app
app_engine = initialize_rag_components()

# STREAMLIT UI — Tabs
st.markdown("---")
tab1, tab2 = st.tabs(["💬 HIPAA Assistant", "🏥 Top 20 Health Companies 2026"])

# ── Tab 1: RAG Assistant ───────────────────────────────────────────────────
with tab1:
    st.markdown("#### Ask a question about HIPAA Cybersecurity Regulations:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 What is Health Insurance Portability and Accountability Act of 1996?"):
            st.session_state.user_query = "What is Health Insurance Portability and Accountability Act of 1996?"
    with col2:
        if st.button("🌐 Who won the world cup 2011 in cricket?"):
            st.session_state.user_query = "who won the world cup 2011 in cricket?"

    user_input = st.text_input(
        "Enter your question here:",
        value=st.session_state.get("user_query", ""),
        key="tab1_input"   # ✅ unique key
    )

    if st.button("Run Adaptive Agent", type="primary", key="tab1_run"):
        if user_input.strip() == "":
            st.error("Please enter a valid query.")
        else:
            if "user_query" in st.session_state:
                del st.session_state.user_query

            with st.spinner("Agent thinking & self-correcting via Adaptive RAG..."):
                try:
                    output  = app_engine.invoke({"question": user_input})
                    res_col, doc_col = st.columns([3, 2])

                    with res_col:
                        st.success("### Final Generation")
                        st.write(output.get("generation", "No generation produced."))
                        st.info(f"**Processed Question:** {output.get('question')}")

                    with doc_col:
                        st.markdown("### Contextual Sources Used")
                        retrieved_docs = output.get("documents", [])
                        if isinstance(retrieved_docs, list):
                            for idx, doc in enumerate(retrieved_docs):
                                meta      = getattr(doc, "metadata", {})
                                page_info = f" | Page {meta.get('page_label', idx)}" if meta.get("page_label") else ""
                                with st.expander(f"Source [{idx+1}]{page_info}"):
                                    st.caption(f"Source: {meta.get('source', 'Local/Search Context')}")
                                    st.write(doc.page_content)
                        else:
                            with st.expander("Web Context Data"):
                                st.write(str(retrieved_docs))
                except Exception as e:
                    st.error(f"An error occurred: {e}")

#Tab 2: Top 20 Health Companies
with tab2:
    st.markdown("#### 📊 Top 20 US Health Companies — Live Financial Data")

    if st.button("🔄 Fetch / Refresh Data", type="primary", key="tab2_refresh"):
        st.cache_data.clear()

    with st.spinner("Fetching live financial data from Yahoo Finance..."):
        df = fetch_health_companies()

    # Summary metrics
    st.markdown("### 📈 Market Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Companies Tracked", len(df))
    m2.metric("Largest by Market Cap", df.iloc[0]["Company"] if not df.empty else "N/A")
    m3.metric("Data Source", "Yahoo Finance")

    st.markdown("---")

    st.markdown("### 🏥 Company Rankings")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name="top20_health_companies_2026.csv",
        mime="text/csv",
        key="tab2_download"
    )
