from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
import operator
import time

class RAGState(TypedDict):
    """Multi-turn conversation state."""
    messages: Annotated[Sequence, operator.add]
    query: str
    context: str
    answer: str
    evidence: list
    retrieval_time: float
    llm_time: float
    total_time: float

class RAGGraph:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.graph = self._build_graph()
    
    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()
    
    def _retrieve_node(self, state: RAGState):
        start = time.time()
        passages, metadata = self.pipeline.retrieve_top_k(state["query"], top_k=10)
        retrieval_time = time.time() - start
        return {
            "context": "\n".join(passages),
            "evidence": list(zip(passages, metadata)),
            "retrieval_time": retrieval_time
        }
    
    def _generate_node(self, state: RAGState):
        start = time.time()
        answer = self.pipeline.query_llm(state["query"], state["context"])
        llm_time = time.time() - start
        return {
            "answer": answer,
            "llm_time": llm_time,
            "total_time": state.get("retrieval_time", 0) + llm_time
        }
    
    def invoke(self, query: str, messages: list = None):
        initial_state = RAGState(
            messages=messages or [],
            query=query,
            context="",
            answer="",
            evidence=[],
            retrieval_time=0.0,
            llm_time=0.0,
            total_time=0.0
        )
        return self.graph.invoke(initial_state)