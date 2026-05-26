import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from src.emotion_analysis.core.router import semantic_router
from src.emotion_analysis.core.agents import run_lexical_agent, run_contextual_agent
from src.emotion_analysis.persistence.vector_store import vector_store

class EmotionState(TypedDict):
    query: str
    cached_result: Optional[dict]
    lexical_summary: Optional[dict]
    contextual_summary: Optional[dict]
    final_output: Optional[dict]

def router_node(state: EmotionState):
    """Checks the semantic cache. If hit, we skip LLM generation."""
    query = state["query"]
    cached_str = semantic_router.check_cache(query)
    if cached_str:
        return {"cached_result": json.loads(cached_str), "final_output": json.loads(cached_str)}
    return {"cached_result": None}

def should_use_cache(state: EmotionState):
    """Conditional edge to route depending on cache hit."""
    if state.get("cached_result"):
        return "end"
    return "lexical_agent"

def lexical_node(state: EmotionState):
    """Runs the Lexical Emotion Agent."""
    result = run_lexical_agent(state["query"])
    return {"lexical_summary": result.model_dump()}

def contextual_node(state: EmotionState):
    """Fetches RAG context and runs Contextual Nuance Agent."""
    query = state["query"]
    
    # Fix: use kwargs `k` and `filter` for VectorStore API
    sarcasm_results = vector_store.similarity_search(query, k=2, filter={"source": "sarcasm_context"})
    sentiment_results = vector_store.similarity_search(query, k=2, filter={"source": "sentiment140_context"})
    
    # Format RAG context
    rag_context = ""
    for res in sarcasm_results + sentiment_results:
        meta = res.get('metadata') or {}
        label = meta.get('label', 'Unknown')
        rag_context += f"- Text: {res['text']} | Label: {label}\n"
        
    result = run_contextual_agent(
        query=query, 
        lexical_summary=state["lexical_summary"],
        rag_context=rag_context
    )
    
    # Synthesize Final Output
    final_output = {
        "surface_emotion": state["lexical_summary"]["surface_emotion"],
        "final_emotion": result.final_emotion,
        "is_sarcastic": result.is_sarcastic,
        "nuance_explanation": result.nuance_explanation,
        "intensity": state["lexical_summary"]["intensity"]
    }
    
    # Save to Semantic Cache for zero-routing next time
    semantic_router.save_to_cache(query, final_output)
    
    return {"contextual_summary": result.model_dump(), "final_output": final_output}

# --- Build the Graph ---
workflow = StateGraph(EmotionState)

workflow.add_node("router", router_node)
workflow.add_node("lexical_agent", lexical_node)
workflow.add_node("contextual_agent", contextual_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", should_use_cache, {"end": END, "lexical_agent": "lexical_agent"})
workflow.add_edge("lexical_agent", "contextual_agent")
workflow.add_edge("contextual_agent", END)

# We export the uncompiled workflow so it can be compiled with a checkpointer dynamically in main.py
emotion_workflow = workflow
