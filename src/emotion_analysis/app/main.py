import time
import hashlib
import asyncio
import logging
from typing import Dict
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Enforce environment variable loading on boot
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr, field_validator

from src.emotion_analysis.core.graph import emotion_workflow
from src.emotion_analysis.persistence.memory import get_checkpointer
from src.emotion_analysis.observability.benchmarks import benchmark_tracker
from src.emotion_analysis.core.router import semantic_router

logger = logging.getLogger("API")

# Background TTL cleanup task
async def cache_cleanup_task():
    while True:
        try:
            await asyncio.sleep(6 * 3600) # 6 hours
            semantic_router.cleanup_expired()
            logger.info("Executed background semantic cache TTL cleanup.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cache cleanup task error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cache_cleanup_task())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Emotion Analysis API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str
    user_id: str = "default_user"
    session_id: str
    
    @field_validator("session_id")
    def validate_session_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("session_id cannot be empty or whitespace-only")
        if len(v) > 128:
            raise ValueError("session_id cannot exceed 128 characters")
        return v

fallback_chain = None
fallback_parser = None

async def run_llm_fallback(text: str) -> Dict:
    global fallback_chain, fallback_parser
    # Cache the fallback chain so we don't reinstantiate ChatGroq on every trip
    if fallback_chain is None:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from src.emotion_analysis.core.agents import ContextualOutput
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_retries=0)
        fallback_parser = JsonOutputParser(pydantic_object=ContextualOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Emotion Analysis Fallback Agent. Your role is to act as the final, reliable safety net in a multi-agent classification pipeline.

### DIRECTIVES
1. Provide a robust, conservative assessment of the text's emotion.
2. If the text is highly ambiguous, contains conflicting signals, or is too short to analyze, you MUST default the emotion to 'neutral'.
3. Do not attempt deep, speculative analysis. Stick to the most obvious, undeniable sentiment.

### OUTPUT INSTRUCTIONS
Output strictly in valid JSON according to this schema. Do not include introductory text, markdown formatting, or explanations outside the JSON structure:
{format_instructions}"""),
            ("human", "{query}")
        ])
        fallback_chain = prompt | llm | fallback_parser

    result = await fallback_chain.ainvoke({"query": text, "format_instructions": fallback_parser.get_format_instructions()})
    
    from src.emotion_analysis.core.agents import ContextualOutput
    return ContextualOutput(**result).model_dump()

@app.post("/analyze")
async def analyze_emotion(req: AnalyzeRequest):
    logger.debug("Entered analyze_emotion")
    start_time = time.time()
    
    thread_id = hashlib.sha256(f"{req.user_id}:{req.session_id}".encode()).hexdigest()
    config = {"configurable": {"thread_id": thread_id}}
    
    is_fallback = False
    cache_hit = False
    final_output = {}
    
    logger.debug("Checking circuit breaker")
    allow_primary = benchmark_tracker.cb.allow_request()
    logger.debug(f"Circuit allow_primary = {allow_primary}")
    
    if allow_primary:
        try:
            logger.debug("Entering get_checkpointer")
            async with get_checkpointer() as checkpointer:
                logger.debug("Compiling graph")
                graph = emotion_workflow.compile(checkpointer=checkpointer)
                logger.debug("Invoking graph")
                result = await graph.ainvoke({"query": req.text, "cached_result": None, "lexical_summary": None, "contextual_summary": None, "final_output": None}, config)
                logger.debug("Graph invoked")
                
                if result.get("cached_result"):
                    cache_hit = True
                final_output = result.get("final_output", {})
                
                benchmark_tracker.cb.record_success()
                
        except Exception as e:
            logger.error(f"Groq Swarm failed: {str(e)}")
            benchmark_tracker.cb.record_failure()
            allow_primary = False # Trigger fallback
            
    if not allow_primary:
        is_fallback = True
        benchmark_tracker.cb.record_fallback()
        try:
            fallback_out = await run_llm_fallback(req.text)
            final_output = {
                "surface_emotion": fallback_out.get("final_emotion"),
                "final_emotion": fallback_out.get("final_emotion"),
                "is_sarcastic": fallback_out.get("is_sarcastic", False),
                "nuance_explanation": fallback_out.get("nuance_explanation", "Generated via Llama 8B Fallback"),
                "intensity": 5
            }
        except Exception as fallback_e:
            latency_ms = (time.time() - start_time) * 1000
            benchmark_tracker.log_request(req.text, {}, latency_ms, is_fallback, cache_hit)
            logger.error(f"CRITICAL: Primary circuit OPEN/Failed. Fallback failed with '{str(fallback_e)}'.")
            raise HTTPException(status_code=500, detail="Internal server error during emotion analysis.")
            
    latency_ms = (time.time() - start_time) * 1000
    benchmark_tracker.log_request(req.text, final_output, latency_ms, is_fallback, cache_hit)
    
    return {
        "status": "success",
        "data": final_output,
        "metrics": {
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "fallback_triggered": is_fallback
        }
    }

@app.get("/metrics")
def get_metrics():
    return benchmark_tracker.get_summary()
