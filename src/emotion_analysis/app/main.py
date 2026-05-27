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
from src.emotion_analysis.core.provider_manager import provider_manager

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

async def run_llm_fallback(text: str) -> Dict:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from src.emotion_analysis.core.agents import ContextualOutput
    
    gemini_key = provider_manager.get_active_gemini_key()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=gemini_key, max_retries=0)
    parser = JsonOutputParser(pydantic_object=ContextualOutput)
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
    fallback_chain = prompt | llm | parser

    result = await fallback_chain.ainvoke({"query": text, "format_instructions": parser.get_format_instructions()})
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
                
                try:
                    result = await graph.ainvoke({"query": req.text, "cached_result": None, "lexical_summary": None, "contextual_summary": None, "final_output": None}, config)
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        logger.warning("Groq Rate Limit hit. Attempting key rotation...")
                        if provider_manager.rotate_groq_key():
                            logger.info("Retrying with backup Groq key.")
                            result = await graph.ainvoke({"query": req.text, "cached_result": None, "lexical_summary": None, "contextual_summary": None, "final_output": None}, config)
                        else:
                            raise e
                    else:
                        raise e
                        
                logger.debug("Graph invoked")
                
                if result.get("cached_result"):
                    cache_hit = True
                final_output = result.get("final_output", {})
                
                benchmark_tracker.cb.record_success()
                
        except Exception as e:
            logger.error(f"Groq Swarm failed completely: {str(e)}")
            benchmark_tracker.cb.record_failure()
            allow_primary = False # Trigger fallback
            
    if not allow_primary:
        is_fallback = True
        benchmark_tracker.cb.record_fallback()
        try:
            try:
                fallback_out = await run_llm_fallback(req.text)
            except Exception as fallback_e:
                if "429" in str(fallback_e) or "rate" in str(fallback_e).lower():
                    logger.warning("Gemini Rate Limit hit. Attempting key rotation...")
                    if provider_manager.rotate_gemini_key():
                        fallback_out = await run_llm_fallback(req.text)
                    else:
                        raise fallback_e
                else:
                    raise fallback_e
                    
            final_output = {
                "surface_emotion": fallback_out.get("final_emotion"),
                "final_emotion": fallback_out.get("final_emotion"),
                "is_sarcastic": fallback_out.get("is_sarcastic", False),
                "nuance_explanation": fallback_out.get("nuance_explanation", "Generated via Gemini Fallback"),
                "intensity": 5
            }
        except Exception as fallback_e:
            latency_ms = (time.time() - start_time) * 1000
            benchmark_tracker.log_request(req.text, {}, latency_ms, is_fallback, cache_hit, provider_manager.get_active_provider_name(is_fallback))
            logger.error(f"CRITICAL: Primary circuit OPEN/Failed. Fallback failed with '{str(fallback_e)}'.")
            raise HTTPException(status_code=500, detail="Internal server error during emotion analysis.")
            
    latency_ms = (time.time() - start_time) * 1000
    active_prov = provider_manager.get_active_provider_name(is_fallback)
    benchmark_tracker.log_request(req.text, final_output, latency_ms, is_fallback, cache_hit, active_prov)
    
    return {
        "status": "success",
        "data": final_output,
        "metrics": {
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "fallback_triggered": is_fallback,
            "active_provider": active_prov
        }
    }

@app.get("/metrics")
def get_metrics():
    return benchmark_tracker.get_summary()
