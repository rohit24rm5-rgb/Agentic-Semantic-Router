import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.emotion_analysis.core.provider_manager import provider_manager

class LexicalOutput(BaseModel):
    surface_emotion: str = Field(description="The primary emotion detected (e.g. Joy, Sadness, Anger)")
    intensity: int = Field(description="Intensity of the emotion from 1 to 10")
    key_phrases: list[str] = Field(description="Words or phrases that led to this conclusion")

class ContextualOutput(BaseModel):
    final_emotion: str = Field(description="The final categorized emotion (e.g. anger, joy, sadness, surprise, neutral, etc.)")
    is_sarcastic: bool = Field(description="True if sarcasm was detected, False otherwise")
    nuance_explanation: str = Field(description="1-2 sentences explaining the emotion and if sarcasm changed the meaning")

# --- Optimized Prompts ---
LEXICAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the Lexical Emotion Agent. Your strict, single-purpose task is to analyze the surface-level, explicit emotional tone of the provided text.\n\n"
               "### DIRECTIVES\n"
               "1. Focus STRICTLY on the literal dictionary definition of the words used.\n"
               "2. Ignore subtext, implied sarcasm, or situational irony.\n"
               "3. If the text lacks explicit emotional vocabulary, classify the emotion as 'Neutral'.\n"
               "4. Do not attempt to guess the user's true intent.\n\n"
               "### OUTPUT INSTRUCTIONS\n"
               "Output strictly in valid JSON according to this schema. Do not include introductory text or markdown formatting:\n"
               "{format_instructions}"),
    ("human", "{query}")
])

def run_lexical_agent(query: str) -> LexicalOutput:
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=provider_manager.get_active_groq_key(), max_retries=0)
    parser = JsonOutputParser(pydantic_object=LexicalOutput)
    prompt = LEXICAL_PROMPT.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser
    result = chain.invoke({"query": query})
    # Enforce Pydantic validation
    return LexicalOutput(**result)

CONTEXTUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the Contextual Nuance Agent. Your task is to synthesize surface-level emotional data with historical context to determine the true, underlying emotion of the text.\n\n"
               "### DIRECTIVES\n"
               "1. Compare the user's text against the provided 'Surface Lexical Analysis'.\n"
               "2. Analyze the 'Historical Contexts (RAG)' to identify patterns of sarcasm, slang, or implicit sentiments.\n"
               "3. CONFLICT RESOLUTION: If the RAG context implies a different emotion than the lexical analysis, weigh the implicit context heavier for your final verdict.\n"
               "4. TEARS OF JOY RULE: Be extremely careful not to confuse 'happy crying' or 'tears of joy' with sadness. If explicit positive emotion is stated alongside a physical reaction like crying, the true emotion remains Joy.\n"
               "5. SARCASM RULE: If the user says something positive like 'Oh great', 'Perfect', or 'Just what I needed', but the context is clearly negative (like rain or bad news), you MUST classify it as Sarcastic (is_sarcastic: true).\n"
               "6. FALLBACK: If the RAG context is empty, rely on your internal logic to detect implicit nuance.\n\n"
               "### OUTPUT INSTRUCTIONS\n"
               "Output strictly in valid JSON according to this schema. Do not include introductory text or markdown formatting:\n"
               "{format_instructions}"),
    ("human", "Analyze the following inputs to determine the final true emotion:\n\n"
              "<user_text>\n{query}\n</user_text>\n\n"
              "<lexical_analysis>\n{lexical_summary}\n</lexical_analysis>\n\n"
              "<rag_context>\n{rag_context}\n</rag_context>")
])

def run_contextual_agent(query: str, lexical_summary: dict, rag_context: str) -> ContextualOutput:
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=provider_manager.get_active_groq_key(), max_retries=0)
    parser = JsonOutputParser(pydantic_object=ContextualOutput)
    prompt = CONTEXTUAL_PROMPT.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser
    # Pass JSON string to avoid malformed python dicts in LLM context
    result = chain.invoke({
        "query": query, 
        "lexical_summary": json.dumps(lexical_summary),
        "rag_context": rag_context
    })
    # Enforce Pydantic validation
    return ContextualOutput(**result)
