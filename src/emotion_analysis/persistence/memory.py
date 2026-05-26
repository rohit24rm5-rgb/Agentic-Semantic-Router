import os
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Compute absolute base path dynamically (Emotion-Analysis root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(BASE_DIR, "persistence", "memory.sqlite"))

@asynccontextmanager
async def get_checkpointer():
    """Returns a LangGraph AsyncSqliteSaver for persistent memory using an async context manager.
    This ensures proper connection closure and triggers the internal .setup() method 
    to build the checkpoint tables asynchronously."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
