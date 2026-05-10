import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer

from src.bielik_llm import BielikChatModel
from src.agent import Agent
from src.mcp_client import MCPClient


bielik = None
tokenizer = None
medical_agent = None
llm_semaphore = None

MODEL_NAME = "speakleash/Bielik-11B-v2"
MCP_CONFIGS = {
    "medical_knowledge_base": {
        "transport": "http",
        "url": "http://127.0.0.1:8002/mcp",
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bielik, tokenizer, medical_agent, llm_semaphore
    bielik = BielikChatModel()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    medical_agent = Agent()
    llm_semaphore = asyncio.Semaphore(5)
    yield


app = FastAPI(title="Medical Agent API", lifespan=lifespan)


class QuestionRequest(BaseModel):
    question: str = Field(..., example="Jaka jest najczęstsza przyczyna zwężenia zastawki trójdzielnej?")

class AgentResponse(BaseModel):
    user_question: str
    final_response: Optional[str]
    total_steps: int
    thoughts_history: Optional[str]
    messages_history: List[Dict[str, Any]] = []


@app.post("/ask", response_model=AgentResponse)
async def ask_question(request: QuestionRequest):
    async with llm_semaphore:
        async with MCPClient(MCP_CONFIGS) as mcp_client:
            try:
                config = {
                    "configurable": {
                        "llm": bielik,
                        "mcp_client": mcp_client,
                        "tokenizer": tokenizer,
                        "max_context_len": 200,
                        "max_steps": 10,
                    }
                }

                result = await medical_agent(
                    {"user_question": request.question},
                    config=config
                )

                raw_messages = result.get("messages", [])
                formatted_messages = []
                for msg in raw_messages:
                    msg_data = {
                        "type": msg.type,
                        "content": msg.content,
                    }
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        msg_data["tool_calls"] = msg.tool_calls
                    if msg.type == "tool":
                        msg_data["tool_name"] = getattr(msg, "name", "unknown_tool")
                    formatted_messages.append(msg_data)

                return AgentResponse(
                    user_question=result["user_question"],
                    final_response=result["final_response"],
                    total_steps=result["step"],
                    thoughts_history=result["thoughts"],
                    messages_history=formatted_messages
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)