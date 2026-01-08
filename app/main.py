import uuid
import sys
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from loguru import logger

from app.config import get_settings
from app.models.schemas import (
    AgentResponse,
    HealthResponse,
    ProcessingStatus,
    InputType,          # ✅ FIXED
)
from app.services.input_processor import InputProcessor
from app.agents.planner_agent import PlannerAgent
from app.agents.executor_agent import ExecutorAgent


# -------------------------------------------------------------------
# Configuration & Logging
# -------------------------------------------------------------------

settings = get_settings()

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
           "<level>{message}</level>",
    level=settings.LOG_LEVEL,
)
logger.add(
    settings.LOG_FILE,
    rotation="100 MB",
    retention="10 days",
    level=settings.LOG_LEVEL,
)

# In-memory session store (clarification flow)
conversation_sessions: Dict[str, Dict] = {}


# -------------------------------------------------------------------
# Lifespan
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic Assistant API")
    logger.info(f"Groq Model: {settings.GROQ_MODEL}")

    if settings.ANTHROPIC_API_KEY:
        logger.info(f"Anthropic Model: {settings.ANTHROPIC_MODEL}")
    else:
        logger.info("Anthropic disabled — using Groq for all tasks")

    yield

    logger.info("Shutting down Agentic Assistant API")


# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------

app = FastAPI(
    title="Agentic Assistant API",
    description="Intelligent agent that processes text, images, PDFs, and audio",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")


# -------------------------------------------------------------------
# Services
# -------------------------------------------------------------------

input_processor = InputProcessor()
planner_agent = PlannerAgent()
executor_agent = ExecutorAgent()


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={
            "input_processor": True,
            "planner_agent": True,
            "executor_agent": True,
        },
    )


@app.post("/api/process", response_model=AgentResponse)
async def process_input(
    text: str = Form(None),
    file: UploadFile = File(None),
    clarification_response: str = Form(None),
    previous_request_id: str = Form(None),
):
    request_id = str(uuid.uuid4())
    logs: list[str] = []

    try:
        logger.info(f"Processing request {request_id}")
        logs.append("Request received")

        # ----------------------------
        # Clarification flow
        # ----------------------------
        if clarification_response and previous_request_id:
            return await handle_clarification(
                previous_request_id,
                clarification_response,
                request_id,
            )

        # ----------------------------
        # Validation
        # ----------------------------
        if not text and not file:
            raise HTTPException(status_code=400, detail="No input provided")

        # ----------------------------
        # Step 1: Input Processing
        # ----------------------------
        extracted_content = await input_processor.process(text=text, file=file)
        logs.append(f"Content extracted ({extracted_content.input_type})")

        # ----------------------------
        # Step 2: Planning
        # ----------------------------
        execution_plan = await planner_agent.create_plan(extracted_content)
        logs.append(f"Plan created: {execution_plan.task_type}")

        if execution_plan.requires_clarification:
            conversation_sessions[request_id] = {
                "extracted_content": extracted_content,
                "plan": execution_plan,
            }

            return AgentResponse(
                request_id=request_id,
                status=ProcessingStatus.NEEDS_CLARIFICATION,
                input_type=extracted_content.input_type,
                extracted_content=extracted_content,
                execution_plan=execution_plan,
                clarification_question=execution_plan.clarification_question,
                logs=logs,
            )

        # ----------------------------
        # Step 3: Execution
        # ----------------------------
        task_result = await executor_agent.execute(
            execution_plan,
            extracted_content,
        )

        logs.append(f"Task completed in {task_result.execution_time_seconds}s")

        return AgentResponse(
            request_id=request_id,
            status=ProcessingStatus.COMPLETED,
            input_type=extracted_content.input_type,
            extracted_content=extracted_content,
            execution_plan=execution_plan,
            result=task_result,
            logs=logs,
            total_cost=execution_plan.estimated_cost,
        )

    # ✅ IMPORTANT: let FastAPI handle HTTP errors
    except HTTPException:
        raise

    # ❌ Only unexpected errors come here
    except Exception as e:
        logger.exception(f"Error processing request {request_id}")
        logs.append(str(e))

        return AgentResponse(
            request_id=request_id,
            status=ProcessingStatus.FAILED,
            input_type=InputType.TEXT,
            error_message="Internal server error",
            logs=logs,
        )


# -------------------------------------------------------------------
# Clarification Handler
# -------------------------------------------------------------------

async def handle_clarification(
    previous_request_id: str,
    clarification: str,
    request_id: str,
) -> AgentResponse:
    if previous_request_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = conversation_sessions.pop(previous_request_id)
    extracted_content = session["extracted_content"]

    execution_plan = await planner_agent.create_plan(
        extracted_content,
        user_clarification=clarification,
    )

    task_result = await executor_agent.execute(
        execution_plan,
        extracted_content,
    )

    return AgentResponse(
        request_id=request_id,
        status=ProcessingStatus.COMPLETED,
        input_type=extracted_content.input_type,
        extracted_content=extracted_content,
        execution_plan=execution_plan,
        result=task_result,
        logs=["Clarification handled"],
        total_cost=execution_plan.estimated_cost,
    )


# -------------------------------------------------------------------
# Local run
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
