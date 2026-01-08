from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class TaskType(str, Enum):
    """Types of tasks the agent can perform."""
    TEXT_EXTRACTION = "text_extraction"
    YOUTUBE_TRANSCRIPT = "youtube_transcript"
    SUMMARIZATION = "summarization"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CODE_EXPLANATION = "code_explanation"
    CONVERSATIONAL = "conversational"
    CLARIFICATION_NEEDED = "clarification_needed"


class InputType(str, Enum):
    """Types of input the system can accept."""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Status of processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"


class ExtractedContent(BaseModel):
    """Content extracted from various input types."""
    text: str
    input_type: InputType
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction_method: Optional[str] = None


class ExecutionPlan(BaseModel):
    """Plan created by the planner agent."""
    task_type: TaskType
    steps: List[str]
    estimated_tokens: int
    estimated_cost: float
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    reasoning: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SummaryResult(BaseModel):
    """Structured summary output."""
    one_line: str
    bullets: List[str] = Field(min_items=3, max_items=3)
    five_sentence: str


class SentimentResult(BaseModel):
    """Sentiment analysis result."""
    label: str  # positive, negative, neutral
    confidence: float = Field(ge=0, le=1)
    justification: str


class CodeExplanationResult(BaseModel):
    """Code explanation result."""
    language: str
    explanation: str
    potential_bugs: List[str] = Field(default_factory=list)
    time_complexity: Optional[str] = None
    space_complexity: Optional[str] = None


class TaskResult(BaseModel):
    """Result of task execution."""
    task_type: TaskType
    output: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: float


class AgentResponse(BaseModel):
    """Complete response from the agent system."""
    request_id: str
    status: ProcessingStatus
    input_type: InputType
    extracted_content: Optional[ExtractedContent] = None
    execution_plan: Optional[ExecutionPlan] = None
    result: Optional[TaskResult] = None
    clarification_question: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    total_cost: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProcessRequest(BaseModel):
    """Request to process input."""
    text: Optional[str] = None
    clarification_response: Optional[str] = None
    previous_request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: Dict[str, bool]