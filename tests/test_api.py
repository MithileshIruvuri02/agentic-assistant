import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.main import app


# ✅ FIXED: use pytest_asyncio.fixture
@pytest_asyncio.fixture
async def client():
    """Async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
class TestAPI:
    """Test suite for API endpoints."""
    
    async def test_health_check(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
    
    async def test_text_input_needs_clarification(self, client):
        response = await client.post(
            "/api/process",
            data={"text": "Hello, this is some random text."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["needs_clarification", "completed"]
    
    async def test_summarization_explicit(self, client):
        text = """
        Artificial Intelligence (AI) has revolutionized many industries.
        Machine learning algorithms can now process vast amounts of data.
        Deep learning uses neural networks to solve complex problems.
        """
        response = await client.post(
            "/api/process",
            data={"text": f"Summarize this: {text}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["execution_plan"]["task_type"] == "summarization"
    
    async def test_sentiment_analysis(self, client):
        response = await client.post(
            "/api/process",
            data={"text": "What is the sentiment of: I absolutely love this product!"}
        )
        assert response.status_code == 200
        data = response.json()
        if data["status"] == "completed":
            assert data["execution_plan"]["task_type"] == "sentiment_analysis"
    
    async def test_youtube_url(self, client):
        response = await client.post(
            "/api/process",
            data={"text": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )
        assert response.status_code == 200
    
    async def test_cost_estimation(self, client):
        response = await client.post(
            "/api/process",
            data={"text": "Summarize: AI is transforming the world."}
        )
        assert response.status_code == 200
        data = response.json()
        if data["status"] == "completed":
            assert data["total_cost"] >= 0
    
    async def test_clarification_flow(self, client):
        response1 = await client.post(
            "/api/process",
            data={"text": "Process this data: Apple, Microsoft, Google"}
        )
        data1 = response1.json()
        if data1["status"] == "needs_clarification":
            response2 = await client.post(
                "/api/process",
                data={
                    "clarification_response": "Analyze the sentiment",
                    "previous_request_id": data1["request_id"]
                }
            )
            data2 = response2.json()
            assert data2["status"] == "completed"
    
    async def test_error_handling_no_input(self, client):
        response = await client.post("/api/process", data={})
        assert response.status_code == 400


@pytest.mark.asyncio
class TestServices:
    async def test_ocr_service(self):
        from app.services.ocr_service import OCRService
        service = OCRService()
        assert service is not None
    
    async def test_summarizer(self):
        from app.services.summarizer import SummarizerService
        summarizer = SummarizerService()
        result = await summarizer.summarize("AI is transforming industries.")
        assert result.one_line
        assert len(result.bullets) == 3


@pytest.mark.asyncio
class TestAgents:
    async def test_planner_agent(self):
        from app.agents.planner_agent import PlannerAgent
        from app.models.schemas import ExtractedContent, InputType
        
        planner = PlannerAgent()
        content = ExtractedContent(
            text="Summarize this: AI is the future.",
            input_type=InputType.TEXT,
            extraction_method="direct"
        )
        plan = await planner.create_plan(content)
        assert plan.task_type is not None
    
    async def test_executor_agent(self):
        from app.agents.executor_agent import ExecutorAgent
        from app.models.schemas import ExecutionPlan, ExtractedContent, TaskType, InputType
        
        executor = ExecutorAgent()
        plan = ExecutionPlan(
            task_type=TaskType.TEXT_EXTRACTION,
            steps=["Extract"],
            estimated_tokens=100,
            estimated_cost=0.0,
            reasoning="Test"
        )
        content = ExtractedContent(
            text="Test content",
            input_type=InputType.TEXT,
            extraction_method="direct"
        )
        result = await executor.execute(plan, content)
        assert result.execution_time_seconds >= 0
