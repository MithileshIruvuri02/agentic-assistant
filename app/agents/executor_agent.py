import time
from typing import Any, Dict
from loguru import logger
from groq import Groq

from app.models.schemas import (
    TaskType, ExecutionPlan, ExtractedContent, TaskResult,
    SummaryResult, SentimentResult, CodeExplanationResult
)
from app.services.summarizer import SummarizerService
from app.services.sentiment_analyzer import SentimentAnalyzerService
from app.services.code_explainer import CodeExplainerService
from app.config import get_settings


class ExecutorAgent:
    """
    Executor agent using Groq (FREE!) for task execution.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.summarizer = SummarizerService()
        self.sentiment_analyzer = SentimentAnalyzerService()
        self.code_explainer = CodeExplainerService()
        self.groq_client = Groq(api_key=self.settings.GROQ_API_KEY)
    
    async def execute(
        self, 
        plan: ExecutionPlan, 
        content: ExtractedContent
    ) -> TaskResult:
        """Execute the planned task using Groq (FREE)."""
        start_time = time.time()
        logger.info(f"Executing task with Groq (FREE): {plan.task_type}")
        
        try:
            # Route to appropriate handler
            if plan.task_type == TaskType.TEXT_EXTRACTION:
                output = await self._handle_text_extraction(content)
            
            elif plan.task_type == TaskType.YOUTUBE_TRANSCRIPT:
                output = await self._handle_youtube_transcript(content)
            
            elif plan.task_type == TaskType.SUMMARIZATION:
                output = await self._handle_summarization(content)
            
            elif plan.task_type == TaskType.SENTIMENT_ANALYSIS:
                output = await self._handle_sentiment_analysis(content)
            
            elif plan.task_type == TaskType.CODE_EXPLANATION:
                output = await self._handle_code_explanation(content)
            
            elif plan.task_type == TaskType.CONVERSATIONAL:
                output = await self._handle_conversational(content)
            
            else:
                raise ValueError(f"Unknown task type: {plan.task_type}")
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Task completed in {execution_time:.2f}s (FREE with Groq)")
            
            return TaskResult(
                task_type=plan.task_type,
                output=output,
                metadata={
                    "steps_completed": len(plan.steps),
                    "content_length": len(content.text),
                    "free_api": True
                },
                execution_time_seconds=round(execution_time, 2)
            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def _handle_text_extraction(self, content: ExtractedContent) -> Dict[str, Any]:
        """Handle text extraction task."""
        return {
            "extracted_text": content.text,
            "confidence": content.confidence,
            "metadata": content.metadata,
            "word_count": len(content.text.split()),
            "character_count": len(content.text)
        }
    
    async def _handle_youtube_transcript(self, content: ExtractedContent) -> Dict[str, Any]:
        """Handle YouTube transcript task."""
        return {
            "transcript": content.text,
            "duration_seconds": content.metadata.get("duration_seconds"),
            "video_id": content.metadata.get("video_id"),
            "word_count": len(content.text.split())
        }
    
    async def _handle_summarization(self, content: ExtractedContent) -> SummaryResult:
        """Handle summarization task using Groq (FREE)."""
        summary = await self.summarizer.summarize(content.text)
        return summary
    
    async def _handle_sentiment_analysis(self, content: ExtractedContent) -> SentimentResult:
        """Handle sentiment analysis using Groq (FREE)."""
        sentiment = await self.sentiment_analyzer.analyze(content.text)
        return sentiment
    
    async def _handle_code_explanation(self, content: ExtractedContent) -> CodeExplanationResult:
        """Handle code explanation using Groq (FREE)."""
        explanation = await self.code_explainer.explain(content.text)
        return explanation
    
    async def _handle_conversational(self, content: ExtractedContent) -> Dict[str, Any]:
        """Handle conversational/Q&A using Groq (FREE)."""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Provide clear, concise, and accurate responses."
                    },
                    {
                        "role": "user",
                        "content": f"Please provide a helpful response to this:\n\n{content.text}"
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "response": response.choices[0].message.content,
                "conversational": True,
                "free_api": True
            }
        except Exception as e:
            logger.error(f"Conversational response failed: {e}")
            return {
                "response": "I'm here to help! Could you please rephrase your question?",
                "conversational": True,
                "error": str(e)
            }