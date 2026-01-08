import json
import re
from typing import Dict, Any

from groq import Groq
from loguru import logger
import tiktoken

from app.models.schemas import TaskType, ExecutionPlan, ExtractedContent
from app.config import get_settings


class PlannerAgent:
    """
    Planner agent using Groq (FREE) for intent understanding and execution planning.
    Enforces grounding rules to prevent hallucinations.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.GROQ_API_KEY)
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    async def create_plan(
        self,
        extracted_content: ExtractedContent,
        user_clarification: str | None = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan based on extracted content.
        """

        logger.info("🧠 PlannerAgent: Creating execution plan")

        # 🚨 HARD GUARD: YouTube transcript failed
        if extracted_content.extraction_method == "youtube_failed":
            logger.warning("🚫 YouTube transcript unavailable — blocking summarization")

            return ExecutionPlan(
                task_type=TaskType.CLARIFICATION_NEEDED,
                steps=[],
                estimated_tokens=0,
                estimated_cost=0.0,
                requires_clarification=True,
                clarification_question=(
                    "I couldn’t access captions for this YouTube video. "
                    "Please upload the transcript, enable captions, or upload the audio/video file."
                ),
                reasoning="YouTube transcript extraction failed; summarization would be ungrounded.",
                metadata={
                    "blocked": True,
                    "reason": "youtube_transcript_unavailable",
                    "free_api": True,
                },
            )

        # Build LLM context
        context = self._build_context(extracted_content, user_clarification)

        # Ask Groq for intent
        plan_data = await self._get_llm_plan(context)

        # Parse into ExecutionPlan
        execution_plan = self._parse_plan(plan_data, extracted_content)

        logger.info(
            f"✅ Plan finalized | task={execution_plan.task_type} | tokens={execution_plan.estimated_tokens}"
        )

        return execution_plan

    def _build_context(
        self,
        content: ExtractedContent,
        clarification: str | None = None,
    ) -> str:
        """
        Build prompt context for the planner LLM.
        """

        context = f"""
Analyze the user's intent strictly based on AVAILABLE content.

INPUT TYPE: {content.input_type}
EXTRACTION METHOD: {content.extraction_method}
CONTENT LENGTH: {len(content.text)} characters

CONTENT (may be empty):
{content.text[:2000]}{"..." if len(content.text) > 2000 else ""}

METADATA:
{json.dumps(content.metadata, indent=2)}
"""

        if clarification:
            context += f"\n\nUSER CLARIFICATION:\n{clarification}"

        return context.strip()

    async def _get_llm_plan(self, context: str) -> Dict[str, Any]:
        """
        Ask Groq to determine user intent and planning steps.
        """

        system_prompt = """
You are a planning agent.

CRITICAL RULES (MUST FOLLOW):
- NEVER summarize content that is empty or unavailable
- NEVER hallucinate missing information
- If text length is 0 and user asks to summarize → clarification_needed
- If extraction method indicates failure → clarification_needed

AVAILABLE TASK TYPES:
- text_extraction
- youtube_transcript
- summarization
- sentiment_analysis
- code_explanation
- conversational
- clarification_needed

Respond ONLY with valid JSON:
{
  "task_type": "...",
  "reasoning": "...",
  "requires_clarification": true/false,
  "clarification_question": "... or null",
  "suggested_steps": ["step 1", "step 2"]
}
"""

        response = self.client.chat.completions.create(
            model=self.settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
            max_tokens=800,
        )

        raw_text = response.choices[0].message.content.strip()
        logger.debug(f"📥 Planner raw response: {raw_text[:300]}")

        # Remove markdown if present
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("❌ Planner JSON parse failed — defaulting to clarification")
            return {
                "task_type": "clarification_needed",
                "reasoning": "Failed to parse planner response",
                "requires_clarification": True,
                "clarification_question": "Could you clarify what you'd like me to do?",
                "suggested_steps": [],
            }

    def _parse_plan(
        self,
        plan_data: Dict[str, Any],
        content: ExtractedContent,
    ) -> ExecutionPlan:
        """
        Convert planner JSON into ExecutionPlan.
        """

        task_map = {
            "text_extraction": TaskType.TEXT_EXTRACTION,
            "youtube_transcript": TaskType.YOUTUBE_TRANSCRIPT,
            "summarization": TaskType.SUMMARIZATION,
            "sentiment_analysis": TaskType.SENTIMENT_ANALYSIS,
            "code_explanation": TaskType.CODE_EXPLANATION,
            "conversational": TaskType.CONVERSATIONAL,
            "clarification_needed": TaskType.CLARIFICATION_NEEDED,
        }

        task_type = task_map.get(
            plan_data.get("task_type"),
            TaskType.CONVERSATIONAL,
        )

        # Token estimation (safe)
        estimated_tokens = (
            len(self.encoder.encode(content.text)) if content.text else 0
        ) + 300

        return ExecutionPlan(
            task_type=task_type,
            steps=plan_data.get("suggested_steps", []),
            estimated_tokens=estimated_tokens,
            estimated_cost=0.0,  # Groq FREE
            requires_clarification=plan_data.get("requires_clarification", False),
            clarification_question=plan_data.get("clarification_question"),
            reasoning=plan_data.get("reasoning", "No reasoning provided"),
            metadata={
                "content_length": len(content.text),
                "extraction_method": content.extraction_method,
                "free_api": True,
            },
        )
