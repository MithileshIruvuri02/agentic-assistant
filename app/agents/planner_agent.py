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
    Planner agent using Groq (FREE!) for intent understanding and planning.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.GROQ_API_KEY)
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    async def create_plan(
        self, 
        extracted_content: ExtractedContent,
        user_clarification: str = None
    ) -> ExecutionPlan:
        """
        Analyze extracted content and create execution plan using Groq (FREE).
        """
        logger.info("Planning task execution with Groq (FREE)")
        
        # Build context for LLM
        context = self._build_context(extracted_content, user_clarification)
        
        # Get plan from Groq
        plan_response = await self._get_llm_plan(context)
        
        # Parse and validate plan
        execution_plan = self._parse_plan(plan_response, extracted_content)
        
        logger.info(f"Plan created: {execution_plan.task_type}, steps: {len(execution_plan.steps)}")
        
        return execution_plan
    
    def _build_context(self, content: ExtractedContent, clarification: str = None) -> str:
        """Build context string for LLM planning."""
        context = f"""Analyze this content and determine the user's intent.

INPUT TYPE: {content.input_type}
EXTRACTION METHOD: {content.extraction_method}
CONTENT LENGTH: {len(content.text)} characters

CONTENT:
{content.text[:2000]}{"..." if len(content.text) > 2000 else ""}

METADATA: {json.dumps(content.metadata, indent=2)}
"""
        
        if clarification:
            context += f"\n\nUSER CLARIFICATION: {clarification}"
        
        return context
    
    async def _get_llm_plan(self, context: str) -> Dict[str, Any]:
        """Get execution plan from Groq (FREE)."""
        
        system_prompt = """You are a planning agent that determines user intent and creates execution plans.

AVAILABLE TASK TYPES:
- text_extraction: User wants to see extracted/transcribed text
- youtube_transcript: User wants YouTube video transcript
- summarization: User wants summary (1-line + 3 bullets + 5 sentences)
- sentiment_analysis: User wants sentiment analysis (label + confidence + justification)
- code_explanation: User wants code explained (language + explanation + bugs + complexity)
- conversational: User is asking questions or having a conversation
- clarification_needed: Intent is unclear

CLARIFICATION RULES:
- If input is JUST text extraction with no specific request, ask what they want
- If multiple tasks are equally plausible, ask which they prefer
- If there are explicit instructions or clear context, proceed without asking
- If content has code and user says "explain", do code_explanation
- If user says "summarize" or "summary", do summarization
- If user asks about sentiment/feeling/emotion, do sentiment_analysis

Respond ONLY with valid JSON (no markdown):
{
  "task_type": "one of the task types above",
  "reasoning": "why you chose this task",
  "requires_clarification": true or false,
  "clarification_question": "question if needed, else null",
  "suggested_steps": ["step 1", "step 2"]
}"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Groq planner response: {response_text[:200]}...")
            
            # Parse JSON - handle markdown
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            elif '```' in response_text:
                response_text = re.sub(r'```\w*\n?', '', response_text).strip()
            
            # Find JSON object
            if not response_text.startswith('{'):
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
            
            plan_data = json.loads(response_text)
            
            # Set defaults
            plan_data.setdefault('requires_clarification', False)
            plan_data.setdefault('clarification_question', None)
            plan_data.setdefault('suggested_steps', ['Process content'])
            
            return plan_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response: {e}")
            logger.debug(f"Response text: {response_text}")
            
            # Default fallback
            return {
                "task_type": "conversational",
                "reasoning": "Failed to parse plan, defaulting to conversational",
                "requires_clarification": False,
                "suggested_steps": ["Provide a helpful response"]
            }
        except Exception as e:
            logger.error(f"Failed to get plan from Groq: {e}")
            raise
    
    def _parse_plan(self, plan_data: Dict[str, Any], content: ExtractedContent) -> ExecutionPlan:
        """Parse LLM response into ExecutionPlan object."""
        
        # Map task type string to enum
        task_type_map = {
            "text_extraction": TaskType.TEXT_EXTRACTION,
            "youtube_transcript": TaskType.YOUTUBE_TRANSCRIPT,
            "summarization": TaskType.SUMMARIZATION,
            "sentiment_analysis": TaskType.SENTIMENT_ANALYSIS,
            "code_explanation": TaskType.CODE_EXPLANATION,
            "conversational": TaskType.CONVERSATIONAL,
            "clarification_needed": TaskType.CLARIFICATION_NEEDED
        }
        
        task_type = task_type_map.get(
            plan_data.get("task_type", "conversational"),
            TaskType.CONVERSATIONAL
        )
        
        # Estimate tokens
        estimated_tokens = len(self.encoder.encode(content.text))
        estimated_tokens += 500  # Overhead
        
        # Groq is FREE, so cost is 0!
        estimated_cost = 0.0
        
        return ExecutionPlan(
            task_type=task_type,
            steps=plan_data.get("suggested_steps", ["Process content"]),
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
            requires_clarification=plan_data.get("requires_clarification", False),
            clarification_question=plan_data.get("clarification_question"),
            reasoning=plan_data.get("reasoning", "No reasoning provided"),
            metadata={
                "content_length": len(content.text),
                "input_type": content.input_type,
                "free_api": True  # Groq is FREE!
            }
        )