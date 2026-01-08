"""
Cost Estimator - Bonus Feature
Calculates approximate token and API costs before execution.
"""

import tiktoken
from typing import Dict, Any
from loguru import logger

from app.models.schemas import TaskType, ExtractedContent
from app.config import get_settings, TOKEN_COSTS


class CostEstimator:
    """
    Cost estimator that predicts API costs before execution.
    This is a bonus feature for the assignment.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    def estimate_cost(
        self, 
        task_type: TaskType, 
        content: ExtractedContent
    ) -> Dict[str, Any]:
        """
        Estimate cost for executing a task.
        
        Args:
            task_type: Type of task to be executed
            content: Extracted content to process
            
        Returns:
            Dictionary with cost breakdown
        """
        logger.info(f"Estimating cost for {task_type}")
        
        # Count input tokens
        input_tokens = self._count_tokens(content.text)
        
        # Estimate output tokens based on task
        output_tokens = self._estimate_output_tokens(task_type)
        
        # Calculate costs
        costs = self._calculate_costs(task_type, input_tokens, output_tokens)
        
        logger.info(f"Estimated cost: ${costs['total_cost']:.4f} "
                   f"({input_tokens} input + {output_tokens} output tokens)")
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": costs["input_cost"],
            "output_cost": costs["output_cost"],
            "total_cost": costs["total_cost"],
            "model": costs["model"],
            "breakdown": costs.get("breakdown", {}),
            "confidence": self._estimate_confidence(task_type, content)
        }
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            tokens = len(self.encoder.encode(text))
            # Add overhead for system prompts and formatting
            overhead = min(500, int(tokens * 0.1))  # 10% overhead, max 500 tokens
            return tokens + overhead
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using character estimate")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4 + 500
    
    def _estimate_output_tokens(self, task_type: TaskType) -> int:
        """Estimate output tokens based on task type."""
        estimates = {
            TaskType.TEXT_EXTRACTION: 100,  # Just returning extracted text
            TaskType.YOUTUBE_TRANSCRIPT: 100,  # Just returning transcript
            TaskType.SUMMARIZATION: 500,  # 1-line + bullets + 5 sentences
            TaskType.SENTIMENT_ANALYSIS: 150,  # Label + confidence + justification
            TaskType.CODE_EXPLANATION: 800,  # Detailed explanation + bugs + complexity
            TaskType.CONVERSATIONAL: 400,  # General response
            TaskType.CLARIFICATION_NEEDED: 100,  # Short clarification question
        }
        return estimates.get(task_type, 300)
    
    def _calculate_costs(
        self, 
        task_type: TaskType, 
        input_tokens: int, 
        output_tokens: int
    ) -> Dict[str, Any]:
        """Calculate costs based on task type and tokens."""
        
        # Determine which model will be used
        if task_type in [TaskType.SUMMARIZATION, TaskType.CODE_EXPLANATION, TaskType.CONVERSATIONAL]:
            model = self.settings.OPENAI_MODEL
            model_key = "gpt-4-turbo-preview"
        else:
            model = self.settings.ANTHROPIC_MODEL
            model_key = "claude-sonnet-4-20250514"
        
        # Get costs for model
        costs = TOKEN_COSTS.get(model_key, TOKEN_COSTS["gpt-4-turbo-preview"])
        
        # Calculate
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        total_cost = input_cost + output_cost
        
        breakdown = {
            "input": {
                "tokens": input_tokens,
                "rate_per_1k": costs["input"],
                "cost": input_cost
            },
            "output": {
                "tokens": output_tokens,
                "rate_per_1k": costs["output"],
                "cost": output_cost
            }
        }
        
        # Add audio transcription cost if applicable
        if task_type == TaskType.TEXT_EXTRACTION and "audio" in str(task_type).lower():
            audio_duration_min = 5  # Default estimate
            audio_cost = audio_duration_min * self.settings.WHISPER_COST_PER_MIN
            total_cost += audio_cost
            breakdown["audio_transcription"] = {
                "duration_min": audio_duration_min,
                "rate_per_min": self.settings.WHISPER_COST_PER_MIN,
                "cost": audio_cost
            }
        
        return {
            "model": model,
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(total_cost, 4),
            "breakdown": breakdown
        }
    
    def _estimate_confidence(self, task_type: TaskType, content: ExtractedContent) -> float:
        """
        Estimate confidence in cost prediction.
        Lower confidence for more complex/variable tasks.
        """
        base_confidence = 0.85
        
        # Reduce confidence for complex tasks
        if task_type in [TaskType.CODE_EXPLANATION, TaskType.CONVERSATIONAL]:
            base_confidence -= 0.15
        
        # Reduce confidence for long content (more variable)
        if len(content.text) > 5000:
            base_confidence -= 0.1
        
        # Increase confidence for simple extraction
        if task_type in [TaskType.TEXT_EXTRACTION, TaskType.YOUTUBE_TRANSCRIPT]:
            base_confidence += 0.10
        
        return max(0.5, min(0.95, base_confidence))
    
    def compare_models(self, task_type: TaskType, content: ExtractedContent) -> Dict[str, Any]:
        """
        Compare costs across different models.
        Useful for showing users cost-performance tradeoffs.
        """
        input_tokens = self._count_tokens(content.text)
        output_tokens = self._estimate_output_tokens(task_type)
        
        comparisons = {}
        for model_name, costs in TOKEN_COSTS.items():
            input_cost = (input_tokens / 1000) * costs["input"]
            output_cost = (output_tokens / 1000) * costs["output"]
            total_cost = input_cost + output_cost
            
            comparisons[model_name] = {
                "total_cost": round(total_cost, 4),
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "relative_performance": self._get_relative_performance(model_name)
            }
        
        return comparisons
    
    def _get_relative_performance(self, model_name: str) -> str:
        """Get relative performance description for model."""
        performance_map = {
            "gpt-4-turbo-preview": "Highest quality, slower",
            "gpt-4": "High quality, moderate speed",
            "gpt-3.5-turbo": "Good quality, fast, economical",
            "claude-sonnet-4-20250514": "Excellent balance of quality and speed",
            "claude-3-opus": "Premium quality, comprehensive analysis",
        }
        return performance_map.get(model_name, "Unknown")
    
    def get_cost_summary(self, estimated_cost: float, actual_cost: float = None) -> str:
        """
        Generate human-readable cost summary.
        """
        summary = f"💰 Estimated cost: ${estimated_cost:.4f}"
        
        if actual_cost:
            difference = actual_cost - estimated_cost
            percentage = (difference / estimated_cost * 100) if estimated_cost > 0 else 0
            summary += f"\n   Actual cost: ${actual_cost:.4f}"
            summary += f"\n   Difference: ${abs(difference):.4f} ({abs(percentage):.1f}%)"
            
            if abs(percentage) < 10:
                summary += " ✅ (Accurate estimate)"
            elif abs(percentage) < 25:
                summary += " ⚠️ (Within acceptable range)"
            else:
                summary += " ❌ (Significant deviation)"
        
        return summary