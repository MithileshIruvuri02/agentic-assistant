import re
import json
from groq import Groq
from loguru import logger
from app.models.schemas import SentimentResult
from app.config import get_settings


class SentimentAnalyzerService:
    """Service for sentiment analysis using Groq (FREE!)."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.GROQ_API_KEY)
    
    async def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment: label + confidence + justification using Groq (FREE)."""
        logger.info(f"Analyzing sentiment with Groq (FREE): {len(text)} chars")
        
        prompt = f"""Analyze the sentiment of the following text.

Text:
{text[:2000]}

Provide:
1. Label: "positive", "negative", or "neutral"
2. Confidence: a number between 0.0 and 1.0
3. Justification: one-line explanation of why

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "label": "positive",
  "confidence": 0.85,
  "justification": "your explanation here"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analysis expert. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"Groq response: {content}")
            
            # Parse JSON - handle markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            elif '```' in content:
                content = re.sub(r'```\w*\n?', '', content).strip()
            
            # Try to find JSON object in the response
            if not content.startswith('{'):
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    content = json_match.group(0)
            
            data = json.loads(content)
            
            # Validate
            if data['label'] not in ['positive', 'negative', 'neutral']:
                data['label'] = 'neutral'
            
            if not (0 <= data['confidence'] <= 1):
                data['confidence'] = 0.5
            
            logger.info(f"✅ Sentiment analyzed (FREE): {data['label']} ({data['confidence']})")
            return SentimentResult(**data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response: {e}")
            logger.error(f"Response was: {content}")
            
            # Fallback sentiment detection
            text_lower = text.lower()
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'wonderful']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'poor', 'horrible']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                label = 'positive'
                confidence = min(0.6 + pos_count * 0.1, 0.9)
            elif neg_count > pos_count:
                label = 'negative'
                confidence = min(0.6 + neg_count * 0.1, 0.9)
            else:
                label = 'neutral'
                confidence = 0.5
            
            return SentimentResult(
                label=label,
                confidence=confidence,
                justification="Sentiment detected based on keyword analysis"
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise