import re
import json
from groq import Groq
from loguru import logger
from app.models.schemas import SummaryResult
from app.config import get_settings


class SummarizerService:
    """Service for text summarization using Groq (FREE!)."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.GROQ_API_KEY)
    
    async def summarize(self, text: str) -> SummaryResult:
        """Generate 1-line + 3 bullets + 5 sentences summary using Groq (FREE)."""
        logger.info(f"Summarizing text with Groq (FREE): {len(text)} chars")
        
        prompt = f"""Summarize the following text in three formats:

1. ONE-LINE SUMMARY (max 20 words)
2. THREE BULLET POINTS (key takeaways)
3. FIVE SENTENCES (comprehensive summary)

Text to summarize:
{text[:4000]}

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "one_line": "your one-line summary here",
  "bullets": ["bullet 1", "bullet 2", "bullet 3"],
  "five_sentence": "your five sentence summary here"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise summaries. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"Groq response: {content}")
            
            # Parse JSON - handle markdown code blocks if present
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            elif '```' in content:
                # Remove any markdown code fences
                content = re.sub(r'```\w*\n?', '', content).strip()
            
            # Parse JSON
            data = json.loads(content)
            
            # Validate structure
            if not all(k in data for k in ['one_line', 'bullets', 'five_sentence']):
                raise ValueError("Missing required fields in summary")
            
            if not isinstance(data['bullets'], list) or len(data['bullets']) != 3:
                raise ValueError("Bullets must be a list of exactly 3 items")
            
            logger.info(f"✅ Summary generated successfully (FREE)")
            return SummaryResult(**data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response as JSON: {e}")
            logger.error(f"Response was: {content}")
            
            # Fallback: create a simple summary
            sentences = text.split('.')[:5]
            return SummaryResult(
                one_line=text[:100],
                bullets=[s.strip() for s in sentences[:3] if s.strip()],
                five_sentence='. '.join(sentences) + '.'
            )
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise