import re
import json
from groq import Groq
from loguru import logger
from app.models.schemas import CodeExplanationResult
from app.config import get_settings


class CodeExplainerService:
    """Service for code explanation using Groq (FREE!)."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.GROQ_API_KEY)
    
    async def explain(self, code: str) -> CodeExplanationResult:
        """Explain code: language + explanation + bugs + complexity using Groq (FREE)."""
        logger.info(f"Explaining code with Groq (FREE): {len(code)} chars")
        
        prompt = f"""Analyze the following code and provide:

1. Programming language
2. Clear explanation of what the code does
3. Any potential bugs or issues (as a list, can be empty if no bugs)
4. Time complexity (Big O notation)
5. Space complexity (Big O notation)

Code:
{code[:3000]}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "language": "Python",
  "explanation": "detailed explanation here",
  "potential_bugs": ["bug 1", "bug 2"],
  "time_complexity": "O(n)",
  "space_complexity": "O(1)"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code analyzer. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"Groq response: {content[:200]}...")
            
            # Parse JSON - handle markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            elif '```' in content:
                content = re.sub(r'```\w*\n?', '', content).strip()
            
            # Try to find JSON object
            if not content.startswith('{'):
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
            
            data = json.loads(content)
            
            # Ensure potential_bugs is a list
            if not isinstance(data.get('potential_bugs'), list):
                data['potential_bugs'] = []
            
            # Set defaults if missing
            data.setdefault('time_complexity', 'O(n)')
            data.setdefault('space_complexity', 'O(1)')
            data.setdefault('language', self._detect_language(code))
            
            logger.info(f"✅ Code explained (FREE): {data['language']}")
            return CodeExplanationResult(**data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response: {e}")
            logger.error(f"Response was: {content[:500]}")
            
            # Fallback explanation
            language = self._detect_language(code)
            return CodeExplanationResult(
                language=language,
                explanation=f"This appears to be {language} code. It contains functions, variables, and control structures.",
                potential_bugs=["Unable to perform detailed analysis"],
                time_complexity="O(n)",
                space_complexity="O(1)"
            )
            
        except Exception as e:
            logger.error(f"Code explanation failed: {e}")
            raise
    
    def _detect_language(self, code: str) -> str:
        """Simple language detection based on syntax."""
        code_lower = code.lower()
        
        if 'def ' in code or 'import ' in code or 'print(' in code:
            return 'Python'
        elif 'function ' in code or 'const ' in code or 'let ' in code or '=>' in code:
            return 'JavaScript'
        elif '#include' in code or 'int main' in code:
            return 'C/C++'
        elif 'public class' in code or 'public static void' in code:
            return 'Java'
        elif 'fn ' in code and 'let mut' in code:
            return 'Rust'
        elif 'func ' in code and 'var ' in code:
            return 'Go'
        else:
            return 'Unknown'