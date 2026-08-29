"""
eval_judge.py: Custom DeepEval judge model that interfaces with Groq (or OpenAI if key is present).
Implements DeepEvalBaseLLM so it can be passed into any DeepEval metric (GEval, Faithfulness, etc.).
"""
import os
import json
import re
from typing import Optional, Union, Type
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from deepeval.models.base_model import DeepEvalBaseLLM

class GroqJudge(DeepEvalBaseLLM):
    """
    Evaluator LLM for DeepEval using Groq.
    Defaults to openai/gpt-oss-120b (fast, large reasoning model available in user's Groq account).
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EVAL_MODEL", "openai/gpt-oss-20b")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment or .env")
        from groq import Groq
        self.client = Groq(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Union[str, BaseModel]:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.0,
        )
        content = chat_completion.choices[0].message.content or ""
        
        if schema:
            # Extract JSON from content if model wrapped it in markdown fences
            cleaned = content.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            json_str = match.group(0) if match else cleaned
            try:
                data = json.loads(json_str)
                return schema(**data)
            except Exception:
                pass
        return content

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Union[str, BaseModel]:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return f"Groq-{self.model_name}"


def get_eval_model() -> Optional[DeepEvalBaseLLM]:
    """
    Returns the judge model for DeepEval metrics.
    If OPENAI_API_KEY is present, returns None (DeepEval will default to gpt-4o natively).
    Otherwise returns GroqJudge.
    """
    if os.getenv("OPENAI_API_KEY"):
        return None
    return GroqJudge()
