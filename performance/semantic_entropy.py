import math
import logging
from typing import List, Dict
from .grounding import GroundingChecker

logger = logging.getLogger(__name__)

class SemanticEntropyChecker:
    def __init__(self, groq_client, grounding_checker: GroundingChecker, n_samples: int = 4):
        self.client = groq_client
        self.nli = grounding_checker
        self.n_samples = n_samples

    def compute_entropy(self, messages: List[Dict[str, str]], model: str = "openai/gpt-oss-20b") -> tuple[float, int]:
        samples = []
        tokens_used = 0
        for _ in range(self.n_samples):
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=512,
                stream=False,
            )
            samples.append(completion.choices[0].message.content.strip())
            if hasattr(completion, "usage") and completion.usage:
                tokens_used += (getattr(completion.usage, "prompt_tokens", 0) or 0) + (getattr(completion.usage, "completion_tokens", 0) or 0)

        clusters: List[List[str]] = []
        for s in samples:
            placed = False
            for cluster in clusters:
                rep = cluster[0]
                results = self.nli.check_batch_entailment([(s, rep), (rep, s)])
                if len(results) == 2 and results[0] and results[1]:
                    cluster.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])

        total = len(samples)
        probs = [len(c) / total for c in clusters]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(total)
        entropy_val = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        return entropy_val, tokens_used
