import math 
from typing import Dict, Any, List
from sentence_transformers import CrossEncoder

class GroundingChecker:
    LABELS = ["contradiction", "entailment", "neutral"]
    def __init__(self, mdl:str = "cross-encoder/nli-deberta-v3-base"):
        self.mdl = CrossEncoder(mdl)
    def checkClaim(self, output:str, chunk:str)->Dict[str,Any]:
        scores = self.mdl.predict([(chunk,output)])[0]
        exp_scores = [math.exp(s) for s in scores]
        sum_e_scores = sum(exp_scores)
        probs = [s/sum_e_scores for s in exp_scores]

        label_idx = int(scores.argnmax()) 
        return {
            "label": self.LABELS[label_idx],
            "entailment_score": float(probs[self.LABELS.index("entailment")]),
            "contradiction_score": float(probs[self.LABELS.index("contradiction")]),
            "neutral_score": float(probs[self.LABELS.index("neutral")]),
        }

    def check_batch_entailment(self, pairs: List[tuple]) -> List[bool]:
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        labels = [self.LABELS[int(s.argmax())] for s in scores]
        return [label == "entailment" for label in labels]