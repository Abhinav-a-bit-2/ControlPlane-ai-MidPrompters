import math
from typing import Dict, Any, List
from sentence_transformers import CrossEncoder

class GroundingChecker:
    def __init__(self, mdl: str = "cross-encoder/nli-deberta-v3-base"):
        self.mdl = CrossEncoder(mdl)
        # Read label mapping directly from the HF model config
        id2label = getattr(self.mdl.model.config, "id2label", None)
        if id2label:
            self.LABELS = [id2label[i].lower() for i in sorted(id2label.keys())]
            print(self.LABELS)
        else:
            # Standard NLI DeBERTa default: 0=contradiction, 1=neutral, 2=entailment
            self.LABELS = ["contradiction", "neutral", "entailment"]

    def checkClaim(self, output: str, chunk: str) -> Dict[str, Any]:
        scores = self.mdl.predict([(chunk, output)])[0]
        
        # Softmax over logits
        exp_scores = [math.exp(s) for s in scores]
        sum_e = sum(exp_scores)
        probs = [s / sum_e for s in exp_scores]

        label_idx = int(scores.argmax())
        print(label_idx)
        return {
            "label": self.LABELS[label_idx],
            "entailment_score": float(probs[self.LABELS.index("entailment")]),
            "contradiction_score": float(probs[self.LABELS.index("contradiction")]),
            "neutral_score": float(probs[self.LABELS.index("neutral")]),
        }

    def check_batch_entailment(self, pairs: List[tuple]) -> List[bool]:
        if not pairs:
            return []
        scores = self.mdl.predict(pairs)
        labels = [self.LABELS[int(s.argmax())] for s in scores]
        return [label == "entailment" for label in labels]
