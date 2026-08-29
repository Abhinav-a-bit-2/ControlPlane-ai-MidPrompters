import math
from typing import Dict, Any, List
from sentence_transformers import CrossEncoder

class GroundingChecker:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GroundingChecker, cls).__new__(cls)
        return cls._instance

    def __init__(self, mdl: str = "cross-encoder/nli-deberta-v3-base"):
        if self._initialized:
            return
        self.mdl = CrossEncoder(mdl)
        id2label = getattr(self.mdl.model.config, "id2label", None)
        if id2label:
            self.LABELS = [id2label[i].lower() for i in sorted(id2label.keys())]
        else:
            self.LABELS = ["contradiction", "neutral", "entailment"]
        self._initialized = True

    def checkClaim(self, output: str, chunk: str) -> Dict[str, Any]:
        scores = self.mdl.predict([(chunk, output)])[0]
        
        # Softmax over logits
        exp_scores = [math.exp(s) for s in scores]
        sum_e = sum(exp_scores)
        probs = [s / sum_e for s in exp_scores]
        p_e = float(probs[self.LABELS.index("entailment")])
        p_c = float(probs[self.LABELS.index("contradiction")])
        p_n = float(probs[self.LABELS.index("neutral")])

        entropy = -sum(p*math.log(p,3) for p in (p_c,p_n,p_e) if p > 0.0)

        label_idx = int(scores.argmax())
        print(label_idx)
        return {
            "label": self.LABELS[label_idx],
            "entailment_score": p_e,
            "contradiction_score": p_c,
            "neutral_score": p_n,
            "entropy": float(entropy)
        }

    def check_batch_entailment(self, pairs: List[tuple]) -> List[bool]:
        if not pairs:
            return []
        scores = self.mdl.predict(pairs)
        labels = [self.LABELS[int(s.argmax())] for s in scores]
        return [label == "entailment" for label in labels]