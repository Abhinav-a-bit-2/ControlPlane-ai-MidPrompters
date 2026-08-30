
import sys
sys.path.append('d:/ControlPlane')
from tests.rag_client import RAGClient

client = RAGClient.get_instance()
q1 = 'In the beginning, what did God create?'
q2 = 'What was upon the face of the deep before God said Be light made?'

for q in [q1, q2]:
    print(f'\nQUERY: {q}')
    res = client.query(q)
    print(f'ANSWER: {repr(res.answer)}')
    print(f'BLOCKED: {res.blocked} (Layer: {res.blocked_at_layer})')

