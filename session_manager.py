"""
Redis data structure manager to manage short-term memory
"""

import os
import json
import redis
import tiktoken
from dotenv import load_dotenv

from langchain_redis.cache import RedisSemanticCache
from langchain_core.outputs import Generation
from base_rag import LocalBGEEmbeddings

class SessionManager:
    def __init__(self, host: str = None, port: int = None, ttl: int = 86400, encoding: str = "cl100k_base"):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", 6379))
        self.ttl = ttl
        self.client = redis.Redis(
            host = self.host,
            port = self.port,
            decode_responses = True,
        )
        self.tokenizer = tiktoken.get_encoding(encoding)
        self.semantic_cache = RedisSemanticCache(
            embeddings=LocalBGEEmbeddings(),
            redis_url=f"redis://{self.host}:{self.port}",
            distance_threshold=0.15,
            ttl=3600
        )


                
    def addChats(self, session_id:str, role:str, content:str)->None:
        key = f"session:{session_id}:history"
        payload = json.dumps({"role":role,"content":content})
        self.client.rpush(key,payload)
        self.client.expire(key,self.ttl)
        
    def recentKChats(self, session_id: str, k:int = 4)->list[dict]:
        key = f"session:{session_id}:history"
        raw_chats = self.client.lrange(key,-k,-1)
        return [json.loads(chat) for chat in raw_chats]
    
    def tokenBudgetedChats(self, session_id:str, token_budget: int = 4096)->list[dict]:
        key = f"session:{session_id}:history"
        raw_chats = self.client.lrange(key,0,-1)
        chats = [json.loads(chat) for chat in raw_chats]
        budgeted_chats = []
        used = 0
        for chat in reversed(chats):
            chat_tokens = len(self.tokenizer.encode(chat["content"]))
            if used + chat_tokens > token_budget:
                break
            budgeted_chats.insert(0,chat)
            used += chat_tokens
        return budgeted_chats
    
    def deleteChat(self, session_id:str)->bool:
        key = f"session:{session_id}:history"
        return bool(self.client.delete(key))

    # --- Caching Methods ---
    def get_cache(self, query: str) -> str:
        """Semantic match cache lookup."""
        result = self.semantic_cache.lookup(prompt=query, llm_string="")
        if result:
            return result[0].text
        return None

    def set_cache(self, query: str, answer: str, ttl: int = 3600) -> None:
        """Store semantic match cache."""
        self.semantic_cache.update(prompt=query, llm_string="", return_val=[Generation(text=answer)])
        

    # --- HITL Queue Methods ---
    def enqueue_hitl(self, session_id: str, query: str, trace_id: str) -> str:
        """Pushes a query to the human-in-the-loop queue."""
        ticket_id = f"ticket:{os.urandom(4).hex()}"
        payload = json.dumps({
            "ticket_id": ticket_id,
            "session_id": session_id,
            "query": query,
            "trace_id": trace_id,
            "status": "pending"
        })
        self.client.hset("hitl:queue", ticket_id, payload)
        return ticket_id
        
    def get_hitl_ticket(self, ticket_id: str) -> dict:
        data = self.client.hget("hitl:queue", ticket_id)
        return json.loads(data) if data else None
        
    def resolve_hitl_ticket(self, ticket_id: str) -> None:
        self.client.hdel("hitl:queue", ticket_id)

