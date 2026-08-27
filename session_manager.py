"""
Redis data structure manager to manage short-term memory
"""

import os
import json
import redis
import tiktoken
from dotenv import load_dotenv

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

