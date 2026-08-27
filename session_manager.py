"""
Redis data structure manager to manage short-term memory
"""

import os
import json
import redis
import tiktoken
from dotenv import load_dotenv


class SessionManager:
    def __init__(self):
        pass
    def addChats(self, session_id:str, role:str, content:str)->None:
        pass
    def recentKChats(self, session_id: str, k:int = 4)->list[dict]:
        pass
    def tokenBudgetedChats(self, session_id:str, token_budget: int )->list[dict]:
        pass

