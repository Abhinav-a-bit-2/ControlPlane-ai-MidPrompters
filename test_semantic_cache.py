from langchain_redis.cache import RedisSemanticCache
from langchain_core.outputs import Generation
from base_rag import LocalBGEEmbeddings

embedding = LocalBGEEmbeddings()
cache = RedisSemanticCache(redis_url="redis://localhost:6379", embeddings=embedding, distance_threshold=0.15)

query = "What is the refund policy?"
answer = "The refund policy is 30 days."

print("Updating cache...")
cache.update(prompt=query, llm_string="", return_val=[Generation(text=answer)])

print("Looking up exact query...")
res = cache.lookup(prompt=query, llm_string="")
print(f"Exact match result: {res[0].text if res else None}")

print("Looking up semantic query...")
res = cache.lookup(prompt="How long do I have to get a refund?", llm_string="")
print(f"Semantic match result: {res[0].text if res else None}")

print("Looking up unrelated query...")
res = cache.lookup(prompt="What is the weather today?", llm_string="")
print(f"Unrelated match result: {res[0].text if res else None}")
