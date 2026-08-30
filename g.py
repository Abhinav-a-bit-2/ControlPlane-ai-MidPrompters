from openai import OpenAI
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key="")
models = client.models.list()
for m in models.data:
    print(m.id)
