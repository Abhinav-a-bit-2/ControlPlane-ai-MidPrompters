# Imports
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from pathlib import Path
from typing import Any
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from groq import Groq

key = os.environ.get("GROQ_API_KEY")
# Embedding model used
class LocalBGEEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = "Represent this sentence for searching relevant passages: "
        self.document_prefix = ""
    
    def embed_documents(self, texts):
        prefixed = [self.document_prefix + t for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()

#source file path and uuid extraction from path
SOURCE = "C:\\Users\\Kunal\\ControlPlane-ai-MidPrompters\\confluence\\dsid_0a2cd37d53ff47d4aced289cd9a76fe8__evidence-driven-offer-evaluation-and-onboarding-trigger-playbook-2028.txt"
uuid = os.path.basename(SOURCE).split('.')[0]
with open(SOURCE,"r",encoding="utf-8") as f:
    text = f.read()


documents = [ Document(page_content = text )]
# defining doc metadeta
def create_metadata(file:str, uuid:str , **extra_metadata:Any)->dict[str,Any]:
    path = Path(file)
    metadata = {
        "uuid": uuid,
        "filename": path.name,
        "source_type": path.parent.name
    }
    metadata.update(extra_metadata)
    return metadata

documents[0].metadata.update(create_metadata(SOURCE,uuid,**documents[0].metadata))

# splitting and indexing chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 450,
    chunk_overlap = 48
)
chunks = splitter.split_documents(documents)
print(f"{len(chunks)} chunks created from {len(documents[0].page_content)}")
for i,c in enumerate(chunks,start=1):
    c.metadata["chunk_id"] = f"chunk-{i}"

embeddings = LocalBGEEmbeddings(device = "cuda" if torch.cuda.is_available() else "cpu")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=uuid    
)
print(f"indexing done in {uuid}")

def retrieve(query:str, k:int = 3):
    return vector_store.similarity_search_with_score(query,k=k)


def build_context(hits)->str:
    blocks = []
    for doc, dist in hits:
        chunk_id = doc.metadata["chunk_id"]
        blocks.append(f"{chunk_id}\n"
                      f"{doc.page_content}")
    return "\n\n".join(blocks)



def final_answer(Question:str,k:int = 3):
    hits = retrieve(Question,k)

    context = build_context(hits)

    SYSTEM_PROMPT = """
    You answer support questions using only the provided context.

    Rules:

    1. Use only information present in the context.
    2. Do not invent facts.
    3. Cite every factual claim using the chunk ID.
    4. Use citations such as [chunk-2].
    5. If the context does not contain the answer, say:

    I do not know based on the provided context.

    Keep the answer concise and directly answer the question.
    """


        
    msg = [
        {
            "role":"system",
            "content":str({SYSTEM_PROMPT})
        },
        {
                "role":"user",
                "content":
    f"""
    Context : {context}
    Question: {Question}
    """
        }
    ]
    client = Groq(api_key=key)
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=msg,
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )
    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""
    return {
        "answer": result,
        "document_ids": uuid, 
    }

