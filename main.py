"""
Entry point. Replace SOURCE with your document path.
"""
from pathlib import Path
from base_rag import RAGEngine
from security.pipeline import SecureRAGPipeline

SOURCE = str(
    Path(__file__).parent
    / "confluence"
    / "dsid_0a2cd37d53ff47d4aced289cd9a76fe8__evidence-driven-offer-evaluation-and-onboarding-trigger-playbook-2028.txt"
)


def main():
    engine = RAGEngine(SOURCE)
    existing_cnt = len(engine.vector_store.get()["ids"])
    if existing_cnt == 0:
        print(f"Collection '{engine.uuid}' is empty. Indexing document...")
        n_chunks = engine.index()
        print(f"{n_chunks} chunks indexed into collection '{engine.uuid}'.")
    else:
        print(f"Collection '{engine.uuid}' already contains {existing_cnt} chunks. Skipping indexing.")

    pipeline = SecureRAGPipeline(engine)

    while True:
        question = input("\nQuestion (or 'quit'): ").strip()
        if question.lower() == "quit":
            break

        result = pipeline.ask(question)

        print(f"\n--- Audit trail ---")
        for entry in result.audit_trail:
            status = "PASS" if entry.passed else "BLOCK"
            print(f"  [{status}] {entry.layer}: {entry.detail} ({entry.latency_ms:.1f}ms)")

        if result.quarantined_chunk_ids:
            print(f"  Quarantined chunks: {result.quarantined_chunk_ids}")

        print(f"\nAnswer: {result.answer}")


if __name__ == "__main__":
    main()
