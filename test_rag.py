from app.rag_engine import retriever

retriever.load()
result = retriever.retrieve("customer reporting unauthorised transaction fraud")

print(f"Retrieved {len(result['rag_context_chunks'])} chunks\n")
for ref in result["policy_references"]:
    print(f"  [{ref['score']}] {ref['source']} (p{ref['page']}) [{ref['doc_type']}]")

print("\nTop chunk preview:")
print(result["rag_context_chunks"][0][:300])
