from llama_index.core import PromptTemplate

QA_PROMPT_TMPL = """
You are a legal assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, state: "This clause does not appear to be present."

CONTEXT:
{context_str}

QUESTION: {query_str}

INSTRUCTIONS:
1. Provide a direct answer in plain English.
2. Quote relevant language exactly.
3. Cite the section number if available.
4. Add Disclaimer: "This is not legal advice."
"""

def get_query_engine(index):
    return index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=PromptTemplate(QA_PROMPT_TMPL),
        # Ensure we only return high-confidence matches
        vector_store_query_mode="default",
        alpha=0.7 # Hybrid search weight if using BM25
    )
