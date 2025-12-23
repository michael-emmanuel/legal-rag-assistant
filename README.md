# Legal Document Q&A Assistant (RAG + LoRA)

A high-accuracy legal assistant using Retrieval-Augmented Generation to answer questions from private legal contracts.

## Tech Stack

- **Framework**: LangChain
- **Vector DB**: FAISS (High-performance similarity search)
- **Fine-Tuning**: LoRA (Low-Rank Adaptation) via Hugging Face PEFT
- **LLM**: GPT-4o / Local LoRA-tuned Llama-3
- **Evaluation**: RAGAS Metrics (Faithfulness, Relevance)

## Getting Started

1. **Setup**: `pip install -r requirements.txt`
2. **Ingest Docs**: Place PDFs in `data/` and run `python src/ingest.py`
3. **Run App**: `streamlit run src/app.py`

## LLM Learnings

- **LoRA Fine-tuning**: Demonstrated in `notebooks/lora_finetune.ipynb`. We use PEFT to tune a base model on legal terminology without updating all billions of parameters.
- **RAG Architecture**: Combines a retrieval system (FAISS) with a generative model to ensure factual accuracy based on provided documents.
- **Evaluation**: We utilize RAGAS to score the model's reliability, ensuring zero hallucinations in legal contexts.
