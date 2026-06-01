"""
Prompt creation for long context task (vanilla LLM mode).
Handles documents by embedding them in the prompt.
"""
import json
from typing import Any, Dict


def create_prompt(problem_data: Dict[str, Any]) -> str:
    """Create prompt for long context problems with documents."""
    problem_statement = problem_data['problem_statement']
    documents = problem_data.get('documents') or problem_data.get('input_files') or {}
    
    if not documents:
        # No documents, just return the problem statement
        return problem_statement
    
    # Handle JSON string (e.g., from HuggingFace datasets)
    if isinstance(documents, str):
        try:
            documents = json.loads(documents)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse documents as JSON: {e}")
    
    # Get document contents as a list
    if isinstance(documents, dict):
        docs = list(documents.values())
    else:
        docs = documents
    
    # Format documents with markers
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" 
        for i, doc in enumerate(docs)
    )
    
    return f"""BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{problem_statement}

END QUESTION
"""
