# Copyright (c) Microsoft. All rights reserved.

"""Prompts for ChartQA agent workflow."""

from langchain_core.prompts import ChatPromptTemplate

EXTRACT_DATA_PROMPT = ChatPromptTemplate(
    [
        (
            "user",
            """Analyze this chart to answer the question. Extract all relevant data.

Question: {question}

Instructions:
1. Identify what data is needed to answer the question
2. Read the values carefully from the chart
3. Output the extracted data as JSON

Output format:
<data>
{{
  "chart_type": "bar/line/pie/...",
  "relevant_values": {{
    "label1": value1,
    "label2": value2,
    ...
  }},
  "notes": "any observations about the data"
}}
</data>

Rules:
- Extract ONLY values needed to answer the question
- Use exact labels/categories from the chart
- For numbers, give your best estimate if not labeled
- Include units in notes if relevant (but not in values)""",
        ),
    ]
)


COMPUTE_ANSWER_PROMPT = ChatPromptTemplate(
    [
        (
            "user",
            """Given the extracted chart data, answer the question.

Question: {question}

Extracted data:
{extracted_data}

Instructions:
1. Determine what calculation is needed (if any)
2. Perform the calculation
3. Give the final answer

Output format:
<think>
[Your reasoning - what operation is needed?]
</think>

<answer>[final answer - number or short phrase, no units]</answer>

Rules:
- Answer must be concise: a number OR max 3 words
- No units, %, or currency symbols in the answer
- If the question asks "which/what", return the label/name
- If the question asks "how many/much", return a number""",
        ),
    ]
)
