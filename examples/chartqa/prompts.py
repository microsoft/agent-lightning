"""Prompts for ChartQA agent workflow."""

from langchain_core.prompts import ChatPromptTemplate

ANALYZE_CHART_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a visual reasoning expert analyzing charts and graphs.
Given a chart image and a question, first carefully observe and describe the chart.

Instructions:
- Identify the chart type (bar chart, line chart, pie chart, scatter plot, etc.)
- Note the axes labels and units (if applicable)
- Describe the data series or categories shown
- Observe key patterns, trends, or noteworthy values
- Pay attention to legends, titles, and annotations

## Output Format ##

Provide your observation inside <observe> and </observe> tags.

Example:
<observe>
Bar chart showing GDP of 5 countries. X-axis shows country names, Y-axis shows GDP in trillions of USD.
Data values: USA appears highest at around 25, China second at around 20, followed by India, UK, and France.
</observe>
""".strip(),
        ),
        ("user", "Question: {question}"),
    ]
)

EXTRACT_DATA_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
Based on your observation of the chart, extract the specific data values needed to answer the question.

Instructions:
- Extract only the data relevant to the question
- Be precise with values (read carefully from the chart)
- Include labels/categories with each value
- Use appropriate units

## Output Format ##

Provide extracted data inside <extract> and </extract> tags.
Format: Label1: Value1, Label2: Value2, ...

Example:
<extract>
USA: 25, China: 20, India: 15, UK: 10, France: 8
</extract>
""".strip(),
        ),
        (
            "user",
            """Observation: {observation}

Question: {question}

Please extract the relevant data values.""",
        ),
    ]
)

CALCULATE_ANSWER_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
Using the extracted data, perform any necessary calculations to answer the question.

Instructions:
- Show your calculation steps clearly
- Use correct mathematical operations
- Pay attention to the question (average, sum, difference, maximum, etc.)
- Provide a precise numerical answer if applicable
- Keep the answer concise (typically 1-10 words)

## Output Format ##

Show calculation inside <calculate> and </calculate> tags (if needed).
Provide final answer inside <answer> and </answer> tags.

Example:
<calculate>
Average = (25 + 20 + 15 + 10 + 8) / 5 = 78 / 5 = 15.6
</calculate>
<answer>
15.6
</answer>
""".strip(),
        ),
        (
            "user",
            """Extracted Data: {extracted_data}

Question: {question}

Please calculate and provide the answer.""",
        ),
    ]
)

CHECK_ANSWER_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a chart analysis expert with strong attention to detail.
Review the answer for potential mistakes.

Common mistakes to check:
- Incorrect data extraction from chart (misread values)
- Arithmetic errors in calculations
- Misunderstanding the question type (average vs. sum vs. difference)
- Wrong number of data points counted
- Incorrect units or scale interpretation
- Off-by-one errors

## Chart Information ##

Observation: {observation}
Extracted Data: {extracted_data}

## Output Format ##

If any mistakes are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps:
- If mistakes are found: `THE ANSWER IS INCORRECT.`
- If no mistakes are found: `THE ANSWER IS CORRECT.`

DO NOT write the corrected answer in this response. You only need to report mistakes.
""".strip(),
        ),
        (
            "user",
            """Question: {question}

Current Answer: {answer}

Calculation shown:
{calculation}

Please review this answer for correctness.""",
        ),
    ]
)

REFINE_ANSWER_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a chart analysis agent.
The previous answer had errors. Based on the feedback, provide a corrected answer.

Instructions:
- Re-examine the chart observation carefully
- Correct any data extraction errors by re-extracting if needed
- Fix calculation mistakes
- Address all points mentioned in the feedback

## Chart Observation ##

{observation}

## Output Format ##

If you need to re-extract data, provide it inside <extract> and </extract> tags.
Show corrected calculation inside <calculate> and </calculate> tags.
Provide corrected answer inside <answer> and </answer> tags.
""".strip(),
        ),
        (
            "user",
            """Question: {question}

## Previous Attempt ##

Extracted Data: {extracted_data}
Calculation: {calculation}
Answer: {answer}

## Feedback ##

{feedback}

Please provide the corrected answer.""",
        ),
    ]
)
