# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Visualization utilities for GenAI Evaluation SDK."""

import json
import logging
from typing import Any, Optional

import pandas as pd
from pydantic import errors

from . import types


logger = logging.getLogger(__name__)


def _is_ipython_env() -> bool:
    """Checks if the code is running in an IPython environment."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _pydantic_serializer(obj: Any) -> Any:
    """Custom serializer for Pydantic models."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _preprocess_df_for_json(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Prepares a DataFrame for JSON serialization by converting complex objects to strings."""
    if df is None:
        return None
    df_copy = df.copy()

    for col in df_copy.columns:
        if (
            df_copy[col].dtype == "object"
            or df_copy[col].apply(lambda x: isinstance(x, (dict, list))).any()
        ):

            def stringify_cell(cell: Any) -> Optional[str]:
                if isinstance(cell, (dict, list)):
                    try:
                        return json.dumps(
                            cell, ensure_ascii=False, default=_pydantic_serializer
                        )
                    except TypeError:
                        return str(cell)
                elif pd.isna(cell):
                    return None
                elif not isinstance(cell, (str, int, float, bool)):
                    if hasattr(cell, "model_dump"):
                        return json.dumps(
                            cell.model_dump(mode="json"), ensure_ascii=False
                        )
                    return str(cell)
                return str(cell)

            df_copy[col] = df_copy[col].apply(stringify_cell)
    return df_copy


def _get_evaluation_html(eval_result_json: str) -> str:
    """Returns a self-contained HTML for single evaluation visualization."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Evaluation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
    <style>
        body {{ font-family: 'Roboto', 'Helvetica', sans-serif; margin: 2em; background-color: #f8f9fa; color: #202124; }}
        .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
        h1, h2, h3 {{ color: #3c4043; }}
        h1 {{ border-bottom: 2px solid #4285F4; padding-bottom: 8px; }}
        h2 {{ border-bottom: 1px solid #dadce0; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #dadce0; padding: 12px; text-align: left; vertical-align: top; }}
        th {{ background-color: #f2f2f2; font-weight: 500; }}
        details {{ border: 1px solid #dadce0; border-radius: 8px; padding: 16px; margin-bottom: 16px; background: #fff; }}
        summary {{ font-weight: 500; font-size: 1.1em; cursor: pointer; }}
        .prompt-container {{ background-color: #e8f0fe; padding: 16px; margin: 12px 0; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; }}
        .reference-container {{ background-color: #e6f4ea; padding: 16px; margin: 12px 0; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; }}
        .agent-info-container {{
            background-color: #f1f3f4;
            padding: 16px;
            margin: 12px 0;
            border-radius: 8px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            font-size: 14px;
         }}
        .agent-info-grid {{
            display: grid;
            grid-template-columns: 120px 1fr;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .agent-info-grid dt {{
            font-weight: 500;
            color: #3c4043;
        }}
        .agent-info-grid dd {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .intermediate-events-container {{ background-color: #f1f3f4; padding: 16px; margin: 12px 0; border-radius: 8px; word-wrap: break-word; overflow-wrap: break-word; max-height: 400px; overflow-y: auto; overflow-x: auto; }}
        .response-container {{ background-color: #f9f9f9; padding: 12px; margin-top: 8px; border-radius: 8px; border: 1px solid #eee; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; }}
        .explanation {{ color: #5f6368; font-style: italic; font-size: 0.9em; padding-top: 6px; }}
        .raw-json-details summary {{ font-size: 0.9em; cursor: pointer; color: #5f6368;}}
        .raw-json-container {{ white-space: pre-wrap; word-wrap: break-word; max-height: 300px; overflow-y: auto; background-color: #f1f1f1; padding: 10px; border-radius: 4px; margin-top: 8px; }}
        .rubric-bubble-container {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .rubric-details {{ border: none; padding: 0; margin: 0; }}
        .rubric-bubble {{
            display: inline-flex;
            align-items: center;
            background-color: #e8f0fe;
            color: #1967d2;
            border-radius: 16px;
            padding: 8px 12px;
            font-size: 0.9em;
            cursor: pointer;
            list-style: none; /* Hide default marker in Safari */
        }}
        .rubric-bubble::-webkit-details-marker {{ display: none; }} /* Hide default marker in Chrome */
        .rubric-bubble::before {{
            content: '►';
            margin-right: 8px;
            font-size: 0.8em;
            transition: transform 0.2s;
        }}
        .rubric-details[open] > .rubric-bubble::before {{
            transform: rotate(90deg);
        }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .case-content-wrapper {{ display: flex; gap: 1rem; }}
        .case-content-main {{ flex: 1; }}
        .case-content-sidebar {{ flex: 1; min-width: 0; }}
        .case-content-sidebar .intermediate-events-container {{
            padding: 0;
            background-color: #F8F9FA;
            border: 1px solid #dadce0;
            border-radius: 4px;
            overflow: auto;
            margin: 0;
        }}
        .trace-event-row {{
            display: flex;
            align-items: center;
            padding: 6px 12px;
            border-bottom: 1px solid #eee;
            font-size: 13px;
            background-color: #F8F9FA;
        }}
        .trace-event-row:last-child {{
            border-bottom: none;
        }}
        .trace-event-row .name {{
            flex-grow: 1;
            color: #3c4043;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .trace-event-row .duration {{
            background-color: #d2e3fc;
            color: #1967d2;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            white-space: nowrap;
        }}
        .trace-event-row .name .icon {{
            margin-right: 8px;
            font-size: 16px;
            line-height: 1;
        }}
        .trace-details {{
            padding: 2px 12px 6px 38px; /* indent to align with text after icon */
            font-size: 13px;
            line-height: 1.4;
            color: #5f6368;
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: #F8F9FA;
            border-bottom: 1px solid #eee;
        }}
        .trace-event-row .name.trace-l1 {{
            padding-left: 20px;
        }}
        .trace-details.details-l1 {{
            padding-left: 58px;
        }}
        .trace-details-wrapper details {{
            border:0;
            padding:0;
            margin:0;
        }}
        .trace-details-wrapper summary {{
            list-style: none;
            cursor: pointer;
        }}
        .trace-details-wrapper summary::-webkit-details-marker {{
            display: none;
        }}
        .tool-declarations-container {{
             background-color: #f1f1f1;
             padding: 10px;
             border-radius: 4px;
             margin-top: 8px;
             max-height: 300px;
             overflow-y: auto;
        }}
        .tool-declaration {{
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .tool-declaration:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Report</h1>
        <div id="summary-section"></div>
        <div id="agent-info-section"></div>
        <div id="details-section"></div>
    </div>
    <script>
        const vizData = {eval_result_json};
        function formatDictVals(obj) {{
            if (typeof obj === 'string') return obj;
            if (obj === undefined || obj === null) return '';
            if (typeof obj !== 'object') return String(obj);
            if (Array.isArray(obj)) return JSON.stringify(obj);
            return Object.entries(obj).map(([k,v]) => `${{k}}=${{formatDictVals(v)}}`).join(', ');
        }}
        function formatIntermediateEvents(events) {{
            let eventsArray = events;
            if (typeof events === 'string') {{
                try {{
                    eventsArray = JSON.parse(events);
                }} catch (e) {{
                    console.error("Failed to parse intermediate_events:", e);
                    return '';
                }}
            }}
            if (!eventsArray || !Array.isArray(eventsArray)) {{
                return '';
            }}

            const agentInfo = vizData.agent_info;

            // If we have agent info, render as trace
            if(agentInfo) {{
                let traceHtml = `<div class="trace-event-row"><div class="name"><span class="icon">🤖</span>agent_run</div></div>`;
                eventsArray.forEach(event => {{
                    if (event.content && event.content.parts && event.content.parts.length > 0) {{
                        event.content.parts.forEach(part => {{
                            if (part.function_call) {{
                                traceHtml += `<div class="trace-details-wrapper"><details><summary><div class="trace-event-row"><div class="name trace-l1"><span class="icon">🛠️</span>function_call</div></div></summary>`;
                                traceHtml += `<div class="trace-details details-l1">function name: ${{part.function_call.name}}</div>`;
                                traceHtml += `<div class="trace-details details-l1">function args: ${{formatDictVals(part.function_call.args)}}</div></details></div>`;
                            }} else if (part.text && event.content.role === 'model') {{
                                traceHtml += `<div class="trace-details-wrapper"><details><summary><div class="trace-event-row"><div class="name trace-l1"><span class="icon">💬</span>call_llm</div></div></summary>`;
                                traceHtml += `<div class="trace-details details-l1">model response: ${{part.text}}</div></details></div>`;
                            }} else if (part.function_response) {{
                                traceHtml += `<div class="trace-details-wrapper"><details><summary><div class="trace-event-row"><div class="name trace-l1"><span class="icon">🛠️</span>function_response</div></div></summary>`;
                                traceHtml += `<div class="trace-details details-l1">function name: ${{part.function_response.name}}</div>`;
                                let response_val = part.function_response.response;
                                if(typeof response_val === 'object' && response_val !== null && response_val.result !== undefined) {{
                                    response_val = response_val.result;
                                }}
                                traceHtml += `<div class="trace-details details-l1">function response: ${{formatDictVals(response_val)}}</div></details></div>`;
                            }} else {{
                                // Skipping user messages and other parts in trace view
                                return;
                            }}
                        }});
                    }}
                }});
                return traceHtml;
            }}

            // Fallback to original conversation view if not agent trace
            return eventsArray.map(event => {{
                const role = event.content.role;
                let contentHtml = '';
                if (event.content && event.content.parts && event.content.parts.length > 0) {{
                    event.content.parts.forEach(part => {{
                        if (part.text) {{
                            contentHtml += DOMPurify.sanitize(marked.parse(String(part.text)));
                        }} else if (part.function_call) {{
                            contentHtml += `<pre class="raw-json-container">${{DOMPurify.sanitize(JSON.stringify(part.function_call, null, 2))}}</pre>`;
                        }} else if (part.function_response) {{
                            contentHtml += `<pre class="raw-json-container">${{DOMPurify.sanitize(JSON.stringify(part.function_response, null, 2))}}</pre>`;
                        }} else {{
                            contentHtml += `<pre class="raw-json-container">${{DOMPurify.sanitize(JSON.stringify(part, null, 2))}}</pre>`;
                        }}
                    }});
                }} else {{
                    contentHtml = `<pre class="raw-json-container">${{DOMPurify.sanitize(JSON.stringify(event.content, null, 2))}}</pre>`;
                }}
                return `<div class="trace-event" style="margin-bottom: 1rem;">
                            <div class="trace-role" style="font-weight: 500;">${{role}}</div>
                            <div class="trace-content">${{contentHtml}}</div>
                        </div>`;
            }}).join('');
        }}

        function formatToolDeclarations(toolDeclarations) {{
            if (!toolDeclarations) {{
                return '';
            }}
            let functions = [];
            if (Array.isArray(toolDeclarations)) {{
                toolDeclarations.forEach(tool => {{
                    if (tool.function_declarations) {{
                        functions = functions.concat(tool.function_declarations);
                    }} else if (tool.name && tool.parameters) {{
                        // It might be a list of function declarations directly
                        functions.push(tool);
                    }}
                }});
            }} else if (typeof toolDeclarations === 'object' && toolDeclarations.function_declarations) {{
                functions = toolDeclarations.function_declarations;
            }}

            if (functions.length === 0) {{
                 return `<pre class="raw-json-container">${{DOMPurify.sanitize(JSON.stringify(toolDeclarations, null, 2))}}</pre>`;
            }}

            let html = '<div class="tool-declarations-container">';
            functions.forEach(func => {{
                html += '<div class="tool-declaration">';
                const params = func.parameters && func.parameters.properties ? func.parameters.properties : {{}};
                const requiredParams = func.parameters && func.parameters.required ? new Set(func.parameters.required) : new Set();
                const paramStrings = Object.keys(params).map(p => `${{p}}: ${{params[p].type}}`).join(', ');
                html += `<strong>${{func.name}}</strong>(${{paramStrings}})<br>`;
                if(func.description) html += `<em>${{func.description}}</em><br>`;
                if(Object.keys(params).length > 0) html += 'Parameters:<br>';
                Object.keys(params).forEach(p => {{
                    html += `&nbsp;&nbsp;- ${{p}}: ${{params[p].description || ''}} ${{requiredParams.has(p) ? '<strong>(required)</strong>' : ''}}<br>`;
                }});
                html += '</div>';
            }});
            html += '</div>';
            return html;
        }}

        function renderSummary(summaryMetrics) {{
            const container = document.getElementById('summary-section');
            let content = '<h2>Summary Metrics</h2>';
            if (!summaryMetrics || summaryMetrics.length === 0) {{ container.innerHTML = content + '<p>No summary metrics.</p>'; return; }}
            let table = '<table><thead><tr><th>Metric</th><th>Mean Score</th><th>Std. Dev.</th></tr></thead><tbody>';
            summaryMetrics.forEach(m => {{
                table += `<tr><td>${{m.metric_name || 'N/A'}}</td><td>${{m.mean_score != null ? m.mean_score.toFixed(4) : 'N/A'}}</td><td>${{m.stdev_score != null ? m.stdev_score.toFixed(4) : 'N/A'}}</td></tr>`;
            }});
            container.innerHTML = content + table + '</tbody></table>';
        }}
        function renderAgentInfo(agentInfo) {{
            const container = document.getElementById('agent-info-section');
            if (!agentInfo) {{
                return;
            }}
            let content = '<h2>Agent Info</h2>';
            content += '<div class="agent-info-container">';
            content += '<dl class="agent-info-grid">';
            if(agentInfo.name) content += `<dt>Name:</dt><dd>${{agentInfo.name}}</dd>`;
            if(agentInfo.instruction) content += `<dt>Instruction:</dt><dd>${{agentInfo.instruction}}</dd>`;
            if(agentInfo.description) content += `<dt>Description:</dt><dd>${{agentInfo.description}}</dd>`;
            content += '</dl>';
            if(agentInfo.tool_declarations) {{
                content += `<div style="margin-top: 12px;"><div style="font-weight: 500; color: #3c4043; margin-bottom: 8px;">Tool Declarations</div>`;
                content += formatToolDeclarations(agentInfo.tool_declarations);
                content += '</div>';
            }}
            content += '</div>';
            container.innerHTML = content;
        }}
        function renderDetails(caseResults, metadata, agentInfo) {{
            const container = document.getElementById('details-section');
            container.innerHTML = '<h2>Detailed Results</h2>';
            if (!caseResults || caseResults.length === 0) {{ container.innerHTML += '<p>No detailed results.</p>'; return; }}
            const datasetRows = metadata && metadata.dataset ? metadata.dataset : [];

            caseResults.forEach((caseResult, i) => {{
                const original_case = datasetRows[caseResult.eval_case_index] || {{}};
                const promptText = original_case.prompt_display_text || '(prompt not found)';
                const promptJson = original_case.prompt_raw_json;
                const reference = original_case.reference || '';
                const responseText = original_case.response_display_text || '(response not found)';
                const responseJson = original_case.response_raw_json;
                const intermediateEvents = original_case.intermediate_events;
                const isAgentEval = agentInfo || intermediateEvents;

                let card = `<details><summary>Case #${{caseResult.eval_case_index != null ? caseResult.eval_case_index : i}}</summary>`;

                card += `<div class="case-content-wrapper">`;

                card += `<div class="case-content-main">`;
                card += `<div class="prompt-container"><strong>Prompt:</strong><br>${{DOMPurify.sanitize(marked.parse(String(promptText)))}}</div>`;
                if (promptJson) {{
                    card += `<details class="raw-json-details"><summary>View Raw Prompt JSON</summary><pre class="raw-json-container">${{DOMPurify.sanitize(promptJson)}}</pre></details>`;
                }}

                if (reference) {{ card += `<div class="reference-container"><strong>Reference:</strong><br>${{DOMPurify.sanitize(marked.parse(String(reference)))}}</div>`; }}

                const responseTitle = isAgentEval ? 'Final Response' : 'Candidate Response';
                card += `<div class="response-container"><h4>${{responseTitle}}</h4>${{DOMPurify.sanitize(marked.parse(String(responseText)))}}</div>`;
                if (responseJson) {{
                    card += `<details class="raw-json-details"><summary>View Raw Response JSON</summary><pre class="raw-json-container">${{DOMPurify.sanitize(responseJson)}}</pre></details>`;
                }}
                card += `</div>`; // case-content-main

                if (isAgentEval) {{
                    card += `<div class="case-content-sidebar">
                                <h4>Traces</h4>
                                <div class="intermediate-events-container">${{formatIntermediateEvents(intermediateEvents)}}</div>
                             </div>`;
                }}

                card += `</div>`; // case-content-wrapper

                let metricTable = '<h4>Metrics</h4><table><tbody>';
                const candidateMetrics = (caseResult.response_candidate_results && caseResult.response_candidate_results[0] && caseResult.response_candidate_results[0].metric_results) || {{}};
                Object.entries(candidateMetrics).forEach(([name, val]) => {{
                    let metricNameCell = name;
                    let explanationHandled = false;
                    let bubbles = '';

                    if (name.startsWith('hallucination') && val.explanation) {{
                        try {{
                            const explanationData = typeof val.explanation === 'string' ? JSON.parse(val.explanation) : val.explanation;
                            if (Array.isArray(explanationData) && explanationData.length > 0) {{
                                let sentenceGroups = [];
                                if (explanationData[0].explanation && Array.isArray(explanationData[0].explanation)) {{
                                    explanationData.forEach(item => {{
                                        if(item.explanation && Array.isArray(item.explanation)) {{
                                            sentenceGroups.push(item.explanation);
                                        }}
                                    }});
                                }} else if (explanationData[0].sentence) {{
                                    sentenceGroups.push(explanationData);
                                }}

                                if(sentenceGroups.length > 0) {{
                                    sentenceGroups.forEach(sentenceList => {{
                                        bubbles += '<div class="rubric-bubble-container" style="margin-top: 8px;">';
                                        sentenceList.forEach(item => {{
                                            let sentence = item.sentence || 'N/A';
                                            const label = item.label ? item.label.toLowerCase() : '';
                                            const isPass = label === 'no_rad' || label === 'supported';
                                            const verdictText = isPass ? '<span class="pass">Pass</span>' : '<span class="fail">Fail</span>';
                                            if (isPass) {{
                                                sentence = `"${{sentence}}" is grounded`;
                                            }}
                                            const rationale = item.rationale || 'N/A';
                                            const itemJson = JSON.stringify(item, null, 2);
                                            bubbles += `
                                                <details class="rubric-details">
                                                    <summary class="rubric-bubble">${{verdictText}}: ${{DOMPurify.sanitize(sentence)}}</summary>
                                                    <div class="explanation" style="padding: 10px 0 0 20px;">${{DOMPurify.sanitize(rationale)}}</div>
                                                    <pre class="raw-json-container">${{DOMPurify.sanitize(itemJson)}}</pre>
                                                </details>`;
                                        }});
                                        bubbles += '</div>';
                                    }});
                                    explanationHandled = true;
                                }}
                            }}
                        }} catch (e) {{
                            console.error("Failed to parse hallucination explanation:", e);
                        }}
                    }} else if (name.startsWith('safety') && val.score != null) {{
                        try {{
                            bubbles += '<div class="rubric-bubble-container" style="margin-top: 8px;">';
                            const verdictText = val.score >= 1.0 ? '<span class="pass">Pass</span>' : '<span class="fail">Fail</span>';
                            const explanation = val.explanation || (val.score >= 1.0 ? 'Safety check passed' : 'Safety check failed');
                            const itemJson = JSON.stringify(val, null, 2);
                            bubbles += `
                                <details class="rubric-details">
                                    <summary class="rubric-bubble">${{verdictText}}: ${{DOMPurify.sanitize(explanation)}}</summary>
                                    <pre class="raw-json-container">${{DOMPurify.sanitize(itemJson)}}</pre>
                                </details>`;
                            bubbles += '</div>';
                            explanationHandled = true;
                        }} catch (e) {{
                            console.error("Failed to process safety metric:", e);
                        }}
                    }}

                    if (!bubbles && val.rubric_verdicts && val.rubric_verdicts.length > 0) {{
                        bubbles += '<div class="rubric-bubble-container" style="margin-top: 8px;">';
                        val.rubric_verdicts.forEach(verdict => {{
                            const rubricDescription = verdict.evaluated_rubric && verdict.evaluated_rubric.content && verdict.evaluated_rubric.content.property ? verdict.evaluated_rubric.content.property.description : 'N/A';
                            const verdictText = verdict.verdict ? '<span class="pass">Pass</span>' : '<span class="fail">Fail</span>';
                            const verdictJson = JSON.stringify(verdict, null, 2);
                            bubbles += `
                                <details class="rubric-details">
                                    <summary class="rubric-bubble">${{verdictText}}: ${{DOMPurify.sanitize(rubricDescription)}}</summary>
                                    <pre class="raw-json-container">${{DOMPurify.sanitize(verdictJson)}}</pre>
                                </details>`;
                        }});
                        bubbles += '</div>';
                    }}

                    if(bubbles) {{
                        metricNameCell += bubbles;
                    }}

                    metricTable += `<tr><td>${{metricNameCell}}</td><td><b>${{val.score != null ? val.score.toFixed(2) : 'N/A'}}</b></td></tr>`;
                    if (val.explanation && !explanationHandled) {{
                        metricTable += `<tr><td colspan="2"><div class="explanation">${{DOMPurify.sanitize(marked.parse(String(val.explanation)))}}</div></td></tr>`;
                    }}
                }});
                card += metricTable + '</tbody></table>';
                container.innerHTML += card + '</details>';
            }});
        }}
        renderSummary(vizData.summary_metrics);
        renderAgentInfo(vizData.agent_info);
        renderDetails(vizData.eval_case_results, vizData.metadata, vizData.agent_info);
    </script>
</body>
</html>
"""


def _get_comparison_html(eval_result_json: str) -> str:
    """Returns a self-contained HTML for a side-by-side eval comparison."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Eval Comparison Report</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
    <style>
        body {{ font-family: 'Roboto', 'Helvetica', sans-serif; margin: 2em; background-color: #f8f9fa; color: #202124; }}
        .container {{ max-width: 95%; margin: 20px auto; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
        h1, h2, h3, h4 {{ color: #3c4043; }}
        h1 {{ border-bottom: 2px solid #4285F4; padding-bottom: 8px; }}
        h2 {{ border-bottom: 1px solid #dadce0; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #dadce0; padding: 12px; text-align: left; vertical-align: top; }}
        th {{ background-color: #f2f2f2; font-weight: 500; }}
        details {{ border: 1px solid #dadce0; border-radius: 8px; padding: 24px; margin-bottom: 24px; background: #fff; }}
        summary {{ font-weight: 500; font-size: 1.2em; cursor: pointer; }}
        .prompt-container {{ background-color: #e8f0fe; padding: 16px; margin-bottom: 16px; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; }}
        .responses-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-top: 16px;}}
        .response-column {{ border: 1px solid #e0e0e0; padding: 16px; border-radius: 8px; background: #f9f9f9; }}
        .response-text-container {{ background-color: #fff; padding: 12px; margin-top: 8px; border-radius: 4px; border: 1px solid #eee; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; overflow-wrap: break-word; }}
        .explanation {{ color: #5f6368; font-style: italic; font-size: 0.9em; padding-top: 8px; }}
        .raw-json-details summary {{ font-size: 0.9em; cursor: pointer; color: #5f6368;}}
        .raw-json-container {{ white-space: pre-wrap; word-wrap: break-word; max-height: 300px; overflow-y: auto; background-color: #f1f1f1; padding: 10px; border-radius: 4px; margin-top: 8px; }}
        .rubric-bubble-container {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .rubric-details {{ border: none; padding: 0; margin: 0; }}
        .rubric-bubble {{
            display: inline-flex;
            align-items: center;
            background-color: #e8f0fe;
            color: #1967d2;
            border-radius: 16px;
            padding: 8px 12px;
            font-size: 0.9em;
            cursor: pointer;
            list-style: none; /* Hide default marker in Safari */
        }}
        .rubric-bubble::-webkit-details-marker {{ display: none; }} /* Hide default marker in Chrome */
        .rubric-bubble::before {{
            content: '►';
            margin-right: 8px;
            font-size: 0.8em;
            transition: transform 0.2s;
        }}
        .rubric-details[open] > .rubric-bubble::before {{
            transform: rotate(90deg);
        }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Eval Comparison Report</h1>
        <div id="summary-section"></div>
        <div id="details-section"></div>
    </div>
    <script>
        const vizData = {eval_result_json};
        function renderSummary(summaryMetrics, metadata) {{
            const container = document.getElementById('summary-section');
            if (!summaryMetrics || summaryMetrics.length === 0) {{ container.innerHTML = '<h2>Summary Metrics</h2><p>No summary metrics.</p>'; return; }}
            const candidateNames = (metadata.candidate_names && metadata.candidate_names.length) ? metadata.candidate_names : null;
            let table = '<h2>Summary Metrics</h2><table><thead><tr><th>Metric</th><th>Mean Score</th><th>Std Dev</th><th>Win/Tie Rates</th></tr></thead><tbody>';
            summaryMetrics.forEach(m => {{
                let winRateText = 'N/A';
                if (m.win_rates) {{
                    winRateText = m.win_rates.map((rate, i) => `<b>${{candidateNames ? candidateNames[i] : `Candidate #${{i+1}}`}}</b> wins: <b>${{(rate * 100).toFixed(1)}}%</b>`).join('<br>');
                    if (m.tie_rate !== undefined) {{ winRateText += `<br>Ties: <b>${{(m.tie_rate * 100).toFixed(1)}}%</b>`; }}
                }}
                table += `<tr><td>${{m.metric_name}}</td><td>${{m.mean_score.toFixed(4)}}</td><td>${{m.stdev_score.toFixed(4)}}</td><td>${{winRateText}}</td></tr>`;
            }});
            container.innerHTML = table + '</tbody></table>';
        }}
        function renderDetails(caseResults, metadata) {{
            const container = document.getElementById('details-section');
            container.innerHTML = '<h2>Detailed Comparison</h2>';
            if (!caseResults || caseResults.length === 0) {{ container.innerHTML += '<p>No detailed results.</p>'; return; }}
            const datasetRows = metadata.dataset || [];
            const candidateNames = (metadata.candidate_names && metadata.candidate_names.length) ? metadata.candidate_names : null;

            caseResults.forEach((caseResult, i) => {{
                const original_case = datasetRows[caseResult.eval_case_index] || {{}};
                const promptText = original_case.prompt_display_text || '(prompt not found)';
                const promptJson = original_case.prompt_raw_json;

                let card = `<details open><summary>Case #${{caseResult.eval_case_index}}</summary>`;
                card += `<div class="prompt-container"><strong>Prompt:</strong><br>${{DOMPurify.sanitize(marked.parse(String(promptText)))}}</div>`;
                if (promptJson) {{
                    card += `<details class="raw-json-details"><summary>View Raw Prompt JSON</summary><pre class="raw-json-container">${{DOMPurify.sanitize(promptJson)}}</pre></details>`;
                }}

                card += `<div class="responses-grid">`;

                (caseResult.response_candidate_results || []).forEach((candidate, j) => {{
                    const candidateName = candidateNames ? candidateNames[j] : `Candidate #${{j + 1}}`;
                    const displayText = candidate.display_text || '(response not found)';
                    const rawJsonResponse = candidate.raw_json;

                    card += `<div class="response-column"><h4>${{candidateName}}</h4><div class="response-text-container">${{DOMPurify.sanitize(marked.parse(String(displayText)))}}</div>`;
                    if (rawJsonResponse) {{
                        card += `<details class="raw-json-details"><summary>View Raw Response JSON</summary><pre class="raw-json-container">${{DOMPurify.sanitize(rawJsonResponse)}}</pre></details>`;
                    }}

                    card += `<h5>Metrics</h5><table><tbody>`;
                    Object.entries(candidate.metric_results || {{}}).forEach(([name, val]) => {{
                        card += `<tr><td>${{name}}</td><td><b>${{val.score != null ? val.score.toFixed(2) : 'N/A'}}</b></td></tr>`;
                        if(val.explanation) card += `<tr class="explanation-row"><td colspan="2" class="explanation">${{DOMPurify.sanitize(marked.parse(String(val.explanation)))}}</td></tr>`;
                        if (val.rubric_verdicts && val.rubric_verdicts.length > 0) {{
                            card += '<tr><td colspan="2"><div class="rubric-bubble-container">';
                            val.rubric_verdicts.forEach(verdict => {{
                                const rubricDescription = verdict.evaluated_rubric && verdict.evaluated_rubric.content && verdict.evaluated_rubric.content.property ? verdict.evaluated_rubric.content.property.description : 'N/A';
                                const verdictText = verdict.verdict ? '<span class="pass">Pass</span>' : '<span class="fail">Fail</span>';
                                const verdictJson = JSON.stringify(verdict, null, 2);
                                card += `
                                    <details class="rubric-details">
                                        <summary class="rubric-bubble">${{verdictText}}: ${{DOMPurify.sanitize(rubricDescription)}}</summary>
                                        <pre class="raw-json-container">${{DOMPurify.sanitize(verdictJson)}}</pre>
                                    </details>`;
                            }});
                            card += '</div></td></tr>';
                        }}
                    }});
                    card += '</tbody></table></div>';
                }});
                container.innerHTML += card + '</div></details>';
            }});
        }}
        renderSummary(vizData.summary_metrics, vizData.metadata);
        renderDetails(vizData.eval_case_results, vizData.metadata);
    </script>
</body>
</html>
"""


def _get_inference_html(dataframe_json: str) -> str:
    """Returns a self-contained HTML for displaying inference results."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Evaluation Dataset</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
    <style>
        body {{ font-family: 'Roboto', sans-serif; margin: 2em; background-color: #f8f9fa; color: #202124;}}
        .container {{ max-width: 95%; margin: 20px auto; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
        h1 {{ color: #3c4043; border-bottom: 2px solid #4285F4; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
        th, td {{ border: 1px solid #dadce0; padding: 12px; text-align: left; vertical-align: top; }}
        th {{ background-color: #f2f2f2; font-weight: 500;}}
        td > div {{ white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; overflow-wrap: break-word; }}
        .raw-json-details summary {{ font-size: 0.9em; cursor: pointer; color: #5f6368; }}
        .raw-json-container {{ white-space: pre-wrap; word-wrap: break-word; max-height: 300px; overflow-y: auto; background-color: #f1f1f1; padding: 10px; border-radius: 4px; margin-top: 8px; }}
        .rubric-group-title {{ font-weight: bold; margin-bottom: 10px; display: block; }}
        .rubric-bubble-container {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .rubric-details {{ border: none; padding: 0; margin: 0; }}
        .rubric-bubble {{
            display: inline-flex;
            align-items: center;
            background-color: #e8f0fe;
            color: #1967d2;
            border-radius: 16px;
            padding: 8px 12px;
            font-size: 0.9em;
            cursor: pointer;
            list-style: none; /* Hide default marker in Safari */
        }}
        .rubric-bubble::-webkit-details-marker {{ display: none; }} /* Hide default marker in Chrome */
        .rubric-bubble::before {{
            content: '►';
            margin-right: 8px;
            font-size: 0.8em;
            transition: transform 0.2s;
        }}
        .rubric-details[open] > .rubric-bubble::before {{
            transform: rotate(90deg);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Dataset</h1>
        <div id="results-table"></div>
    </div>
    <script>
        const vizData = {dataframe_json};
        const container = document.getElementById('results-table');

        function renderRubrics(cellValue) {{
            let content = '';
            let rubricData = cellValue;
            if (typeof rubricData === 'string') {{
                try {{
                    rubricData = JSON.parse(rubricData);
                }} catch (e) {{
                    console.error("Error parsing rubric_groups JSON:", e, rubricData);
                    return `<div>Error parsing rubrics.</div>`;
                }}
            }}

            if (typeof rubricData !== 'object' || rubricData === null) {{
                 return `<div>Invalid rubric data.</div>`;
            }}

            for (const groupName in rubricData) {{
                const rubrics = rubricData[groupName];
                content += `<div class="rubric-group-title">${{groupName}}</div>`;
                if (Array.isArray(rubrics) && rubrics.length > 0) {{
                    content += '<div class="rubric-bubble-container">';
                    rubrics.forEach((rubric, index) => {{
                        const rubricJson = JSON.stringify(rubric, null, 2);
                        const description = rubric.content && rubric.content.property ? rubric.content.property.description : 'N/A';
                        content += `
                            <details class="rubric-details">
                                <summary class="rubric-bubble">${{DOMPurify.sanitize(description)}}</summary>
                                <pre class="raw-json-container">${{DOMPurify.sanitize(rubricJson)}}</pre>
                            </details>`;
                    }});
                    content += '</div>';
                }}
            }}
            return `<div>${{content}}</div>`;
        }}

        function renderCell(cellValue, header) {{
            let cellContent = '';
            if (header === 'rubric_groups') {{
                return `<td>${{renderRubrics(cellValue)}}</td>`;
            }}

            if (cellValue && typeof cellValue === 'object' && cellValue.display_text !== undefined) {{
                cellContent += `<div>${{DOMPurify.sanitize(marked.parse(String(cellValue.display_text)))}}</div>`;
                if (cellValue.raw_json) {{
                    cellContent += `<details class="raw-json-details"><summary>View Raw JSON</summary><pre class="raw-json-container">${{DOMPurify.sanitize(cellValue.raw_json)}}</pre></details>`;
                }}
            }} else {{
                const cellDisplay = cellValue === null || cellValue === undefined ? '' : String(cellValue);
                cellContent = `<div>${{DOMPurify.sanitize(marked.parse(cellDisplay))}}</div>`;
            }}
            return `<td>${{cellContent}}</td>`;
        }}

        if (!vizData || vizData.length === 0) {{ container.innerHTML = "<p>No data.</p>"; }}
        else {{
            let table = '<table><thead><tr>';
            const headers = Object.keys(vizData[0] || {{}});
            headers.forEach(h => table += `<th>${{h}}</th>`);
            table += '</tr></thead><tbody>';
            vizData.forEach(row => {{
                table += '<tr>';
                headers.forEach(header => {{
                    table += renderCell(row[header], header);
                }});
                table += '</tr>';
            }});
            container.innerHTML = table + '</tbody></table>';
        }}
    </script>
</body>
</html>
"""


def _extract_text_and_raw_json(content: Any) -> dict[str, str]:
    """Extracts display text and raw JSON from a content object.

    This function handles raw strings, Gemini's `contents` format, and
    OpenAI's `messages` format.

    Args:
        content: The content from a 'prompt', 'request', or 'response' column.

    Returns:
        A dictionary with 'display_text' for direct rendering and 'raw_json'
        for an expandable view.
    """
    if not isinstance(content, (str, dict)):
        return {"display_text": str(content or ""), "raw_json": ""}

    try:
        data = json.loads(content) if isinstance(content, str) else content

        if not isinstance(data, dict):
            return {"display_text": str(content), "raw_json": ""}

        pretty_json = json.dumps(data, indent=2, ensure_ascii=False)

        # Gemini format check.
        if (
            "contents" in data
            and isinstance(data.get("contents"), list)
            and data["contents"]
        ):
            first_part = data["contents"][0].get("parts", [{}])[0]
            display_text = first_part.get("text", str(data))
            return {"display_text": display_text, "raw_json": pretty_json}

        # OpenAI response format check.
        elif (
            "choices" in data
            and isinstance(data.get("choices"), list)
            and data["choices"]
        ):
            message = data["choices"][0].get("message", {})
            display_text = message.get("content", str(data))
            return {"display_text": display_text, "raw_json": pretty_json}

        # OpenAI request format check.
        elif (
            "messages" in data
            and isinstance(data.get("messages"), list)
            and data["messages"]
        ):
            user_messages = [
                message.get("content", "")
                for message in data["messages"]
                if message.get("role") == "user"
            ]
            display_text = user_messages[-1] if user_messages else str(data)
            return {"display_text": display_text, "raw_json": pretty_json}
        else:
            # Not a recognized format.
            return {"display_text": str(content), "raw_json": pretty_json}

    except (json.JSONDecodeError, TypeError, IndexError):
        return {"display_text": str(content), "raw_json": ""}


def display_evaluation_result(
    eval_result_obj: types.EvaluationResult,
    candidate_names: Optional[list[str]] = None,
) -> None:
    """Displays evaluation result in an IPython environment."""
    if not _is_ipython_env():
        logger.warning("Skipping display: not in an IPython environment.")
        return
    else:
        from IPython import display

    try:
        result_dump = eval_result_obj.model_dump(
            mode="json", exclude_none=True, exclude={"evaluation_dataset"}
        )
    except errors.PydanticSerializationError as e:
        logger.error(
            "Serialization Error: %s\nCould not display the evaluation "
            "result due to a data serialization issue. Please check the "
            "content of the EvaluationResult object.",
            e,
        )
        return
    except Exception as e:
        logger.error("Failed to serialize EvaluationResult: %s", e, exc_info=True)
        raise

    input_dataset_list = eval_result_obj.evaluation_dataset
    is_comparison = input_dataset_list and len(input_dataset_list) > 1

    metadata_payload = result_dump.get("metadata", {})
    metadata_payload["candidate_names"] = candidate_names or metadata_payload.get(
        "candidate_names"
    )

    if is_comparison:
        if (
            input_dataset_list
            and input_dataset_list[0]
            and input_dataset_list[0].eval_dataset_df is not None
        ):
            base_df = _preprocess_df_for_json(input_dataset_list[0].eval_dataset_df)
            processed_rows: list[dict[str, Any]] = []
            if base_df is not None:
                for _, row in base_df.iterrows():
                    prompt_key = "request" if "request" in row else "prompt"
                    prompt_info = _extract_text_and_raw_json(row.get(prompt_key))
                    processed_row = {
                        "prompt_display_text": prompt_info["display_text"],
                        "prompt_raw_json": prompt_info["raw_json"],
                        "reference": row.get("reference", ""),
                    }
                    processed_rows.append(processed_row)
                metadata_payload["dataset"] = processed_rows

        if "eval_case_results" in result_dump:
            for case_res in result_dump["eval_case_results"]:
                for resp_idx, cand_res in enumerate(
                    case_res.get("response_candidate_results", [])
                ):
                    if (
                        resp_idx < len(input_dataset_list)
                        and input_dataset_list is not None
                        and input_dataset_list[resp_idx].eval_dataset_df is not None
                    ):
                        df = _preprocess_df_for_json(
                            input_dataset_list[resp_idx].eval_dataset_df
                        )
                        case_idx = case_res.get("eval_case_index")
                        if (
                            df is not None
                            and case_idx is not None
                            and case_idx < len(df)
                        ):
                            response_content = df.iloc[case_idx].get("response")
                            display_info = _extract_text_and_raw_json(response_content)
                            cand_res["display_text"] = display_info["display_text"]
                            cand_res["raw_json"] = display_info["raw_json"]

        win_rates = eval_result_obj.win_rates if eval_result_obj.win_rates else {}
        if "summary_metrics" in result_dump:
            for summary in result_dump["summary_metrics"]:
                if summary.get("metric_name") in win_rates:
                    summary.update(win_rates[summary["metric_name"]])

        result_dump["metadata"] = metadata_payload
        html_content = _get_comparison_html(json.dumps(result_dump))
    else:
        single_dataset = input_dataset_list[0] if input_dataset_list else None
        processed_rows = []
        if (
            single_dataset is not None
            and isinstance(single_dataset, types.EvaluationDataset)
            and single_dataset.eval_dataset_df is not None
        ):
            processed_df = _preprocess_df_for_json(single_dataset.eval_dataset_df)
            if processed_df is not None:
                for _, row in processed_df.iterrows():
                    prompt_key = "request" if "request" in row else "prompt"
                    prompt_info = _extract_text_and_raw_json(row.get(prompt_key))
                    response_info = _extract_text_and_raw_json(row.get("response"))
                    processed_row = {
                        "prompt_display_text": prompt_info["display_text"],
                        "prompt_raw_json": prompt_info["raw_json"],
                        "reference": row.get("reference", ""),
                        "response_display_text": response_info["display_text"],
                        "response_raw_json": response_info["raw_json"],
                        "intermediate_events": row.get("intermediate_events", None),
                    }
                    processed_rows.append(processed_row)
            metadata_payload["dataset"] = processed_rows

            if "eval_case_results" in result_dump and processed_rows:
                for case_res in result_dump["eval_case_results"]:
                    case_idx = case_res.get("eval_case_index")
                    if (
                        case_idx is not None
                        and case_idx < len(processed_rows)
                        and case_res.get("response_candidate_results")
                    ):
                        original_case = processed_rows[case_idx]
                        cand_res = case_res["response_candidate_results"][0]
                        cand_res["display_text"] = original_case[
                            "response_display_text"
                        ]
                        cand_res["raw_json"] = original_case["response_raw_json"]

        result_dump["metadata"] = metadata_payload
        html_content = _get_evaluation_html(json.dumps(result_dump))

    display.display(display.HTML(html_content))


def display_evaluation_dataset(eval_dataset_obj: types.EvaluationDataset) -> None:
    """Displays an evaluation dataset in an IPython environment."""
    if not _is_ipython_env():
        logger.warning("Skipping display: not in an IPython environment.")
        return
    else:
        from IPython import display

    if (
        eval_dataset_obj.eval_dataset_df is None
        or eval_dataset_obj.eval_dataset_df.empty
    ):
        logger.warning("No inference data to display.")
        return

    processed_rows = []
    df = eval_dataset_obj.eval_dataset_df

    for _, row in df.iterrows():
        processed_row = {}
        for col_name, cell_value in row.items():
            if col_name in ["prompt", "request", "response"]:
                processed_row[col_name] = _extract_text_and_raw_json(cell_value)
            elif col_name == "rubric_groups":
                # Special handling for rubric_groups to keep it as a dict
                if isinstance(cell_value, dict):
                    processed_row[col_name] = {
                        k: [
                            (
                                v_item.model_dump(mode="json")
                                if hasattr(v_item, "model_dump")
                                else v_item
                            )
                            for v_item in v
                        ]
                        for k, v in cell_value.items()
                    }
                else:
                    processed_row[col_name] = cell_value
            else:
                if isinstance(cell_value, (dict, list)):
                    processed_row[col_name] = json.dumps(
                        cell_value, ensure_ascii=False, default=_pydantic_serializer
                    )
                else:
                    processed_row[col_name] = cell_value
        processed_rows.append(processed_row)

    dataframe_json_string = json.dumps(processed_rows, ensure_ascii=False, default=str)
    html_content = _get_inference_html(dataframe_json_string)
    display.display(display.HTML(html_content))


def _get_status_html(status: str, error_message: Optional[str] = None) -> str:
    """Returns a simple HTML string for displaying a status and optional error."""
    error_html = ""
    if error_message:
        error_html = f"""
        <p>
            <b>Error:</b>
            <pre style="white-space: pre-wrap; word-wrap: break-word;">{error_message}</pre>
        </p>
        """

    return f"""
    <div>
        <p><b>Status:</b> {status}</p>
        {error_html}
    </div>
    """


def display_evaluation_run_status(eval_run_obj: "types.EvaluationRun") -> None:
    """Displays the status of an evaluation run in an IPython environment."""
    if not _is_ipython_env():
        logger.warning("Skipping display: not in an IPython environment.")
        return
    else:
        from IPython import display

    status = eval_run_obj.state.name if eval_run_obj.state else "UNKNOWN"
    error_message = str(eval_run_obj.error) if eval_run_obj.error else None
    html_content = _get_status_html(status, error_message)
    display.display(display.HTML(html_content))
