
"""
UltraRAG adapter for Agent Lightning.
Uses UltraRAG components (retrieval/generation) with the AGL training interface.
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from agentlightning import LLM, LitAgent, NamedResources, Trainer, setup_logging
from agentlightning.reward import reward

try:
    from ultrarag.client import UltraData, Configuration
    from fastmcp import Client
    ULTRARAG_AVAILABLE = True
except ImportError:
    print("Warning: UltraRAG components are unavailable")
    ULTRARAG_AVAILABLE = False

from qa_em import compute_score_em, em_check

setup_logging()


@reward
async def eval(prediction: str, ground_truth: List[str]) -> float:
    has_answer_tag = "<answer>" in prediction
    if not has_answer_tag:
        reward_score = float(em_check(prediction, ground_truth))
    else:
        reward_score = float(compute_score_em(prediction, ground_truth))
    print(f"pred: {prediction} | gold_answer: {ground_truth} | has_tag: {has_answer_tag} | res: {reward_score}")
    return reward_score


def extract_answer_from_response(response_text: str) -> str:
    """Extract the final answer from the response."""
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return response_text.strip()


class UltraRAGPipelineExecutor:
    """
    UltraRAG Pipeline executor.
    
    Wraps the UltraRAG pipeline execution into a single-query interface.
    """
    
    def __init__(
        self,
        config_path: str,
        param_path: Optional[str] = None,
        generation_endpoint: Optional[str] = None,
        generation_model: Optional[str] = None,
    ):
        """
        Args:
            config_path: UltraRAG pipeline config path.
            param_path: UltraRAG parameter config path.
            generation_endpoint: Generation model API endpoint (override config).
            generation_model: Generation model name (override config).
        """
        self.config_path = Path(config_path)
        self.param_path = Path(param_path) if param_path else None
        self.generation_endpoint = generation_endpoint
        self.generation_model = generation_model
        
        self.cfg = Configuration()
        self.pipeline_config = None
        self.param_config = None
        self._load_configs()
    
    def _load_configs(self):
        """Load configuration files."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        self.pipeline_config = self.cfg.load_config(str(self.config_path))
        
        if self.param_path and self.param_path.exists():
            self.param_config = self.cfg.load_parameter_config(str(self.param_path))
        else:
            param_file = self.config_path.parent / "parameter" / f"{self.config_path.stem}_parameter.yaml"
            if param_file.exists():
                self.param_config = self.cfg.load_parameter_config(str(param_file))
    
    async def execute_single_query(
        self,
        question: str,
        llm_endpoint: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 1.0,
        max_iterations: int = 8,
    ) -> Dict[str, Any]:
        """
        Run a single query via the UltraRAG pipeline.
        
        Args:
            question: Question text.
            llm_endpoint: LLM API endpoint (override).
            llm_model: LLM model name (override).
            temperature: Sampling temperature.
            max_iterations: Max iterations (from pipeline config or this argument).
        
        Returns:
            Dict with answer, reasoning steps, and retrieved nodes.
        """
        if not ULTRARAG_AVAILABLE:
            raise ImportError("UltraRAG components unavailable; full pipeline mode disabled.")
        
        if not self.pipeline_config:
            raise ValueError("Pipeline config not loaded")
        
        from ultrarag.client import run as ultrarag_run
        import tempfile
        import json
        import yaml
        
        temp_param_file = None
        temp_benchmark_file = None
        
        try:
            param_config = self.param_config.copy() if self.param_config else {}
            
            if llm_endpoint:
                if "generation" not in param_config:
                    param_config["generation"] = {}
                if "backend_configs" not in param_config["generation"]:
                    param_config["generation"]["backend_configs"] = {}
                if "openai" not in param_config["generation"]["backend_configs"]:
                    param_config["generation"]["backend_configs"]["openai"] = {}
                
                param_config["generation"]["backend_configs"]["openai"]["base_url"] = llm_endpoint
                if llm_model:
                    param_config["generation"]["backend_configs"]["openai"]["model_name"] = llm_model
                param_config["generation"]["backend_configs"]["openai"]["use_completions"] = True
            
            if "generation" not in param_config:
                param_config["generation"] = {}
            if "sampling_params" not in param_config["generation"]:
                param_config["generation"]["sampling_params"] = {}
            param_config["generation"]["sampling_params"]["temperature"] = temperature
            
            pipeline_steps = self.pipeline_config.get("pipeline", [])
            has_benchmark = any(
                (isinstance(step, str) and step == "benchmark.get_data") or
                (isinstance(step, dict) and "benchmark" in str(step))
                for step in pipeline_steps
            )
            
            temp_benchmark_file = None
            if has_benchmark:
                temp_benchmark_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
                )
                benchmark_data = {
                    "question": question,
                    "golden_answers": []
                }
                temp_benchmark_file.write(json.dumps(benchmark_data, ensure_ascii=False) + "\n")
                temp_benchmark_file.close()
                
                if "benchmark" not in param_config:
                    param_config["benchmark"] = {}
                param_config["benchmark"]["benchmark"] = {
                    "path": temp_benchmark_file.name,
                    "limit": 1,
                    "key_map": {
                        "q_ls": "question",
                        "gt_ls": "golden_answers"
                    }
                }
            else:
                pass
            
            temp_param_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False, encoding='utf-8'
            )
            yaml.dump(param_config, temp_param_file, allow_unicode=True)
            temp_param_file.close()
            
            result = await ultrarag_run(
                str(self.config_path),
                param_path=temp_param_file.name,
                return_all=True  # return all intermediate results for reasoning steps
            )
            
            all_results = result.get("all_results", [])
            final_result = result.get("final_result", None)
            
            answer = ""
            if final_result:
                if isinstance(final_result, dict):
                    ans_ls = final_result.get("ans_ls", [])
                    answer = ans_ls[0] if ans_ls else ""
                elif isinstance(final_result, list) and final_result:
                    answer = final_result[0] if isinstance(final_result[0], str) else str(final_result[0])
                else:
                    answer = str(final_result) if final_result else ""
            
            reasoning_steps = []
            retrieved_nodes = []
            rollout_content_parts = []
            
            for snapshot in all_results:
                if "ans_ls" in snapshot:
                    ans_list = snapshot.get("ans_ls", [])
                    if ans_list and ans_list[-1]:
                        reasoning_steps.append(str(ans_list[-1]))
                        rollout_content_parts.append(str(ans_list[-1]))
                
                if "retrieved_docs" in snapshot:
                    retrieved = snapshot.get("retrieved_docs", [])
                    if retrieved:
                        retrieved_nodes.extend(retrieved)
            
            rollout_content = "\n".join(rollout_content_parts) if rollout_content_parts else answer
            
            return {
                "answer": answer,
                "response": rollout_content,  # full response (for RL training)
                "steps": reasoning_steps,
                "retrieved_nodes": retrieved_nodes,
            }
            
        except Exception as e:
            print(f"UltraRAG pipeline execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": "",
                "response": "",
                "steps": [],
                "retrieved_nodes": [],
            }
        finally:
            if temp_param_file:
                try:
                    os.unlink(temp_param_file.name)
                except:
                    pass
            if temp_benchmark_file:
                try:
                    os.unlink(temp_benchmark_file.name)
                except:
                    pass


class UltraRAGAgent(LitAgent[Any]):
    """UltraRAG Agent for UltraRAG + AGL training."""
    """
    Agent that integrates UltraRAG with Agent Lightning.
    
    Uses UltraRAG core components to process queries.
    """
    
    def __init__(
        self,
        ultrarag_config_path: Optional[str] = None,
        ultrarag_param_path: Optional[str] = None,
        use_simplified_interface: bool = False,
    ):
        """
        Args:
            ultrarag_config_path: UltraRAG pipeline config path.
            ultrarag_param_path: UltraRAG parameter config path.
            use_simplified_interface: Whether to use the simplified interface (direct retrieve/generate without full pipeline).
        """
        super().__init__()
        self.use_simplified_interface = use_simplified_interface
        
        if ultrarag_config_path:
            self.ultrarag_config_path = Path(ultrarag_config_path)
        else:
            default_path = Path(__file__).parent / "search_r1_rl.yaml"
            if not default_path.exists():
                default_path = Path(__file__).parent / "r1_searcher.yaml"
            self.ultrarag_config_path = default_path if default_path.exists() else None
        
        if ultrarag_param_path:
            self.ultrarag_param_path = Path(ultrarag_param_path)
        else:
            default_param = Path(__file__).parent / "search_r1_rl_parameter.yaml"
            if not default_param.exists():
                default_param = Path(__file__).parent / "r1_searcher_parameter.yaml"
            self.ultrarag_param_path = default_param if default_param.exists() else None
        
        if not use_simplified_interface and self.ultrarag_config_path:
            self.pipeline_executor = UltraRAGPipelineExecutor(
                str(self.ultrarag_config_path),
                str(self.ultrarag_param_path) if self.ultrarag_param_path else None,
                generation_endpoint=None,  # set at runtime
                generation_model=None,  # set at runtime
            )
        else:
            self.pipeline_executor = None
    
    async def _act_with_ultrarag(
        self,
        question: str,
        llm_endpoint: str,
        llm_model: str,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Execute query with UltraRAG core components.
        
        Return format:
        {
            "response": str,  # full response (for RL training)
            "steps": List[str],  # reasoning steps (for RL reward)
            "retrieved_nodes": List[Dict],  # retrieved nodes
        }
        """
        if self.pipeline_executor and not self.use_simplified_interface:
            result = await self.pipeline_executor.execute_single_query(
                question,
                llm_endpoint=llm_endpoint,
                llm_model=llm_model,
                temperature=temperature,
            )
            return {
                "response": result.get("response", ""),  # Use full response, not just the answer.
                "steps": result.get("steps", []),
                "retrieved_nodes": result.get("retrieved_nodes", []),
            }
        else:
            return await self._act_with_simplified_interface(
                question, llm_endpoint, llm_model, temperature
            )
    
    async def _act_with_simplified_interface(
        self,
        question: str,
        llm_endpoint: str,
        llm_model: str,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Execute query with the simplified interface.
        
        Similar to search_r1_agent but keeps hooks for full UltraRAG pipeline.
        """
        from openai import AsyncOpenAI
        import requests
        
        INSTRUCTION_FORMAT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """
        
        client = AsyncOpenAI(
            base_url=llm_endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        )
        
        async def call_llm(content: str, max_tokens: int = 500) -> str:
            """Call LLM via chat.completions (Instruct models support chat templates).
            
            AsyncOpenAI so AgentOpsTracer can capture calls and generate triplets.
            """
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"return_token_ids": True},  # vLLM needs this to return token_ids.
            )
            return response.choices[0].message.content or ""
        
        def extract_action(response: str) -> Tuple[Optional[str], str]:
            pattern = r"<(search|answer)>(.*?)</\1>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1), match.group(2).strip()
            return None, ""
        
        def postprocess_response(response: str) -> str:
            if "</search>" in response:
                return response.split("</search>")[0] + "</search>"
            elif "</answer>" in response:
                return response.split("</answer>")[0] + "</answer>"
            return response
        
        retrieval_endpoint = os.environ.get("RETRIEVAL_ENDPOINT", "http://127.0.0.1:8002/retrieve")
        
        prompt = INSTRUCTION_FORMAT + question
        rollout_content = ""
        reasoning_steps = []
        retrieved_nodes = []
        turn_id = 0
        finished = False
        
        while turn_id < 4 and not finished:
            turn_id += 1
            turn_response = await call_llm(prompt + rollout_content)
            valid_response = postprocess_response(turn_response)
            reasoning_steps.append(valid_response)
            
            action, content = extract_action(valid_response)
            if action == "answer":
                finished = True
                rollout_content += valid_response
            elif action == "search":
                payload = {"queries": [content], "topk": 3, "return_scores": True}
                try:
                    resp = requests.post(retrieval_endpoint, json=payload, timeout=10)
                    resp.raise_for_status()
                    json_resp = resp.json()
                    retrieval_result = json_resp["result"][0]
                    retrieved_nodes.extend(retrieval_result)
                    
                    format_ref = ""
                    for idx, doc_item in enumerate(retrieval_result):
                        doc = doc_item.get("document", doc_item)
                        content_str = doc.get("contents", str(doc)) if isinstance(doc, dict) else str(doc)
                        lines = content_str.split("\n")
                        title = lines[0] if lines else ""
                        text = "\n".join(lines[1:]) if len(lines) > 1 else content_str
                        format_ref += f"Doc {idx+1}(Title: {title}) {text}\n"
                    
                    env_feedback = f"\n\n<information>{format_ref}</information>\n\n"
                except Exception as e:
                    print(f"Retrieval error: {e}")
                    env_feedback = "\n\n<information>retrieval failed</information>\n\n"
                
                rollout_content += valid_response + env_feedback
            else:
                error_msg = "\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
                rollout_content += valid_response + error_msg
        
        if not finished:
            final_response = await call_llm(prompt + rollout_content)
            rollout_content += final_response
            reasoning_steps.append(final_response)
        
        return {
            "response": rollout_content,
            "steps": reasoning_steps,
            "retrieved_nodes": retrieved_nodes,
        }
    
    async def training_rollout_async(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any,
        temperature: float = 1.0,
    ) -> Any:
        question = task["question"]
        answer_list: List[str] = cast(List[str], task["golden_answers"])
        llm: LLM = cast(LLM, resources.get("main_llm"))
        
        result = await self._act_with_ultrarag(
            question, llm.endpoint, llm.model, temperature
        )
        
        pred_answer = extract_answer_from_response(result["response"])
        
        reward_score = await eval(pred_answer, answer_list)
        print(
            f"question: {question} "
            f"pred_answer: {pred_answer} "
            f"ground_truth: {answer_list} "
            f"reward: {reward_score}"
        )
        
        return reward_score
    
    async def validation_rollout_async(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any,
    ) -> Any:
        reward_score = await self._validation_with_save(task, resources, rollout)
        return reward_score
    
    async def _validation_with_save(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any,
    ) -> float:
        """Run validation and save results."""
        import json
        import os
        from pathlib import Path
        from datetime import datetime
        
        question = task["question"]
        answer_list: List[str] = cast(List[str], task["golden_answers"])
        llm: LLM = cast(LLM, resources.get("main_llm"))
        
        result = await self._act_with_ultrarag(
            question, llm.endpoint, llm.model, temperature=0.0
        )
        
        pred_answer = extract_answer_from_response(result["response"])
        
        reward_score = await eval(pred_answer, answer_list)
        
        try:
            checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints/ultrarag_agl_checkpoints/ultrarag_agl")
            step = int(os.environ.get("CURRENT_STEP", "0"))
            is_val_before_train = (step == 0)
            
            if is_val_before_train:
                val_dir = Path(checkpoint_dir) / "val_before_train"
            else:
                val_dir = Path(checkpoint_dir) / f"validation_step_{step}"
            val_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = val_dir / "results.jsonl"
            validation_result = {
                "question": question,
                "golden_answers": answer_list,
                "prediction": pred_answer,  # extracted final answer
                "rollout_content": result["response"],  # full reasoning trace
                "steps": result.get("steps", []),  # reasoning steps
                "retrieved_nodes": result.get("retrieved_nodes", []),  # retrieved nodes
                "reward": float(reward_score),
                "step": step,
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(validation_result, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error while saving validation results: {e}")
        
        return reward_score


if __name__ == "__main__":
    Trainer(n_workers=128).fit_v0(
        UltraRAGAgent(use_simplified_interface=False), 
        "http://localhost:9999/"
    )

