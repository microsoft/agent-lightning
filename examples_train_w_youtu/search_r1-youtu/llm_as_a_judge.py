import json
import logging
import traceback
from copy import deepcopy
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

assert "REWARD_MODEL_URL" in os.environ, "Environment variable REWARD_MODEL_URL must be set"
assert "REWARD_MODEL_NAME" in os.environ, "Environment variable REWARD_MODEL_NAME must be set"
reward_model_url = os.getenv("REWARD_MODEL_URL")
reward_model_name = os.getenv("REWARD_MODEL_NAME")
api_key = os.getenv("REWARD_MODEL_API_KEY", "xxx")

global client
client = OpenAI(base_url=reward_model_url, api_key=api_key, timeout=60)


EVALUATION_PROMPT_EN="""You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""


EVALUATION_PROMPT_ZH="""你是一个专家，能精确判断学生回答是否正确。我会给你一个[问题]、这个问题的[答案]、以及学生对这个问题的[学生回答]。
你需要首先输出一句话简短的思考，然后给出[学生回答]是否正确的评判。
注意：
1. 用“[]”框起最终的评判：“[正确]”或“[错误]”。
2. 如果学生回答和答案矛盾，那学生回答是错误的。
3. 如果学生回答在答案的基础上进行了适当的拓展，那可以认为学生回答是正确的。

请你帮我评判学生的回答：
[问题]
{query}
[答案]
{answer}
[学生回答]
{result}
[你的判断]"""



def compute_score(solution_str, ground_truth, user_query, timeout=60, *args, **kwargs):
    """Compute the reward score for questions with reference responses
    """
    solution = deepcopy(solution_str)
    if "<answer>" in solution and "</answer>" in solution:
        solution = solution[solution.rfind("<answer>")+len("<answer>"):]
        if "</answer>" in solution:
            solution = solution[:solution.find("</answer>")]
    
    res = {"pred": solution, "score": -1.0, "acc": False}
    if type(ground_truth) is list:
        assert(len(ground_truth))
        if len(ground_truth) == 1:
            ground_truth = ground_truth[0]
        else:
            ground_truth = ", ".join(ground_truth)

    prompt = EVALUATION_PROMPT_ZH.replace("{query}",\
        str(user_query)).replace("{result}",\
            str(solution)).replace("{answer}", str(ground_truth))
    global client
    global reward_model_name
    if len(prompt) > 20000:
        return -1.0
    solution = solution.strip()
    if len(solution) == 0:
        return -1.0
    def get_response_from_openai(prompt):
        messages = [
            {"role":"user", "content":prompt}
        ]
        response = client.chat.completions.create(
                    model=reward_model_name,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.7,
                    top_p=0.8,
                    stream=False).to_dict()
        # For 235b Temperature=0.7, TopP=0.8, TopK=20
        assert "choices" in response
        result = response["choices"][0]["message"]["content"]
        return result

    try:
        # print("prompt", prompt)
        result = get_response_from_openai(prompt)
        assert (result is not None)
        result = str(result)
        # print(f"{reward_model_name} 输出结果: ", result)
        if "[正确]" in result or ("[错误]" not in result and "正确" in result):
            res["score"] = 1.0
            res["acc"] = True
            return res
        else:
            res["score"] = -1.0
            return res

    except Exception as e:
        logger.error(f"Error during remote reward modeling compute_score: {e}")
        traceback.print_exc()
        res["score"] = -1.0
    return res
