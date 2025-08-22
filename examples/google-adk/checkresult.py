import json
import os, glob

def get_chat_messages(prompt_text):
    """
    Call the OpenAI ChatCompletion API to get the model's response.
    """
    
    import os
    from openai import OpenAI

    # read API key from .abacusagent/env.json
    if os.path.isfile("/root/.abacusagent/env.json"):
        envs = json.load(open("/root/.abacusagent/env.json", "r"))
        api_key = envs.get("LLM_API_KEY")
        base_url = envs.get("LLM_BASE_URL")
        model = envs.get("LLM_MODEL").split("/")[-1]  # transfer openai/qwen-turbo to qwen-turbo
    else:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        model = os.getenv("LLM_MODEL")


    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        completion = client.chat.completions.create(
            model=model,
            messages=prompt_text,
            max_tokens=2000,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        print(completion.model_dump_json())
        
        content = completion.choices[0].message.content.strip()
        if not content:
            raise ValueError("Empty response from OpenAI API")
        return content
    except Exception as e:
        print(f"Error during API call: {e}")
        return None



def evaluate_answer(user_content, ref_response, test_response):
    """Evaluate the correctness of a test response against a reference response.
    
    Args:
        user_content (str): The user content for which the responses are generated.
        ref_response (str): The reference response to compare against.
        test_response (str): The test response to evaluate.
        
    Returns:
        bool: True if the test response matches the reference response, False otherwise.
    """
    question = user_content
    response = test_response
    # Get the standard answer
    correct_answer = ref_response
    # Build the prompt
    prompt = f"""You will be given a scientific question, a standard answer, and a response of unknown correctness. Please evaluate the response based on the question and the standard answer, using the following criteria:

If the value is a prediction or a calculation result, it allows for a margin of error, then deviations within the allowable error range (-10% to +10%) can be considered "correct". Significant deviations are considered "incorrect".
As long as the response conveys the same core concepts, regardless of specific wording, it can be considered "correct". If the core results is inconsistent or incorrect, label it as "incorrect."

Question: {question}
Standard answer: {correct_answer}
Response: {response}

The response only needs to contain the correct answer; it does not have to be identical.
Please make your judgment based on the above criteria and respond only with JSON format like:
{{
"question": [The question you are evaluating.],
"reason": [The reason why you think the response is correct or incorrect.],
"correctness": [correct/incorrect]
}}"""

    prompt_text = [
        {"role": "system", "content": "You are an expert in science."},
        {"role": "user", "content": prompt}
    ]
    try:
        # Get the model's response
        response_content = get_chat_messages(prompt_text)
        if not response_content:
            raise ValueError("No response content received.")
        # Try to parse the response as JSON
        try:
            response_json = json.loads(response_content)
        except json.JSONDecodeError as json_error:
            print(f"Invalid JSON response: {response_content}")
            response_json = {"correctness": "incorrect", "reason": "Invalid JSON response"}
        # Record the result
        correctness = response_json.get("correctness", "incorrect")
        return {
            "reason": response_json.get("reason", "No reason provided"),
            "correctness": correctness
        }
        
    except Exception as e:
        print(f"Error processing question: {question}, Error: {e}")
        return {
            "reason": str(e),
            "correctness": "error"
        }

def check_args(ref_args, test_args):
    if len(ref_args) != len(test_args):
        return False
    for iarg in ref_args:
        if iarg not in test_args:
            return False
        
        # do not compare path
        if iarg.endswith("_path") or iarg.endswith("_dir") or iarg.endswith("_file"):
            continue
        
        if ref_args[iarg] != test_args[iarg]:
            return False
    return True

def calculate_metrics(response_correctness, ref_tool_uses, test_tool_uses):
    """Calculate evaluation metrics based on reference and test tool uses and final responses.
    
    Args:
        ref_final_response (str): The reference final response.
        test_final_response (str): The test final response.
        ref_tool_uses (List[Dict]): List of reference tool uses, each containing 'name' and 'args'.
        test_tool_uses (List[Dict]): List of test tool uses, each containing 'name' and 'args'.
        
    Returns:
        Dict[str, Any]: A dictionary containing the evaluation metrics:
            - "tool_order_correct" (int): 1 if the order of tool uses is correct, 0 otherwise.
            - "tool_args_correct" (List[int]): A list indicating whether the arguments for each tool use are correct (1) or not (0).
            - "final_response_correct" (int): 1 if the final responses match, 0 otherwise.
    """
    
    # 1. check the tool use order and names
    ref_tool_names = [tool["name"] for tool in ref_tool_uses]
    test_tool_names = [tool["name"] for tool in test_tool_uses]
    tool_order_correct = ref_tool_names == test_tool_names
    
    # 2. check the tool use arguments
    tool_args_correct = []
    for i, testname in enumerate(test_tool_names):
        if len(ref_tool_names) <= i or testname != ref_tool_names[i]:
            break

        if not check_args(ref_tool_uses[i]["args"], 
                          test_tool_uses[i]["args"]):
            tool_args_correct.append(0)
        else:
            tool_args_correct.append(1)

    
    # 3. check the final response
    
    
    return {
        "Tool-Order-Accuracy": tool_order_correct,
        "Tool-Param-Accuracy": tool_args_correct,
        "Respond-Correct-Rate": response_correctness["correctness"] == "correct"
    }



def compress_evalset(eval_set_id, eval_id):
    return eval_set_id + "." + eval_id

def collect_evalset(agent_path):
    evalset = {}
    
    for i in glob.glob(os.path.join(agent_path, "*evalset.json")):
        c = json.load(open(i, "r"))
        eval_set_id = c["eval_set_id"]
        for eval_set in c["eval_cases"]:
            eval_name = compress_evalset(eval_set_id, eval_set["eval_id"])
            
            evalset[eval_name] = {
                "user_content": eval_set["conversation"][0]["user_content"]["parts"][0]["text"],
                "final_response": eval_set["conversation"][0]["final_response"]["parts"][0]["text"],
                "tool_uses":[]
            }
            
            for inter in eval_set["conversation"][0]["intermediate_data"]["tool_uses"]:
                evalset[eval_name]["intermediate_data"].append({
                    "name": inter["name"],
                    "args": inter["args"]
                })
    
    return evalset

def collect_results(results_path):
    results = {}
    
    for i in glob.glob(os.path.join(results_path, "*.evalset_result.json")):
        with open(i, "r",encoding="utf-8") as f: line = f.read()
        c = json.loads(json.loads(line))
        
        
        eval_set_id = c["eval_set_id"]
        for eval_set in c["eval_case_results"]:
            eval_name = compress_evalset(eval_set_id, eval_set["eval_id"])
            
            if eval_name not in results:
                results[eval_name] = {
                    "user_content": eval_set["eval_metric_result_per_invocation"][0]["expected_invocation"]["user_content"]["parts"][0]["text"],
                    "ref_final_response": eval_set["eval_metric_result_per_invocation"][0]["expected_invocation"]["final_response"]["parts"][0]["text"],
                    "ref_tool_uses":[{
                        "name": inter["name"],
                        "args": inter["args"]
                    } for inter in eval_set["eval_metric_result_per_invocation"][0]["expected_invocation"]["intermediate_data"]["tool_uses"]],
                    "test_results":[]
                     
                }

            results[eval_name]["test_results"].append({
                "file": i,
                "final_response": eval_set["eval_metric_result_per_invocation"][0]["actual_invocation"]["final_response"]["parts"][0]["text"],
                "tool_uses": [{
                    "name": inter["name"],
                    "args": inter["args"]
                } for inter in eval_set["eval_metric_result_per_invocation"][0]["actual_invocation"]["intermediate_data"]["tool_uses"]]
            })
            
            results[eval_name]["test_results"][-1]["response_correctness"] = evaluate_answer(
                user_content=results[eval_name]["user_content"],
                ref_response=results[eval_name]["ref_final_response"],
                test_response=results[eval_name]["test_results"][-1]["final_response"]
                )
            
            results[eval_name]["test_results"][-1]["metrics"] = calculate_metrics(
                results[eval_name]["test_results"][-1]["response_correctness"],
                results[eval_name]["ref_tool_uses"],
                results[eval_name]["test_results"][-1]["tool_uses"]
            )
            
    return results


def cal_true_ratio(lst):
    """Calculate the ratio of True values in a list or list of lists.
    """        
    lst_all = []
    
    def flatten(i):
        """Flatten a list of lists into a single list."""
        if isinstance(i, list):
            for j in i:
                flatten(j)
        else:
            lst_all.append(i)
        
    flatten(lst)
    return lst_all.count(True) / len(lst_all) if lst_all else 0
    
    
    

def summary_results(results):
    r = {}
    
    metrics_name = list(results[list(results.keys())[0]]["test_results"][0]["metrics"].keys())
    total_m = {
        m: [] for m in metrics_name
    }
    
    for eval_name, eval_data in results.items():
        r[eval_name] = { }
        for m in metrics_name:
            r_m = [t["metrics"][m] for t in eval_data["test_results"]]
            r[eval_name][m] = cal_true_ratio(r_m)
            total_m[m].append(r_m)
        r[eval_name]["Tool_Number"] = len(eval_data["ref_tool_uses"])
        r[eval_name]["Total_Run"] = len(eval_data["test_results"]) 
        r[eval_name]["User_Content"] = eval_data["user_content"]
    
    # calculate the total metrics
    total_run_times = sum([r[eval_name]["Total_Run"] for eval_name in r])
    r["total"] = {
        m: cal_true_ratio(total_m[m]) for m in metrics_name
    }
    
    
    r["total"]["Total_Run"] = total_run_times
    return r

if __name__ == "__main__":
    # Please set the environment variables LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    # to use the OpenAI API to compare the responsed.
    
    results = collect_results(".")
    metrics = summary_results(results)
    json.dump(results, open("results.json", "w"), indent=4, ensure_ascii=False)
    json.dump(metrics, open("metrics.json", "w"), indent=4, ensure_ascii=False)
    
    import pandas as pd
    df = pd.DataFrame(metrics).T
    # sort by eval_name
    df = df.sort_index()
    df_display = df.drop(columns=['User_Content'] if 'User_Content' in df.columns else [])
    print(df_display)
    
    
  
    
        
    
