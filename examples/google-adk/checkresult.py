import json
import os, glob

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
            results[eval_name]["test_results"][-1]["metrics"] = calculate_metrics(
                results[eval_name]["ref_final_response"],
                results[eval_name]["test_results"][-1]["final_response"],
                results[eval_name]["ref_tool_uses"],
                results[eval_name]["test_results"][-1]["tool_uses"]
            )
            
    return results

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

def calculate_metrics(ref_final_response, test_final_response, ref_tool_uses, test_tool_uses):
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
        "tool_order_correct": tool_order_correct,
        "tool_args_correct": tool_args_correct,
        "final_response_correct": ref_final_response == test_final_response
    }

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

        r[eval_name]["run_times"] = len(eval_data["test_results"])   
    
    # calculate the total metrics
    total_run_times = sum([r[eval_name]["run_times"] for eval_name in r])
    r["total"] = {
        m: cal_true_ratio(total_m[m]) for m in metrics_name
    }
    
    r["total"]["run_times"] = total_run_times
    return r

if __name__ == "__main__":
    
    results = collect_results(".")
    metrics = summary_results(results)
    json.dump(results, open("results.json", "w"), indent=4, ensure_ascii=False)
    json.dump(metrics, open("metrics.json", "w"), indent=4, ensure_ascii=False)
    
    import pandas as pd
    df = pd.DataFrame(metrics).T
    # sort by eval_name
    df = df.sort_index()
    print(df)
    
    
  
    
        
    
