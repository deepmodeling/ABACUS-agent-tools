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

def compare_args(ref_args, test_args):
    if len(ref_args) != len(test_args):
        return False
    for iarg in ref_args:
        if iarg not in test_args:
            return False
        
        # do not compare path
        if iarg.endswith("_path") or iarg.endswith("_dir"):
            continue
        
        if ref_args[iarg] != test_args[iarg]:
            return False
    return True

def calculate_metrics(ref_final_response, test_final_response, ref_tool_uses, test_tool_uses):
    
    # 1. check the tool use order and names
    ref_tool_names = [tool["name"] for tool in ref_tool_uses]
    test_tool_names = [tool["name"] for tool in test_tool_uses]
    tool_order_correct = ref_tool_names == test_tool_names
    
    # 2. check the tool use arguments
    if tool_order_correct:
        tool_args_correct = True 
        for ref_tool, test_tool in zip(ref_tool_uses, test_tool_uses):
            if not compare_args(ref_tool["args"], test_tool["args"]):
                tool_args_correct = False
                break
    else:
        tool_args_correct = False
    
    # 3. check the final response
    
    return {
        "tool_order_correct": tool_order_correct,
        "tool_args_correct": tool_args_correct,
        "final_response_correct": ref_final_response == test_final_response
    }

        

def summary_results(results):
    r = {}
    
    for eval_name, eval_data in results.items():
        metrics_name = eval_data["test_results"][0]["metrics"].keys()
        run_times = len(eval_data["test_results"])
        r[eval_name] = {
            m : [results[eval_name]["test_results"][i]["metrics"][m] for i in range(run_times)].count(True) / run_times for m in metrics_name
        }
        
        r[eval_name]["run_times"] = run_times   
    
    # calculate the total metrics
    total_run_times = sum([r[eval_name]["run_times"] for eval_name in r])
    r["total"] = {
        m: sum([r[eval_name][m] * r[eval_name]["run_times"] for eval_name in r]) / total_run_times for m in metrics_name
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
    
    
  
    
        
    
