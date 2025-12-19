import asyncio
import aiohttp
import time
import random
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm

API_ENDPOINT = "http://10.137.80.46:8000/v1/completions" 
MODELS = ['Chatbot-A', 'Chatbot-B', 'Chatbot-C']

# Test duration (seconds)
TEST_DURATION = 60

# Base concurrency (Requests Per Second)
TARGET_RPS = 3

# Input Prompt (length should be fixed to exclude input length interference, focusing on switching)
PROMPT_TEXT = "Define the Service Level Objective in one sentence." * 14 # token: 9
MAX_TOKENS = 64

async def send_request(session, request_id, model_name, scenario_name):
    """
    Send a single request, using Streaming mode to measure TTFT and ITL
    """
    url = API_ENDPOINT
    payload = {
        "model": model_name,
        "prompt": PROMPT_TEXT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True
    }
    
    start_time = time.time()
    ttft = 0
    total_latency = 0
    token_count = 0
    status = "FAIL"
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                first_token_time = None
                last_token_time = None
                
                async for line in response.content:
                    if line:
                        current_time = time.time()
                        
                        # Capture TTFT (Time of first data received)
                        if first_token_time is None:
                            first_token_time = current_time
                            ttft = (first_token_time - start_time) * 1000 # ms
                        
                        last_token_time = current_time
                        token_count += 1
                
                # Calculate total latency
                if last_token_time:
                    end_time = last_token_time
                    total_latency = (end_time - start_time) * 1000 # ms
                    status = "SUCCESS"
                else:
                    # 200 OK but no content
                    status = "EMPTY_RESPONSE"
            else:
                # Use tqdm.write to avoid breaking the progress bar
                tqdm.write(f"Error {response.status}") 
                pass
                
    except Exception as e:
        tqdm.write(f"Request failed: {e}")
        pass
    
    avg_itl = 0
    if token_count > 1 and status == "SUCCESS":
        avg_itl = (total_latency - ttft) / (token_count - 1)

    return {
        "request_id": request_id,
        "scenario": scenario_name,
        "model": model_name,
        "start_time": start_time,
        "ttft_ms": ttft,
        "total_latency_ms": total_latency,
        "avg_itl_ms": avg_itl,
        "token_count": token_count,
        "status": status
    }

async def run_scenario_round_robin(session, results):
    """Scenario 1: Extreme Switching (Round Robin)"""
    print(f"--- Starting Scenario: Round Robin (Worst Case) ---")
    start_test = time.time()
    req_id = 0
    
    # Use tqdm to create a progress bar, total amount is test seconds
    with tqdm(total=TEST_DURATION, desc="Round Robin Progress", unit="s") as pbar:
        last_update_time = start_test
        
        while time.time() - start_test < TEST_DURATION:
            model = MODELS[req_id % len(MODELS)]
            task = asyncio.create_task(send_request(session, f"RR-{req_id}", model, "round_robin"))
            results.append(task)
            req_id += 1
            
            await asyncio.sleep(1.0 / TARGET_RPS)
            
            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_scenario_zipfian(session, results, seed):
    """Scenario 2: Real Distribution (Zipfian / Weighted)"""
    print(f"--- Starting Scenario: Zipfian (Real World) with seed {seed} ---")
    # Set weights: First model 80%, the rest share the remaining 20%
    weights = [0.8] + [(0.2 / (len(MODELS)-1))] * (len(MODELS)-1)
    
    # Initialize random generators
    random_gen = random.Random(seed)
    np_gen = np.random.default_rng(seed)

    start_test = time.time()
    req_id = 0
    
    with tqdm(total=TEST_DURATION, desc="Zipfian Progress", unit="s") as pbar:
        last_update_time = start_test
        
        while time.time() - start_test < TEST_DURATION:
            # Randomly select based on weights
            model = random_gen.choices(MODELS, weights=weights, k=1)[0]
            task = asyncio.create_task(send_request(session, f"ZIPF-{req_id}", model, "zipfian"))
            results.append(task)
            req_id += 1
            # Use Poisson process interval time (closer to real traffic)
            sleep_time = np_gen.exponential(1.0 / TARGET_RPS)
            await asyncio.sleep(sleep_time)
            
            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_scenario_bursty(session, results, seed):
    """Scenario 3: Bursty Traffic (Bursty)"""
    print(f"--- Starting Scenario: Bursty (Stress Test) with seed {seed} ---")
    
    # Initialize random generator
    random_gen = random.Random(seed)

    start_test = time.time()
    req_id = 0
    
    with tqdm(total=TEST_DURATION, desc="Bursty Progress", unit="s") as pbar:
        last_update_time = start_test
        
        while time.time() - start_test < TEST_DURATION:
            current_elapsed = time.time() - start_test
            
            # Generate burst traffic every 10 seconds
            if int(current_elapsed) % 10 == 0 and int(current_elapsed) > 0:
                # Use tqdm.write to print logs to avoid progress bar confusion
                tqdm.write(f"!!! BURST INCOMING at {int(current_elapsed)}s !!!")
                
                burst_size = 10  # Inject 10 requests at once
                # Bursty traffic is usually mixed, here randomly mixed
                burst_models = random_gen.choices(MODELS, k=burst_size)
                
                for model in burst_models:
                    task = asyncio.create_task(send_request(session, f"BURST-{req_id}", model, "bursty"))
                    results.append(task)
                    req_id += 1
                
                # Rest a bit after burst to avoid instant overload causing Client crash
                await asyncio.sleep(1) 
            else:
                # Background traffic (low load)
                model = random_gen.choice(MODELS)
                task = asyncio.create_task(send_request(session, f"BG-{req_id}", model, "bursty"))
                results.append(task)
                req_id += 1
                await asyncio.sleep(1.0 / (TARGET_RPS / 2)) # Background traffic set to half of the target
            
            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_scenario_single_model(session, results, model_index=0):
    """Scenario 5: Single Model Test (Baseline)"""
    model_name = MODELS[model_index]
    print(f"--- Starting Scenario: Single Model Test (Model: {model_name}) ---")
    
    start_test = time.time()
    req_id = 0
    
    with tqdm(total=TEST_DURATION, desc=f"Single Model ({model_name}) Progress", unit="s") as pbar:
        last_update_time = start_test
        
        while time.time() - start_test < TEST_DURATION:
            task = asyncio.create_task(send_request(session, f"SINGLE-{req_id}", model_name, "single_model"))
            results.append(task)
            req_id += 1
            
            await asyncio.sleep(1.0 / TARGET_RPS)
            
            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_single_test(session, test_case, seed, model_index=0):
    all_tasks = []
    case_name = ""
    
    if test_case == 1:
        case_name = "round_robin"
        await run_scenario_round_robin(session, all_tasks)
    elif test_case == 2:
        case_name = "zipfian"
        await run_scenario_zipfian(session, all_tasks, seed)
    elif test_case == 3:
        case_name = "bursty"
        await run_scenario_bursty(session, all_tasks, seed)
    elif test_case == 5:
        case_name = f"single_model_{MODELS[model_index]}"
        await run_scenario_single_model(session, all_tasks, model_index)
    
    print(f"\nAll requests dispatched for {case_name}. Waiting for pending responses...")
    
    responses = []
    for f in tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc="Collecting Responses", unit="req"):
        responses.append(await f)
    
    df = pd.DataFrame(responses)
    filename = f"benchmark_results_{case_name}.csv"
    df.to_csv(filename, index=False)
    print(f"Done! Results saved to {filename}")
    
    print(f"\n=== Quick Summary ({case_name}) ===")
    if not df.empty:
        print(df.groupby("model")["total_latency_ms"].describe())
    else:
        print("No data collected.")

async def main(seed):
    print(f"Running in RPS{TARGET_RPS}")
    print(f"Running in Random Seed: {seed}")
    print("input test_case number:")
    print("1. Round Robin")
    print("2. Zipfian (Real Distribution)")
    print("3. Bursty")
    print("4. Run All (Multi-Model)")
    print("5. Single Model Test")
    
    try:
        test_case = int(input().strip())
    except ValueError:
        print("Invalid input")
        return

    if test_case not in [1, 2, 3, 4, 5]:
        print(f"Please enter option 1, 2, 3, 4, or 5")
        return
    
    model_index = 0
    if test_case == 5:
        print("\nSelect model to test:")
        for i, model in enumerate(MODELS):
            print(f"{i}. {model}")
        try:
            model_index = int(input().strip())
            if model_index < 0 or model_index >= len(MODELS):
                print(f"Invalid model index. Please enter 0-{len(MODELS)-1}")
                return
        except ValueError:
            print("Invalid input")
            return

    async with aiohttp.ClientSession() as session:
        if test_case == 4:
            scenarios = [1, 2, 3]
            for i, scenario in enumerate(scenarios):
                await run_single_test(session, scenario, seed)
                if i < len(scenarios) - 1:
                    print("\nWaiting 10 seconds before next test...")
                    await asyncio.sleep(10)
        else:
            await run_single_test(session, test_case, seed, model_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for experiments")
    args = parser.parse_args()
    
    asyncio.run(main(args.seed))
