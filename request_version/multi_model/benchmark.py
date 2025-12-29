import asyncio
import aiohttp
import time
import random
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm
import random
import string

# vLLM random input generation
from vllm.benchmarks.datasets import RandomDataset, RandomMultiModalDataset
from vllm.transformers_utils.tokenizer import get_tokenizer

API_ENDPOINT = "http://localhost:8000/v1/completions"
MODELS =  ['Chatbot', 'VisionProcessor', 'Embedding']
# MODELS =  ['Chatbot', 'Chatbot']

# Test duration (seconds)
TEST_DURATION = 300

# Request timeout (seconds)
REQUEST_TIMEOUT = 1000

# Base concurrency (Requests Per Second)
TARGET_RPS = 4

# Input Prompt (length should be fixed to exclude input length interference, focusing on switching)
PROMPT_TEXT = "Define the Service Level Objective in one sentence." * 14 # token: 9
MAX_TOKENS = 300


# add params endpoint, payload                                                                                                                                                                                                                 
# 看 vllm 怎麼 random input 的                                                                                                                                                                                                                 
# set random int 500 and set min token output                                                                                                                                                                                                  
# Default endpoints for different model types

DEFAULT_ENDPOINTS = {
    "llm": "http://localhost:8000/v1/completions",
    "vlm": "http://localhost:8000/v1/chat/completions",
    "embedding": "http://localhost:8000/v1/embeddings",
}

# Global random input manager instance (initialized in main)
random_input_manager = None


class RandomInputManager:
    """Manages random input generation using vLLM's RandomDataset classes."""

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        # Initialize datasets with seed
        self.text_dataset = RandomDataset(random_seed=args.seed)
        self.mm_dataset = RandomMultiModalDataset(random_seed=args.seed)

        # Sample pools
        self.llm_samples = []
        self.vlm_samples = []
        self.embedding_samples = []

        # Index counters for accessing samples
        self.llm_idx = 0
        self.vlm_idx = 0
        self.embedding_idx = 0

    def _calculate_pool_size(self):
        """Calculate dynamic pool size based on test parameters."""
        return TEST_DURATION * TARGET_RPS * 4

    def _parse_bucket_config(self, config_str):
        """Parse bucket config string to dict."""
        import ast
        try:
            return ast.literal_eval(config_str)
        except:
            # Fallback to default
            return {(256, 256, 1): 0.5, (720, 1280, 1): 0.5}

    def generate_sample_pool(self):
        """Pre-generate sample pools for all model types."""
        pool_size = self._calculate_pool_size()

        # Generate LLM samples (text only)
        self.llm_samples = self.text_dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=pool_size,
            input_len=self.args.random_input_len,
            output_len=self.args.random_output_len,
            range_ratio=self.args.random_range_ratio,
            batchsize=1
        )
        # Shuffle LLM samples
        random.shuffle(self.llm_samples)

        # Generate VLM samples (multimodal)
        bucket_config = self._parse_bucket_config(self.args.random_mm_bucket_config)
        self.vlm_samples = self.mm_dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=pool_size,
            input_len=self.args.random_input_len,
            output_len=self.args.random_output_len,
            range_ratio=self.args.random_range_ratio,
            base_items_per_request=self.args.random_mm_base_items_per_request,
            num_mm_items_range_ratio=self.args.random_mm_num_mm_items_range_ratio,
            bucket_config=bucket_config,
            limit_mm_per_prompt={"image": 255, "video": 0}
        )
        # Shuffle VLM samples
        random.shuffle(self.vlm_samples)

        # Generate Embedding samples (text only, batchsize=1)
        self.embedding_samples = self.text_dataset.sample(
            tokenizer=self.tokenizer,
            num_requests=pool_size,
            input_len=self.args.random_input_len,
            output_len=self.args.random_output_len,
            range_ratio=self.args.random_range_ratio,
            batchsize=1
        )
        # Shuffle Embedding samples
        random.shuffle(self.embedding_samples)

    def get_llm_sample(self):
        """Get next LLM sample (text prompt)."""
        if not self.llm_samples:
            return None

        # Get sample at current index
        sample = self.llm_samples[self.llm_idx % len(self.llm_samples)]
        self.llm_idx += 1

        # If we've used all samples, reshuffle
        if self.llm_idx >= len(self.llm_samples):
            random.shuffle(self.llm_samples)
            self.llm_idx = 0

        return sample.prompt

    def get_vlm_sample(self):
        """Get next VLM sample (text prompt + multimodal data)."""
        if not self.vlm_samples:
            return None, None

        # Get sample at current index
        sample = self.vlm_samples[self.vlm_idx % len(self.vlm_samples)]
        self.vlm_idx += 1

        # If we've used all samples, reshuffle
        if self.vlm_idx >= len(self.vlm_samples):
            random.shuffle(self.vlm_samples)
            self.vlm_idx = 0

        # Return prompt and multimodal data
        return sample.prompt, sample.multi_modal_data

    def get_embedding_sample(self):
        """Get next Embedding sample (text input)."""
        if not self.embedding_samples:
            return None

        # Get sample at current index
        sample = self.embedding_samples[self.embedding_idx % len(self.embedding_samples)]
        self.embedding_idx += 1

        # If we've used all samples, reshuffle
        if self.embedding_idx >= len(self.embedding_samples):
            random.shuffle(self.embedding_samples)
            self.embedding_idx = 0

        return sample.prompt


# Default payload generators for different model types
def get_default_llm_payload(model_name, prompt=None, max_tokens=None):
    """Default payload for LLM (text completion) models"""
    global random_input_manager

    # Use random input if manager is initialized, otherwise fallback
    if prompt is None and random_input_manager is not None:
        prompt = random_input_manager.get_llm_sample()
    elif prompt is None:
        prompt = PROMPT_TEXT  # Fallback to original constant

    # Use configured max_tokens from args if available
    if max_tokens is None and random_input_manager is not None:
        max_tokens = random_input_manager.args.random_output_len
    elif max_tokens is None:
        max_tokens = MAX_TOKENS

    return {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "ignore_eos": True
    }

def get_default_vlm_payload(model_name, image_url=None, prompt=None, max_tokens=None):
    """Default payload for VLM (vision-language) models"""
    global random_input_manager

    # Get random multimodal content if manager is initialized
    if image_url is None and prompt is None and random_input_manager is not None:
        text_prompt, mm_data = random_input_manager.get_vlm_sample()

        # Build content list with images + text
        content = []

        # Add all images from multi_modal_data
        if mm_data and isinstance(mm_data, list):
            for item in mm_data:
                if item.get("type") == "image_url":
                    content.append(item)

        # Add text prompt
        content.append({
            "type": "text",
            "text": text_prompt
        })
    else:
        # Fallback to original behavior
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url or "http://localhost:9000/1.png"
                }
            },
            {
                "type": "text",
                "text": prompt or "What is in this image?"
            }
        ]

    # Use configured max_tokens from args if available
    if max_tokens is None and random_input_manager is not None:
        max_tokens = random_input_manager.args.random_output_len
    elif max_tokens is None:
        max_tokens = MAX_TOKENS

    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }

def get_default_embedding_payload(model_name, input_text=None):
    """Default payload for embedding models"""
    global random_input_manager

    # Use random input if manager is initialized
    if input_text is None and random_input_manager is not None:
        input_text = random_input_manager.get_embedding_sample()
    elif input_text is None:
        input_text = PROMPT_TEXT  # Fallback

    return {
        "model": model_name,
        "input": input_text,
        "encoding_format": "float"
    }


# Model type mapping for RAG scenario
MODEL_TYPE_MAP = {
    "Chatbot": "llm",
    "VisionProcessor": "vlm",
    "Embedding": "embedding",
}

def get_endpoint_and_payload_for_model(model_name):
    """
    Get the appropriate endpoint and payload based on model name.
    Returns (endpoint, payload) tuple.
    """
    model_type = MODEL_TYPE_MAP.get(model_name, "llm")
    if model_type == "llm":
        return DEFAULT_ENDPOINTS["llm"], get_default_llm_payload(model_name)
    elif model_type == "vlm":
        return DEFAULT_ENDPOINTS["vlm"], get_default_vlm_payload(model_name)
    elif model_type == "embedding":
        return DEFAULT_ENDPOINTS["embedding"], get_default_embedding_payload(model_name)
    else:
        # Default to LLM
        return DEFAULT_ENDPOINTS["llm"], get_default_llm_payload(model_name)


async def send_request(session, request_id, model_name, scenario_name, endpoint=None, payload=None):
    """
    Send a single request, using Streaming mode to measure TTFT and ITL

    Args:
        session: aiohttp ClientSession
        request_id: Unique identifier for the request
        model_name: Name of the model to use
        scenario_name: Name of the test scenario
        endpoint: Custom endpoint URL (defaults to API_ENDPOINT if None)
        payload: Custom payload dict (defaults to LLM payload if None)
    """
    url = endpoint if endpoint is not None else API_ENDPOINT

    # Use custom payload if provided, otherwise use default LLM payload
    if payload is None:
        payload = get_default_llm_payload(model_name)

    # Check if this is a streaming request
    is_streaming = payload.get("stream", False)

    start_time = time.time()
    ttft = 0
    total_latency = 0
    token_count = 0
    status = "FAIL"
    error_reason = ""

    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                if is_streaming:
                    # Streaming response handling (for LLM/VLM)
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
                        error_reason = "Empty response"
                else:
                    # Non-streaming response handling (for embedding)
                    result = await response.json()
                    end_time = time.time()
                    total_latency = (end_time - start_time) * 1000 # ms
                    ttft = total_latency  # For non-streaming, TTFT equals total latency
                    token_count = 1  # Treat as single response
                    status = "SUCCESS"
            else:
                # Use tqdm.write to avoid breaking the progress bar
                tqdm.write(f"Error {response.status}")
                error_reason = f"HTTP {response.status}"

    except asyncio.TimeoutError:
        error_reason = "Timeout"
        tqdm.write(f"Request timeout: {request_id}")
    except aiohttp.ClientConnectorError:
        error_reason = "Connection refused"
        tqdm.write(f"Connection refused: {request_id}")
    except aiohttp.ServerDisconnectedError:
        error_reason = "Server disconnected"
        tqdm.write(f"Server disconnected: {request_id}")
    except Exception as e:
        # Extract brief error reason from exception
        error_type = type(e).__name__
        error_reason = error_type
        tqdm.write(f"Request failed: {error_type}")

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
        "status": status,
        "error_reason": error_reason
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

async def run_scenario_rag(session, results, seed):
    """Scenario 6: RAG Workload (Chatbot:4, VisionProcessor:4, Embedding:2)"""
    print(f"--- Starting Scenario: RAG Workload with seed {seed} ---")

    # RAG models and their weights
    # Chatbot (LLM): 4, VisionProcessor (VLM): 4, Embedding: 2
    rag_models = ["Chatbot", "VisionProcessor", "Embedding"]
    weights = [40, 10, 50]  # Will be normalized by random.choices

    # Initialize random generators
    random_gen = random.Random(seed)
    np_gen = np.random.default_rng(seed)

    start_test = time.time()
    req_id = 0

    with tqdm(total=TEST_DURATION, desc="RAG Progress", unit="s") as pbar:
        last_update_time = start_test

        while time.time() - start_test < TEST_DURATION:
            # Randomly select model based on weights
            model = random_gen.choices(rag_models, weights=weights, k=1)[0]

            # Get appropriate endpoint and payload for the model type
            endpoint, payload = get_endpoint_and_payload_for_model(model)

            task = asyncio.create_task(
                send_request(session, f"RAG-{req_id}", model, "rag", endpoint, payload)
            )
            results.append(task)
            req_id += 1

            # Use Poisson process interval time (closer to real traffic)
            sleep_time = np_gen.exponential(1.0 / TARGET_RPS)
            await asyncio.sleep(sleep_time)

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
    elif test_case == 6:
        case_name = "rag"
        await run_scenario_rag(session, all_tasks, seed)

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
        # Count failed requests
        total_requests = len(df)
        success_count = (df['status'] == 'SUCCESS').sum()
        failed_count = total_requests - success_count

        print(f"\n--- Request Statistics ---")
        print(f"Total Requests: {total_requests}")
        print(f"Success: {success_count} ({success_count/total_requests*100:.1f}%)")
        print(f"Failed: {failed_count} ({failed_count/total_requests*100:.1f}%)")

        # Show failed breakdown by error reason if there are failures
        if failed_count > 0:
            print(f"\n--- Failed Breakdown by Reason ---")
            failed_df = df[df['status'] == 'FAIL']
            error_counts = failed_df['error_reason'].value_counts()
            for reason, count in error_counts.items():
                print(f"  {reason}: {count}")

        print(f"\n--- Latency Statistics ---")
        print(df.groupby("model")["total_latency_ms"].describe([.50,.90,.95,.99]))
    else:
        print("No data collected.")

async def main(args):
    global random_input_manager

    print(f"Running in RPS{TARGET_RPS}")
    print(f"Running in Random Seed: {args.seed}")

    # Initialize tokenizer and random input manager
    print(f"Loading tokenizer: {args.tokenizer_name}")
    try:
        tokenizer = get_tokenizer(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code
        )

        print("Initializing random input generator...")
        random_input_manager = RandomInputManager(args, tokenizer)
        pool_size = random_input_manager._calculate_pool_size()
        print(f"Calculated pool size: {pool_size} (TEST_DURATION={TEST_DURATION}s × TARGET_RPS={TARGET_RPS} × 4)")
        random_input_manager.generate_sample_pool()
        print(f"Generated sample pools: {len(random_input_manager.llm_samples)} LLM, "
              f"{len(random_input_manager.vlm_samples)} VLM, "
              f"{len(random_input_manager.embedding_samples)} Embedding")
    except Exception as e:
        print(f"Warning: Failed to initialize random input manager: {e}")
        print("Falling back to legacy input generation")
        random_input_manager = None


    breakpoint()

    print("input test_case number:")
    print("1. Round Robin")
    print("2. Zipfian (Real Distribution)")
    print("3. Bursty")
    print("4. Run All (Multi-Model)")
    print("5. Single Model Test")
    print("6. RAG Workload (LLM:4, VLM:4, Embedding:2)")
    
    try:
        test_case = int(input().strip())
    except ValueError:
        print("Invalid input")
        return

    if test_case not in [1, 2, 3, 4, 5, 6]:
        print(f"Please enter option 1, 2, 3, 4, 5, or 6")
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

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if test_case == 4:
            scenarios = [1, 2, 3]
            for i, scenario in enumerate(scenarios):
                await run_single_test(session, scenario, args.seed)
                if i < len(scenarios) - 1:
                    print("\nWaiting 10 seconds before next test...")
                    await asyncio.sleep(10)
        else:
            await run_single_test(session, test_case, args.seed, model_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for experiments")

    # Random input generation arguments
    parser.add_argument("--random-input-len", type=int, default=500,
                        help="Number of input tokens for random generation (default: 500)")
    parser.add_argument("--random-output-len", type=int, default=500,
                        help="Number of output tokens for random generation (default: 500)")
    parser.add_argument("--random-range-ratio", type=float, default=0.0,
                        help="Range ratio for input/output length variability [0.0-1.0]. "
                             "0.0 = fixed length, 1.0 = max variability (default: 0.0)")

    # Vision model specific arguments
    parser.add_argument("--random-mm-base-items-per-request", type=int, default=1,
                        help="number of images per VLM request (default: 1)")
    parser.add_argument("--random-mm-num-mm-items-range-ratio", type=float, default=0.0,
                        help="Range ratio for number of images per request (default: 0.0)")
    parser.add_argument("--random-mm-bucket-config", type=str,
                        default="{(256,256,1):0.5,(720,1280,1):0.5}",
                        help="Image bucket config as dict string, e.g., "
                             "'{(256,256,1):0.5,(720,1280,1):0.5}' for 50%% 256x256 "
                             "and 50%% 720x1280 images (default: mixed sizes)")

    # Tokenizer arguments
    parser.add_argument("--tokenizer-name", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace tokenizer name for random input generation "
                             "(default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code when loading tokenizer")

    args = parser.parse_args()

    asyncio.run(main(args))
