import asyncio
import aiohttp
import time
import random
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm

# vLLM random input generation
from vllm.benchmarks.datasets import RandomDataset, RandomMultiModalDataset
from vllm.transformers_utils.tokenizer import get_tokenizer

# Configuration classes
from config import BenchmarkConfig, ModelConfig, ScenarioConfig


class RandomInputManager:
    """Manages random input generation using vLLM's RandomDataset classes."""

    def __init__(self, config: BenchmarkConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Initialize datasets with seed
        self.text_dataset = RandomDataset(random_seed=config.seed)
        self.mm_dataset = RandomMultiModalDataset(random_seed=config.seed)

        # Model-based sample pools (not type-based)
        self.model_samples = {}
        self.model_indices = {}

        # Initialize pools and indices for each model
        for model_cfg in config.models:
            self.model_samples[model_cfg.name] = []
            self.model_indices[model_cfg.name] = 0

    def _calculate_pool_size(self):
        """Calculate dynamic pool size based on test parameters."""
        return self.config.test_duration * self.config.target_rps * 1

    def _parse_bucket_config(self, config_str):
        """Parse bucket config string to dict."""
        import ast
        try:
            return ast.literal_eval(config_str)
        except (ValueError, SyntaxError) as e:
            # Fallback to default on parsing error
            print(f"Warning: Failed to parse bucket config '{config_str}': {e}. Using default config.")
            return {(256, 256, 1): 0.5, (720, 1280, 1): 0.5}

    def generate_sample_pool(self):
        """Pre-generate sample pools for all models using model-specific configurations."""
        pool_size = self._calculate_pool_size()

        for model_cfg in self.config.models:
            if model_cfg.type == "llm":
                # Generate text samples for LLM
                samples = self.text_dataset.sample(
                    tokenizer=self.tokenizer,
                    num_requests=pool_size,
                    input_len=model_cfg.input_len,
                    output_len=model_cfg.output_len,
                    range_ratio=model_cfg.range_ratio,
                    batchsize=1
                )

            elif model_cfg.type == "vlm":
                # Generate multimodal samples for VLM
                bucket_config = self._parse_bucket_config(model_cfg.mm_bucket_config)
                samples = self.mm_dataset.sample(
                    tokenizer=self.tokenizer,
                    num_requests=pool_size,
                    input_len=model_cfg.input_len,
                    output_len=model_cfg.output_len,
                    range_ratio=model_cfg.range_ratio,
                    base_items_per_request=model_cfg.mm_base_items_per_request,
                    num_mm_items_range_ratio=model_cfg.mm_num_mm_items_range_ratio,
                    bucket_config=bucket_config,
                    limit_mm_per_prompt={"image": 255, "video": 0}
                )

            elif model_cfg.type == "embedding":
                # Generate text samples for embedding
                samples = self.text_dataset.sample(
                    tokenizer=self.tokenizer,
                    num_requests=pool_size,
                    input_len=model_cfg.input_len,
                    output_len=model_cfg.output_len,
                    range_ratio=model_cfg.range_ratio,
                    batchsize=1
                )

            else:
                raise ValueError(f"Unknown model type: {model_cfg.type} for model: {model_cfg.name}")

            random.shuffle(samples)
            self.model_samples[model_cfg.name] = samples

        print(f"Generated sample pools:")
        for model_name, samples in self.model_samples.items():
            print(f"  {model_name}: {len(samples)} samples")

    def get_sample(self, model_name: str):
        """Get next sample for the specified model.

        Returns:
            For LLM/Embedding: prompt string
            For VLM: (prompt, multi_modal_data) tuple
        """
        if model_name not in self.model_samples:
            raise ValueError(f"Unknown model: {model_name}")

        samples = self.model_samples[model_name]
        if not samples:
            return None

        idx = self.model_indices[model_name]
        sample = samples[idx]

        # Increment and wrap around (循環使用，不 reshuffle)
        self.model_indices[model_name] = (idx + 1) % len(samples)

        # Get model config to determine return type
        model_cfg = self.config.model_map[model_name]

        if model_cfg.type == "vlm":
            return sample.prompt, sample.multi_modal_data
        else:  # llm or embedding
            return sample.prompt


# Payload builders for different model types
def build_llm_payload(model_config: ModelConfig, prompt: str):
    """Build payload for LLM models using model-specific config"""
    return {
        "model": model_config.name,
        "prompt": prompt,
        "max_tokens": model_config.output_len,
        "temperature": model_config.temperature,
        "stream": model_config.stream,
        "ignore_eos": model_config.ignore_eos,
        **model_config.extra_params
    }

def build_vlm_payload(model_config: ModelConfig, prompt: str, mm_data: dict):
    """Build payload for VLM models using model-specific config"""
    content = []

    # Add all images from multi_modal_data
    if mm_data and isinstance(mm_data, list):
        for item in mm_data:
            if item.get("type") == "image_url":
                content.append(item)

    # Add text prompt
    content.append({"type": "text", "text": prompt})

    return {
        "model": model_config.name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": model_config.output_len,
        "temperature": model_config.temperature,
        "stream": model_config.stream,
        "ignore_eos": model_config.ignore_eos,
        **model_config.extra_params
    }

def build_embedding_payload(model_config: ModelConfig, input_text: str):
    """Build payload for embedding models using model-specific config"""
    return {
        "model": model_config.name,
        "input": input_text,
        "encoding_format": model_config.encoding_format,
        **model_config.extra_params
    }


def get_endpoint_and_payload_for_model(model_name: str, config: BenchmarkConfig,
                                        random_input_manager: RandomInputManager):
    """Get endpoint and payload based on model name from config"""
    if model_name not in config.model_map:
        raise ValueError(f"Unknown model: {model_name}. Valid models: {list(config.model_map.keys())}")

    model_config = config.model_map[model_name]

    if model_config.type == "llm":
        prompt = random_input_manager.get_sample(model_name)
        payload = build_llm_payload(model_config, prompt)
    elif model_config.type == "vlm":
        prompt, mm_data = random_input_manager.get_sample(model_name)
        payload = build_vlm_payload(model_config, prompt, mm_data)
    elif model_config.type == "embedding":
        prompt = random_input_manager.get_sample(model_name)
        payload = build_embedding_payload(model_config, prompt)
    else:
        raise ValueError(f"Unknown model type: {model_config.type} for model: {model_name}")

    return model_config.endpoint, payload


async def send_request(session, request_id, model_name, scenario_name, endpoint=None, payload=None):
    """
    Send a single request, using Streaming mode to measure TTFT and ITL

    Args:
        session: aiohttp ClientSession
        request_id: Unique identifier for the request
        model_name: Name of the model to use
        scenario_name: Name of the test scenario
        endpoint: Custom endpoint URL (uses get_endpoint_and_payload_for_model if None)
        payload: Custom payload dict (uses get_endpoint_and_payload_for_model if None)
    """
    # Get endpoint and payload if not provided
    if endpoint is None or payload is None:
        endpoint, payload = get_endpoint_and_payload_for_model(model_name)

    url = endpoint

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

async def run_scenario_round_robin(session, results, config: BenchmarkConfig,
                                    random_input_manager: RandomInputManager):
    """Scenario 1: Extreme Switching (Round Robin)"""
    print(f"--- Starting Scenario: Round Robin (Worst Case) ---")
    start_test = time.time()
    req_id = 0

    # Get model list from config
    models = [m.name for m in config.models]

    with tqdm(total=config.test_duration, desc="Round Robin Progress", unit="s") as pbar:
        last_update_time = start_test

        while time.time() - start_test < config.test_duration:
            model_name = models[req_id % len(models)]

            # Get endpoint and payload using config
            endpoint, payload = get_endpoint_and_payload_for_model(model_name, config, random_input_manager)

            task = asyncio.create_task(
                send_request(session, f"RR-{req_id}", model_name, "round_robin", endpoint, payload)
            )
            results.append(task)
            req_id += 1

            await asyncio.sleep(1.0 / config.target_rps)

            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_scenario_zipfian(session, results, config: BenchmarkConfig,
                                random_input_manager: RandomInputManager):
    """Scenario 2: Real Distribution (Zipfian / Weighted)"""
    print(f"--- Starting Scenario: Zipfian (Real World) with seed {config.seed} ---")

    # Get models and weights from config
    models = [m.name for m in config.models]
    weights = config.scenarios.zipfian_weights

    # Validate weights length
    if len(weights) != len(models):
        raise ValueError(f"Zipfian weights length ({len(weights)}) must match number of models ({len(models)})")

    # Initialize random generators
    random_gen = random.Random(config.seed)
    np_gen = np.random.default_rng(config.seed)

    start_test = time.time()
    req_id = 0

    with tqdm(total=config.test_duration, desc="Zipfian Progress", unit="s") as pbar:
        last_update_time = start_test

        while time.time() - start_test < config.test_duration:
            # Randomly select based on weights
            model_name = random_gen.choices(models, weights=weights, k=1)[0]

            # Get endpoint and payload using config
            endpoint, payload = get_endpoint_and_payload_for_model(model_name, config, random_input_manager)

            task = asyncio.create_task(
                send_request(session, f"ZIPF-{req_id}", model_name, "zipfian", endpoint, payload)
            )
            results.append(task)
            req_id += 1

            # Use Poisson process interval time
            sleep_time = np_gen.exponential(1.0 / config.target_rps)
            await asyncio.sleep(sleep_time)

            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_scenario_bursty(session, results, config: BenchmarkConfig,
                               random_input_manager: RandomInputManager):
    """Scenario 3: Bursty Traffic (Bursty)"""
    print(f"--- Starting Scenario: Bursty (Stress Test) with seed {config.seed} ---")

    # Get models from config
    models = [m.name for m in config.models]

    # Get bursty parameters from config
    burst_interval = config.scenarios.burst_interval
    burst_size = config.scenarios.burst_size
    background_rps = config.target_rps * config.scenarios.bursty_background_rps_ratio

    # Initialize random generator
    random_gen = random.Random(config.seed)

    start_test = time.time()
    req_id = 0

    with tqdm(total=config.test_duration, desc="Bursty Progress", unit="s") as pbar:
        last_update_time = start_test

        while time.time() - start_test < config.test_duration:
            current_elapsed = time.time() - start_test

            # Generate burst traffic at configured interval
            if int(current_elapsed) % burst_interval == 0 and int(current_elapsed) > 0:
                tqdm.write(f"!!! BURST INCOMING at {int(current_elapsed)}s !!!")

                # Bursty traffic is mixed
                burst_models = random_gen.choices(models, k=burst_size)

                for model_name in burst_models:
                    endpoint, payload = get_endpoint_and_payload_for_model(model_name, config, random_input_manager)
                    task = asyncio.create_task(
                        send_request(session, f"BURST-{req_id}", model_name, "bursty", endpoint, payload)
                    )
                    results.append(task)
                    req_id += 1

                await asyncio.sleep(1)
            else:
                # Background traffic
                model_name = random_gen.choice(models)
                endpoint, payload = get_endpoint_and_payload_for_model(model_name, config, random_input_manager)
                task = asyncio.create_task(
                    send_request(session, f"BG-{req_id}", model_name, "bursty", endpoint, payload)
                )
                results.append(task)
                req_id += 1
                await asyncio.sleep(1.0 / background_rps)

            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def run_scenario_rag(session, results, config: BenchmarkConfig,
                            random_input_manager: RandomInputManager):
    """Scenario 6: RAG Workload"""
    print(f"--- Starting Scenario: RAG Workload with seed {config.seed} ---")

    # Get RAG models and weights from config
    models = [m.name for m in config.models]
    weights = config.scenarios.rag_weights

    # Validate weights length
    if len(weights) != len(models):
        raise ValueError(f"RAG weights length ({len(weights)}) must match number of models ({len(models)})")

    # Initialize random generators
    random_gen = random.Random(config.seed)
    np_gen = np.random.default_rng(config.seed)

    start_test = time.time()
    req_id = 0

    with tqdm(total=config.test_duration, desc="RAG Progress", unit="s") as pbar:
        last_update_time = start_test

        while time.time() - start_test < config.test_duration:
            # Randomly select model based on weights
            model_name = random_gen.choices(models, weights=weights, k=1)[0]

            # Get appropriate endpoint and payload for the model
            endpoint, payload = get_endpoint_and_payload_for_model(model_name, config, random_input_manager)

            task = asyncio.create_task(
                send_request(session, f"RAG-{req_id}", model_name, "rag", endpoint, payload)
            )
            results.append(task)
            req_id += 1

            # Use Poisson process interval time
            sleep_time = np_gen.exponential(1.0 / config.target_rps)
            await asyncio.sleep(sleep_time)

            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now


async def run_scenario_single_model(session, results, config: BenchmarkConfig,
                                    random_input_manager: RandomInputManager):
    """Scenario 5: Single Model Test (Baseline)"""

    # Validate: must have exactly one model
    if len(config.models) != 1:
        raise ValueError(
            f"Single model scenario requires exactly 1 model in config, but got {len(config.models)} models. "
            f"Models: {[m.name for m in config.models]}"
        )

    model_name = config.models[0].name
    print(f"--- Starting Scenario: Single Model Test (Model: {model_name}) ---")

    start_test = time.time()
    req_id = 0

    with tqdm(total=config.test_duration, desc=f"Single Model ({model_name}) Progress", unit="s") as pbar:
        last_update_time = start_test

        while time.time() - start_test < config.test_duration:
            endpoint, payload = get_endpoint_and_payload_for_model(model_name, config, random_input_manager)
            task = asyncio.create_task(
                send_request(session, f"SINGLE-{req_id}", model_name, "single_model", endpoint, payload)
            )
            results.append(task)
            req_id += 1

            await asyncio.sleep(1.0 / config.target_rps)

            now = time.time()
            pbar.update(now - last_update_time)
            last_update_time = now

async def execute_scenario(session, scenario_name: str, config: BenchmarkConfig,
                           random_input_manager: RandomInputManager):
    """Execute a single benchmark scenario and collect results

    Args:
        session: aiohttp session
        scenario_name: Name of scenario (round_robin, zipfian, bursty, single_model, rag)
        config: BenchmarkConfig object
        random_input_manager: RandomInputManager instance
    """
    all_tasks = []

    if scenario_name == "round_robin":
        await run_scenario_round_robin(session, all_tasks, config, random_input_manager)
    elif scenario_name == "zipfian":
        await run_scenario_zipfian(session, all_tasks, config, random_input_manager)
    elif scenario_name == "bursty":
        await run_scenario_bursty(session, all_tasks, config, random_input_manager)
    elif scenario_name == "single_model":
        await run_scenario_single_model(session, all_tasks, config, random_input_manager)
    elif scenario_name == "rag":
        await run_scenario_rag(session, all_tasks, config, random_input_manager)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    print(f"\nAll requests dispatched for {scenario_name}. Waiting for pending responses...")

    responses = []
    for f in tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc="Collecting Responses", unit="req"):
        responses.append(await f)

    df = pd.DataFrame(responses)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{scenario_name}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Done! Results saved to {filename}")

    print(f"\n=== Quick Summary ({scenario_name}) ===")
    if not df.empty:
        # Count failed requests
        total_requests = len(df)
        success_count = (df['status'] == 'SUCCESS').sum()
        failed_count = total_requests - success_count

        print(f"\n--- Request Statistics ---")
        print(f"Total Requests: {total_requests}")
        print(f"Success: {success_count} ({success_count/total_requests*100:.1f}%)")
        print(f"Failed: {failed_count} ({failed_count/total_requests*100:.1f}%)")

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

async def main(config_file: str):
    """Main function to run benchmark scenarios based on YAML configuration

    Args:
        config_file: Path to YAML configuration file
    """
    # Load configuration from YAML
    print(f"Loading configuration from: {config_file}")
    config = BenchmarkConfig.from_yaml(config_file)

    print(f"\n=== Benchmark Configuration ===")
    print(f"Test Duration: {config.test_duration}s")
    print(f"Target RPS: {config.target_rps}")
    print(f"Request Timeout: {config.request_timeout}s")
    print(f"Random Seed: {config.seed}")
    print(f"\nModels ({len(config.models)}):")
    for model in config.models:
        print(f"  - {model.name} ({model.type}): input={model.input_len}, output={model.output_len}")

    # Initialize tokenizer and random input manager
    print(f"\nLoading tokenizer: {config.tokenizer_name}")
    try:
        tokenizer = get_tokenizer(
            config.tokenizer_name,
            trust_remote_code=config.trust_remote_code
        )

        print("Initializing random input generator...")
        random_input_manager = RandomInputManager(config, tokenizer)
        pool_size = random_input_manager._calculate_pool_size()
        print(f"Calculated pool size: {pool_size} samples per model")
        random_input_manager.generate_sample_pool()

    except Exception as e:
        print(f"Error: Failed to initialize random input manager: {e}")
        return

    # Collect enabled scenarios
    enabled_scenarios = []
    if config.scenarios.round_robin_enabled:
        enabled_scenarios.append("round_robin")
    if config.scenarios.zipfian_enabled:
        enabled_scenarios.append("zipfian")
    if config.scenarios.bursty_enabled:
        enabled_scenarios.append("bursty")
    if config.scenarios.single_model_enabled:
        enabled_scenarios.append("single_model")
    if config.scenarios.rag_enabled:
        enabled_scenarios.append("rag")

    if not enabled_scenarios:
        print("\nWarning: No scenarios are enabled in the configuration!")
        print("Please enable at least one scenario in the YAML config file.")
        return

    print(f"\nEnabled scenarios: {', '.join(enabled_scenarios)}")

    # Run enabled scenarios
    timeout = aiohttp.ClientTimeout(total=config.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, scenario_name in enumerate(enabled_scenarios):
            print(f"\n{'='*60}")
            print(f"Running scenario {i+1}/{len(enabled_scenarios)}: {scenario_name}")
            print(f"{'='*60}")

            try:
                await execute_scenario(session, scenario_name, config, random_input_manager)
            except Exception as e:
                print(f"Error running scenario {scenario_name}: {e}")
                import traceback
                traceback.print_exc()

            # Wait between scenarios (except after the last one)
            if i < len(enabled_scenarios) - 1:
                wait_time = 10
                print(f"\nWaiting {wait_time} seconds before next scenario...")
                await asyncio.sleep(wait_time)

    print(f"\n{'='*60}")
    print("All scenarios completed!")
    print(f"{'='*60}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-model benchmark tool for vLLM workloads"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    asyncio.run(main(args.config))
