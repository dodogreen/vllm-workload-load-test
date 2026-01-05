# Benchmark Configuration Refactoring - 實作指南

## 概述

將 benchmark.py 中的硬編碼參數完全移到 YAML 配置檔案中，實現配置驅動的基準測試系統。

---

## 階段 1: 更新配置結構

### 1.1 更新 `config.py` 中的 `ScenarioConfig`

**檔案**: `config.py` (lines 34-42)

**操作**: 替換整個 `ScenarioConfig` 類別

```python
@dataclass
class ScenarioConfig:
    """Scenario-specific configuration"""

    # Zipfian scenario - 需要權重參數
    zipfian_weights: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])

    # Bursty scenario - 需要 burst 參數
    burst_interval: int = 10
    burst_size: int = 10
    bursty_background_rps_ratio: float = 0.5

    # RAG scenario - 需要權重參數
    rag_weights: List[int] = field(default_factory=lambda: [40, 10, 50])

    # Run All scenario - 決定要執行哪些場景
    run_all_scenarios: List[str] = field(default_factory=lambda: ["round_robin", "zipfian", "bursty"])
```

**說明**:
- 移除所有 `enabled` 欄位（執行時互動式選擇場景）
- Round Robin 和 Single Model 不需要配置參數

---

### 1.2 更新 `BenchmarkConfig.from_yaml()` 方法

**檔案**: `config.py` (lines 86-92)

**操作**: 替換 scenarios 載入邏輯

```python
# Load scenarios
scenario_data = data.get('scenarios', {})
config.scenarios = ScenarioConfig(
    # Zipfian
    zipfian_weights=scenario_data.get('zipfian', {}).get('weights', [0.8, 0.1, 0.1]),

    # Bursty
    burst_interval=scenario_data.get('bursty', {}).get('burst_interval', 10),
    burst_size=scenario_data.get('bursty', {}).get('burst_size', 10),
    bursty_background_rps_ratio=scenario_data.get('bursty', {}).get('background_rps_ratio', 0.5),

    # RAG
    rag_weights=scenario_data.get('rag', {}).get('weights', [40, 10, 50]),

    # Run All
    run_all_scenarios=scenario_data.get('run_all', {}).get('scenarios', ['round_robin', 'zipfian', 'bursty'])
)
```

TODO: 應該直接在設定檔案決定要使用哪一個 scenario 來執行

---

## 階段 2: 建立 YAML 配置檔案

### 2.1 建立 configs 目錄

```bash
mkdir -p configs/
```

### 2.2 建立 `configs/full_benchmark.yaml`

```yaml
# Global benchmark settings
test_duration: 60
target_rps: 4
request_timeout: 2000
seed: 42

# Tokenizer configuration
tokenizer:
  name: "Qwen/Qwen3-0.6B"
  trust_remote_code: false

# Model configurations
models:
  - name: "Chatbot"
    type: "llm"
    endpoint: "http://0.0.0.0:8000/v1/completions"
    input_len: 500
    output_len: 500
    range_ratio: 0.0
    temperature: 0.7
    stream: true
    ignore_eos: true

  - name: "VisionProcessor"
    type: "vlm"
    endpoint: "http://0.0.0.0:8000/v1/chat/completions"
    input_len: 500
    output_len: 500
    temperature: 0.7
    stream: true
    ignore_eos: true
    mm_base_items_per_request: 1
    mm_num_mm_items_range_ratio: 0.0
    mm_bucket_config: "{(256,256,1):0.5,(720,1280,1):0.5}"

  - name: "Embedding"
    type: "embedding"
    endpoint: "http://0.0.0.0:8000/v1/embeddings"
    input_len: 500
    output_len: 500
    encoding_format: "float"

# Scenario configurations
scenarios:
  # Round Robin: 輪詢所有模型
  round_robin:
    enabled: true

  # Zipfian: 真實世界的分佈（需要指定各模型的權重）
  zipfian:
    enabled: true
    weights: [0.8, 0.1, 0.1]  # 對應 models 列表順序

  # Bursty: 突發流量測試
  bursty:
    enabled: true
    burst_interval: 10        # 每 N 秒產生一次 burst
    burst_size: 10            # 每次 burst 的請求數量
    background_rps_ratio: 0.5 # 背景流量 = target_rps * ratio

  # Single Model: 單一模型測試（config.models 必須只有一個模型）
  single_model:
    enabled: false

  # RAG: RAG 工作負載
  rag:
    enabled: false
    weights: [40, 10, 50]     # 對應 models 列表順序

# 注意：只有 enabled: true 的場景會被執行
# 執行順序：round_robin -> zipfian -> bursty -> single_model -> rag
```

### 2.3 建立 `configs/quick_test.yaml`

```yaml
# Quick test configuration
test_duration: 30
target_rps: 2
request_timeout: 2000
seed: 42

tokenizer:
  name: "Qwen/Qwen3-0.6B"
  trust_remote_code: false

models:
  - name: "Chatbot"
    type: "llm"
    endpoint: "http://0.0.0.0:8000/v1/completions"
    input_len: 100
    output_len: 100
    temperature: 0.7
    stream: true
    ignore_eos: true

  - name: "Embedding"
    type: "embedding"
    endpoint: "http://0.0.0.0:8000/v1/embeddings"
    input_len: 100
    output_len: 100
    encoding_format: "float"

scenarios:
  round_robin:
    enabled: true

  zipfian:
    enabled: true
    weights: [0.8, 0.2]

  bursty:
    enabled: false
    burst_interval: 10
    burst_size: 5

  single_model:
    enabled: false

  rag:
    enabled: false
    weights: [50, 50]
```

### 2.4 建立 `configs/rag_workload.yaml`

```yaml
# RAG workload configuration
test_duration: 120
target_rps: 8
request_timeout: 2000
seed: 42

tokenizer:
  name: "Qwen/Qwen3-0.6B"
  trust_remote_code: false

models:
  - name: "Chatbot"
    type: "llm"
    endpoint: "http://0.0.0.0:8000/v1/completions"
    input_len: 800
    output_len: 300
    range_ratio: 0.2
    temperature: 0.7
    stream: true
    ignore_eos: true

  - name: "VisionProcessor"
    type: "vlm"
    endpoint: "http://0.0.0.0:8000/v1/chat/completions"
    input_len: 400
    output_len: 200
    range_ratio: 0.1
    temperature: 0.7
    stream: true
    ignore_eos: true
    mm_base_items_per_request: 2
    mm_num_mm_items_range_ratio: 0.3
    mm_bucket_config: "{(256,256,1):0.3,(512,512,1):0.4,(720,1280,1):0.3}"

  - name: "Embedding"
    type: "embedding"
    endpoint: "http://0.0.0.0:8000/v1/embeddings"
    input_len: 512
    output_len: 1
    range_ratio: 0.1
    encoding_format: "float"

scenarios:
  round_robin:
    enabled: false

  zipfian:
    enabled: false
    weights: [0.4, 0.1, 0.5]

  bursty:
    enabled: false
    burst_interval: 15
    burst_size: 20
    background_rps_ratio: 0.5

  single_model:
    enabled: false

  rag:
    enabled: true
    weights: [40, 10, 50]
```

### 2.5 建立 `configs/single_model_test.yaml`

```yaml
# Single model test configuration
test_duration: 60
target_rps: 4
request_timeout: 2000
seed: 42

tokenizer:
  name: "Qwen/Qwen3-0.6B"
  trust_remote_code: false

# 注意：Single Model scenario 只能有一個模型
models:
  - name: "Chatbot"
    type: "llm"
    endpoint: "http://0.0.0.0:8000/v1/completions"
    input_len: 500
    output_len: 500
    temperature: 0.7
    stream: true
    ignore_eos: true

scenarios:
  round_robin:
    enabled: false

  zipfian:
    enabled: false
    weights: [1.0]

  bursty:
    enabled: false

  single_model:
    enabled: true  # 只測試單一模型

  rag:
    enabled: false
    weights: [100]
```

---

## 階段 3: 重構 benchmark.py

### 3.1 刪除硬編碼的全域變數

**檔案**: `benchmark.py` (lines 18-36)

**操作**: 刪除以下行

```python
# 刪除 line 18
MODELS =  ['Chatbot', 'VisionProcessor', 'Embedding']

# 刪除 lines 20-27
TEST_DURATION = 3
REQUEST_TIMEOUT = 2000
TARGET_RPS = 4

# 刪除 lines 29-33
MODEL_TYPE_ENDPOINTS = {
    "llm": "http://0.0.0.0:8000/v1/completions",
    "vlm": "http://0.0.0.0:8000/v1/chat/completions",
    "embedding": "http://0.0.0.0:8000/v1/embeddings",
}

# 刪除 line 36
random_input_manager = None
```

---

### 3.2 清理 RandomInputManager 死程式碼

**檔案**: `benchmark.py` (lines 153-193)

**操作**: 刪除三個未使用的方法

```python
# 刪除 lines 153-193
def get_llm_sample(self):
    ...

def get_vlm_sample(self):
    ...

def get_embedding_sample(self):
    ...
```

---

### 3.3 修正 Payload 建構函數

**檔案**: `benchmark.py`

#### 3.3.1 更新 `build_llm_payload` (約 line 197)

**替換整個函數**:

```python
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
```

#### 3.3.2 更新 `build_vlm_payload` (約 line 218)

**替換整個函數**:

```python
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
```

#### 3.3.3 更新 `build_embedding_payload` (約 line 273)

**替換整個函數**:

```python
def build_embedding_payload(model_config: ModelConfig, input_text: str):
    """Build payload for embedding models using model-specific config"""
    return {
        "model": model_config.name,
        "input": input_text,
        "encoding_format": model_config.encoding_format,
        **model_config.extra_params
    }
```

#### 3.3.4 更新 `get_endpoint_and_payload_for_model` (約 line 295)

**替換整個函數**:

```python
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
```

---

### 3.4 更新場景函數使用 config

#### 3.4.1 更新 `run_scenario_round_robin`

**更新函數簽名和內容**:

```python
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
```

#### 3.4.2 更新 `run_scenario_zipfian`

**更新函數簽名和內容** (重點: 使用 config.scenarios.zipfian_weights):

```python
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
```

#### 3.4.3 更新 `run_scenario_bursty`

**更新函數簽名和內容** (重點: 使用 config.scenarios.burst_* 參數):

```python
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
```

#### 3.4.4 更新 `run_scenario_single_model`

**更新函數簽名和內容** (重點: 從 config.models 讀取，必須只有一個模型):

```python
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
```

#### 3.4.5 更新 `run_scenario_rag`

**更新函數簽名和內容** (重點: 使用 config.scenarios.rag_weights):

```python
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
```

---

### 3.5 更新 `run_single_test` 函數

**更新函數簽名和邏輯** (移除 model_name 參數):

```python
async def run_single_test(session, scenario_name: str, config: BenchmarkConfig,
                          random_input_manager: RandomInputManager):
    """Run a single test scenario

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
    filename = f"benchmark_results_{scenario_name}.csv"
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
```

---

### 3.6 重寫 `main()` 函數

**完全替換 main() 函數和 __main__ 區塊** (lines 634-741):

**移除所有互動式選擇，直接根據 config 中的 enabled 參數執行場景**

```python
async def main(config_path: str):
    # Load configuration from YAML
    print(f"Loading configuration from: {config_path}")
    config = BenchmarkConfig.from_yaml(config_path)

    print(f"\n=== Benchmark Configuration ===")
    print(f"Test Duration: {config.test_duration}s")
    print(f"Target RPS: {config.target_rps}")
    print(f"Request Timeout: {config.request_timeout}s")
    print(f"Random Seed: {config.seed}")
    print(f"Models: {[m.name for m in config.models]}")

    # Initialize tokenizer and random input manager
    print(f"\nLoading tokenizer: {config.tokenizer_name}")
    try:
        tokenizer = get_tokenizer(
            config.tokenizer_name,
            trust_remote_code=config.trust_remote_code
        )

        print("Initializing random input generator...")
        random_input_manager = RandomInputManager(config, tokenizer)
        random_input_manager.generate_sample_pool()
    except Exception as e:
        print(f"Error: Failed to initialize random input manager: {e}")
        return

    # Determine which scenarios to run based on config
    scenarios_to_run = []
    if config.scenarios.round_robin_enabled:
        scenarios_to_run.append("round_robin")
    if config.scenarios.zipfian_enabled:
        scenarios_to_run.append("zipfian")
    if config.scenarios.bursty_enabled:
        scenarios_to_run.append("bursty")
    if config.scenarios.single_model_enabled:
        scenarios_to_run.append("single_model")
    if config.scenarios.rag_enabled:
        scenarios_to_run.append("rag")

    if not scenarios_to_run:
        print("\nWarning: No scenarios enabled in config. Please enable at least one scenario.")
        return

    print(f"\n=== Enabled Scenarios ===")
    for i, scenario in enumerate(scenarios_to_run, 1):
        print(f"{i}. {scenario}")

    # Run all enabled scenarios
    timeout = aiohttp.ClientTimeout(total=config.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, scenario_name in enumerate(scenarios_to_run):
            print(f"\n{'='*60}")
            print(f"Running scenario {i+1}/{len(scenarios_to_run)}: {scenario_name}")
            print(f"{'='*60}")

            await run_single_test(session, scenario_name, config, random_input_manager)

            # Wait between scenarios (except for the last one)
            if i < len(scenarios_to_run) - 1:
                print("\nWaiting 10 seconds before next scenario...")
                await asyncio.sleep(10)

    print(f"\n{'='*60}")
    print("All scenarios completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <config_yaml_path>")
        print("Example: python benchmark.py configs/full_benchmark.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    if not config_path.endswith(('.yaml', '.yml')):
        print("Error: Config file must be a YAML file (.yaml or .yml)")
        sys.exit(1)

    asyncio.run(main(config_path))
```

**刪除**: lines 707-741 所有 argparse 相關程式碼

---

## 測試步驟

### 1. 快速測試 (Round Robin + Zipfian)

```bash
python benchmark.py configs/quick_test.yaml
# 自動執行 enabled 的場景：round_robin 和 zipfian
```

### 2. 完整測試 (多種場景)

```bash
python benchmark.py configs/full_benchmark.yaml
# 自動執行所有 enabled 的場景：round_robin, zipfian, bursty
```

### 3. 測試 Single Model

```bash
python benchmark.py configs/single_model_test.yaml
# 自動執行 single_model 場景
# 注意：config 中只能有一個模型
```

### 4. 測試 RAG 工作負載

```bash
python benchmark.py configs/rag_workload.yaml
# 自動執行 rag 場景
```

### 5. 自訂場景組合

修改 config 中的 enabled 參數來決定要執行哪些場景：

```yaml
scenarios:
  round_robin:
    enabled: true   # 執行
  zipfian:
    enabled: false  # 跳過
  bursty:
    enabled: true   # 執行
  single_model:
    enabled: false  # 跳過
  rag:
    enabled: false  # 跳過
```

---

## 驗證清單

- [ ] config.py 的 ScenarioConfig 已更新（加回所有 enabled 參數）
- [ ] config.py 的 from_yaml() 已更新（載入所有 enabled 參數）
- [ ] configs/ 目錄已建立
- [ ] 4 個 YAML 配置檔案已建立（full, quick, rag, single_model）
- [ ] benchmark.py 硬編碼全域變數已刪除 (lines 18-36)
- [ ] RandomInputManager 死程式碼已刪除 (lines 153-193)
- [ ] 3 個 payload 建構函數已更新
- [ ] get_endpoint_and_payload_for_model 已更新
- [ ] 5 個場景函數已更新使用 config
- [ ] run_scenario_single_model 從 config.models 讀取（檢查只有一個模型）
- [ ] run_single_test 已更新（移除 model_name 參數，改用 scenario_name）
- [ ] main() 函數已完全重寫（移除互動式選擇，根據 enabled 自動執行）
- [ ] argparse 程式碼已刪除
- [ ] 所有測試場景可正常執行

---

## 使用方式對比

### 舊方式 (已移除)
```bash
# 需要命令列參數
python benchmark.py --seed 42 --random-input-len 500 --random-output-len 500

# 需要互動式選擇場景
input test_case number:
1. Round Robin
2. Zipfian (Real Distribution)
...
> 1
```

### 新方式
```bash
# 所有配置都在 YAML 中，直接執行
python benchmark.py configs/full_benchmark.yaml

# 自動執行所有 enabled 的場景，無需互動
```

---

## 注意事項

1. **權重長度必須匹配模型數量**: Zipfian 和 RAG 的 weights 長度必須等於 models 列表長度
2. **模型名稱必須唯一**: config 中的模型名稱不可重複
3. **endpoint 格式**: 確保 endpoint URL 正確且可訪問
4. **timeout 單位**: request_timeout 單位為毫秒 (ms)
5. **背景流量比例**: bursty_background_rps_ratio 應介於 0 到 1 之間
6. **Single Model 限制**: 當 single_model_enabled: true 時，config.models 必須只有一個模型
7. **場景執行順序**: 按照 round_robin -> zipfian -> bursty -> single_model -> rag 的順序執行 enabled 的場景
8. **無互動式操作**: 所有場景設定都在 YAML 中，執行時無需用戶輸入

---

## 常見錯誤處理

### 錯誤: "Zipfian weights length must match number of models"
**解決**: 確保 scenarios.zipfian.weights 列表長度等於 models 列表長度

### 錯誤: "Unknown model: xxx"
**解決**: 檢查 model_name 是否存在於 config.models 中

### 錯誤: "Config file must be a YAML file"
**解決**: 確保配置檔案副檔名為 .yaml 或 .yml

### 錯誤: "Failed to initialize random input manager"
**解決**: 檢查 tokenizer_name 是否正確，是否需要設定 trust_remote_code

### 錯誤: "Single model scenario requires exactly 1 model in config"
**解決**: 使用 single_model 場景時，config.models 只能有一個模型。請建立專用的 single_model_test.yaml 配置檔

### 錯誤: "No scenarios enabled in config"
**解決**: 至少啟用一個場景（設定 enabled: true）

---

## 完成！

執行完所有步驟後，你的 benchmark 系統將完全由 YAML 配置檔案驅動，無需修改程式碼即可調整測試參數。
