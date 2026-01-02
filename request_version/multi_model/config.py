"""
Configuration classes for benchmark
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any
import yaml


@dataclass
class ModelConfig:
    """Per-model configuration"""
    name: str
    type: str  # llm, vlm, embedding
    endpoint: str
    input_len: int
    output_len: int
    range_ratio: float = 0.0
    temperature: float = 0.7
    stream: bool = True
    ignore_eos: bool = True

    # VLM-specific parameters
    mm_base_items_per_request: int = 1
    mm_num_mm_items_range_ratio: float = 0.0
    mm_bucket_config: str = "{(256,256,1):0.5,(720,1280,1):0.5}"

    # Embedding-specific parameters
    encoding_format: str = "float"

    # Extra parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


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

    # Scenario enabled flags
    round_robin_enabled: bool = False
    zipfian_enabled: bool = False
    bursty_enabled: bool = False
    single_model_enabled: bool = False
    rag_enabled: bool = False


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration"""
    test_duration: int = 60
    target_rps: int = 4
    request_timeout: int = 1800  # timeout in seconds
    seed: int = 42

    tokenizer_name: str = "Qwen/Qwen3-0.6B"
    trust_remote_code: bool = False

    models: List[ModelConfig] = field(default_factory=list)
    model_map: Dict[str, ModelConfig] = field(default_factory=dict)

    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BenchmarkConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        # Load global settings
        config.test_duration = data.get('test_duration', 60)
        config.target_rps = data.get('target_rps', 4)
        config.request_timeout = data.get('request_timeout', 2000)
        config.seed = data.get('seed', 42)

        # Load tokenizer config
        tokenizer_cfg = data.get('tokenizer', {})
        config.tokenizer_name = tokenizer_cfg.get('name', 'Qwen/Qwen3-0.6B')
        config.trust_remote_code = tokenizer_cfg.get('trust_remote_code', False)

        # Load models
        for model_data in data.get('models', []):
            model_cfg = ModelConfig(**model_data)
            config.models.append(model_cfg)
            config.model_map[model_cfg.name] = model_cfg

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

            # Enabled flags
            round_robin_enabled=scenario_data.get('round_robin', {}).get('enabled', False),
            zipfian_enabled=scenario_data.get('zipfian', {}).get('enabled', False),
            bursty_enabled=scenario_data.get('bursty', {}).get('enabled', False),
            single_model_enabled=scenario_data.get('single_model', {}).get('enabled', False),
            rag_enabled=scenario_data.get('rag', {}).get('enabled', False)
        )

        return config
