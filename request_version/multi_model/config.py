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
    # Bursty scenario
    burst_interval: int = 10
    burst_size: int = 10

    # RAG scenario
    rag_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration"""
    test_duration: int = 60
    target_rps: int = 4
    request_timeout: int = 2000
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
            burst_interval=scenario_data.get('burst_interval', 10),
            burst_size=scenario_data.get('burst_size', 10),
            rag_weights=scenario_data.get('rag_weights', {})
        )

        return config
