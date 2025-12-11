import dataclasses
from typing import Optional, Any

@dataclasses.dataclass
class Config:
    project_id: Optional[str] = None
    metric_prefix: str = "custom.googleapis.com/jax/monitoring"
    monitored_resource_type: str = "global"
    monitored_resource_labels: dict[str, str] = dataclasses.field(default_factory=dict)
    job_name: str = "jax_job"

# Initialize with default values
_GLOBAL_CONFIG = Config()

def get_config() -> Config:
    return _GLOBAL_CONFIG

def replace_config(config: Config) -> None:
    """Replace the global configuration. Useful for initialization in worker processes."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config


def set_config(
    project_id: Optional[str] = None,
    metric_prefix: Optional[str] = None,
    monitored_resource_type: Optional[str] = None,
    monitored_resource_labels: Optional[dict[str, str]] = None,
    job_name: Optional[str] = None,
):
    global _GLOBAL_CONFIG
    
    if project_id is not None:
        _GLOBAL_CONFIG.project_id = project_id
    if metric_prefix is not None:
        _GLOBAL_CONFIG.metric_prefix = metric_prefix
    if monitored_resource_type is not None:
        _GLOBAL_CONFIG.monitored_resource_type = monitored_resource_type
    if monitored_resource_labels is not None:
        _GLOBAL_CONFIG.monitored_resource_labels = monitored_resource_labels
    if job_name is not None:
        _GLOBAL_CONFIG.job_name = job_name
