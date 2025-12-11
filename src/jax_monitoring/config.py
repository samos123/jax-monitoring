import dataclasses
from typing import Optional, Any
import urllib.error
import urllib.request

@dataclasses.dataclass
class Config:
    project_id: Optional[str] = None
    metric_prefix: str = "custom.googleapis.com/jax/monitoring"
    monitored_resource_type: str = "global"
    monitored_resource_labels: dict[str, str] = dataclasses.field(default_factory=dict)
    job_name: str = "jax_job"

def _get_metadata(path: str) -> Optional[str]:
    """Fetch metadata from GCE metadata server."""
    url = f"http://metadata.google.internal/computeMetadata/v1/{path}"
    req = urllib.request.Request(url)
    req.add_header("Metadata-Flavor", "Google")
    
    try:
        # Short timeout to avoid blocking startup in non-GCE environments
        with urllib.request.urlopen(req, timeout=0.5) as response:
            return response.read().decode("utf-8").strip()
    except (urllib.error.URLError, Exception):
        return None

# Global cache for GCE config to avoid repeated metadata server calls
_GCE_CONFIG_CACHE: Optional[dict[str, Any]] = None

def _detect_gce_config() -> dict[str, Any]:
    """Detect GCE configuration from metadata server."""
    global _GCE_CONFIG_CACHE
    if _GCE_CONFIG_CACHE is not None:
        return _GCE_CONFIG_CACHE

    config_updates = {}
    
    # Check if we are on GCE by trying to fetch instance ID
    instance_id = _get_metadata("instance/id")
    if instance_id:
        config_updates["monitored_resource_type"] = "gce_instance"
        
        # Get zone (format: projects/PROJECT_NUM/zones/ZONE)
        # We need to extract just the zone name
        zone_full = _get_metadata("instance/zone")
        zone = zone_full.split("/")[-1] if zone_full else "unknown"
        
        # Get instance name as requested by user
        instance_name = _get_metadata("instance/name") 
        
        config_updates["monitored_resource_labels"] = {
            "instance_id": instance_id,
            "zone": zone
        }
        if instance_name:
            config_updates["monitored_resource_labels"]["instance_name"] = instance_name
    
    _GCE_CONFIG_CACHE = config_updates
    return config_updates

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
    
    if gce_config := _detect_gce_config():
        for key, value in gce_config.items():
            setattr(_GLOBAL_CONFIG, key, value)

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
