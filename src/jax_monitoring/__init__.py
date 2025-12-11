from typing import Optional

from .config import set_config
from .listeners import register_listeners
from .gcp_utils import initialize_client

def init(
    project_id: Optional[str] = None,
    metric_prefix: str = "custom.googleapis.com/jax/monitoring",
    monitored_resource_type: Optional[str] = None,
    monitored_resource_labels: Optional[dict[str, str]] = None,
    job_name: str = "jax_job",
):
    """
    Initialize JAX Cloud Monitoring integration.
    
    Args:
        project_id: Google Cloud Project ID. If None, inferred from environment.
        metric_prefix: Prefix for Cloud Monitoring metrics.
        monitored_resource_type: Monitored Resource type (e.g., 'gce_instance', 'global').
                                 If None, attempts to auto-detect.
        monitored_resource_labels: Labels for the Monitored Resource.
        job_name: Name of the job to associate metrics with. Defaults to 'jax_job'.
    """
    # 1. Configure global settings
    set_config(
        project_id=project_id,
        metric_prefix=metric_prefix,
        monitored_resource_type=monitored_resource_type,
        monitored_resource_labels=monitored_resource_labels,
        job_name=job_name,
    )

    # 2. Initialize Cloud Monitoring Client (and detect metadata if needed)
    initialize_client()

    # 3. Register JAX monitoring listeners
    register_listeners()
