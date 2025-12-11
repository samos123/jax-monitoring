from google.cloud import monitoring_v3
from google.auth.exceptions import DefaultCredentialsError
import google.auth
import logging
from typing import Optional
from .config import get_config

_CLIENT: Optional[monitoring_v3.MetricServiceClient] = None
_PROJECT_PATH: Optional[str] = None

logger = logging.getLogger(__name__)

def initialize_client():
    global _CLIENT, _PROJECT_PATH
    config = get_config()
    
    try:
        # Get credentials and project_id if not provided
        credentials, project_id = google.auth.default()
        if config.project_id is None:
            if project_id:
                config.project_id = project_id
            else:
                logger.warning("Project ID not found in credentials and not provided. Metrics might fail to export.")
        
        _CLIENT = monitoring_v3.MetricServiceClient(credentials=credentials)
        if config.project_id:
            _PROJECT_PATH = _CLIENT.common_project_path(config.project_id)
            
    except DefaultCredentialsError:
        logger.error("Could not load Application Default Credentials. JAX Cloud Monitoring will not export metrics.")
    except Exception as e:
        logger.error(f"Error initializing Cloud Monitoring client: {e}")

def get_client() -> Optional[monitoring_v3.MetricServiceClient]:
    return _CLIENT

def get_project_path() -> Optional[str]:
    return _PROJECT_PATH
