import jax.monitoring
import time
import logging
import multiprocessing
import multiprocessing.synchronize
from typing import Any, Optional
from google.cloud import monitoring_v3
from google.api import metric_pb2
from google.protobuf import timestamp_pb2
from .config import get_config, set_config, replace_config, Config
from .gcp_utils import initialize_client, get_client, get_project_path

logger = logging.getLogger(__name__)

# Queue for asynchronous metric export
# We initialize it lazily in register_listeners to avoid creation on import
_METRIC_QUEUE: Optional[multiprocessing.Queue] = None
_WORKER_PROCESS: Optional[multiprocessing.Process] = None
_STOP_EVENT: Optional[multiprocessing.synchronize.Event] = None


def _metric_worker(config: Config, queue: multiprocessing.Queue, stop_event: multiprocessing.synchronize.Event):
    """Background worker to export metrics. Runs in a separate process."""
    # Re-initialize configuration and client in this process to ensure they are available in the worker process
    replace_config(config)
    initialize_client()
    
    logger.info("Worker process started")
    
    while not stop_event.is_set():
        client = get_client()
        project_path = get_project_path()
        
        if not client or not project_path:
            # Wait a bit and retry
            time.sleep(0.1)
            continue

        try:
            # Prepare batch
            batch = []
            try:
                # Drain queue up to 200 items (sync/non-blocking pull)
                for _ in range(200):
                    batch.append(queue.get_nowait())
            except multiprocessing.queues.Empty:
                pass

            # Export if we have data
            if batch:
                series_list = []
                for metric_data in batch:
                    series = _create_time_series(metric_data)
                    if series:
                        series_list.append(series)

                if series_list:
                    try:
                        client.create_time_series(
                            request={"name": project_path, "time_series": series_list}
                        )
                    except Exception as e:
                        logger.error(f"Failed to export metrics: {e}")
            
            # Wait 5 seconds before next cycle (rate limiting)
            stop_event.wait(timeout=5)
            
        except Exception as e:
            logger.error(f"Error in metric worker: {e}")

def _create_time_series(data: dict):
    config = get_config()
    event_name = data['event']
    if event_name.startswith('/'):
        event_name = event_name[1:]
    metric_type = f"{config.metric_prefix}/{event_name}"
    
    series = monitoring_v3.TimeSeries()
    series.metric.type = metric_type
    
    # Add labels
    labels = data.get('kwargs', {})
    # Flatten kwargs to string labels
    for k, v in labels.items():
        series.metric.labels[k] = str(v)
    
    # Add Job Name label
    series.metric.labels['job_name'] = config.job_name

    # Resource
    series.resource.type = config.monitored_resource_type
    for k, v in config.monitored_resource_labels.items():
        series.resource.labels[k] = v
    
    # Ensure project_id is present for global resource
    if config.monitored_resource_type == 'global' and 'project_id' not in series.resource.labels:
        if config.project_id:
            series.resource.labels['project_id'] = config.project_id
    
    # Point
    point = monitoring_v3.Point()
    
    # Use timestamp from data if available, or fallback to now
    timestamp_val = data.get('timestamp', time.time())
    seconds = int(timestamp_val)
    nanos = int((timestamp_val - seconds) * 10**9)
    
    # Create Timestamp object explicitly
    end_time = timestamp_pb2.Timestamp()
    end_time.seconds = seconds
    end_time.nanos = nanos
    point.interval.end_time = end_time
    
    # Value
    value = data['value']
    
    # Convert to microseconds to preserve precision in int64
    point.value.int64_value = int(value * 1e6)
    series.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
    
    series.points.append(point)
    return series

def _on_duration(event: str, duration_secs: float, **kwargs):
    # print(f"DEBUG: Listener received event: {event}")
    if _METRIC_QUEUE is not None:
        try:
            _METRIC_QUEUE.put_nowait({
                'event': event,
                'value': duration_secs,
                'timestamp': time.time(),
                'kwargs': kwargs
            })
        except multiprocessing.queues.Full:
            # Drop metric if queue is full to avoid blocking main process
            logger.warning("Metric queue full, dropping metric")

def register_listeners():
    global _WORKER_PROCESS, _METRIC_QUEUE, _STOP_EVENT
    
    if _METRIC_QUEUE is None:
        _METRIC_QUEUE = multiprocessing.Queue(maxsize=10000)
        _STOP_EVENT = multiprocessing.Event()

    if _STOP_EVENT:
        _STOP_EVENT.clear()

    if _WORKER_PROCESS is None or not _WORKER_PROCESS.is_alive():
        current_config = get_config()
        
        _WORKER_PROCESS = multiprocessing.Process(
            target=_metric_worker, 
            args=(current_config, _METRIC_QUEUE, _STOP_EVENT),
            daemon=True
        )
        _WORKER_PROCESS.start()

    jax.monitoring.register_event_duration_secs_listener(_on_duration)
    logger.info("JAX Cloud Monitoring listeners registered (multiprocessing).")

def stop_listeners():
    """Stop the background worker process."""
    global _WORKER_PROCESS, _METRIC_QUEUE, _STOP_EVENT
    
    if _STOP_EVENT:
        _STOP_EVENT.set()
    
    if _WORKER_PROCESS and _WORKER_PROCESS.is_alive():
        _WORKER_PROCESS.join(timeout=2.0)
        if _WORKER_PROCESS.is_alive():
            _WORKER_PROCESS.terminate()
            _WORKER_PROCESS.join()
    
    _WORKER_PROCESS = None
