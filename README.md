# JAX Cloud Monitoring

This library provides Google Cloud Monitoring (Stackdriver) integration for JAX. It captures JAX monitoring events (like compilation time) and exports them as custom metrics to Google Cloud Monitoring.

## Key Features

- **Non-blocking Export (Multiprocessing):** metric export happens in a separate background process (`multiprocessing`), ensuring that network I/O to the Cloud Monitoring API does not block your JAX workload or contend for the Global Interpreter Lock (GIL).
- **Easy Integration:** hooks into `jax.monitoring` with a simple registration call.
- **Configurable:** customize metric prefixes, monitored resources, and labels.

## Installation

```bash
pip install jax-monitoring
```

*Note: You may need to install from source or a private registry if this package is not published to PyPI.*

## Usage

```python
import jax
import jax_monitoring as jm

def main():
    jm.init(
        job_name="my-training-job",
    )

    # Run jax inside jit to capture metrics compilation metrics.
    # Check for backend_compile_duration metric in Cloud Monitoring > Metrics Explorer.
    x = jax.jit(lambda x: x + x)(jax.numpy.ones((1000, 1000)))
    x.block_until_ready()
    
if __name__ == "__main__":
    main()
```

## Configuration

You can configure the behavior using `jm.init()`:

-   `project_id` (str): GCP Project ID. If not provided, the library will attempt to infer it from the environment.
-   `metric_prefix` (str): Prefix for all exported metrics. Default: `custom.googleapis.com/jax/monitoring`.
-   `job_name` (str): A label added to all metrics to identify the job. Default: `jax_job`.
-   `monitored_resource_type` (str): The Stackdriver monitored resource type. Default: `global` or `gce_instance` when running on GCE.
-   `monitored_resource_labels` (dict): Labels for the monitored resource.

## How it Works

When `jm.init()` is called, the library:
1.  Starts a background `multiprocessing.Process`.
2.  Registers a callback with `jax.monitoring`.
3.  When JAX triggers an event (e.g., a compilation finishes), the callback puts the event data into a `multiprocessing.Queue`.
4.  The background worker picks up events from the queue and batches them to the Google Cloud Monitoring API using the efficient `create_time_series` call.

This architecture ensures that the main training loop is never blocked by HTTP requests to the monitoring backend.

## Disclaimer

This is a proof of concept and is not production ready. You may get a huge cloud bill and you are fully responsible for the usage of the cloud resources.
