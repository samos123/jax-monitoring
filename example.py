import jax
from jax_cloud_monitoring import init as jax_cloud_monitoring_init

def main():
    jax_cloud_monitoring_init(
        job_name="my-training-job",
    )

    # Run jax inside jit to capture metrics compilation metrics.
    # Check for backend_compile_duration metric in Cloud Monitoring > Metrics Explorer.
    jax.jit(lambda x: x + x)(jax.numpy.ones((1000, 1000)))

if __name__ == "__main__":
    main()
