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
    
    print("Program finished. Metrics should be flushed automatically.")

if __name__ == "__main__":
    main()
