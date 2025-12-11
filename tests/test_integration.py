import unittest
import jax
import jax.numpy as jnp
import jax.monitoring
import time
import uuid
import logging
from google.cloud import monitoring_v3
import jax_monitoring as jm
from jax_monitoring import config, listeners, gcp_utils

class TestLiveIntegration(unittest.TestCase):
    def setUp(self):
        # Generate a unique event name to avoid collisions and easy lookup
        self.unique_id = str(uuid.uuid4())
        # Use a fixed event name to avoid creating too many custom metric descriptors (quota limit)
        # Updated to v2 to avoid type conflict with legacy INT64 metric (now using DOUBLE)
        self.event_name = "integration_test_event_v3"
        self.metric_type = None # Constructed dynamically in test using prefix
        self.job_name = f"integration_test_{self.unique_id}"
        
        # Use a unique metric prefix to avoid conflicts with existing metrics (e.g. incompatible value types)
        # This effectively sandboxes the test run.
        self.metric_prefix = f"custom.googleapis.com/jax/monitoring/test_{self.unique_id}"
        
        # Initialize the library
        # We assume ADC is set up in the environment
        jm.init(job_name=self.job_name, metric_prefix=self.metric_prefix)
        
        self.project_id = config.get_config().project_id
        if not self.project_id:
            self.skipTest("Project ID could not be determined. Skipping live test.")

        self.client = gcp_utils.get_client()
        if not self.client:
             self.skipTest("GCP Client could not be initialized. Skipping live test.")

    def tearDown(self):
        listeners.stop_listeners()

    def test_end_to_end_metric_export(self):
        print(f"\nRunning live test with event: {self.event_name}")
        # Construct expected metric type using the dynamic prefix
        self.metric_type = f"{self.metric_prefix}/{self.event_name}"
        print(f"Target metric type: {self.metric_type}")
        
        # Record an event duration
        jax.monitoring.record_event_duration_secs(self.event_name, 1.23, test_id=self.unique_id)
        
        self._wait_and_verify_metric(self.metric_type)

    def test_jax_compilation_metric(self):
        print(f"\nRunning live test for JAX compilation metric")
        # Construct expected metric type using the dynamic prefix
        # Note: listeners.py handles the slash logic, usually appending /{event} to prefix
        target_metric = f"{self.metric_prefix}/jax/core/compile/backend_compile_duration"
        print(f"Target metric type: {target_metric}")

        # Trigger JAX compilation
        @jax.jit
        def matmul(x, y):
            return jnp.dot(x, y)

        x = jnp.ones((8, 8))
        # First call triggers compilation
        _ = matmul(x, x).block_until_ready()
        
        self._wait_and_verify_metric(target_metric)

    def _wait_and_verify_metric(self, metric_type):
        # Wait for the worker to pick it up and export
        print("Waiting for metric export...")
        # Reduce initial wait but keep retry loop
        time.sleep(2)
        
        project_name = f"projects/{self.project_id}"
        interval = monitoring_v3.TimeInterval()
        now = time.time()
        
        from google.protobuf import timestamp_pb2
        end_time = timestamp_pb2.Timestamp()
        end_time.seconds = int(now) + 60
        interval.end_time = end_time
        
        start_time = timestamp_pb2.Timestamp()
        start_time.seconds = int(now) - 60
        interval.start_time = start_time
        
        # We'll retry a few times
        found = False
        for i in range(10):
            print(f"Query attempt {i+1}...")
            try:
                # Add job_name filter to isolate this test run
                filter_str = f'metric.type = "{metric_type}" AND metric.labels.job_name = "{self.job_name}"'
                results = self.client.list_time_series(
                    request={
                        "name": project_name,
                        "filter": filter_str,
                        "interval": interval,
                        "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                    }
                )
                for result in results:
                    if result.metric.type == metric_type:
                        print(f"Metric found: {metric_type}")
                        found = True
                        break
            except Exception as e:
                print(f"Query failed: {e}")
            
            if found:
                break
            time.sleep(3)
            
        if not found:
            print("Metric not found yet (this is expected due to ingestion latency).")
            print("Check the Cloud Monitoring console later.")
            self.fail("Metric not found after 10 attempts.")
        else:
            print("Successfully verified metric export!")

if __name__ == '__main__':
    unittest.main()
