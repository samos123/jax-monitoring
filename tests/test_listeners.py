import unittest
from unittest.mock import MagicMock, patch
import jax.monitoring
import time
import multiprocessing
import queue
from jax_monitoring import listeners, config

class TestListeners(unittest.TestCase):
    def setUp(self):
        # Reset config
        config.set_config(project_id="test-project")
        
        # We will test components individually to avoids multiprocessing complexity in unit tests
        # 1. Test _on_duration puts to queue
        # 2. Test _metric_worker reads from queue and calls client

        # Clear any existing listeners/state if possible
        # We can't easily "unregister" JAX listeners, but we can reset our internal state
        listeners._METRIC_QUEUE = None
        listeners._WORKER_PROCESS = None
        listeners._STOP_EVENT = None

    def tearDown(self):
        listeners.stop_listeners()

    def test_on_duration_plumbing(self):
        # Setup a queue locally
        test_queue = multiprocessing.Queue()
        listeners._METRIC_QUEUE = test_queue
        
        # Trigger listener
        listeners._on_duration("test_event", 1.23, label="val")
        
        # Verify
        try:
            item = test_queue.get(timeout=1.0)
            self.assertEqual(item['event'], "test_event")
            self.assertEqual(item['value'], 1.23)
            self.assertIn('timestamp', item)
            self.assertIsInstance(item['timestamp'], float)
            self.assertEqual(item['kwargs'], {'label': 'val'})
        except queue.Empty:
            self.fail("Queue should not be empty")

    @patch('jax_monitoring.listeners.get_client')
    @patch('jax_monitoring.listeners.get_project_path')
    @patch('jax_monitoring.listeners.initialize_client')
    def test_worker_logic(self, mock_init, mock_get_path, mock_get_client):
        # Prepare mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_path.return_value = "projects/test-project"
        
        # Prepare inputs
        q = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        
        config_data = config.Config(
            project_id='test-project',
            metric_prefix='custom',
            monitored_resource_type='global',
            monitored_resource_labels={},
            job_name='test_job'
        )
        
        # Put some data
        q.put({
            'event': 'test_event',
            'value': 5.0,
            'kwargs': {'l': 'v'}
        })
        
        # Run worker function in a separate thread (to simulate process without process isolation)
        # or just run it for a short time?
        # Since _metric_worker loops, we need to set stop_event eventually.
        # We can run it in a thread, or just modify _metric_worker to handle 1 item? No, it loops.
        # Let's run it in a thread.
        import threading
        t = threading.Thread(target=listeners._metric_worker, args=(config_data, q, stop_event))
        t.start()
        
        # Wait a bit for processing
        time.sleep(0.5)
        
        # Stop
        stop_event.set()
        t.join(timeout=1.0)
        
        # Verify
        mock_client.create_time_series.assert_called()
        call_args = mock_client.create_time_series.call_args
        request = call_args[1]['request']
        series = request['time_series'][0]
        self.assertEqual(series.metric.type, "custom/test_event")
        from google.api import metric_pb2
        self.assertEqual(series.unit, "s")
        self.assertEqual(series.value_type, metric_pb2.MetricDescriptor.ValueType.DOUBLE)
        self.assertEqual(series.points[0].value.double_value, 5.0)

if __name__ == '__main__':
    unittest.main()
