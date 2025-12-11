
import unittest
from unittest.mock import patch, MagicMock
import sys
import importlib
from jax_cloud_monitoring import config

class TestGCEMetadata(unittest.TestCase):
    
    def setUp(self):
        # Reset config to defaults before each test
        config._GLOBAL_CONFIG = config.Config()

    @patch('urllib.request.urlopen')
    def test_detect_gce_config_success(self, mock_urlopen):
        # Setup mock for instance/id and instance/zone
        def side_effect(req, timeout=None):
            url = req.full_url
            mock_response = MagicMock()
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            
            if "instance/id" in url:
                mock_response.read.return_value = b"123456789"
            elif "instance/zone" in url:
                mock_response.read.return_value = b"projects/123/zones/us-central1-a"
            elif "instance/name" in url:
                mock_response.read.return_value = b"my-instance"
            else:
                raise urllib.error.URLError("Not found")
            return mock_response
        
        mock_urlopen.side_effect = side_effect

        # We can't easily mock the module-level execution without reloading, 
        # but we can test the function `_detect_gce_config` directly if we access it.
        # Since it's private, we access it via 'config'.
        
        # ACT
        detected_config = config._detect_gce_config()
        
        # ASSERT
        self.assertEqual(detected_config['monitored_resource_type'], 'gce_instance')
        self.assertEqual(detected_config['monitored_resource_labels']['instance_id'], '123456789')
        self.assertEqual(detected_config['monitored_resource_labels']['zone'], 'us-central1-a')
        self.assertEqual(detected_config['monitored_resource_labels']['instance_name'], 'my-instance')

    @patch('urllib.request.urlopen')
    def test_detect_gce_config_not_gce(self, mock_urlopen):
        # Setup mock to fail (simulate timeout or connectivity issue)
        mock_urlopen.side_effect = Exception("Timeout")
        
        # ACT
        detected_config = config._detect_gce_config()
        
        # ASSERT
        self.assertEqual(detected_config, {})

    @patch('urllib.request.urlopen')
    def test_module_initialization(self, mock_urlopen):
        # We need to simulate GCE environment via urlopen mock
        def side_effect(req, timeout=None):
            url = req.full_url
            mock_response = MagicMock()
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            if "instance/id" in url:
                mock_response.read.return_value = b"999"
            elif "instance/zone" in url:
                mock_response.read.return_value = b"projects/123/zones/us-west1-b"
            elif "instance/name" in url:
                mock_response.read.return_value = b"reloaded-instance"
            else:
                raise urllib.error.URLError("Not found")
            return mock_response
            
        mock_urlopen.side_effect = side_effect
        
        # Reload the module to trigger the top-level code
        importlib.reload(config)
        
        # ASSERT
        cfg = config.get_config()
        self.assertEqual(cfg.monitored_resource_type, 'gce_instance')
        self.assertEqual(cfg.monitored_resource_labels['instance_id'], '999')
        self.assertEqual(cfg.monitored_resource_labels['zone'], 'us-west1-b')

if __name__ == '__main__':
    unittest.main()
