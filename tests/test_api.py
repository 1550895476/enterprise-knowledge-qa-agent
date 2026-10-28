import unittest

from fastapi.testclient import TestClient

from src.api import app
from src.runtime_config import ModelConfigError, RuntimeModelConfig


class WebInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_serves_product_interface(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("企业知识问答 Agent", response.text)
        self.assertIn("上传知识文件", response.text)
        self.assertIn("模型配置", response.text)

    def test_model_config_is_redacted_and_validated(self):
        config = RuntimeModelConfig()
        public = config.update("offline", "gpt-4o-mini", "https://api.openai.com/v1")
        self.assertEqual(public["provider"], "offline")
        self.assertNotIn("api_key", public)

        empty_config = RuntimeModelConfig()
        empty_config._config.api_key = ""
        with self.assertRaises(ModelConfigError):
            empty_config.update("openai", "gpt-4o-mini", "https://api.openai.com/v1")


if __name__ == "__main__":
    unittest.main()
