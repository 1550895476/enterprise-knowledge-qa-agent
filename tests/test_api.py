import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.decision import OpenAICompatibleDecisionModel
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

    def test_model_config_keeps_protocol_and_output_options(self):
        config = RuntimeModelConfig()
        config._config.api_key = "stored-key"
        public = config.update(
            "custom",
            "relay-model",
            "https://relay.example.com/v1/responses",
            protocol="responses",
            reasoning_effort="low",
            response_format="json_object",
        )
        self.assertEqual(public["base_url"], "https://relay.example.com/v1")
        self.assertEqual(public["protocol"], "responses")
        self.assertEqual(public["reasoning_effort"], "low")
        self.assertEqual(public["response_format"], "json_object")
        self.assertNotIn("api_key", public)

    def test_offline_model_connection_test(self):
        result = RuntimeModelConfig().test_connection("offline", "", "")
        self.assertTrue(result["ok"])
        self.assertEqual(result["latency_ms"], 0)

    def test_deepseek_uses_openai_compatible_chat_endpoint(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"model":"deepseek-flash","choices":[{"message":{"content":"OK"}}]}'

        with patch("urllib.request.urlopen", return_value=FakeResponse()) as mocked:
            result = RuntimeModelConfig().test_connection(
                "deepseek", "deepseek-flash", "https://api.deepseek.com", "test-key"
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["response_preview"], "OK")
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")

    def test_responses_protocol_uses_selected_reasoning_and_json_output(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"model":"deepseek-v4-pro","output_text":"{\\"status\\":\\"OK\\"}"}'

        with patch("urllib.request.urlopen", return_value=FakeResponse()) as mocked:
            result = RuntimeModelConfig().test_connection(
                "deepseek",
                "deepseek-v4-pro",
                "https://api.deepseek.com/responses",
                "test-key",
                protocol="responses",
                reasoning_effort="low",
                response_format="json_object",
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["protocol"], "responses")
        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.deepseek.com/responses")
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["text"], {"format": {"type": "json_object"}})

    def test_responses_protocol_drives_agent_action_selection(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"output_text":"{\\"action\\":\\"search\\",\\"query\\":\\"expense deadline\\",\\"reason\\":\\"need evidence\\"}"}'

        model = OpenAICompatibleDecisionModel(
            api_key="test-key",
            base_url="https://relay.example.com/v1",
            model="reasoning-model",
            protocol="responses",
            reasoning_effort="medium",
        )
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as mocked:
            action = model.choose({"question": "报销期限多久", "history": []}, None)
        self.assertEqual(action.name, "search")
        self.assertEqual(action.query, "expense deadline")
        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://relay.example.com/v1/responses")
        self.assertEqual(payload["reasoning"], {"effort": "medium"})

    def test_model_discovery_lists_relay_models(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"object":"list","data":[{"id":"gpt-4o-mini"},{"id":"deepseek-chat"},{"id":"gpt-4o-mini"}]}'

        with patch("urllib.request.urlopen", return_value=FakeResponse()) as mocked:
            result = RuntimeModelConfig().list_models(
                "openai", "https://relay.example.com/v1/chat/completions", "test-key"
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["models"], ["deepseek-chat", "gpt-4o-mini"])
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://relay.example.com/v1/models")


if __name__ == "__main__":
    unittest.main()
