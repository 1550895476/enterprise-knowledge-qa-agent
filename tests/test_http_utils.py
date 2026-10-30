import urllib.error
import urllib.request
import unittest
from unittest.mock import MagicMock, patch

from src.http_utils import urlopen_with_retry


class HttpRetryTests(unittest.TestCase):
    @patch("src.http_utils.time.sleep", return_value=None)
    @patch("src.http_utils.urllib.request.urlopen")
    def test_transient_tls_style_error_is_retried(self, mocked_open, _mocked_sleep):
        response = MagicMock()
        mocked_open.side_effect = [
            urllib.error.URLError("temporary TLS EOF"),
            response,
        ]

        actual = urlopen_with_retry(urllib.request.Request("https://example.com"), timeout=5)

        self.assertIs(actual, response)
        self.assertEqual(mocked_open.call_count, 2)

    @patch("src.http_utils.time.sleep", return_value=None)
    @patch("src.http_utils.urllib.request.urlopen")
    def test_authentication_error_is_not_retried(self, mocked_open, _mocked_sleep):
        mocked_open.side_effect = urllib.error.HTTPError(
            "https://example.com", 401, "Unauthorized", {}, None
        )

        with self.assertRaises(urllib.error.HTTPError):
            urlopen_with_retry(urllib.request.Request("https://example.com"), timeout=5)

        self.assertEqual(mocked_open.call_count, 1)
