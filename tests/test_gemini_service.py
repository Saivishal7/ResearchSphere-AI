import importlib
import sys
import unittest


class GeminiServiceImportTests(unittest.TestCase):
    def test_gemini_service_imports_without_sdk(self):
        sys.modules.pop("tools.gemini_service", None)

        gemini_service = importlib.import_module("tools.gemini_service")
        service = gemini_service.GeminiService()

        self.assertTrue(service._initialized)
        self.assertTrue(hasattr(service, "generate_json"))


if __name__ == "__main__":
    unittest.main()
