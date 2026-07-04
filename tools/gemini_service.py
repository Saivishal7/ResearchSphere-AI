# tools/gemini_service.py
import json
import logging
from typing import Any, Dict, Union

import requests

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - exercised when dependency is absent
    genai = None

try:
    from google.api_core.exceptions import GoogleAPICallError, GoogleAPIError
except Exception:  # pragma: no cover - exercised when dependency is absent
    class GoogleAPICallError(Exception):
        pass

    class GoogleAPIError(Exception):
        pass

from config import GEMINI_API_KEY, GEMINI_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GeminiService:
    """
    A production-quality Gemini AI service that acts as the single interface
    between our application and the Gemini API.
    """
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.model_name = GEMINI_MODEL
        self._initialized = False
        self._sdk_available = genai is not None

        if not self.api_key:
            logger.error("GEMINI_API_KEY is missing from environment/configuration.")
        else:
            if self._sdk_available:
                try:
                    genai.configure(api_key=self.api_key)
                    self._initialized = True
                    logger.info(f"GeminiService successfully initialized with model: {self.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to configure Gemini SDK: {e}. Falling back to REST calls.")
                    self._initialized = True
            else:
                logger.warning("google-generativeai SDK is not installed. Falling back to Gemini REST API calls.")
                self._initialized = True

    def _get_model(self) -> Any:
        """
        Helper method to retrieve the configured GenerativeModel.
        Raises an exception if the service is not initialized.
        """
        if not self._initialized:
            raise ValueError(
                "GeminiService is not initialized. Please check your GEMINI_API_KEY."
            )
        if not self._sdk_available:
            raise RuntimeError("Gemini SDK is unavailable. Use the REST fallback path instead.")
        return genai.GenerativeModel(self.model_name)

    def _call_rest_api(self, prompt: str, response_mime_type: str = "text/plain") -> str:
        """Calls the Gemini REST endpoint directly when the SDK is unavailable."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment/configuration.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        if response_mime_type != "text/plain":
            payload["generationConfig"] = {"responseMimeType": response_mime_type}

        logger.info("Using Gemini REST fallback for generation request.")
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise RuntimeError("Gemini REST response was not a JSON object.")

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini REST response did not include any candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("Gemini REST response did not include any content parts.")

        text = parts[0].get("text", "")
        if not isinstance(text, str):
            raise RuntimeError("Gemini REST response contained non-text content.")
        return text

    def generate_response(self, prompt: str) -> str:
        """
        Generates a clean text response from the Gemini model.
        
        Args:
            prompt (str): The prompt string to send to the model.
            
        Returns:
            str: The clean text response from the model.
            
        Raises:
            ValueError: If the service is uninitialized or prompt is empty.
            RuntimeError: If there is an API communication issue.
        """
        if not prompt or not prompt.strip():
            logger.warning("generate_response called with empty prompt.")
            return ""

        try:
            if self._sdk_available:
                model = self._get_model()
                logger.info(f"Sending prompt to Gemini ({self.model_name})...")
                response = model.generate_content(prompt)

                if response and getattr(response, "text", None):
                    return response.text.strip()
                logger.warning("Gemini returned an empty response.")
                return ""

            return self._call_rest_api(prompt).strip()

        except GoogleAPIError as e:
            logger.error(f"Gemini API failure: {e}")
            raise RuntimeError(f"Gemini API Call Failed: {e}") from e
        except GoogleAPICallError as e:
            logger.error(f"Google API call error: {e}")
            raise RuntimeError(f"Google API Call Failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {e}")
            raise RuntimeError(f"Unexpected Gemini Service Failure: {e}") from e

    def generate_json(self, prompt: str) -> Union[Dict[str, Any], list]:
        """
        Generates a structured JSON response from the Gemini model and returns the parsed object.
        Forces the output to be application/json.
        
        Args:
            prompt (str): The prompt string to send to the model.
            
        Returns:
            Union[Dict[str, Any], list]: Parsed JSON object (dictionary or list).
            
        Raises:
            ValueError: If the service is uninitialized, prompt is empty, or JSON parsing fails.
            RuntimeError: If there is an API communication issue.
        """
        if not prompt or not prompt.strip():
            logger.warning("generate_json called with empty prompt.")
            return {}

        try:
            if self._sdk_available:
                model = self._get_model()
                logger.info(f"Sending JSON prompt to Gemini ({self.model_name})...")

                generation_config = genai.GenerationConfig(
                    response_mime_type="application/json"
                )

                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                if not response or not getattr(response, "text", None):
                    logger.warning("Gemini returned an empty JSON response.")
                    return {}

                json_text = response.text.strip()
            else:
                logger.info("Using REST fallback for JSON generation.")
                json_text = self._call_rest_api(prompt, response_mime_type="application/json").strip()

            # Clean up potential markdown code block backticks if present
            if json_text.startswith("```json"):
                json_text = json_text.split("```json", 1)[1]
                if json_text.endswith("```"):
                    json_text = json_text.rsplit("```", 1)[0]
            elif json_text.startswith("```"):
                json_text = json_text.split("```", 1)[1]
                if json_text.endswith("```"):
                    json_text = json_text.rsplit("```", 1)[0]

            json_text = json_text.strip()

            try:
                parsed_json = json.loads(json_text)
                return parsed_json
            except json.JSONDecodeError as jde:
                logger.error(f"Failed to parse returned JSON string: {json_text}. Error: {jde}")
                raise ValueError(f"Returned content was not valid JSON: {jde}") from jde

        except GoogleAPIError as e:
            logger.error(f"Gemini API failure during JSON generation (status code: {e.code}): {e.message}")
            raise RuntimeError(f"Gemini API JSON Call Failed: {e.message}") from e
        except GoogleAPICallError as e:
            logger.error(f"Google API call error during JSON generation: {e}")
            raise RuntimeError(f"Google API JSON Call Failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in generate_json: {e}")
            raise RuntimeError(f"Unexpected Gemini JSON Service Failure: {e}") from e

    def health_check(self) -> bool:
        """
        Checks the health of the Gemini service.
        Verifies initialization and verifies if a basic ping to the API succeeds.
        
        Returns:
            bool: True if the service is fully functional, False otherwise.
        """
        if not self._initialized:
            logger.error("health_check failed: Service not initialized due to missing/invalid API key.")
            return False

        try:
            if self._sdk_available:
                model = self._get_model()
                response = model.generate_content("ping")
                if response and getattr(response, "text", None):
                    logger.info("health_check passed: Gemini API successfully connected.")
                    return True
                logger.warning("health_check warning: Empty response received.")
                return False

            text = self._call_rest_api("ping")
            if text:
                logger.info("health_check passed: Gemini API successfully connected.")
                return True
            logger.warning("health_check warning: Empty response received.")
            return False
        except Exception as e:
            logger.error(f"health_check failed: Could not communicate with Gemini API: {e}")
            return False
