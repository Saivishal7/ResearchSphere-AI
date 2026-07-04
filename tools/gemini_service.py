# tools/gemini_service.py
import json
import logging
from typing import Optional, Any, Dict, Union
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, APIError

from config import GEMINI_API_KEY, GEMINI_MODEL

# Configure logging
logger = logging.getLogger("GeminiService")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class GeminiService:
    """
    A production-quality Gemini AI service that acts as the single interface
    between our application and the Gemini API.
    """
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.model_name = GEMINI_MODEL
        self._initialized = False

        if not self.api_key:
            logger.error("GEMINI_API_KEY is missing from environment/configuration.")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self._initialized = True
                logger.info(f"GeminiService successfully initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to configure Gemini SDK: {e}")

    def _get_model(self) -> genai.GenerativeModel:
        """
        Helper method to retrieve the configured GenerativeModel.
        Raises an exception if the service is not initialized.
        """
        if not self._initialized:
            raise ValueError(
                "GeminiService is not initialized. Please check your GEMINI_API_KEY."
            )
        return genai.GenerativeModel(self.model_name)

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
            model = self._get_model()
            logger.info(f"Sending prompt to Gemini ({self.model_name})...")
            response = model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("Gemini returned an empty response.")
                return ""
                
        except APIError as e:
            logger.error(f"Gemini API failure (status code: {e.code}): {e.message}")
            raise RuntimeError(f"Gemini API Call Failed: {e.message}") from e
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
            model = self._get_model()
            logger.info(f"Sending JSON prompt to Gemini ({self.model_name})...")
            
            # Request JSON output using generation config
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json"
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response or not response.text:
                logger.warning("Gemini returned an empty JSON response.")
                return {}

            json_text = response.text.strip()
            
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
                
        except APIError as e:
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
            model = self._get_model()
            # Simple latency-efficient verification prompt
            response = model.generate_content("ping")
            if response and response.text:
                logger.info("health_check passed: Gemini API successfully connected.")
                return True
            logger.warning("health_check warning: Empty response received.")
            return False
        except Exception as e:
            logger.error(f"health_check failed: Could not communicate with Gemini API: {e}")
            return False
