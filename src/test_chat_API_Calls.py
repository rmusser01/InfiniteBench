# test_chat_API_Calls.py
# Test file for testing the integration of the LLM API calls with the Chat APIs.
#
# Usage:
# python -m unittest test_chat_API_Calls.py

import unittest
import configparser
import os
from LLM_API_Calls import (
    chat_with_openai,
    chat_with_anthropic,
    chat_with_cohere,
    chat_with_groq,
    chat_with_openrouter,
    chat_with_huggingface,
    chat_with_deepseek,
    chat_with_mistral
)

class TestLLMAPICallsIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load configuration from config.txt
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.txt')
        config.read(config_path)

        # Load API keys from config file
        cls.openai_api_key = config.get('API', 'openai_api_key', fallback=None)
        cls.anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
        cls.cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
        cls.groq_api_key = config.get('API', 'groq_api_key', fallback=None)
        cls.openrouter_api_key = config.get('API', 'openrouter_api_key', fallback=None)
        cls.huggingface_api_key = config.get('API', 'huggingface_api_key', fallback=None)
        cls.deepseek_api_key = config.get('API', 'deepseek_api_key', fallback=None)
        cls.mistral_api_key = config.get('API', 'mistral_api_key', fallback=None)

        # Load models from config file
        cls.openai_model = config.get('API', 'openai_model', fallback='gpt-4o')
        cls.anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-5-sonnet-20240620')
        cls.cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
        cls.groq_model = config.get('API', 'groq_model', fallback='llama3-70b-8192')
        cls.openrouter_model = config.get('API', 'openrouter_model', fallback='mistralai/mistral-7b-instruct:free')
        cls.deepseek_model = config.get('API', 'deepseek_model', fallback='deepseek-chat')
        cls.mistral_model = config.get('API', 'mistral_model', fallback='mistral-large-latest')

    def test_chat_with_openai(self):
        if not self.openai_api_key:
            self.skipTest("OpenAI API key not available")
        response = chat_with_openai(self.openai_api_key, "Hello, how are you?", "Respond briefly", model=self.openai_model)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_anthropic(self):
        if not self.anthropic_api_key:
            self.skipTest("Anthropic API key not available")
        response = chat_with_anthropic(self.anthropic_api_key, "Hello, how are you?", self.anthropic_model, "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_cohere(self):
        if not self.cohere_api_key:
            self.skipTest("Cohere API key not available")
        response = chat_with_cohere(self.cohere_api_key, "Hello, how are you?", self.cohere_model, "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_groq(self):
        if not self.groq_api_key:
            self.skipTest("Groq API key not available")
        response = chat_with_groq(self.groq_api_key, "Hello, how are you?", "Respond briefly", model=self.groq_model)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_openrouter(self):
        if not self.openrouter_api_key:
            self.skipTest("OpenRouter API key not available")
        response = chat_with_openrouter(self.openrouter_api_key, "Hello, how are you?", "Respond briefly", model=self.openrouter_model)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_huggingface(self):
        if not self.huggingface_api_key:
            self.skipTest("HuggingFace API key not available")
        response = chat_with_huggingface(self.huggingface_api_key, "Hello, how are you?", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_deepseek(self):
        if not self.deepseek_api_key:
            self.skipTest("DeepSeek API key not available")
        response = chat_with_deepseek(self.deepseek_api_key, "Hello, how are you?", "Respond briefly", model=self.deepseek_model)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_mistral(self):
        if not self.mistral_api_key:
            self.skipTest("Mistral API key not available")
        response = chat_with_mistral(self.mistral_api_key, "Hello, how are you?", "Respond briefly", model=self.mistral_model)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()