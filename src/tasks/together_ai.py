"""
Together.ai integration for LLM functionality.
"""
import os
from dotenv import load_dotenv
try:
    # Try to import from langchain-together first
    from langchain_together import Together
except ImportError:
    # Fall back to community version if langchain-together is not installed
    from langchain_community.llms import Together

# Load environment variables
load_dotenv()

class TogetherAIManager:
    """Class for managing Together.ai models."""

    def __init__(self, model_name=None, api_key=None):
        """
        Initialize with model parameters.

        Args:
            model_name: Together.ai model name
            api_key: Together.ai API key
        """
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables or parameters")

        # Set model name
        self.model_name = model_name or os.getenv("TOGETHER_MODEL_NAME", "deepseek-ai/DeepSeek-V3")

        print(f"Using Together.ai model: {self.model_name}")

        # Set API key in environment for LangChain
        os.environ["TOGETHER_API_KEY"] = self.api_key

    def get_llm(self, temperature=0.7, max_tokens=2048):
        """
        Get Together.ai LLM.

        Args:
            temperature: Temperature for text generation
            max_tokens: Maximum number of tokens to generate

        Returns:
            Together LLM instance
        """
        return Together(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def generate_text(self, prompt, temperature=0.7, max_tokens=2048):
        """
        Generate text using Together.ai.

        Args:
            prompt: Text prompt
            temperature: Temperature for text generation
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text
        """
        llm = self.get_llm(temperature=temperature, max_tokens=max_tokens)
        return llm.invoke(prompt)

