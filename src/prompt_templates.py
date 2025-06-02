#!/usr/bin/env python3
"""
Model-specific prompt templates for better response quality
"""

from typing import Dict, Optional, Tuple

class PromptTemplate:
    """Base class for prompt templates"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def format_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format the prompt for the model"""
        raise NotImplementedError
    
    def extract_response(self, generated_text: str, original_prompt: str) -> str:
        """Extract the response from generated text"""
        raise NotImplementedError


class QwenTemplate(PromptTemplate):
    """Template for Qwen models"""
    
    def format_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt in Qwen chat format"""
        if system_prompt:
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def extract_response(self, generated_text: str, original_prompt: str) -> str:
        """Extract response from Qwen format"""
        # Look for the assistant response after the last assistant tag
        if "<|im_start|>assistant\n" in generated_text:
            response = generated_text.split("<|im_start|>assistant\n")[-1]
            # Remove any end tokens
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            return response.strip()
        # Fallback: remove the original prompt if it appears at the start
        elif generated_text.startswith(original_prompt):
            return generated_text[len(original_prompt):].strip()
        return generated_text.strip()


class DeepSeekTemplate(PromptTemplate):
    """Template for DeepSeek models"""
    
    def format_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt in DeepSeek format"""
        # DeepSeek R1 models use a specific format
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        else:
            return f"User: {user_prompt}\n\nAssistant:"
    
    def extract_response(self, generated_text: str, original_prompt: str) -> str:
        """Extract response from DeepSeek format"""
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1]
            # Remove any subsequent "User:" parts
            if "\nUser:" in response:
                response = response.split("\nUser:")[0]
            return response.strip()
        # Fallback
        elif generated_text.startswith(original_prompt):
            return generated_text[len(original_prompt):].strip()
        return generated_text.strip()


class DefaultTemplate(PromptTemplate):
    """Default template for models without specific formatting"""
    
    def format_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simple concatenation for default models"""
        if system_prompt:
            return f"{system_prompt}\n\n{user_prompt}\n\n"
        return user_prompt
    
    def extract_response(self, generated_text: str, original_prompt: str) -> str:
        """Try to extract response by removing the prompt"""
        if generated_text.startswith(original_prompt):
            return generated_text[len(original_prompt):].strip()
        return generated_text.strip()


class PromptTemplateFactory:
    """Factory for creating appropriate prompt templates"""
    
    @staticmethod
    def get_template(model_name: str) -> PromptTemplate:
        """Get the appropriate template for a model"""
        model_lower = model_name.lower()
        
        # Use simpler templates for now since complex chat formats may not work
        if "deepseek" in model_lower and "r1" in model_lower:
            return DeepSeekTemplate(model_name)
        else:
            return DefaultTemplate(model_name)
    
    @staticmethod
    def get_system_prompt(task_type: str = "general") -> str:
        """Get a system prompt for a specific task type"""
        system_prompts = {
            "general": "You are a helpful AI assistant. Provide clear, concise, and accurate responses.",
            "code": "You are a coding assistant. Provide clear code examples and explanations.",
            "creative": "You are a creative writing assistant. Be imaginative and engaging.",
            "analysis": "You are an analytical assistant. Provide detailed analysis and insights.",
        }
        return system_prompts.get(task_type, system_prompts["general"])


# Model-specific generation parameters
MODEL_GENERATION_PARAMS = {
    "qwen2.5-1.5b": {
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "do_sample": True,
        "no_repeat_ngram_size": 3,
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "do_sample": True,
        "no_repeat_ngram_size": 2,
    },
    "default": {
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "no_repeat_ngram_size": 0,
    }
}


def get_model_params(model_name: str, user_params: Dict) -> Dict:
    """Merge model-specific params with user params"""
    base_params = MODEL_GENERATION_PARAMS.get(
        model_name, 
        MODEL_GENERATION_PARAMS["default"]
    ).copy()
    
    # User params override defaults
    base_params.update(user_params)
    return base_params