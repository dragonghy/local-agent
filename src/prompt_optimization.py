#!/usr/bin/env python3
"""
Advanced prompt engineering and optimization strategies
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass

try:
    from .inference import ModelManager
    from .prompt_templates import PromptTemplateFactory
except ImportError:
    from inference import ModelManager
    from prompt_templates import PromptTemplateFactory

logger = logging.getLogger(__name__)

@dataclass
class PromptVariation:
    """Container for prompt variations and their performance"""
    original_prompt: str
    variation: str
    strategy: str
    temperature: float
    top_p: float
    performance_score: float
    response_quality: float
    response_time: float
    tokens_per_second: float

class PromptOptimizer:
    """Advanced prompt engineering and optimization"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Advanced prompt strategies
        self.advanced_strategies = {
            "chain_of_thought": {
                "prefix": "Let's think through this step by step:\n1. ",
                "system": "Break down complex problems into logical steps."
            },
            "few_shot": {
                "prefix": "Here are some examples:\nQ: What is 2+2? A: 4\nQ: What is the capital of France? A: Paris\nQ: ",
                "system": "Learn from examples and apply the pattern."
            },
            "role_playing": {
                "prefix": "As an expert in this field, I would say: ",
                "system": "You are a knowledgeable expert. Demonstrate your expertise."
            },
            "critical_thinking": {
                "prefix": "Let me analyze this carefully, considering different perspectives: ",
                "system": "Think critically and consider multiple viewpoints."
            },
            "structured_response": {
                "prefix": "I'll organize my response as follows:\n**Main Point:** ",
                "system": "Provide well-structured, organized responses."
            },
            "creative_expansion": {
                "prefix": "Let me explore this creatively and think outside the box: ",
                "system": "Be creative and think unconventionally."
            },
            "socratic_method": {
                "prefix": "To understand this better, let me ask: What if we consider... ",
                "system": "Use questioning to deepen understanding."
            },
            "metacognitive": {
                "prefix": "I need to think about how I'm thinking about this problem: ",
                "system": "Reflect on your thought processes while reasoning."
            }
        }
        
        # Parameter optimization ranges
        self.param_ranges = {
            "temperature": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
            "top_p": [0.7, 0.8, 0.9, 0.95, 0.99],
            "repetition_penalty": [1.0, 1.05, 1.1, 1.15],
            "max_tokens": [50, 100, 200, 300, 500]
        }
        
        # Task-specific optimizations
        self.task_optimizations = {
            "reasoning": {
                "preferred_strategies": ["chain_of_thought", "critical_thinking", "metacognitive"],
                "optimal_temp": 0.3,
                "optimal_top_p": 0.9,
                "target_length": "medium"
            },
            "creativity": {
                "preferred_strategies": ["creative_expansion", "role_playing"],
                "optimal_temp": 0.8,
                "optimal_top_p": 0.95,
                "target_length": "long"
            },
            "knowledge": {
                "preferred_strategies": ["structured_response", "few_shot"],
                "optimal_temp": 0.2,
                "optimal_top_p": 0.8,
                "target_length": "short"
            },
            "code": {
                "preferred_strategies": ["chain_of_thought", "structured_response"],
                "optimal_temp": 0.1,
                "optimal_top_p": 0.9,
                "target_length": "medium"
            }
        }
    
    def generate_prompt_variations(self, base_prompt: str, task_type: str = "general") -> List[str]:
        """Generate multiple prompt variations using different strategies"""
        variations = []
        
        # Get task-specific preferred strategies
        if task_type in self.task_optimizations:
            strategies = self.task_optimizations[task_type]["preferred_strategies"]
        else:
            strategies = list(self.advanced_strategies.keys())[:4]  # Use first 4 as default
        
        # Apply each strategy
        for strategy_name in strategies:
            strategy = self.advanced_strategies[strategy_name]
            
            # Basic variation
            variation = f"{strategy['prefix']}{base_prompt}"
            variations.append((variation, strategy_name, strategy.get('system')))
            
            # Enhanced variations
            if strategy_name == "chain_of_thought":
                enhanced = f"Let me break this down systematically:\n1. First, I'll identify the key components\n2. Then I'll analyze each part\n3. Finally, I'll synthesize the answer\n\nQuestion: {base_prompt}\n\nStep 1:"
                variations.append((enhanced, f"{strategy_name}_enhanced", strategy.get('system')))
            
            elif strategy_name == "few_shot" and task_type == "reasoning":
                enhanced = f"Here are similar problems and solutions:\nExample 1: If a store sells 5 apples for $2, how much do 3 apples cost? Answer: $1.20 (5 apples = $2, so 1 apple = $0.40, therefore 3 apples = $1.20)\nExample 2: A car travels 120 miles in 2 hours. What's its speed? Answer: 60 mph (120 miles ÷ 2 hours = 60 mph)\n\nNow solve: {base_prompt}"
                variations.append((enhanced, f"{strategy_name}_reasoning", strategy.get('system')))
        
        return variations
    
    def optimize_parameters_for_model(self, model_name: str, prompt: str, task_type: str = "general") -> Dict[str, Any]:
        """Find optimal parameters for a specific model and prompt"""
        best_config = None
        best_score = 0
        results = []
        
        # Get base parameters for task
        base_params = self.task_optimizations.get(task_type, {})
        
        logger.info(f"Optimizing parameters for {model_name} on {task_type} task")
        
        # Test temperature variations
        temperatures = self.param_ranges["temperature"]
        if task_type in self.task_optimizations:
            # Focus around optimal temperature
            optimal_temp = base_params.get("optimal_temp", 0.7)
            temperatures = [t for t in temperatures if abs(t - optimal_temp) <= 0.4]
        
        for temp in temperatures[:3]:  # Limit to 3 temperatures for efficiency
            for top_p in self.param_ranges["top_p"][:3]:  # Limit to 3 top_p values
                try:
                    start_time = time.time()
                    result = self.model_manager.generate_text(
                        model_name,
                        prompt,
                        temperature=temp,
                        top_p=top_p,
                        max_tokens=150
                    )
                    
                    if "error" in result:
                        continue
                    
                    # Evaluate result quality
                    response = result["response"]
                    quality_score = self._evaluate_response_quality(response, task_type)
                    speed_score = min(result.get("tokens_per_second", 0) / 20, 1.0)  # Normalize speed
                    
                    # Combined score (70% quality, 30% speed)
                    combined_score = 0.7 * quality_score + 0.3 * speed_score
                    
                    config = {
                        "temperature": temp,
                        "top_p": top_p,
                        "quality_score": quality_score,
                        "speed_score": speed_score,
                        "combined_score": combined_score,
                        "response_time": time.time() - start_time,
                        "tokens_per_second": result.get("tokens_per_second", 0),
                        "response": response[:100] + "..." if len(response) > 100 else response
                    }
                    
                    results.append(config)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_config = config
                
                except Exception as e:
                    logger.error(f"Error testing params temp={temp}, top_p={top_p}: {e}")
                    continue
        
        return {
            "best_config": best_config,
            "all_results": sorted(results, key=lambda x: x["combined_score"], reverse=True),
            "optimization_summary": {
                "total_configs_tested": len(results),
                "best_score": best_score,
                "parameter_insights": self._analyze_parameter_trends(results)
            }
        }
    
    def _evaluate_response_quality(self, response: str, task_type: str) -> float:
        """Evaluate response quality based on task type"""
        if not response or len(response.strip()) < 3:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length appropriateness
        word_count = len(response.split())
        if task_type == "reasoning" and 20 <= word_count <= 150:
            score += 0.2
        elif task_type == "creativity" and word_count >= 30:
            score += 0.2
        elif task_type == "knowledge" and 5 <= word_count <= 80:
            score += 0.2
        elif task_type == "code" and any(keyword in response for keyword in ["def ", "function", "class ", "import "]):
            score += 0.3
        
        # Quality indicators
        if task_type == "reasoning":
            reasoning_words = ["because", "therefore", "since", "thus", "consequently", "first", "then", "finally"]
            if any(word in response.lower() for word in reasoning_words):
                score += 0.2
        
        # Coherence (avoid repetition)
        words = response.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.7:
                score += 0.1
        
        # Completeness
        if response.strip().endswith(('.', '!', '?', '"', "'", ':')):
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_parameter_trends(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in parameter optimization results"""
        if not results:
            return {}
        
        # Group by temperature
        temp_scores = {}
        top_p_scores = {}
        
        for result in results:
            temp = result["temperature"]
            top_p = result["top_p"]
            score = result["combined_score"]
            
            if temp not in temp_scores:
                temp_scores[temp] = []
            temp_scores[temp].append(score)
            
            if top_p not in top_p_scores:
                top_p_scores[top_p] = []
            top_p_scores[top_p].append(score)
        
        # Calculate averages
        temp_avg = {temp: np.mean(scores) for temp, scores in temp_scores.items()}
        top_p_avg = {top_p: np.mean(scores) for top_p, scores in top_p_scores.items()}
        
        return {
            "best_temperature": max(temp_avg.items(), key=lambda x: x[1]),
            "best_top_p": max(top_p_avg.items(), key=lambda x: x[1]),
            "temperature_trend": sorted(temp_avg.items()),
            "top_p_trend": sorted(top_p_avg.items()),
            "insights": self._generate_parameter_insights(temp_avg, top_p_avg)
        }
    
    def _generate_parameter_insights(self, temp_avg: Dict, top_p_avg: Dict) -> List[str]:
        """Generate insights about parameter performance"""
        insights = []
        
        # Temperature insights
        temps = sorted(temp_avg.items())
        if temps[0][1] > temps[-1][1]:
            insights.append("Lower temperatures perform better - model benefits from more focused responses")
        elif temps[-1][1] > temps[0][1]:
            insights.append("Higher temperatures perform better - model benefits from more creative responses")
        
        # Top-p insights
        top_ps = sorted(top_p_avg.items())
        if max(top_p_avg.values()) == top_p_avg.get(0.9):
            insights.append("Top-p of 0.9 is optimal - good balance of diversity and focus")
        elif max(top_p_avg.values()) == top_p_avg.get(0.95):
            insights.append("Higher top-p (0.95) works best - model benefits from more diverse token selection")
        
        return insights
    
    async def comprehensive_prompt_optimization(self, model_name: str, base_prompts: List[str], task_type: str = "general") -> Dict[str, Any]:
        """Run comprehensive prompt optimization for a model"""
        all_results = []
        
        logger.info(f"Running comprehensive prompt optimization for {model_name}")
        
        for prompt in base_prompts:
            logger.info(f"Optimizing prompt: {prompt[:50]}...")
            
            # Generate prompt variations
            variations = self.generate_prompt_variations(prompt, task_type)
            
            # Test each variation
            for variation, strategy, system_prompt in variations:
                try:
                    # Test with optimal parameters for task
                    base_params = self.task_optimizations.get(task_type, {})
                    temp = base_params.get("optimal_temp", 0.7)
                    top_p = base_params.get("optimal_top_p", 0.9)
                    
                    start_time = time.time()
                    result = self.model_manager.generate_text(
                        model_name,
                        variation,
                        system_prompt=system_prompt,
                        temperature=temp,
                        top_p=top_p,
                        max_tokens=200
                    )
                    
                    if "error" in result:
                        continue
                    
                    # Evaluate performance
                    quality_score = self._evaluate_response_quality(result["response"], task_type)
                    
                    prompt_result = PromptVariation(
                        original_prompt=prompt,
                        variation=variation,
                        strategy=strategy,
                        temperature=temp,
                        top_p=top_p,
                        performance_score=quality_score,
                        response_quality=quality_score,
                        response_time=time.time() - start_time,
                        tokens_per_second=result.get("tokens_per_second", 0)
                    )
                    
                    all_results.append({
                        "original_prompt": prompt,
                        "variation": variation,
                        "strategy": strategy,
                        "temperature": temp,
                        "top_p": top_p,
                        "quality_score": quality_score,
                        "response_time": time.time() - start_time,
                        "tokens_per_second": result.get("tokens_per_second", 0),
                        "response": result["response"]
                    })
                    
                except Exception as e:
                    logger.error(f"Error testing variation: {e}")
                    continue
        
        # Analyze results
        analysis = self._analyze_prompt_optimization_results(all_results, task_type)
        
        return {
            "model_name": model_name,
            "task_type": task_type,
            "total_variations_tested": len(all_results),
            "results": all_results,
            "analysis": analysis,
            "recommendations": self._generate_optimization_recommendations(analysis, model_name)
        }
    
    def _analyze_prompt_optimization_results(self, results: List[Dict], task_type: str) -> Dict[str, Any]:
        """Analyze prompt optimization results"""
        if not results:
            return {}
        
        # Group by strategy
        strategy_performance = {}
        for result in results:
            strategy = result["strategy"]
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(result["quality_score"])
        
        # Calculate strategy averages
        strategy_avg = {strategy: np.mean(scores) for strategy, scores in strategy_performance.items()}
        
        # Find best performing variations
        sorted_results = sorted(results, key=lambda x: x["quality_score"], reverse=True)
        top_variations = sorted_results[:5]
        
        # Performance insights
        best_strategy = max(strategy_avg.items(), key=lambda x: x[1])
        worst_strategy = min(strategy_avg.items(), key=lambda x: x[1])
        
        return {
            "best_strategy": {"name": best_strategy[0], "avg_score": best_strategy[1]},
            "worst_strategy": {"name": worst_strategy[0], "avg_score": worst_strategy[1]},
            "strategy_rankings": sorted(strategy_avg.items(), key=lambda x: x[1], reverse=True),
            "top_variations": top_variations,
            "overall_stats": {
                "avg_quality": np.mean([r["quality_score"] for r in results]),
                "avg_speed": np.mean([r["tokens_per_second"] for r in results]),
                "quality_std": np.std([r["quality_score"] for r in results])
            }
        }
    
    def _generate_optimization_recommendations(self, analysis: Dict, model_name: str) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not analysis:
            return ["Insufficient data for recommendations"]
        
        # Strategy recommendations
        best_strategy = analysis["best_strategy"]
        recommendations.append(f"Use '{best_strategy['name']}' strategy for best results (avg score: {best_strategy['avg_score']:.2f})")
        
        # Performance recommendations
        overall_stats = analysis.get("overall_stats", {})
        avg_quality = overall_stats.get("avg_quality", 0)
        
        if avg_quality < 0.6:
            recommendations.append(f"{model_name} may benefit from fine-tuning or different prompt templates")
        elif avg_quality > 0.8:
            recommendations.append(f"{model_name} shows excellent prompt responsiveness")
        
        # Strategy-specific insights
        strategy_rankings = analysis.get("strategy_rankings", [])
        if len(strategy_rankings) >= 2:
            top_two = strategy_rankings[:2]
            recommendations.append(f"Top strategies: {top_two[0][0]} ({top_two[0][1]:.2f}) and {top_two[1][0]} ({top_two[1][1]:.2f})")
        
        # Speed vs quality trade-offs
        avg_speed = overall_stats.get("avg_speed", 0)
        if avg_speed < 10:
            recommendations.append("Consider reducing max_tokens or using quantization for faster inference")
        
        return recommendations


# Example usage function
def run_optimization_example():
    """Example of how to use the prompt optimizer"""
    model_manager = ModelManager()
    optimizer = PromptOptimizer(model_manager)
    
    # Example prompts for different tasks
    test_prompts = {
        "reasoning": [
            "What is 47 * 23?",
            "If a train travels 60 mph for 2.5 hours, how far does it go?"
        ],
        "creativity": [
            "Write a short story about a robot",
            "Describe a color to someone who has never seen"
        ],
        "knowledge": [
            "Who wrote Pride and Prejudice?",
            "What is the capital of Australia?"
        ]
    }
    
    # Test with available models
    available_models = ["deepseek-r1-distill-qwen-1.5b", "qwen2.5-1.5b"]
    
    for model in available_models:
        print(f"\n=== Optimizing {model} ===")
        
        for task_type, prompts in test_prompts.items():
            print(f"\nTask: {task_type}")
            
            # Parameter optimization
            param_results = optimizer.optimize_parameters_for_model(model, prompts[0], task_type)
            print(f"Best config: {param_results['best_config']}")
            
            # Prompt optimization (async would need proper event loop)
            # optimization_results = await optimizer.comprehensive_prompt_optimization(model, prompts, task_type)
            # print(f"Best strategy: {optimization_results['analysis']['best_strategy']}")


if __name__ == "__main__":
    run_optimization_example()