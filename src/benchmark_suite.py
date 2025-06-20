#!/usr/bin/env python3
"""
Comprehensive benchmark suite for evaluating model performance across multiple dimensions
"""

import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from datetime import datetime
import psutil
import gc
from PIL import Image

try:
    from .inference import ModelManager
    from .prompt_templates import PromptTemplateFactory, get_model_params
except ImportError:
    from inference import ModelManager
    from prompt_templates import PromptTemplateFactory, get_model_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_name: str
    task_type: str
    task_name: str
    prompt_strategy: str
    score: float
    metrics: Dict[str, Any]
    timestamp: str

class BenchmarkSuite:
    """Comprehensive benchmarking system"""
    
    def __init__(self, models_dir: str = "models", results_dir: str = "benchmarks"):
        self.model_manager = ModelManager(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
        
        # Define benchmark tasks
        self.text_tasks = {
            "reasoning": [
                "What is 47 * 23 + 15?",
                "If a train travels 60 mph for 2.5 hours, how far does it go?",
                "Sarah has 3 times as many apples as John. If John has 8 apples, how many do they have together?",
                "A square has a perimeter of 24 cm. What is its area?",
                "If it takes 3 people 4 hours to paint a fence, how long would it take 6 people?"
            ],
            "language": [
                "Translate to French: 'The quick brown fox jumps over the lazy dog'",
                "Correct the grammar: 'Me and my friend goes to store yesterday'",
                "Explain the difference between 'affect' and 'effect'",
                "Write a haiku about artificial intelligence",
                "What is the past tense of 'run'?"
            ],
            "knowledge": [
                "Who wrote 'Pride and Prejudice'?",
                "What is the capital of Australia?",
                "When did World War II end?",
                "What is photosynthesis?",
                "Name three programming languages"
            ],
            "creativity": [
                "Write a short story about a robot learning to paint",
                "Create a recipe for a sandwich using only fruits",
                "Invent a new sport and explain its rules",
                "Describe a color to someone who has never seen",
                "Write a poem about the internet"
            ],
            "code": [
                "Write a Python function to find the largest number in a list",
                "Explain what a for loop does in programming",
                "How do you reverse a string in Python?",
                "What is the difference between a list and a tuple?",
                "Write a function to check if a number is prime"
            ]
        }
        
        self.vision_tasks = {
            "image_description": [
                "Describe what you see in this image",
                "What objects are visible in this image?",
                "What is the main subject of this image?",
                "Describe the colors and lighting in this image",
                "What emotions does this image convey?"
            ],
            "visual_reasoning": [
                "Count the number of objects in this image",
                "What is the relationship between the objects?",
                "What might happen next in this scene?",
                "Is this image taken indoors or outdoors?",
                "What time of day was this photo likely taken?"
            ]
        }
        
        # Prompt strategies to test
        self.prompt_strategies = {
            "direct": {
                "system_prompt": None,
                "prefix": "",
                "suffix": ""
            },
            "thinking": {
                "system_prompt": "Think step by step and explain your reasoning.",
                "prefix": "Let me think about this carefully: ",
                "suffix": ""
            },
            "expert": {
                "system_prompt": "You are an expert in this field. Provide a detailed and accurate response.",
                "prefix": "As an expert, ",
                "suffix": ""
            },
            "concise": {
                "system_prompt": "Provide brief, direct answers.",
                "prefix": "In brief: ",
                "suffix": ""
            }
        }
    
    def create_test_images(self) -> List[str]:
        """Create simple test images for vision benchmarks"""
        test_images = []
        
        # Create geometric patterns
        for i, (name, pattern) in enumerate([
            ("red_square", lambda: self._create_colored_square((255, 0, 0))),
            ("blue_circle", lambda: self._create_circle((0, 0, 255))),
            ("gradient", lambda: self._create_gradient()),
            ("checkerboard", lambda: self._create_checkerboard()),
            ("mixed_shapes", lambda: self._create_mixed_shapes())
        ]):
            img_path = self.results_dir / f"test_{name}.png"
            if not img_path.exists():
                img = pattern()
                img.save(img_path)
            test_images.append(str(img_path))
        
        return test_images
    
    def _create_colored_square(self, color: Tuple[int, int, int]) -> Image.Image:
        """Create a colored square"""
        img = Image.new('RGB', (224, 224), color)
        return img
    
    def _create_circle(self, color: Tuple[int, int, int]) -> Image.Image:
        """Create a circle"""
        from PIL import ImageDraw
        img = Image.new('RGB', (224, 224), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse([50, 50, 174, 174], fill=color)
        return img
    
    def _create_gradient(self) -> Image.Image:
        """Create a gradient"""
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            img_array[:, i, 0] = int(255 * i / 224)  # Red gradient
        return Image.fromarray(img_array)
    
    def _create_checkerboard(self) -> Image.Image:
        """Create a checkerboard pattern"""
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        square_size = 28
        for i in range(0, 224, square_size):
            for j in range(0, 224, square_size):
                if (i // square_size + j // square_size) % 2:
                    img_array[i:i+square_size, j:j+square_size] = 255
        return Image.fromarray(img_array)
    
    def _create_mixed_shapes(self) -> Image.Image:
        """Create mixed geometric shapes"""
        from PIL import ImageDraw
        img = Image.new('RGB', (224, 224), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Red rectangle
        draw.rectangle([20, 20, 100, 80], fill=(255, 0, 0))
        # Blue circle
        draw.ellipse([120, 30, 180, 90], fill=(0, 0, 255))
        # Green triangle
        draw.polygon([(50, 120), (100, 200), (0, 200)], fill=(0, 255, 0))
        
        return img
    
    def evaluate_text_response(self, task_type: str, prompt: str, response: str) -> float:
        """Evaluate text response quality (simplified scoring)"""
        if not response or len(response.strip()) < 5:
            return 0.0
        
        # Basic quality indicators
        score = 0.5  # Base score for valid response
        
        # Length appropriateness
        response_len = len(response.split())
        if task_type == "reasoning":
            if 10 <= response_len <= 100:
                score += 0.2
        elif task_type == "creativity":
            if response_len >= 20:
                score += 0.2
        elif task_type == "knowledge":
            if 3 <= response_len <= 50:
                score += 0.2
        
        # Content quality indicators
        if task_type == "reasoning" and any(word in response.lower() for word in ["because", "therefore", "so", "thus"]):
            score += 0.1
        
        if task_type == "code" and any(word in response for word in ["def ", "function", "return", "for ", "if "]):
            score += 0.2
        
        # Completeness (not cut off abruptly)
        if response.strip().endswith(('.', '!', '?', '"', "'")):
            score += 0.1
        
        # Avoid repetition
        words = response.lower().split()
        if len(set(words)) / len(words) > 0.7:  # Good word diversity
            score += 0.1
        
        return min(score, 1.0)
    
    def evaluate_vision_response(self, task_type: str, image_path: str, response: str) -> float:
        """Evaluate vision response quality"""
        if not response or len(response.strip()) < 3:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check for relevant visual terms
        visual_terms = ["image", "picture", "photo", "see", "shows", "visible", "color", "shape", "object"]
        if any(term in response.lower() for term in visual_terms):
            score += 0.2
        
        # Check for specific color mentions if image has distinct colors
        image_name = Path(image_path).stem
        if "red" in image_name and "red" in response.lower():
            score += 0.1
        if "blue" in image_name and "blue" in response.lower():
            score += 0.1
        if "circle" in image_name and "circle" in response.lower():
            score += 0.1
        if "square" in image_name and ("square" in response.lower() or "rectangle" in response.lower()):
            score += 0.1
        
        return min(score, 1.0)
    
    async def benchmark_text_model(self, model_name: str) -> List[BenchmarkResult]:
        """Benchmark a text generation model"""
        results = []
        
        for strategy_name, strategy in self.prompt_strategies.items():
            logger.info(f"Testing {model_name} with {strategy_name} strategy")
            
            for task_type, prompts in self.text_tasks.items():
                task_scores = []
                task_metrics = []
                
                for prompt in prompts:
                    # Format prompt with strategy
                    formatted_prompt = f"{strategy['prefix']}{prompt}{strategy['suffix']}"
                    
                    try:
                        # Measure performance
                        start_time = time.time()
                        result = self.model_manager.generate_text(
                            model_name, 
                            formatted_prompt,
                            system_prompt=strategy["system_prompt"],
                            max_tokens=150,
                            temperature=0.7
                        )
                        end_time = time.time()
                        
                        if "error" in result:
                            logger.warning(f"Error in {model_name}: {result['error']}")
                            continue
                        
                        # Evaluate response
                        score = self.evaluate_text_response(task_type, prompt, result["response"])
                        task_scores.append(score)
                        
                        # Collect metrics
                        metrics = {
                            "latency": end_time - start_time,
                            "tokens_generated": result.get("tokens_generated", 0),
                            "tokens_per_second": result.get("tokens_per_second", 0),
                            "response_length": len(result["response"]),
                            "prompt": formatted_prompt,
                            "response": result["response"]
                        }
                        task_metrics.append(metrics)
                        
                        # Memory cleanup
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                        
                        # Small delay to prevent overheating
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Error testing {model_name} on {prompt}: {e}")
                        continue
                
                if task_scores:
                    # Aggregate results for this task
                    avg_score = np.mean(task_scores)
                    avg_metrics = {
                        "avg_latency": np.mean([m["latency"] for m in task_metrics]),
                        "avg_tokens_per_second": np.mean([m["tokens_per_second"] for m in task_metrics]),
                        "avg_response_length": np.mean([m["response_length"] for m in task_metrics]),
                        "sample_count": len(task_scores),
                        "score_std": np.std(task_scores),
                        "examples": task_metrics[:2]  # Keep first 2 examples
                    }
                    
                    result = BenchmarkResult(
                        model_name=model_name,
                        task_type="text",
                        task_name=task_type,
                        prompt_strategy=strategy_name,
                        score=avg_score,
                        metrics=avg_metrics,
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(result)
        
        return results
    
    async def benchmark_vision_model(self, model_name: str) -> List[BenchmarkResult]:
        """Benchmark a vision-language model"""
        results = []
        test_images = self.create_test_images()
        
        for strategy_name, strategy in self.prompt_strategies.items():
            logger.info(f"Testing vision model {model_name} with {strategy_name} strategy")
            
            for task_type, prompts in self.vision_tasks.items():
                task_scores = []
                task_metrics = []
                
                for image_path in test_images:
                    for prompt in prompts:
                        # Format prompt with strategy
                        formatted_prompt = f"{strategy['prefix']}{prompt}{strategy['suffix']}"
                        
                        try:
                            start_time = time.time()
                            result = self.model_manager.analyze_image(
                                model_name,
                                image_path,
                                formatted_prompt
                            )
                            end_time = time.time()
                            
                            if "error" in result:
                                logger.warning(f"Error in {model_name}: {result['error']}")
                                continue
                            
                            # Evaluate response
                            score = self.evaluate_vision_response(task_type, image_path, result["response"])
                            task_scores.append(score)
                            
                            # Collect metrics
                            metrics = {
                                "latency": end_time - start_time,
                                "response_length": len(result["response"]),
                                "image_path": image_path,
                                "prompt": formatted_prompt,
                                "response": result["response"]
                            }
                            task_metrics.append(metrics)
                            
                            # Memory cleanup
                            gc.collect()
                            if torch.backends.mps.is_available():
                                torch.mps.empty_cache()
                            
                            await asyncio.sleep(0.2)  # Longer delay for vision models
                            
                        except Exception as e:
                            logger.error(f"Error testing {model_name} on vision task: {e}")
                            continue
                
                if task_scores:
                    # Aggregate results
                    avg_score = np.mean(task_scores)
                    avg_metrics = {
                        "avg_latency": np.mean([m["latency"] for m in task_metrics]),
                        "avg_response_length": np.mean([m["response_length"] for m in task_metrics]),
                        "sample_count": len(task_scores),
                        "score_std": np.std(task_scores),
                        "examples": task_metrics[:2]
                    }
                    
                    result = BenchmarkResult(
                        model_name=model_name,
                        task_type="vision",
                        task_name=task_type,
                        prompt_strategy=strategy_name,
                        score=avg_score,
                        metrics=avg_metrics,
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(result)
        
        return results
    
    def measure_system_metrics(self) -> Dict[str, Any]:
        """Measure system resource usage"""
        return {
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "gpu_memory_mb": torch.mps.driver_allocated_memory() / (1024**2) if torch.backends.mps.is_available() else 0
        }
    
    async def run_comprehensive_benchmark(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive benchmarks on specified models"""
        if models is None:
            models = list(self.model_manager.model_configs.keys())
        
        all_results = []
        system_metrics = []
        
        logger.info(f"Starting comprehensive benchmark of {len(models)} models")
        start_time = time.time()
        
        for model_name in models:
            logger.info(f"Benchmarking {model_name}")
            model_start = time.time()
            
            # Measure system state before model
            pre_metrics = self.measure_system_metrics()
            
            try:
                model_config = self.model_manager.model_configs[model_name]
                
                if model_config["type"] == "text":
                    results = await self.benchmark_text_model(model_name)
                elif model_config["type"] == "vision-language":
                    results = await self.benchmark_vision_model(model_name)
                else:
                    logger.info(f"Skipping {model_name} - type {model_config['type']} not benchmarked")
                    continue
                
                all_results.extend(results)
                
                # Measure system state after model
                post_metrics = self.measure_system_metrics()
                
                system_metrics.append({
                    "model": model_name,
                    "duration": time.time() - model_start,
                    "pre_metrics": pre_metrics,
                    "post_metrics": post_metrics
                })
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                continue
            
            # Cleanup between models
            if model_name in self.model_manager.loaded_models:
                del self.model_manager.loaded_models[model_name]
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        total_time = time.time() - start_time
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        
        benchmark_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_time,
            "models_tested": models,
            "results": [
                {
                    "model_name": r.model_name,
                    "task_type": r.task_type,
                    "task_name": r.task_name,
                    "prompt_strategy": r.prompt_strategy,
                    "score": r.score,
                    "metrics": r.metrics,
                    "timestamp": r.timestamp
                }
                for r in all_results
            ],
            "system_metrics": system_metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        logger.info(f"Benchmark complete! Results saved to {results_file}")
        return benchmark_data
    
    def analyze_results(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results and provide insights"""
        results = results_data["results"]
        
        # Group results by model and strategy
        model_performance = {}
        strategy_performance = {}
        task_performance = {}
        
        for result in results:
            model = result["model_name"]
            strategy = result["prompt_strategy"]
            task = result["task_name"]
            
            # Model performance
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(result["score"])
            
            # Strategy performance
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(result["score"])
            
            # Task performance
            if task not in task_performance:
                task_performance[task] = {}
            if model not in task_performance[task]:
                task_performance[task][model] = []
            task_performance[task][model].append(result["score"])
        
        # Calculate averages
        model_avg = {model: np.mean(scores) for model, scores in model_performance.items()}
        strategy_avg = {strategy: np.mean(scores) for strategy, scores in strategy_performance.items()}
        
        # Find best combinations
        best_model = max(model_avg.items(), key=lambda x: x[1])
        best_strategy = max(strategy_avg.items(), key=lambda x: x[1])
        
        # Performance insights
        insights = {
            "overall_performance": {
                "best_model": {"name": best_model[0], "avg_score": best_model[1]},
                "best_strategy": {"name": best_strategy[0], "avg_score": best_strategy[1]},
                "model_rankings": sorted(model_avg.items(), key=lambda x: x[1], reverse=True),
                "strategy_rankings": sorted(strategy_avg.items(), key=lambda x: x[1], reverse=True)
            },
            "task_specific": {
                task: {
                    "best_model": max(models.items(), key=lambda x: np.mean(x[1]))[0],
                    "avg_scores": {model: np.mean(scores) for model, scores in models.items()}
                }
                for task, models in task_performance.items()
            },
            "recommendations": self._generate_recommendations(results)
        }
        
        return insights
    
    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Group by model for analysis
        model_data = {}
        for result in results:
            model = result["model_name"]
            if model not in model_data:
                model_data[model] = {"scores": [], "latencies": [], "strategies": {}}
            
            model_data[model]["scores"].append(result["score"])
            if "avg_latency" in result["metrics"]:
                model_data[model]["latencies"].append(result["metrics"]["avg_latency"])
            
            strategy = result["prompt_strategy"]
            if strategy not in model_data[model]["strategies"]:
                model_data[model]["strategies"][strategy] = []
            model_data[model]["strategies"][strategy].append(result["score"])
        
        # Generate recommendations
        for model, data in model_data.items():
            avg_score = np.mean(data["scores"])
            avg_latency = np.mean(data["latencies"]) if data["latencies"] else 0
            
            # Best strategy for this model
            best_strategy = max(data["strategies"].items(), key=lambda x: np.mean(x[1]))
            
            recommendations.append(f"{model}: Use '{best_strategy[0]}' strategy (avg score: {np.mean(best_strategy[1]):.2f})")
            
            if avg_latency > 5:
                recommendations.append(f"{model}: Consider reducing max_tokens or using quantization for faster inference")
            
            if avg_score < 0.6:
                recommendations.append(f"{model}: Scores below 0.6 - consider prompt engineering or fine-tuning")
        
        return recommendations


async def main():
    """Run benchmarks from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive model benchmarking")
    parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    parser.add_argument("--output", help="Output directory for results")
    
    args = parser.parse_args()
    
    benchmark = BenchmarkSuite(results_dir=args.output or "benchmarks")
    results = await benchmark.run_comprehensive_benchmark(args.models)
    insights = benchmark.analyze_results(results)
    
    print("\n=== BENCHMARK INSIGHTS ===")
    print(f"Best Model: {insights['overall_performance']['best_model']['name']}")
    print(f"Best Strategy: {insights['overall_performance']['best_strategy']['name']}")
    print("\nRecommendations:")
    for rec in insights['recommendations']:
        print(f"- {rec}")


if __name__ == "__main__":
    asyncio.run(main())