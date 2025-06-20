#!/usr/bin/env python3
"""
Quick benchmark script for immediate results
"""

import sys
import json
import time
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from inference import ModelManager
from prompt_optimization import PromptOptimizer
# from weight_optimization import WeightOptimizer  # Skip for now due to dependencies

async def run_quick_benchmark():
    """Run a quick benchmark of key models"""
    print("=== QUICK BENCHMARK SUITE ===\n")
    
    # Initialize components
    model_manager = ModelManager()
    prompt_optimizer = PromptOptimizer(model_manager)
    # weight_optimizer = WeightOptimizer()  # Skip for now
    
    # Test models
    test_models = ["llama-3.2-3b"]
    
    # Quick test prompts
    test_prompts = {
        "reasoning": "What is 15 * 8?",
        "knowledge": "Who wrote Romeo and Juliet?",
        "creativity": "Write a haiku about computers"
    }
    
    results = {}
    
    for model_name in test_models:
        print(f"--- Testing {model_name} ---")
        model_results = {}
        
        # Test basic performance
        for task, prompt in test_prompts.items():
            try:
                start_time = time.time()
                result = model_manager.generate_text(
                    model_name, 
                    prompt,
                    max_tokens=100,
                    temperature=0.7
                )
                end_time = time.time()
                
                if "error" not in result:
                    model_results[task] = {
                        "prompt": prompt,
                        "response": result["response"][:200] + "..." if len(result["response"]) > 200 else result["response"],
                        "time": end_time - start_time,
                        "tokens_per_second": result.get("tokens_per_second", 0),
                        "tokens_generated": result.get("tokens_generated", 0)
                    }
                    print(f"  {task}: {result.get('tokens_per_second', 0):.1f} tokens/sec")
                else:
                    print(f"  {task}: ERROR - {result['error']}")
                    model_results[task] = {"error": result["error"]}
                
            except Exception as e:
                print(f"  {task}: FAILED - {str(e)}")
                model_results[task] = {"error": str(e)}
        
        # Test prompt optimization (quick version)
        print(f"  Testing prompt strategies...")
        try:
            prompt_results = prompt_optimizer.optimize_parameters_for_model(
                model_name, 
                "What is artificial intelligence?", 
                "knowledge"
            )
            
            if prompt_results and prompt_results.get("best_config"):
                best_config = prompt_results["best_config"]
                model_results["optimization"] = {
                    "best_temperature": best_config.get("temperature"),
                    "best_top_p": best_config.get("top_p"),
                    "quality_score": best_config.get("quality_score"),
                    "speed_score": best_config.get("speed_score")
                }
                print(f"  Best params: temp={best_config.get('temperature')}, top_p={best_config.get('top_p')}")
            
        except Exception as e:
            print(f"  Optimization failed: {e}")
            model_results["optimization"] = {"error": str(e)}
        
        results[model_name] = model_results
        print()
    
    # Test vision models (if available)
    vision_models = ["blip2-base", "blip-base"]
    for model_name in vision_models:
        if model_name in model_manager.model_configs:
            print(f"--- Testing Vision Model {model_name} ---")
            try:
                # Create a simple test image
                from PIL import Image
                import numpy as np
                
                # Create red square
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                img_array[:, :, 0] = 255  # Red
                img = Image.fromarray(img_array)
                test_image_path = "/tmp/test_red_square.png"
                img.save(test_image_path)
                
                start_time = time.time()
                result = model_manager.analyze_image(
                    model_name,
                    test_image_path,
                    "What color is this image?"
                )
                end_time = time.time()
                
                if "error" not in result:
                    results[model_name] = {
                        "response": result["response"],
                        "time": end_time - start_time,
                        "type": "vision"
                    }
                    print(f"  Response: {result['response'][:100]}...")
                    print(f"  Time: {end_time - start_time:.2f}s")
                else:
                    print(f"  ERROR: {result['error']}")
                    results[model_name] = {"error": result["error"]}
                    
            except Exception as e:
                print(f"  Vision test failed: {e}")
                results[model_name] = {"error": str(e)}
            print()
    
    # Weight optimization analysis (skipped due to dependencies)
    print("--- Weight Optimization Analysis ---")
    print("  Skipped due to missing dependencies (datasets library)")
    results["weight_optimization"] = {"skipped": "Missing dependencies"}
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmarks/quick_benchmark_{timestamp}.json"
    
    Path("benchmarks").mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== BENCHMARK COMPLETE ===")
    print(f"Results saved to: {results_file}")
    
    return results

def analyze_quick_results(results):
    """Analyze and summarize quick benchmark results"""
    print("\n=== QUICK ANALYSIS ===")
    
    text_models = {k: v for k, v in results.items() if k.endswith("1.5b")}
    vision_models = {k: v for k, v in results.items() if "blip" in k}
    
    print("\nText Model Performance:")
    for model, data in text_models.items():
        if "error" not in data:
            avg_speed = 0
            task_count = 0
            
            for task, task_data in data.items():
                if task != "optimization" and "tokens_per_second" in task_data:
                    avg_speed += task_data["tokens_per_second"]
                    task_count += 1
            
            if task_count > 0:
                avg_speed /= task_count
                print(f"  {model}: {avg_speed:.1f} avg tokens/sec")
                
                # Check optimization results
                if "optimization" in data and "best_temperature" in data["optimization"]:
                    opt = data["optimization"]
                    print(f"    Best settings: temp={opt['best_temperature']}, quality={opt.get('quality_score', 0):.2f}")
    
    print("\nVision Model Performance:")
    for model, data in vision_models.items():
        if "error" not in data and "time" in data:
            print(f"  {model}: {data['time']:.2f}s response time")
    
    print("\nKey Findings:")
    
    # Compare text models
    if len(text_models) >= 2:
        model_speeds = {}
        for model, data in text_models.items():
            speeds = []
            for task, task_data in data.items():
                if task != "optimization" and "tokens_per_second" in task_data:
                    speeds.append(task_data["tokens_per_second"])
            if speeds:
                model_speeds[model] = sum(speeds) / len(speeds)
        
        if model_speeds:
            fastest = max(model_speeds.items(), key=lambda x: x[1])
            print(f"- Fastest text model: {fastest[0]} ({fastest[1]:.1f} tokens/sec)")
    
    # Check for optimization benefits
    optimization_benefits = []
    for model, data in text_models.items():
        if "optimization" in data and "quality_score" in data["optimization"]:
            quality = data["optimization"]["quality_score"]
            if quality > 0.7:
                optimization_benefits.append(f"{model} shows good optimization potential (quality: {quality:.2f})")
    
    for benefit in optimization_benefits:
        print(f"- {benefit}")
    
    # Weight optimization insights
    if "weight_optimization" in results and "recommendations" in results["weight_optimization"]:
        print("- Weight optimization opportunities identified")
        for rec in results["weight_optimization"]["recommendations"][:3]:  # Top 3
            print(f"  * {rec}")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        results = await run_quick_benchmark()
        analyze_quick_results(results)
    
    asyncio.run(main())