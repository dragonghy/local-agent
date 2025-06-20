#!/usr/bin/env python3
"""
Weight optimization and model fine-tuning utilities
"""

import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import time

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Utilities for model quantization and compression"""
    
    @staticmethod
    def quantize_model_int8(model: nn.Module) -> nn.Module:
        """Apply INT8 quantization to reduce memory usage"""
        try:
            if hasattr(torch, 'quantization'):
                # PyTorch quantization (if available)
                model.eval()
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied INT8 quantization")
                return quantized_model
            else:
                logger.warning("PyTorch quantization not available")
                return model
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    @staticmethod
    def measure_model_size(model: nn.Module) -> Dict[str, float]:
        """Measure model memory usage"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            "total_mb": total_size / 1024 / 1024,
            "params_mb": param_size / 1024 / 1024,
            "buffers_mb": buffer_size / 1024 / 1024,
            "total_params": sum(p.numel() for p in model.parameters())
        }

class LoRAOptimizer:
    """Low-Rank Adaptation (LoRA) for efficient fine-tuning"""
    
    def __init__(self, rank: int = 16, alpha: float = 32):
        self.rank = rank
        self.alpha = alpha
        self.lora_layers = {}
    
    def add_lora_to_linear(self, module: nn.Linear, name: str) -> nn.Module:
        """Add LoRA adaptation to a linear layer"""
        in_features = module.in_features
        out_features = module.out_features
        
        # Create LoRA matrices
        lora_A = nn.Parameter(torch.randn(in_features, self.rank) * 0.01)
        lora_B = nn.Parameter(torch.zeros(self.rank, out_features))
        
        # Store original weights (frozen)
        module.weight.requires_grad = False
        
        # Add LoRA parameters
        module.lora_A = lora_A
        module.lora_B = lora_B
        module.lora_alpha = self.alpha
        module.lora_rank = self.rank
        
        # Modify forward pass
        original_forward = module.forward
        
        def lora_forward(x):
            result = original_forward(x)
            lora_result = x @ module.lora_A @ module.lora_B * (module.lora_alpha / module.lora_rank)
            return result + lora_result
        
        module.forward = lora_forward
        self.lora_layers[name] = module
        
        return module
    
    def apply_lora_to_model(self, model: nn.Module, target_modules: List[str] = None) -> nn.Module:
        """Apply LoRA to specified modules in the model"""
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Common attention modules
        
        for name, module in model.named_modules():
            if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
                logger.info(f"Adding LoRA to {name}")
                self.add_lora_to_linear(module, name)
        
        return model
    
    def get_trainable_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count trainable parameters after LoRA application"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }

class WeightOptimizer:
    """Main weight optimization class"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.optimization_results = {}
    
    def analyze_model_efficiency(self, model_name: str, model: nn.Module, tokenizer) -> Dict[str, Any]:
        """Analyze current model efficiency metrics"""
        
        # Model size analysis
        size_metrics = ModelQuantizer.measure_model_size(model)
        
        # Parameter analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory efficiency test
        memory_metrics = self._measure_memory_efficiency(model, tokenizer)
        
        # Layer analysis
        layer_analysis = self._analyze_model_layers(model)
        
        return {
            "model_name": model_name,
            "size_metrics": size_metrics,
            "parameter_metrics": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": total_params - trainable_params
            },
            "memory_metrics": memory_metrics,
            "layer_analysis": layer_analysis,
            "optimization_opportunities": self._identify_optimization_opportunities(
                size_metrics, layer_analysis
            )
        }
    
    def _measure_memory_efficiency(self, model: nn.Module, tokenizer) -> Dict[str, float]:
        """Measure memory usage during inference"""
        model.eval()
        
        # Test with different input lengths
        test_lengths = [10, 50, 100, 200]
        memory_usage = {}
        
        for length in test_lengths:
            try:
                # Create test input
                test_input = "This is a test prompt. " * (length // 5)
                inputs = tokenizer(test_input, return_tensors="pt", max_length=length, truncation=True)
                
                # Measure memory before
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    initial_memory = torch.mps.current_allocated_memory()
                else:
                    initial_memory = 0
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Measure memory after
                if torch.backends.mps.is_available():
                    peak_memory = torch.mps.current_allocated_memory()
                    memory_used = (peak_memory - initial_memory) / 1024 / 1024  # MB
                else:
                    memory_used = 0
                
                memory_usage[f"length_{length}"] = memory_used
                
                # Cleanup
                del inputs, outputs
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
            except Exception as e:
                logger.error(f"Memory measurement failed for length {length}: {e}")
                memory_usage[f"length_{length}"] = -1
        
        return memory_usage
    
    def _analyze_model_layers(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model layer structure and efficiency"""
        layer_info = {}
        total_params_by_type = {}
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            if module_type not in total_params_by_type:
                total_params_by_type[module_type] = 0
            
            if hasattr(module, 'weight') and module.weight is not None:
                params = module.weight.numel()
                total_params_by_type[module_type] += params
                
                layer_info[name] = {
                    "type": module_type,
                    "parameters": params,
                    "shape": list(module.weight.shape) if hasattr(module.weight, 'shape') else None
                }
        
        # Identify largest layers
        largest_layers = sorted(
            [(name, info) for name, info in layer_info.items()],
            key=lambda x: x[1]["parameters"],
            reverse=True
        )[:10]
        
        return {
            "total_layers": len(layer_info),
            "parameters_by_type": total_params_by_type,
            "largest_layers": largest_layers,
            "layer_distribution": self._calculate_layer_distribution(total_params_by_type)
        }
    
    def _calculate_layer_distribution(self, params_by_type: Dict[str, int]) -> Dict[str, float]:
        """Calculate parameter distribution across layer types"""
        total_params = sum(params_by_type.values())
        return {
            layer_type: (params / total_params) * 100
            for layer_type, params in params_by_type.items()
        }
    
    def _identify_optimization_opportunities(self, size_metrics: Dict, layer_analysis: Dict) -> List[str]:
        """Identify potential optimization opportunities"""
        opportunities = []
        
        # Size-based recommendations
        if size_metrics["total_mb"] > 10000:  # > 10GB
            opportunities.append("Model is very large - consider quantization or pruning")
        
        if size_metrics["total_mb"] > 5000:  # > 5GB
            opportunities.append("Consider INT8 quantization to reduce memory usage")
        
        # Layer-based recommendations
        params_by_type = layer_analysis["parameters_by_type"]
        total_params = sum(params_by_type.values())
        
        if params_by_type.get("Linear", 0) / total_params > 0.8:
            opportunities.append("Model is linear layer heavy - excellent candidate for LoRA fine-tuning")
        
        if params_by_type.get("Embedding", 0) / total_params > 0.3:
            opportunities.append("Large embedding layers - consider vocabulary pruning")
        
        # Performance recommendations
        largest_layers = layer_analysis["largest_layers"]
        if largest_layers and largest_layers[0][1]["parameters"] > total_params * 0.1:
            opportunities.append(f"Layer '{largest_layers[0][0]}' contains >10% of parameters - prime target for optimization")
        
        return opportunities
    
    def create_quantized_version(self, model_name: str) -> Dict[str, Any]:
        """Create and test quantized version of a model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = self.models_dir / model_name
            if not model_path.exists():
                return {"error": f"Model path {model_path} not found"}
            
            logger.info(f"Loading model for quantization: {model_name}")
            
            # Load original model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Measure original size
            original_metrics = ModelQuantizer.measure_model_size(model)
            
            # Apply quantization
            quantized_model = ModelQuantizer.quantize_model_int8(model)
            quantized_metrics = ModelQuantizer.measure_model_size(quantized_model)
            
            # Test performance
            test_prompt = "What is artificial intelligence?"
            
            # Original model performance
            start_time = time.time()
            inputs = tokenizer(test_prompt, return_tensors="pt")
            with torch.no_grad():
                original_outputs = model.generate(**inputs, max_new_tokens=50)
            original_time = time.time() - start_time
            original_response = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
            
            # Quantized model performance
            start_time = time.time()
            with torch.no_grad():
                quantized_outputs = quantized_model.generate(**inputs, max_new_tokens=50)
            quantized_time = time.time() - start_time
            quantized_response = tokenizer.decode(quantized_outputs[0], skip_special_tokens=True)
            
            # Calculate compression metrics
            size_reduction = (1 - quantized_metrics["total_mb"] / original_metrics["total_mb"]) * 100
            speed_change = (quantized_time / original_time - 1) * 100
            
            return {
                "model_name": model_name,
                "quantization_successful": True,
                "original_size_mb": original_metrics["total_mb"],
                "quantized_size_mb": quantized_metrics["total_mb"],
                "size_reduction_percent": size_reduction,
                "original_inference_time": original_time,
                "quantized_inference_time": quantized_time,
                "speed_change_percent": speed_change,
                "quality_comparison": {
                    "original_response": original_response,
                    "quantized_response": quantized_response,
                    "responses_identical": original_response == quantized_response
                },
                "recommendation": "Deploy quantized version" if size_reduction > 20 and abs(speed_change) < 50 else "Keep original version"
            }
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return {"error": str(e), "quantization_successful": False}
    
    def setup_lora_fine_tuning(self, model_name: str, rank: int = 16) -> Dict[str, Any]:
        """Set up LoRA fine-tuning for a model"""
        try:
            model_path = self.models_dir / model_name
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Apply LoRA
            lora_optimizer = LoRAOptimizer(rank=rank)
            model_with_lora = lora_optimizer.apply_lora_to_model(model)
            
            # Get parameter statistics
            param_stats = lora_optimizer.get_trainable_parameters(model_with_lora)
            
            # Create sample training configuration
            training_config = {
                "learning_rate": 2e-4,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "warmup_steps": 100,
                "save_steps": 500,
                "evaluation_strategy": "steps",
                "eval_steps": 500,
                "logging_steps": 100,
                "output_dir": f"./lora_output_{model_name}",
                "lora_rank": rank,
                "lora_alpha": 32
            }
            
            return {
                "model_name": model_name,
                "lora_setup_successful": True,
                "parameter_statistics": param_stats,
                "lora_configuration": {
                    "rank": rank,
                    "alpha": 32,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
                },
                "training_configuration": training_config,
                "memory_efficiency": f"Reduced trainable parameters by {100 - param_stats['trainable_percentage']:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"LoRA setup failed: {e}")
            return {"error": str(e), "lora_setup_successful": False}
    
    def comprehensive_weight_analysis(self, model_names: List[str]) -> Dict[str, Any]:
        """Run comprehensive weight optimization analysis"""
        results = {}
        
        for model_name in model_names:
            logger.info(f"Analyzing {model_name}")
            
            try:
                model_path = self.models_dir / model_name
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Base analysis
                analysis = self.analyze_model_efficiency(model_name, model, tokenizer)
                
                # Quantization analysis
                quantization_results = self.create_quantized_version(model_name)
                
                # LoRA analysis
                lora_results = self.setup_lora_fine_tuning(model_name)
                
                results[model_name] = {
                    "efficiency_analysis": analysis,
                    "quantization_results": quantization_results,
                    "lora_results": lora_results,
                    "optimization_score": self._calculate_optimization_score(analysis, quantization_results, lora_results)
                }
                
                # Cleanup
                del model, tokenizer
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
            except Exception as e:
                logger.error(f"Analysis failed for {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return {
            "analysis_results": results,
            "summary": self._generate_optimization_summary(results),
            "recommendations": self._generate_weight_optimization_recommendations(results)
        }
    
    def _calculate_optimization_score(self, efficiency: Dict, quantization: Dict, lora: Dict) -> float:
        """Calculate overall optimization potential score"""
        score = 0.5  # Base score
        
        # Size optimization potential
        if efficiency["size_metrics"]["total_mb"] > 5000:
            score += 0.2
        
        # Quantization benefits
        if quantization.get("quantization_successful") and quantization.get("size_reduction_percent", 0) > 20:
            score += 0.2
        
        # LoRA benefits
        if lora.get("lora_setup_successful"):
            trainable_pct = lora.get("parameter_statistics", {}).get("trainable_percentage", 100)
            if trainable_pct < 10:  # Less than 10% trainable parameters
                score += 0.1
        
        return min(score, 1.0)
    
    def _generate_optimization_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary of optimization analysis"""
        successful_analyses = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_analyses:
            return {"error": "No successful analyses"}
        
        # Calculate averages
        avg_size = np.mean([
            r["efficiency_analysis"]["size_metrics"]["total_mb"] 
            for r in successful_analyses.values()
        ])
        
        quantization_success_rate = sum(
            1 for r in successful_analyses.values() 
            if r.get("quantization_results", {}).get("quantization_successful", False)
        ) / len(successful_analyses)
        
        lora_success_rate = sum(
            1 for r in successful_analyses.values()
            if r.get("lora_results", {}).get("lora_setup_successful", False)
        ) / len(successful_analyses)
        
        return {
            "models_analyzed": len(successful_analyses),
            "average_model_size_mb": avg_size,
            "quantization_success_rate": quantization_success_rate,
            "lora_success_rate": lora_success_rate,
            "optimization_potential": "High" if avg_size > 5000 else "Medium" if avg_size > 2000 else "Low"
        }
    
    def _generate_weight_optimization_recommendations(self, results: Dict) -> List[str]:
        """Generate weight optimization recommendations"""
        recommendations = []
        
        for model_name, result in results.items():
            if "error" in result:
                continue
            
            efficiency = result.get("efficiency_analysis", {})
            quantization = result.get("quantization_results", {})
            lora = result.get("lora_results", {})
            
            # Model-specific recommendations
            size_mb = efficiency.get("size_metrics", {}).get("total_mb", 0)
            
            if size_mb > 10000:
                recommendations.append(f"{model_name}: Large model (>10GB) - prioritize quantization and consider model pruning")
            elif size_mb > 5000:
                recommendations.append(f"{model_name}: Medium-large model - good candidate for INT8 quantization")
            
            if quantization.get("quantization_successful"):
                size_reduction = quantization.get("size_reduction_percent", 0)
                if size_reduction > 30:
                    recommendations.append(f"{model_name}: Quantization achieved {size_reduction:.1f}% size reduction - recommended for deployment")
            
            if lora.get("lora_setup_successful"):
                trainable_pct = lora.get("parameter_statistics", {}).get("trainable_percentage", 100)
                recommendations.append(f"{model_name}: LoRA reduces trainable parameters to {trainable_pct:.1f}% - excellent for fine-tuning")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("All models analyzed are already well-optimized")
        
        return recommendations


# Example usage
def run_weight_optimization_example():
    """Example of weight optimization workflow"""
    optimizer = WeightOptimizer()
    
    # Available models to test
    test_models = ["deepseek-r1-distill-qwen-1.5b", "qwen2.5-1.5b"]
    
    # Run comprehensive analysis
    results = optimizer.comprehensive_weight_analysis(test_models)
    
    print("=== Weight Optimization Analysis ===")
    print(f"Summary: {results['summary']}")
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"- {rec}")


if __name__ == "__main__":
    run_weight_optimization_example()