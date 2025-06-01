# Tests

Model validation and testing utilities.

## Files

- **test_models.py** - Comprehensive model testing suite
- **test_image.png** - Sample image for vision model testing (red/blue split)

## Usage

```bash
# Run all tests
python tests/test_models.py

# Individual model testing via inference.py
python src/inference.py --model <model-name> --prompt "Test prompt"
```

## Test Coverage

### Text Models
- DeepSeek-R1-Distill-Qwen 1.5B
  - Math problems
  - Creative writing
  - Technical explanations

### Vision Models  
- BLIP base - Image captioning
- BLIP-2 - Visual question answering
- DeepSeek-VL2 - Multi-modal understanding

### Speech Models
- Whisper base - Placeholder (audio loading not implemented)

## Adding Tests

To add new tests:
1. Create a new test function following the pattern
2. Add appropriate test cases
3. Include performance metrics tracking