# MMLongBench Test Runner

Standalone test script for running MMLongBench evaluation with NodeRAG.

## Quick Start

### Basic Usage

Run with default settings (uses config and samples from default locations):

```bash
python test_mmlongbench.py
```

### Custom Paths

Specify custom configuration and samples:

```bash
python test_mmlongbench.py \
    --config ./data/Node_config.yaml \
    --samples ./MMLongBench/data/samples.json
```

### Test Mode

Test with a limited number of samples (useful for quick testing):

```bash
# Test with first 3 samples only
python test_mmlongbench.py --max-samples 3

# Test with first 10 samples
python test_mmlongbench.py --max-samples 10
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to Node_config.yaml | `./data/Node_config.yaml` |
| `--samples` | Path to samples.json | `./MMLongBench/data/samples.json` |
| `--max-samples` | Maximum number of samples to test | All samples |

## Output

Results are saved in `mmlongbench_results/<timestamp>/`:

- **results.json** - Raw results with questions and predictions
- **results_scored.json** - Results with scores (if evaluation module available)
- **results_progress.json** - Intermediate progress (updated during run)
- **report.txt** - Detailed evaluation report

## Example Output

```
================================================================================
                        MMLongBench Evaluation Test                        
================================================================================

Loading NodeRAG components...
✓ NodeRAG modules imported successfully

Loading configuration from: ./data/Node_config.yaml
✓ Configuration loaded
  LLM: gpt-4o
  Embedding: text-embedding-3-small

Initializing search engine...
✓ Search engine initialized

Loading samples from: ./MMLongBench/data/samples.json
✓ Loaded 50 samples

Results will be saved to: mmlongbench_results/20251105_143022

================================================================================
                     Starting Evaluation - 50 Samples                     
================================================================================

────────────────────────────────────────────────────────────────────────────────
[Sample 1/50]
Document: document1.pdf
Question: What is the main topic discussed in the document?
  Searching...
  ✓ Search complete
  Generating answer...
..........
  ✓ Generated answer (234 chars)

Ground Truth: Machine Learning
Prediction: The main topic discussed in the document is Machine Learning...

[Sample 2/50]
...
```

## Requirements

- NodeRAG installed and configured
- Valid Node_config.yaml with API keys
- MMLongBench samples.json file
- Python 3.10+

## Tips

1. **Start small**: Use `--max-samples 1` to test your setup first
2. **Monitor progress**: Results are saved incrementally in `results_progress.json`
3. **Interrupt safely**: Press Ctrl+C to stop - partial results will be saved
4. **Check output**: Each run creates a timestamped directory with all results

## Troubleshooting

### "Config file not found"
Make sure the path to `Node_config.yaml` is correct. Use absolute paths if needed.

### "Samples file not found"  
Verify the path to `samples.json` is correct. The default expects it in `MMLongBench/data/`.

### "Failed to import NodeRAG"
Make sure you're in the NodeRAG directory and your virtual environment is activated:
```bash
cd NodeRAG
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python test_mmlongbench.py
```

### Evaluation scores not computed
The script will still save results even if the MMLongBench evaluation module isn't available. You can manually score results later using the MMLongBench eval scripts.

## Integration with Streamlit

Once the test runner is working correctly, you can integrate it with the Streamlit app by:

1. Using the same logic in `NodeRAG/utils/mmlongbench_eval.py`
2. Adding a progress bar in Streamlit UI
3. Displaying results in real-time
4. Adding export functionality

The test runner provides a solid foundation for understanding the evaluation flow before integrating with the UI.
