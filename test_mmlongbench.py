"""
Standalone test script for MMLongBench evaluation with NodeRAG

This script runs MMLongBench evaluation independently from the Streamlit app.

CUSTOMIZATION POINTS:
---------------------
1. System Prompt (lines ~36-108):
   - Modify SYSTEM_PROMPT to change the AI assistant's behavior
   - This sets the overall tone and instructions for answer generation
   - The system prompt is now properly passed to OpenAI models as a separate system message

2. Query Instruction (line ~111):
   - Modify QUERY_INSTRUCTION to add specific instructions to each query
   - Currently used in the formatted_question to add context before questions

Usage:
    python test_mmlongbench.py --config path/to/Node_config.yaml --samples path/to/samples.json
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add NodeRAG to path
sys.path.insert(0, str(Path(__file__).parent))

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def run_mmlongbench_test(config_path: str, samples_path: str, max_samples: int | None = None):
    """
    Run MMLongBench evaluation test - TWO-STEP PROCESS:
    Step 1: Generate analysis from GraphRAG retrieval (using default NodeRAG prompt)
    Step 2: Extract formatted answer from the analysis (using MMLongBench extraction prompt)
    
    Args:
        config_path: Path to Node_config.yaml
        samples_path: Path to samples.json
        max_samples: Maximum number of samples to test (None = all)
    """
    print_banner("MMLongBench Evaluation Test")
    
    # MMLongBench answer extraction prompt (used in Step 2)
    # This extracts the final answer from the analysis in the required format
    EXTRACTION_SYSTEM_PROMPT = """Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. 
If you find the analysis the question can not be answered from the given documents, type "Not answerable". 
Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
```
Extracted answer: [answer]
Answer format: [answer format]
```

Please read the following example, then extract the answer from the model response and type it at the end of the prompt.

---
Question: List the primary questions asked about the services in this report.
Analysis: The primary questions asked about the services in the report for The Limes Residential Home are:
1. Is the service safe?
2. Is the service effective?
3. Is the service caring?
4. Is the service responsive?
5. Is the service well-led?
Extracted answer: ['Is the servife safe?', 'Is the service effective','Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']
Answer format: List

---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. 
Here are the specifics:
1. Regulation 13: Safeguarding service users from abuse and improper treatment
2. Regulation 12: Safe care and treatment
3. Regulation 18: Staffing
4. Regulation 11: Need for consent
5. Regulation 10: Dignity and respect
6. Regulation 9: Person-centred care
7. Regulation 17: Good governance
8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents
9. Regulation 18: Failure to maintain an accurate and up-to-date care plan
10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively

These breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer

---
Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump's election. 
The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. 
If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that!
Extracted answer: Not answerable
Answer format: String

---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately.
If you can provide a clearer image or more context, I might be able to help you with your question.
Extracted answer: Fail to answer
Answer format: String

---
"""
    
    # Import NodeRAG components
    print("Loading NodeRAG components...")
    sys.stdout.flush()
    
    try:
        from NodeRAG.config import NodeConfig
        from NodeRAG.search.search import NodeSearch
        print("✓ NodeRAG modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NodeRAG: {e}")
        print("\nMake sure you're in the NodeRAG directory and dependencies are installed.")
        return
    
    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    sys.stdout.flush()
    
    try:
        # Load YAML configuration file
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = NodeConfig(config_dict)
        print("✓ Configuration loaded")
        print(f"  LLM: {config_dict['config'].get('LLM', {}).get('model_name', 'N/A')}")
        print(f"  Embedding: {config_dict['config'].get('Embedding', {}).get('model_name', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize search engine
    print("\nInitializing search engine...")
    sys.stdout.flush()
    
    try:
        search_engine = NodeSearch(config)
        print("✓ Search engine initialized")
        
        # Note: Custom system prompt is now passed directly to stream_answer()
        # No need to override the config's prompt_manager
        
    except Exception as e:
        print(f"✗ Failed to initialize search engine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load samples
    print(f"\nLoading samples from: {samples_path}")
    sys.stdout.flush()
    
    try:
        with open(samples_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"✓ Loaded {len(samples)} samples")
        
        # Limit samples if requested
        if max_samples and max_samples < len(samples):
            samples = samples[:max_samples]
            print(f"  (Limited to first {max_samples} samples for testing)")
    except Exception as e:
        print(f"✗ Failed to load samples: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "mmlongbench_results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    
    # Run evaluation
    print_banner(f"Starting Evaluation - {len(samples)} Samples")
    
    results = []
    
    for idx, sample in enumerate(samples, 1):
        try:
            print(f"\n{'─' * 80}")
            print(f"[Sample {idx}/{len(samples)}]")
            print(f"Document: {sample.get('doc_id', 'N/A')}")
            
            question = sample['question']
            question_preview = question[:100] + "..." if len(question) > 100 else question
            print(f"Question: {question_preview}")
            sys.stdout.flush()
            
            # STEP 1: Search and generate analysis using GraphRAG
            print("  [Step 1/2] Searching GraphRAG...")
            sys.stdout.flush()
            searched = search_engine.search(question)
            print("  ✓ Search complete")
            sys.stdout.flush()
            
            print("  [Step 1/2] Generating analysis...")
            sys.stdout.flush()
            # Generate analysis without custom system prompt (use default NodeRAG prompt)
            analysis_stream = search_engine.stream_answer(
                question, 
                searched.structured_prompt
            )
            
            # Collect the analysis
            analysis_parts = []
            for chunk in analysis_stream:
                analysis_parts.append(chunk)
                if len(analysis_parts) % 10 == 0:
                    print(".", end="", flush=True)
            
            analysis_text = ''.join(analysis_parts)
            print()  # New line after dots
            print(f"  ✓ Analysis generated ({len(analysis_text)} chars)")
            sys.stdout.flush()
            
            # STEP 2: Extract formatted answer from the analysis
            print("  [Step 2/2] Extracting formatted answer...")
            sys.stdout.flush()
            
            # Format the extraction prompt with question and analysis
            extraction_user_prompt = f"""Question: {question}
Analysis: {analysis_text}"""
            
            # Use the OpenAI client directly to extract the answer
            extraction_response = search_engine.config.API_client.request({
                'system_prompt': EXTRACTION_SYSTEM_PROMPT,
                'query': extraction_user_prompt
            })
            
            print(f"  ✓ Answer extracted")
            sys.stdout.flush()
            
            # Store result with both analysis and extracted answer
            result = {
                'doc_id': sample.get('doc_id', ''),
                'question': question,
                'ground_truth': sample.get('answer', ''),
                'analysis': analysis_text,  # Store the intermediate analysis
                'prediction': extraction_response,  # The final extracted answer
                'answer_format': sample.get('answer_format', 'Str'),
                'evidence_pages': sample.get('evidence_pages', '[]'),
                'evidence_sources': sample.get('evidence_sources', '[]'),
                'doc_type': sample.get('doc_type', '')
            }
            
            # Show preview
            print(f"\nGround Truth: {result['ground_truth']}")
            analysis_preview = analysis_text[:100] + "..." if len(analysis_text) > 100 else analysis_text
            print(f"Analysis: {analysis_preview}")
            pred_preview = extraction_response[:150] + "..." if len(extraction_response) > 150 else extraction_response
            print(f"Extracted Answer: {pred_preview}")
            
            results.append(result)
            
            # Save intermediate results
            intermediate_path = output_dir / "results_progress.json"
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'completed': len(results),
                    'total': len(samples),
                    'results': results
                }, f, indent=2)
            
        except KeyboardInterrupt:
            print("\n\n✗ Evaluation interrupted by user")
            break
        except Exception as e:
            print(f"\n  ✗ Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'doc_id': sample.get('doc_id', ''),
                'question': sample.get('question', ''),
                'ground_truth': sample.get('answer', ''),
                'prediction': f"Error: {str(e)}",
                'answer_format': sample.get('answer_format', 'Str'),
                'error': str(e)
            }
            results.append(result)
    
    # Save final results
    print_banner("Evaluation Complete")
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'total_samples': len(samples),
            'completed_samples': len(results),
            'config_path': config_path,
            'samples_path': samples_path,
            'results': results
        }, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    
    # Try to compute scores if evaluation module is available
    try:
        # Add MMLongBench to path
        mmlongbench_path = Path(__file__).parent / 'MMLongBench'
        if str(mmlongbench_path) not in sys.path:
            sys.path.insert(0, str(mmlongbench_path))
        
        from eval.eval_score import eval_score, eval_acc_and_f1, show_results  # type: ignore
        
        print("\nComputing evaluation scores...")
        sys.stdout.flush()
        
        # Score each result
        for result in results:
            if 'error' not in result:
                try:
                    result['score'] = eval_score(
                        result['ground_truth'],
                        result['prediction'],
                        result['answer_format']
                    )
                except Exception as e:
                    print(f"  Warning: Could not score result: {e}")
                    result['score'] = 0.0
            else:
                result['score'] = 0.0
        
        # Prepare for eval_acc_and_f1 (expects 'answer' and 'pred' keys)
        # Also include all fields needed by show_results
        eval_format_results = []
        for r in results:
            eval_format_results.append({
                'doc_id': r.get('doc_id', ''),
                'question': r.get('question', ''),
                'answer': r['ground_truth'],
                'pred': r['prediction'],
                'score': r.get('score', 0.0),
                'answer_format': r.get('answer_format', 'Str'),
                'evidence_pages': r.get('evidence_pages', '[]'),
                'evidence_sources': r.get('evidence_sources', '[]'),
                'doc_type': r.get('doc_type', '')
            })
        
        # Calculate overall metrics
        acc, f1 = eval_acc_and_f1(eval_format_results)
        
        print(f"\n{'=' * 80}")
        print("Overall Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Samples: {len(results)}/{len(samples)}")
        print(f"{'=' * 80}\n")
        
        # Save scored results
        scored_path = output_dir / 'results_scored.json'
        with open(scored_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'accuracy': acc,
                'f1_score': f1,
                'total_samples': len(samples),
                'completed_samples': len(results),
                'results': results
            }, f, indent=2)
        
        print(f"✓ Scored results saved to: {scored_path}")
        
        # Generate detailed report
        report_path = output_dir / 'report.txt'
        show_results(eval_format_results, str(report_path))
        print(f"✓ Detailed report saved to: {report_path}")
        
    except ImportError:
        print("\n⚠ MMLongBench evaluation module not found - skipping scoring")
        print("  Results saved without scores.")
    except Exception as e:
        print(f"\n⚠ Error during scoring: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✓ All outputs saved in: {output_dir}")
    print("\nTest complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Test MMLongBench evaluation with NodeRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default paths
  python test_mmlongbench.py
  
  # Run with custom config and samples
  python test_mmlongbench.py --config ./data/Node_config.yaml --samples ./MMLongBench/data/samples.json
  
  # Test with first 5 samples only
  python test_mmlongbench.py --max-samples 5
        """
    )
    
    # Get default paths
    script_dir = Path(__file__).parent
    default_config = script_dir / "data" / "Node_config.yaml"
    default_samples = script_dir / "MMLongBench" / "data" / "samples.json"
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help=f"Path to Node_config.yaml (default: {default_config})"
    )
    
    parser.add_argument(
        "--samples",
        type=str,
        default=str(default_samples),
        help=f"Path to samples.json (default: {default_samples})"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to test (default: all)"
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.config).exists():
        print(f"✗ Config file not found: {args.config}")
        print("\nPlease provide a valid config file with --config")
        sys.exit(1)
    
    if not Path(args.samples).exists():
        print(f"✗ Samples file not found: {args.samples}")
        print("\nPlease provide a valid samples file with --samples")
        sys.exit(1)
    
    # Run the test
    try:
        run_mmlongbench_test(args.config, args.samples, args.max_samples)
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
