"""
MMLongBench Evaluation Integration
"""
import json
from typing import List, Dict, Iterator, Tuple
from pathlib import Path

def load_mmlongbench_samples(samples_path: str) -> List[Dict]:
    """Load MMLongBench samples from JSON file"""
    with open(samples_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_sample_generator(search_engine, samples: List[Dict]) -> Iterator[Tuple[int, Dict, str]]:
    """
    Generator that yields evaluation results one sample at a time
    
    Args:
        search_engine: The search engine instance
        samples: List of sample dictionaries
        
    Yields:
        Tuple of (index, result_dict, status_message)
    """
    total_samples = len(samples)
    
    for idx, sample in enumerate(samples):
        try:
            # Get question
            question = sample['question']
            status_msg = f"Processing {idx+1}/{total_samples}: {question[:60]}..."
            
            # Search for answer
            searched = search_engine.search(question)
            
            # Generate answer
            answer = search_engine.stream_answer(question, searched.structured_prompt)
            
            # For evaluation, we need the final answer string
            if hasattr(answer, '__iter__') and not isinstance(answer, str):
                answer_text = ''.join(answer)
            else:
                answer_text = str(answer)
            
            # Store result
            result = {
                'doc_id': sample.get('doc_id', ''),
                'question': question,
                'answer': sample.get('answer', ''),
                'pred': answer_text,
                'answer_type': sample.get('answer_format', 'Str'),
                'evidence_pages': sample.get('evidence_pages', '[]'),
                'evidence_sources': sample.get('evidence_sources', '[]'),
                'doc_type': sample.get('doc_type', '')
            }
            
            yield (idx, result, status_msg)
            
        except Exception as e:
            error_msg = f"Error processing sample {idx}: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            result = {
                'doc_id': sample.get('doc_id', ''),
                'question': sample.get('question', ''),
                'answer': sample.get('answer', ''),
                'pred': f"Error: {str(e)}",
                'answer_type': sample.get('answer_format', 'Str'),
                'evidence_pages': sample.get('evidence_pages', '[]'),
                'evidence_sources': sample.get('evidence_sources', '[]'),
                'doc_type': sample.get('doc_type', '')
            }
            
            yield (idx, result, error_msg)

def run_mmlongbench_evaluation(search_engine, samples: List[Dict], output_dir: str, progress_callback=None) -> Dict:
    """
    Run evaluation on MMLongBench samples
    
    Args:
        search_engine: The search engine instance
        samples: List of sample dictionaries
        output_dir: Directory to save results
        progress_callback: Optional callback function to report progress (current_idx, total)
        
    Returns:
        Dictionary with evaluation results
    """
    import sys
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = len(samples)
    print(f"\n{'='*80}")
    print(f"Starting MMLongBench Evaluation")
    print(f"Total samples: {total_samples}")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    for idx, sample in enumerate(samples):
        try:
            # Report progress
            if progress_callback:
                progress_callback(idx, total_samples, sample)
            
            # Print progress to console
            question_preview = sample['question'][:60] + "..." if len(sample['question']) > 60 else sample['question']
            print(f"[{idx+1}/{total_samples}] Processing: {question_preview}")
            sys.stdout.flush()
            
            # Get question
            question = sample['question']
            
            # Search for answer
            searched = search_engine.search(question)
            
            # Generate answer
            answer = search_engine.stream_answer(question, searched.structured_prompt)
            
            # For evaluation, we need the final answer string
            if hasattr(answer, '__iter__') and not isinstance(answer, str):
                answer_text = ''.join(answer)
            else:
                answer_text = str(answer)
            
            print(f"  ✓ Generated answer ({len(answer_text)} chars)")
            sys.stdout.flush()
            
            # Store result
            result = {
                'doc_id': sample.get('doc_id', ''),
                'question': question,
                'answer': sample.get('answer', ''),
                'pred': answer_text,
                'answer_type': sample.get('answer_format', 'Str'),
                'evidence_pages': sample.get('evidence_pages', '[]'),
                'evidence_sources': sample.get('evidence_sources', '[]'),
                'doc_type': sample.get('doc_type', '')
            }
            results.append(result)
            
        except Exception as e:
            error_msg = f"Error processing sample {idx}: {e}"
            print(f"  ✗ {error_msg}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            result = {
                'doc_id': sample.get('doc_id', ''),
                'question': sample.get('question', ''),
                'answer': sample.get('answer', ''),
                'pred': f"Error: {str(e)}",
                'answer_type': sample.get('answer_format', 'Str'),
                'evidence_pages': sample.get('evidence_pages', '[]'),
                'evidence_sources': sample.get('evidence_sources', '[]'),
                'doc_type': sample.get('doc_type', '')
            }
            results.append(result)
    
    # Evaluate results
    try:
        # Try to import from the MMLongBench directory
        import sys
        mmlongbench_path = Path(__file__).parent.parent.parent.parent / 'MMLongBench'
        if str(mmlongbench_path) not in sys.path:
            sys.path.insert(0, str(mmlongbench_path))
        from eval.eval_score import eval_score, eval_acc_and_f1, show_results
        
        # Score each result
        for result in results:
            result['score'] = eval_score(
                result['answer'],
                result['pred'],
                result['answer_type']
            )
        
        # Calculate overall metrics
        acc, f1 = eval_acc_and_f1(results)
        
        # Save results
        results_path = output_dir / 'mmlongbench_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'overall_accuracy': acc,
                'overall_f1': f1,
                'total_samples': len(results)
            }, f, indent=2)
        
        # Generate detailed report
        report_path = output_dir / 'mmlongbench_report.txt'
        show_results(results, str(report_path))
        
        return {
            'accuracy': acc,
            'f1': f1,
            'total_samples': len(results),
            'results_path': str(results_path),
            'report_path': str(report_path)
        }
    except ImportError:
        # Fallback if eval_score not available
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'total_samples': len(results),
            'results_path': str(output_dir / 'mmlongbench_results.json'),
            'error': 'Evaluation module not available'
        }

