"""
Run all FireAnt examples.
This script demonstrates all the features of FireAnt with comprehensive examples.
"""

import sys
import traceback
from pathlib import Path


def run_example(example_name: str, example_path: Path):
    """Run a single example and handle errors gracefully."""
    print(f"\n{'='*60}")
    print(f"Running {example_name}")
    print(f"{'='*60}")
    
    try:
        # Import and run the example
        spec = __import__(f"{example_path.stem}", fromlist=['main'])
        if hasattr(spec, 'main'):
            spec.main()
        else:
            print(f"Warning: No main() function found in {example_name}")
        
        print(f"SUCCESS: {example_name} completed successfully!")
        return True
        
    except ImportError as e:
        print(f"ERROR: Failed to import {example_name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {example_name} failed with error:")
        print(f"   {e}")
        if "--verbose" in sys.argv:
            print(f"   Traceback:")
            traceback.print_exc()
        return False


def main():
    """Run all examples."""
    print("FireAnt Examples Runner")
    print("This will run all examples demonstrating FireAnt's features.")
    
    # List of examples to run
    examples = [
        ("Error Handling & Retries", Path("error_handling_example.py")),
        ("Monitoring & Logging", Path("monitoring_example.py")),
        ("State Persistence", Path("persistence_example.py")),
        ("Async Support", Path("async_example.py")),
        ("Testing Framework", Path("testing_example.py")),
        ("Configuration Management", Path("config_example.py")),
        ("Performance Metrics", Path("metrics_example.py")),
        ("Agent Lifecycle Management", Path("lifecycle_example.py")),
    ]
    
    # Check if examples directory exists
    examples_dir = Path(__file__).parent
    if not examples_dir.exists():
        print(f"ERROR: Examples directory not found: {examples_dir}")
        return 1
    
    # Change to examples directory
    original_cwd = Path.cwd()
    sys.path.insert(0, str(examples_dir.parent))
    
    try:
        # Run each example
        results = {}
        for name, path in examples:
            example_path = examples_dir / path
            if example_path.exists():
                results[name] = run_example(name, example_path)
            else:
                print(f"Warning: Example file not found: {example_path}")
                results[name] = False
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXAMPLES SUMMARY")
        print(f"{'='*60}")
        
        total = len(results)
        successful = sum(1 for success in results.values() if success)
        failed = total - successful
        
        for name, success in results.items():
            status = "PASSED" if success else "FAILED"
            print(f"   {name}: {status}")
        
        print(f"\nTotal: {total}, Successful: {successful}, Failed: {failed}")
        
        if failed > 0:
            print(f"\nWarning: {failed} example(s) failed. Run with --verbose for details.")
            return 1
        else:
            print(f"\nSUCCESS: All examples completed successfully!")
            return 0
            
    finally:
        # Restore original working directory
        sys.path.remove(str(examples_dir.parent))
        import os
        os.chdir(original_cwd)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)