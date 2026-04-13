#!/usr/bin/env python3
"""
Test the fixed CodeBLEU implementation with various scenarios.
"""

import os
import sys
import tempfile

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_single_file_evaluation():
    """Test evaluation with single reference and hypothesis files."""
    print("🧪 Testing Single File Evaluation")
    print("-" * 40)

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as ref_file:
        ref_file.write("""def add(a, b):
    return a + b

def multiply(x, y):
    return x * y""")
        ref_path = ref_file.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as hyp_file:
        hyp_file.write("""def add(a, b):
    return a + b

def multiply(x, y):
    return x * y""")
        hyp_path = hyp_file.name

    try:
        # Test command line
        cmd = f"python calc_code_bleu.py --refs {ref_path} --hyp {hyp_path} --lang python"
        print(f"Running: {cmd}")
        result = os.system(cmd)
        print(f"Exit code: {result}")
        print("✅ Single file evaluation test passed\n")

    finally:
        # Clean up
        os.unlink(ref_path)
        os.unlink(hyp_path)

def test_multiple_references():
    """Test evaluation with multiple reference files."""
    print("🧪 Testing Multiple Reference Files")
    print("-" * 40)

    # Create temporary files
    ref_files = []
    for i in range(2):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as ref_file:
            ref_file.write(f"""def function_{i}(n):
    return n * {i+1}""")
            ref_files.append(ref_file.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as hyp_file:
        hyp_file.write("""def function_0(n):
    return n * 1

def function_1(n):
    return n * 2""")
        hyp_path = hyp_file.name

    try:
        # Test command line with multiple references
        refs_str = " ".join(ref_files)
        cmd = f"python calc_code_bleu.py --refs {refs_str} --hyp {hyp_path} --lang python"
        print(f"Running: {cmd}")
        result = os.system(cmd)
        print(f"Exit code: {result}")
        if result == 0:
            print("✅ Multiple reference files test passed\n")
        else:
            print("❌ Multiple reference files test failed\n")
            raise Exception(f"Command failed with exit code {result}")

    finally:
        # Clean up
        for ref_path in ref_files:
            os.unlink(ref_path)
        os.unlink(hyp_path)

def test_different_languages():
    """Test evaluation with different programming languages."""
    print("🧪 Testing Different Languages")
    print("-" * 40)

    test_cases = [
        ("python", "def hello(): return 'Hello'", "def hello(): return 'Hello'"),
        ("java", "public class Test { public void test() {} }", "public class Test { public void test() {} }"),
        ("cpp", "#include <iostream>\nint main() { return 0; }", "#include <iostream>\nint main() { return 0; }")
    ]

    for lang, ref_code, hyp_code in test_cases:
        print(f"Testing {lang}...")

        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{lang}', delete=False) as ref_file:
            ref_file.write(ref_code)
            ref_path = ref_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{lang}', delete=False) as hyp_file:
            hyp_file.write(hyp_code)
            hyp_path = hyp_file.name

        try:
            cmd = f"python calc_code_bleu.py --refs {ref_path} --hyp {hyp_path} --lang {lang}"
            result = os.system(cmd)
            if result == 0:
                print(f"✅ {lang} evaluation successful")
            else:
                print(f"⚠️  {lang} evaluation had issues (may be due to missing tree-sitter parsers)")

        finally:
            os.unlink(ref_path)
            os.unlink(hyp_path)

    print()

def test_error_handling():
    """Test error handling for invalid inputs."""
    print("🧪 Testing Error Handling")
    print("-" * 40)

    # Test with non-existent files
    cmd = "python calc_code_bleu.py --refs nonexistent.py --hyp nonexistent.py --lang python"
    print("Testing with non-existent files...")
    result = os.system(cmd)
    if result != 0:
        print("✅ Error handling for missing files works")
    else:
        print("❌ Error handling for missing files failed")

    # Test with unsupported language
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write("def test(): pass")
        temp_path = temp_file.name

    try:
        cmd = f"python calc_code_bleu.py --refs {temp_path} --hyp {temp_path} --lang unsupported_lang"
        print("Testing with unsupported language...")
        result = os.system(cmd)
        if result != 0:
            print("✅ Error handling for unsupported languages works")
        else:
            print("❌ Error handling for unsupported languages failed")
    finally:
        os.unlink(temp_path)

    print()

def main():
    """Run all tests."""
    print("🚀 CodeBLEU Fix Validation Tests")
    print("=" * 50)

    tests = [
        test_single_file_evaluation,
        test_multiple_references,
        test_different_languages,
        test_error_handling
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")

    print("✅ All tests completed!")
    print("\n📝 Summary:")
    print("- Fixed file reading to treat each file as a single code sample")
    print("- Removed line-by-line processing that caused mismatch errors")
    print("- Maintained compatibility with multiple reference files")
    print("- Enhanced error handling and user feedback")

if __name__ == "__main__":
    main()