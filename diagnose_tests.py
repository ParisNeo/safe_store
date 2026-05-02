#!/usr/bin/env python3
"""
Diagnose test failures by running pytest via subprocess and capturing detailed failure information.
"""

import subprocess
import sys
from pathlib import Path


def run_pytest_diagnosis():
    """Run pytest programmatically and capture failures."""
    
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print(f"ERROR: Tests directory '{tests_dir}' not found.")
        sys.exit(1)
    
    # Build pytest command with detailed output
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",  # verbose mode
        "--tb=long",  # full traceback
        "--color=no",  # disable ANSI colors for cleaner parsing
        "-p", "no:warnings",  # suppress warnings
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run pytest via subprocess, capturing stdout and stderr
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    # Print full output
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)
    print("=" * 80)
    
    # Parse and extract failure details
    failures = []
    current_failure = None
    lines = result.stdout.split("\n")
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect failure header: "FAILED tests/...::test_name - ..."
        if stripped.startswith("FAILED "):
            parts = stripped.split(" - ", 1)
            test_path = parts[0].replace("FAILED ", "").strip()
            error_msg = parts[1].strip() if len(parts) > 1 else "Unknown error"
            current_failure = {
                "test": test_path,
                "error_summary": error_msg,
                "details": []
            }
            failures.append(current_failure)
        
        # Capture assertion/details lines after failure indication
        elif current_failure is not None:
            # Stop capturing when we hit a new test or result summary
            if stripped.startswith("PASSED ") or stripped.startswith("FAILED "):
                current_failure = None
            elif stripped.startswith("=") and "FAILURES" in stripped:
                current_failure = None
            elif stripped.startswith("__________"):
                current_failure = None
            else:
                current_failure["details"].append(line)
    
    # Also parse from pytest output for test names in failure sections
    failure_sections = []
    in_failure = False
    current_section = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect failure section start
        if stripped.startswith("__________ ") and " __________" in stripped:
            in_failure = True
            test_name_raw = stripped.replace("_", "").strip()
            current_section = {
                "header": stripped,
                "test_name": test_name_raw,
                "content": []
            }
            failure_sections.append(current_section)
        
        # Detect end of failure section (next test or summary)
        elif in_failure and stripped.startswith("=") and "ERRORS" not in stripped and "FAILURES" not in stripped:
            in_failure = False
            current_section = None
        
        elif in_failure and current_section is not None:
            current_section["content"].append(line)
    
    # Print diagnosis summary
    print("\n" + "=" * 80)
    print("FAILURE DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    if result.returncode == 0:
        print("All tests PASSED.")
        return 0
    
    print(f"Exit code: {result.returncode}")
    print(f"Total failures found: {len(failures)}")
    print("-" * 80)
    
    for idx, failure in enumerate(failures, 1):
        print(f"\n[{idx}] FAILED TEST: {failure['test']}")
        print(f"    Error: {failure['error_summary']}")
        # Print relevant detail lines (filter empty)
        details = [d for d in failure["details"] if d.strip()]
        if details:
            print("    Details:")
            for d in details[:20]:  # limit output
                print(f"      {d}")
    
    # Print failure sections with full tracebacks
    print("\n" + "=" * 80)
    print("DETAILED FAILURE SECTIONS")
    print("=" * 80)
    
    for idx, section in enumerate(failure_sections, 1):
        print(f"\n--- Failure Section [{idx}]: {section['header']} ---")
        content = [c for c in section["content"] if c.strip()]
        for line in content:
            print(line)
    
    # Print summary line counts
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    # Count passed/failed from output
    passed_count = result.stdout.count("PASSED")
    failed_count = result.stdout.count("FAILED")
    error_count = result.stdout.count("ERROR")
    
    print(f"Tests PASSED: {passed_count}")
    print(f"Tests FAILED: {failed_count}")
    print(f"Tests ERROR:  {error_count}")
    
    # Identify the F.F..F pattern specifically
    print("\n" + "=" * 80)
    print("PROGRESS PATTERN ANALYSIS")
    print("=" * 80)
    
    # Look for progress dots pattern in stderr or stdout
    for source, name in [(result.stdout, "stdout"), (result.stderr, "stderr")]:
        for line in source.split("\n"):
            stripped = line.strip()
            if any(c in stripped for c in ".FEsx") and len(stripped) > 3:
                # Likely a progress line
                dots = stripped.replace(" ", "")
                if set(dots).issubset({".", "F", "E", "s", "x", "S", "X", "p", "P", "f"}):
                    print(f"Progress pattern found in {name}: {dots}")
                    failure_positions = [i+1 for i, c in enumerate(dots) if c == "F"]
                    error_positions = [i+1 for i, c in enumerate(dots) if c == "E"]
                    if failure_positions:
                        print(f"  Failure positions (1-indexed): {failure_positions}")
                    if error_positions:
                        print(f"  Error positions (1-indexed): {error_positions}")
                    break
    
    return result.returncode


if __name__ == "__main__":
    exit_code = run_pytest_diagnosis()
    sys.exit(exit_code)