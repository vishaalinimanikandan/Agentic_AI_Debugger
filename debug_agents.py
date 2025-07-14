"""
AI Agents That Debug Each Other - A Multi-Agent Debugging System

This system consists of four AI agents that work together to find and fix bugs:
1. BugDetectorAgent - Finds issues in code
2. FixGeneratorAgent - Proposes solutions
3. ValidatorAgent - Tests the fixes
4. LearningAgent - Improves based on results
"""

import json
import asyncio
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import subprocess
import tempfile
import os

# Note: You'll need to install openai: pip install openai
# For this demo, we'll simulate AI responses, but you can replace with actual API calls

@dataclass
class Bug:
    """Represents a detected bug"""
    type: str
    line: int
    description: str
    severity: str
    code_snippet: str

@dataclass
class Fix:
    """Represents a proposed fix"""
    bug_id: str
    fixed_code: str
    explanation: str
    confidence: float

@dataclass
class ValidationResult:
    """Represents validation outcome"""
    fix_id: str
    success: bool
    error_message: Optional[str]
    execution_time: float

class BugDetectorAgent:
    """Agent responsible for finding bugs in code"""
    
    def __init__(self):
        self.name = "Bug Detective "
        self.bugs_found = 0
        
    def analyze_code(self, code: str) -> List[Bug]:
        """Analyze code and return list of detected bugs"""
        print(f"{self.name}: Analyzing code for potential issues...")
        
        bugs = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Simple heuristic-based bug detection
            # In real implementation, this would use AI/LLM
            
            if 'len(numbers)' in line and 'sum(numbers)' in line and 'if' not in code:
                bugs.append(Bug(
                    type="ZeroDivisionError",
                    line=i,
                    description="Division by length of list without checking if list is empty",
                    severity="high",
                    code_snippet=line.strip()
                ))
            
            if 'open(' in line and 'close()' not in code:
                bugs.append(Bug(
                    type="ResourceLeak",
                    line=i,
                    description="File opened but never explicitly closed",
                    severity="medium",
                    code_snippet=line.strip()
                ))
            
            if 'import *' in line:
                bugs.append(Bug(
                    type="BadPractice",
                    line=i,
                    description="Wildcard import can cause namespace pollution",
                    severity="low",
                    code_snippet=line.strip()
                ))
                
            if 'except:' in line and 'except Exception:' not in line:
                bugs.append(Bug(
                    type="BroadException",
                    line=i,
                    description="Bare except clause catches all exceptions",
                    severity="medium",
                    code_snippet=line.strip()
                ))
        
        self.bugs_found += len(bugs)
        print(f"{self.name}: Found {len(bugs)} potential issues")
        
        return bugs

class FixGeneratorAgent:
    """Agent responsible for generating fixes for detected bugs"""
    
    def __init__(self):
        self.name = "Fix Master "
        self.fixes_generated = 0
        
    def generate_fixes(self, code: str, bugs: List[Bug]) -> List[Fix]:
        """Generate fixes for the detected bugs"""
        print(f"{self.name}: Generating fixes for {len(bugs)} bugs...")
        
        fixes = []
        
        for bug in bugs:
            fix = self._generate_fix_for_bug(code, bug)
            if fix:
                fixes.append(fix)
                
        self.fixes_generated += len(fixes)
        print(f"{self.name}: Generated {len(fixes)} potential fixes")
        
        return fixes
    
    def _generate_fix_for_bug(self, code: str, bug: Bug) -> Optional[Fix]:
        """Generate a specific fix for a bug"""
        
        if bug.type == "ZeroDivisionError":
            # Fix division by zero issues
            fixed_code = code.replace(
                bug.code_snippet,
                f"    if not numbers:\n        return 0\n    {bug.code_snippet}"
            )
            return Fix(
                bug_id=f"{bug.type}_{bug.line}",
                fixed_code=fixed_code,
                explanation="Added check for empty list before division",
                confidence=0.9
            )
            
        elif bug.type == "ResourceLeak":
            # Fix file handling
            lines = code.split('\n')
            fixed_lines = []
            for line in lines:
                if 'open(' in line:
                    fixed_lines.append(line.replace('open(', 'with open(') + ':')
                    # Add proper indentation for the with block
                    continue
                fixed_lines.append('    ' + line if line.strip() else line)
            
            return Fix(
                bug_id=f"{bug.type}_{bug.line}",
                fixed_code='\n'.join(fixed_lines),
                explanation="Used context manager for file handling",
                confidence=0.8
            )
            
        elif bug.type == "BroadException":
            # Fix broad exception handling
            fixed_code = code.replace('except:', 'except Exception as e:')
            return Fix(
                bug_id=f"{bug.type}_{bug.line}",
                fixed_code=fixed_code,
                explanation="Made exception handling more specific",
                confidence=0.7
            )
        
        return None

class ValidatorAgent:
    """Agent responsible for testing and validating fixes"""
    
    def __init__(self):
        self.name = "Validator"
        self.validations_performed = 0
        
    def validate_fixes(self, original_code: str, fixes: List[Fix]) -> List[ValidationResult]:
        """Validate each fix by testing the code"""
        print(f"{self.name}: Validating {len(fixes)} fixes...")
        
        results = []
        
        for fix in fixes:
            result = self._validate_single_fix(fix)
            results.append(result)
            
        self.validations_performed += len(results)
        successful = sum(1 for r in results if r.success)
        print(f"{self.name}: {successful}/{len(results)} fixes passed validation")
        
        return results
    
    def _validate_single_fix(self, fix: Fix) -> ValidationResult:
        """Validate a single fix by attempting to execute it"""
        start_time = datetime.now()
        
        try:
            # Create a temporary file with the fixed code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(fix.fixed_code)
                temp_file = f.name
            
            # Try to compile the code
            with open(temp_file, 'r') as f:
                code = f.read()
                compile(code, temp_file, 'exec')
            
            # Basic syntax check passed
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Clean up
            os.unlink(temp_file)
            
            return ValidationResult(
                fix_id=fix.bug_id,
                success=True,
                error_message=None,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Clean up
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            return ValidationResult(
                fix_id=fix.bug_id,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )

class LearningAgent:
    """Agent responsible for learning from validation results"""
    
    def __init__(self):
        self.name = "Learner"
        self.knowledge_base = {
            "successful_patterns": [],
            "failed_patterns": [],
            "bug_fix_success_rate": {}
        }
        
    def learn_from_results(self, bugs: List[Bug], fixes: List[Fix], results: List[ValidationResult]):
        """Learn from the validation results to improve future performance"""
        print(f"{self.name}: Learning from {len(results)} validation results...")
        
        for bug, fix, result in zip(bugs, fixes, results):
            bug_type = bug.type
            
            # Update success rates
            if bug_type not in self.knowledge_base["bug_fix_success_rate"]:
                self.knowledge_base["bug_fix_success_rate"][bug_type] = {"success": 0, "total": 0}
            
            self.knowledge_base["bug_fix_success_rate"][bug_type]["total"] += 1
            
            if result.success:
                self.knowledge_base["bug_fix_success_rate"][bug_type]["success"] += 1
                self.knowledge_base["successful_patterns"].append({
                    "bug_type": bug_type,
                    "fix_approach": fix.explanation,
                    "confidence": fix.confidence
                })
            else:
                self.knowledge_base["failed_patterns"].append({
                    "bug_type": bug_type,
                    "fix_approach": fix.explanation,
                    "error": result.error_message
                })
        
        self._print_learning_summary()
    
    def _print_learning_summary(self):
        """Print what the agent has learned"""
        print(f"\n{self.name}: Learning Summary")
        print("=" * 40)
        
        for bug_type, stats in self.knowledge_base["bug_fix_success_rate"].items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{bug_type}: {success_rate:.1%} success rate ({stats['success']}/{stats['total']})")
        
        print(f"\nTotal successful patterns learned: {len(self.knowledge_base['successful_patterns'])}")
        print(f"Total failed patterns learned: {len(self.knowledge_base['failed_patterns'])}")

class DebuggingSystem:
    """Main orchestrator for the multi-agent debugging system"""
    
    def __init__(self):
        self.bug_detector = BugDetectorAgent()
        self.fix_generator = FixGeneratorAgent()
        self.validator = ValidatorAgent()
        self.learner = LearningAgent()
        
    def debug_code(self, code: str) -> Dict[str, Any]:
        """Run the complete debugging process"""
        print(" Starting Multi-Agent Debugging Process")
        print("=" * 50)
        
        # Step 1: Detect bugs
        bugs = self.bug_detector.analyze_code(code)
        
        if not bugs:
            print(" No bugs detected! Code looks good.")
            return {"status": "clean", "bugs": [], "fixes": [], "results": []}
        
        # Step 2: Generate fixes
        fixes = self.fix_generator.generate_fixes(code, bugs)
        
        # Step 3: Validate fixes
        results = self.validator.validate_fixes(code, fixes)
        
        # Step 4: Learn from results
        self.learner.learn_from_results(bugs, fixes, results)
        
        # Return the best fix
        successful_fixes = [
            (fix, result) for fix, result in zip(fixes, results) 
            if result.success
        ]
        
        if successful_fixes:
            best_fix = max(successful_fixes, key=lambda x: x[0].confidence)
            print(f"\n Best fix found with {best_fix[0].confidence:.1%} confidence")
            return {
                "status": "fixed",
                "bugs": bugs,
                "fixes": fixes,
                "results": results,
                "best_fix": best_fix[0]
            }
        else:
            print(f"\n No successful fixes found")
            return {
                "status": "failed",
                "bugs": bugs,
                "fixes": fixes,
                "results": results,
                "best_fix": None
            }

# Demo function
def run_demo():
    """Run a demo of the debugging system"""
    
    # Test cases with bugs
    buggy_codes = [
        # Test case 1: Division by zero
        '''
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

result = calculate_average([1, 2, 3, 4, 5])
print(f"Average: {result}")
''',
        
        # Test case 2: Resource leak
        '''
def read_file(filename):
    file = open(filename, 'r')
    content = file.read()
    return content

data = read_file('test.txt')
print(data)
''',
        
        # Test case 3: Broad exception
        '''
def divide_numbers(a, b):
    try:
        return a / b
    except:
        return None

result = divide_numbers(10, 0)
print(result)
'''
    ]
    
    system = DebuggingSystem()
    
    for i, code in enumerate(buggy_codes, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}")
        print(f"{'='*60}")
        print("Original code:")
        print(code)
        print("\n" + "-"*60)
        
        result = system.debug_code(code)
        
        if result["best_fix"]:
            print("\n FIXED CODE:")
            print("-" * 30)
            print(result["best_fix"].fixed_code)
            print(f"\nExplanation: {result['best_fix'].explanation}")

if __name__ == "__main__":
    run_demo()