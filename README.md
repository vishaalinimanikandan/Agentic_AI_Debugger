# AI Debugging Agents 

> A multi-agent AI system where agents collaborate to find and fix bugs in code automatically.

##  What This Does

Four AI agents work together like a debugging team:
- ** Bug Detective**: Scans code for issues
- ** Fix Master**: Generates solutions  
- ** Validator**: Tests if fixes work
- ** Learner**: Improves from results

##  Quick Start

```bash
# Clone the repo
git clone [your-repo-url]
cd ai-debugging-agents

# Run the demo
python test_debug_agents.py

# Or run full test suite
python debug_agents.py
```

##  Demo Results

The system successfully detected and fixed bugs with these results:

- **ZeroDivisionError**: 100% success rate (2/2)
- **BroadException**: 100% success rate (1/1)  
- **ResourceLeak**: 0% success rate (0/1) - *learning opportunity*

**Overall: 67% automatic fix success rate**

##  Sample Output

```
 Starting Multi-Agent Debugging Process
==================================================
Bug Detective : Analyzing code for potential issues...
Bug Detective : Found 1 potential issues
Fix Master : Generating fixes for 1 bugs...
Fix Master : Generated 1 potential fixes
Validator : Validating 1 fixes...
Validator : 1/1 fixes passed validation

 Best fix found with 90.0% confidence

 FIXED CODE:
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
```

*See `demo_output.txt` for complete terminal output.*

##  Technical Details

### Architecture
- **Multi-agent coordination** with specialized roles
- **Real code execution** and validation
- **Learning loops** that track success patterns
- **Confidence scoring** for fix reliability

### Bug Types Detected
- Division by zero vulnerabilities
- Resource leak issues  
- Broad exception handling
- *Easily extensible for more types*

### Technologies
- Python 3.7+
- No external dependencies for basic demo
- Modular design for easy extension

##  Results & Learning

| Bug Type | Success Rate | Fixes Applied |
|----------|-------------|---------------|
| ZeroDivisionError | 100% | 2/2 |
| BroadException | 100% | 1/1 |
| ResourceLeak | 0% | 0/1 |

The system learns from both successes and failures, building knowledge for future debugging sessions.

##  Future Enhancements

- [ ] Integration with OpenAI/Claude APIs for smarter detection
- [ ] Web interface for easier demos
- [ ] GitHub integration for automatic PR reviews
- [ ] More sophisticated bug pattern recognition
- [ ] Team collaboration features

##  Key Insights

1. **Multi-agent coordination** works better than single AI for complex tasks
2. **Learning from failures** is as valuable as learning from successes  
3. **Confidence scoring** makes AI systems more trustworthy
4. **Specialized agents** outperform generalist approaches

##  Article

Read the full story behind this project: [Medium Article Link]

##  Contributing

Want to add new bug types or improve the agents? Check out the issues or submit a PR!

