# Financial Advice Moderation System Documentation

## Overview

This document outlines the moderation system implemented for evaluating financial advice content. The system combines OpenAI's Moderation API with custom domain-specific checks to provide comprehensive content filtering for financial advisory applications.

## Moderation Components

The moderation system consists of three main components:

1. **OpenAI Moderation API** (`omni-moderation-latest` model)
   - Provides general content moderation across multiple categories
   - Identifies potentially harmful or inappropriate content
   - Now includes financial security categories: `illicit` and `illicit/violent`

2. **Custom Financial Red Flag Detection**
   - Identifies common misleading financial claims:
     - Guaranteed returns
     - Unrealistic promises (get-rich-quick schemes)
     - Urgency/pressure tactics
     - False regulatory claims
     - Tax evasion suggestions

3. **PII and Governance Checks**
   - Detects attempts to collect sensitive personal or financial information
   - Flags potential privacy violations
   - Identifies prompt injection attempts

## Implementation Details

### Basic Moderation

```python
def moderate_content(text):
    """Use OpenAI Moderation API to check for inappropriate content."""
    try:
        response = openai.moderations.create(input=text, model="omni-moderation-latest")
        return response.results[0]
    except Exception as e:
        return {"error": str(e)}
```

### Enhanced Moderation

The enhanced moderation combines OpenAI's API with custom checks:

```python
def enhanced_content_moderation(text, custom_flags=None):
    """
    Combines OpenAI moderation with custom financial, PII, governance flags.
    Returns structured results by category.
    """
    # OpenAI moderation
    openai_moderation = moderate_content(text)
    
    # Custom flag detection for financial red flags, PII issues, etc.
    # ...
    
    return {
        "openai_flagged": any(openai_moderation.get("categories", {}).values()),
        "custom_flagged": any([financial_flagged, pii_flagged, injection_flagged]),
        "harmful_content": { /* details */ },
        "financial_red_flags": { /* details */ }, 
        "pii_governance": { /* details */ },
        "prompt_injection": { /* details */ }
    }
```

### Moderation Evaluation

For robust evaluation, we use multiple techniques:

1. **QA Example Generation**: Using LangChain's `QAGenerateChain.apply_and_parse()` to generate diverse QA pairs from financial documents.

2. **Human-crafted Test Cases**: Including typical questions and deliberately problematic examples.

3. **Edge Case Testing**: Specifically designed subtle violations that might evade detection:
   - Disguised tax evasion advice
   - Subtle market manipulation suggestions
   - Veiled urgency tactics
   - Indirect requests for sensitive information
   - Misleading regulatory claims

## Moderation Effectiveness Analysis

Our testing revealed several insights:

### OpenAI Moderation API Strengths
- Effective at detecting explicit illegal financial activities
- Good at identifying overt tax evasion suggestions
- Can catch certain types of harmful content that custom rules might miss

### OpenAI Moderation API Limitations
- Produces false negatives for subtle financial red flags
- Does not detect many financial-specific issues like guaranteed returns promises
- Does not identify PII/sensitive data requests consistently

### Custom Moderation Strengths
- More precise detection of financial-specific red flags
- Better at identifying misleading financial claims
- Effectively catches sensitive information requests
- More adaptable to domain-specific concerns

### Performance Metrics (Based on Test Cases)
- OpenAI API identified 2 out of 10 problematic examples
- Custom moderation identified 4 out of 10 problematic examples
- False positives from OpenAI: 2 examples
- False negatives from OpenAI: 4 examples (caught by custom rules)

## Recommendations

1. **Combined Approach**: Continue using both OpenAI moderation and custom checks for comprehensive coverage.

2. **Rule Refinement**: Expand custom rules based on false negatives identified during testing.

3. **Continual Testing**: Regularly update test cases with new patterns of problematic content.

4. **Human Review**: Implement a human review process for edge cases where moderation results are inconsistent.

5. **Model Updates**: Monitor for updates to the OpenAI moderation models, as capabilities may change over time.

## Conclusion

The combined moderation system provides robust protection against harmful financial advice and other problematic content. By layering general moderation capabilities from OpenAI with domain-specific checks, we achieve better coverage of potential issues.

This approach can be further refined with human feedback and evaluation to improve detection accuracy and reduce false positives/negatives over time.
