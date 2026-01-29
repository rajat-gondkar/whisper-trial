"""
Tests for the classification engine.
"""

import pytest
from src.classifier import CodingClassifier, TEST_CASES


@pytest.fixture
def classifier():
    """Create classifier with default config."""
    config = {
        'classification': {
            'default_class': 'NON-CODING',
            'confidence_threshold': 0.7
        }
    }
    return CodingClassifier(config)


class TestCodingClassifier:
    """Test cases for CodingClassifier."""
    
    def test_empty_input(self, classifier):
        """Test empty input returns NON-CODING."""
        result = classifier.classify("")
        assert result.classification == "NON-CODING"
        assert result.confidence == 1.0
    
    def test_whitespace_input(self, classifier):
        """Test whitespace input returns NON-CODING."""
        result = classifier.classify("   ")
        assert result.classification == "NON-CODING"
    
    # CODING test cases
    @pytest.mark.parametrize("text", [
        "Write a Python function to reverse a linked list",
        "Give me Java code for Dijkstra's algorithm",
        "Implement binary search in C++",
        "Write SQL query to fetch duplicate rows",
        "Create a REST API using Flask",
        "Write a React component that displays a list",
        "Write Python code for BFS",
        "Implement recursion for factorial",
        "Create a compiler in C",
        "Implement HTTP server in Node.js",
        "Write recursive factorial function",
        "Write merge sort in python",
        "Build a web scraper in Python",
        "Generate code for sorting algorithm",
        "Program a calculator in JavaScript",
        "Develop a chat application",
        "Code a binary tree implementation",
    ])
    def test_coding_classification(self, classifier, text):
        """Test that coding requests are classified as CODING."""
        result = classifier.classify(text)
        assert result.classification == "CODING", \
            f"Expected CODING for '{text}', got {result.classification} ({result.reasoning})"
    
    # NON-CODING test cases
    @pytest.mark.parametrize("text", [
        "What is machine learning?",
        "Explain how blockchain works",
        "What is OOP?",
        "How does recursion work?",
        "What is the difference between stack and queue?",
        "Can you explain quicksort?",
        "Is Python better than Java?",
        "How do APIs work?",
        "What is overfitting?",
        "Tell me about AI",
        "How to prepare for coding interviews?",
        "Explain BFS algorithm",
        "How does HTTP work?",
        "What is recursion?",
        "What is a compiler?",
        "Explain merge sort",
        "What are the advantages of Python?",
        "Describe the difference between GET and POST",
        "How do databases work?",
        "What is the time complexity of quicksort?",
    ])
    def test_non_coding_classification(self, classifier, text):
        """Test that conceptual questions are classified as NON-CODING."""
        result = classifier.classify(text)
        assert result.classification == "NON-CODING", \
            f"Expected NON-CODING for '{text}', got {result.classification} ({result.reasoning})"
    
    def test_all_test_cases(self, classifier):
        """Run all predefined test cases."""
        passed = 0
        failed = 0
        failures = []
        
        for text, expected in TEST_CASES:
            result = classifier.classify(text)
            if result.classification == expected:
                passed += 1
            else:
                failed += 1
                failures.append((text, expected, result.classification))
        
        # Report failures
        if failures:
            print("\nFailed test cases:")
            for text, expected, got in failures:
                print(f"  '{text}': expected {expected}, got {got}")
        
        total = passed + failed
        accuracy = passed / total * 100
        
        assert accuracy >= 95, f"Accuracy {accuracy:.1f}% below threshold (95%)"
    
    def test_confidence_reasonable(self, classifier):
        """Test that confidence values are reasonable."""
        # Strong coding request
        result = classifier.classify("Write a Python function for sorting")
        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence >= 0.7  # Should be confident for clear requests
        
        # Strong non-coding request
        result = classifier.classify("What is a function?")
        assert 0.0 <= result.confidence <= 1.0
    
    def test_edge_case_mixed_signals(self, classifier):
        """Test edge cases with mixed signals - should default to NON-CODING."""
        # Mentions coding but asks for explanation
        result = classifier.classify("Explain how to write a function")
        assert result.classification == "NON-CODING"
        
        # Asks about coding concept
        result = classifier.classify("What does the write function do?")
        assert result.classification == "NON-CODING"


class TestClassifierPatterns:
    """Test pattern matching functionality."""
    
    def test_coding_verbs_detected(self, classifier):
        """Test that coding verbs trigger detection."""
        for verb in ["write", "implement", "create", "build", "generate"]:
            text = f"{verb} a sorting algorithm in Python"
            result = classifier.classify(text)
            assert any("coding" in p.lower() for p in result.matched_patterns), \
                f"Should detect coding pattern for verb '{verb}'"
    
    def test_non_coding_indicators_detected(self, classifier):
        """Test that non-coding indicators are detected."""
        for indicator in ["what is", "explain", "how does", "describe"]:
            text = f"{indicator} a sorting algorithm"
            result = classifier.classify(text)
            assert result.classification == "NON-CODING", \
                f"Should classify as NON-CODING for indicator '{indicator}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
