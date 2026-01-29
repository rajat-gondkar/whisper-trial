"""
Strict CODING vs NON-CODING classification engine.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Classification(Enum):
    CODING = "CODING"
    NON_CODING = "NON-CODING"


@dataclass
class ClassificationResult:
    """Result of classification."""
    classification: str
    confidence: float
    reasoning: str
    matched_patterns: List[str]


class CodingClassifier:
    """
    Strict classifier for distinguishing CODING from NON-CODING questions.
    
    CRITICAL RULES:
    1. ONLY classify as CODING if explicit code implementation is requested
    2. Default to NON-CODING for any ambiguity
    3. Theoretical/conceptual questions are ALWAYS NON-CODING
    4. Must have explicit coding intent verbs + implementation context
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize classifier with configuration."""
        class_config = config.get('classification', {})
        
        self.default_class = class_config.get('default_class', 'NON-CODING')
        self.confidence_threshold = class_config.get('confidence_threshold', 0.7)
        
        # Coding intent verbs (MUST be present for CODING)
        self.coding_verbs = set(
            v.lower() for v in class_config.get('coding_verbs', [
                "write", "implement", "create", "build", "generate",
                "code", "program", "develop", "make", "design",
                "solve", "fix", "debug", "complete", "finish", "find"
            ])
        )
        
        # Strong coding phrases
        self.coding_phrases = [
            phrase.lower() for phrase in class_config.get('coding_phrases', [
                "write code", "write a function", "write a program",
                "write a script", "implement a", "create a function",
                "create a class", "create an api", "build a",
                "generate code", "code for", "code to", "program to",
                "program for", "give me code", "show me code",
                "write me", "can you code", "can you write",
                # Solving/completing code problems
                "solve this", "solve the", "solve it", "solve these",
                "find the solution", "find solution", "give solution",
                "solve this coding", "solve the coding", "solve this problem",
                "solve the problem", "solve this question", "solve the question",
                "coding question", "coding problem", "programming question",
                "programming problem", "dsa question", "dsa problem",
                "leetcode", "hackerrank", "codeforces", "codechef",
                "algorithm question", "algorithm problem",
                "on the screen", "given on screen", "shown on screen",
                "complete the code", "finish the code", "fix this code",
                "debug this", "fix this", "correct this code"
            ])
        ]
        
        # Programming languages
        self.programming_languages = set(
            lang.lower() for lang in class_config.get('programming_languages', [
                "python", "java", "javascript", "typescript", "c++", "cpp",
                "c#", "csharp", "ruby", "go", "golang", "rust", "swift",
                "kotlin", "scala", "php", "perl", "r", "matlab", "sql",
                "html", "css", "bash", "shell", "powershell"
            ])
        )
        
        # Code structure terms
        self.code_structures = set(
            term.lower() for term in class_config.get('code_structures', [
                "function", "method", "class", "api", "endpoint", "component",
                "module", "script", "algorithm", "data structure", "loop",
                "array", "list", "dictionary", "hashmap", "tree", "graph",
                "linked list", "stack", "queue", "database", "query",
                "server", "client", "rest", "graphql",
                # DSA/Problem solving terms
                "dsa", "leetcode", "hackerrank", "recursion", "dynamic programming",
                "sorting", "searching", "binary search", "bfs", "dfs",
                "two pointer", "sliding window", "backtracking", "greedy",
                "coding question", "coding problem", "programming question"
            ])
        )
        
        # NON-CODING indicators (strong signals for non-coding)
        self.non_coding_indicators = [
            indicator.lower() for indicator in class_config.get('non_coding_indicators', [
                "what is", "what are", "explain", "describe", "tell me about",
                "how does", "how do", "why is", "why does", "when should",
                "difference between", "compare", "versus", "vs", "define",
                "definition", "meaning of", "concept of", "theory", "principle",
                "best practice", "advantage", "disadvantage", "pros and cons",
                "is it better", "should i use", "which is better"
            ])
        ]
        
        # Build regex patterns
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for matching."""
        # Pattern for coding verbs
        verb_pattern = '|'.join(re.escape(v) for v in self.coding_verbs)
        self.coding_verb_pattern = re.compile(
            rf'\b({verb_pattern})\b',
            re.IGNORECASE
        )
        
        # Pattern for programming languages
        lang_pattern = '|'.join(re.escape(l) for l in self.programming_languages)
        self.language_pattern = re.compile(
            rf'\b({lang_pattern})\b',
            re.IGNORECASE
        )
        
        # Pattern for code structures
        struct_pattern = '|'.join(re.escape(s) for s in self.code_structures)
        self.structure_pattern = re.compile(
            rf'\b({struct_pattern})\b',
            re.IGNORECASE
        )
        
        # Pattern for strong coding requests
        # e.g., "write ... code", "implement ... function", etc.
        self.strong_coding_pattern = re.compile(
            r'\b(write|implement|create|build|generate|develop|code|program|solve|fix|debug|complete)\b'
            r'.*\b(code|function|method|class|program|script|api|algorithm|'
            r'server|client|component|module|query|question|problem|dsa|leetcode)\b',
            re.IGNORECASE
        )
        
        # Pattern for solving coding/DSA questions
        self.solve_coding_pattern = re.compile(
            r'\b(solve|find|give|show|help|do|complete|finish)\b'
            r'.*\b(coding|dsa|algorithm|programming|leetcode|hackerrank|'
            r'question|problem|solution|this|it|these)\b',
            re.IGNORECASE
        )
        
        # Pattern for "it's a DSA/coding question" type phrases
        self.dsa_question_pattern = re.compile(
            r"(it'?s\s+a\s+(dsa|coding|algorithm|programming|leetcode)\s*(question|problem)?|"
            r"(dsa|coding|algorithm)\s*(question|problem)|"
            r"about\s+(dsa|coding|algorithm|programming)|"
            r"(question|problem)\s+(about|on|for)\s+(dsa|coding|algorithm))",
            re.IGNORECASE
        )
        
        # Pattern for screen-based coding questions
        self.screen_pattern = re.compile(
            r'(on\s*(the)?\s*screen|given\s*on|shown\s*on|this\s*question|this\s*problem)',
            re.IGNORECASE
        )
        
        # Pattern to detect questions (often non-coding)
        self.question_pattern = re.compile(
            r'^(what|how|why|when|where|who|which|can you explain|'
            r'could you explain|please explain|tell me)\b',
            re.IGNORECASE
        )
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as CODING or NON-CODING.
        
        Args:
            text: Transcribed text to classify
            
        Returns:
            ClassificationResult with classification and confidence
        """
        if not text or not text.strip():
            return ClassificationResult(
                classification="NON-CODING",
                confidence=1.0,
                reasoning="Empty input",
                matched_patterns=[]
            )
        
        text_lower = text.lower().strip()
        matched_patterns = []
        
        # Step 1: Check for strong NON-CODING indicators first
        non_coding_score, non_coding_matches = self._check_non_coding_indicators(text_lower)
        matched_patterns.extend(non_coding_matches)
        
        # Step 2: Check for CODING indicators
        coding_score, coding_matches = self._check_coding_indicators(text_lower)
        matched_patterns.extend(coding_matches)
        
        # Step 3: Apply strict classification logic
        classification, confidence, reasoning = self._apply_classification_logic(
            text_lower, coding_score, non_coding_score, coding_matches, non_coding_matches
        )
        
        return ClassificationResult(
            classification=classification,
            confidence=confidence,
            reasoning=reasoning,
            matched_patterns=matched_patterns
        )
    
    def _check_non_coding_indicators(self, text: str) -> Tuple[float, List[str]]:
        """Check for NON-CODING indicators."""
        score = 0.0
        matches = []
        
        for indicator in self.non_coding_indicators:
            if indicator in text:
                score += 0.3
                matches.append(f"non_coding: '{indicator}'")
        
        # Question patterns are often conceptual
        if self.question_pattern.match(text):
            # But not if it's "can you write/solve/code" type
            coding_action_phrases = [
                "can you write", "can you code", "can you create", "can you implement",
                "can you solve", "can you fix", "can you complete", "can you debug",
                "can you help me solve", "can you help solve", "could you solve",
                "are you able to solve", "are you able to find"
            ]
            if not any(phrase in text for phrase in coding_action_phrases):
                score += 0.2
                matches.append("non_coding: question_pattern")
        
        return min(score, 1.0), matches
    
    def _check_coding_indicators(self, text: str) -> Tuple[float, List[str]]:
        """Check for CODING indicators."""
        score = 0.0
        matches = []
        
        # Check for coding verbs
        verb_matches = self.coding_verb_pattern.findall(text)
        if verb_matches:
            score += 0.3
            matches.append(f"coding_verb: {verb_matches}")
        
        # Check for programming languages
        lang_matches = self.language_pattern.findall(text)
        if lang_matches:
            score += 0.15
            matches.append(f"language: {lang_matches}")
        
        # Check for code structures
        struct_matches = self.structure_pattern.findall(text)
        if struct_matches:
            score += 0.15
            matches.append(f"structure: {struct_matches}")
        
        # Check for strong coding phrases
        for phrase in self.coding_phrases:
            if phrase in text:
                score += 0.4
                matches.append(f"coding_phrase: '{phrase}'")
                break  # Only count once
        
        # Check for strong coding pattern (verb + code structure)
        if self.strong_coding_pattern.search(text):
            score += 0.3
            matches.append("strong_coding_pattern")
        
        # Check for solve coding pattern (solve/help with coding questions)
        if self.solve_coding_pattern.search(text):
            score += 0.4
            matches.append("solve_coding_pattern")
        
        # Check for screen-based coding questions
        if self.screen_pattern.search(text):
            # Screen reference + coding-related terms = likely a coding request
            if any(term in text for term in ["coding", "dsa", "algorithm", "question", "problem", "solve"]):
                score += 0.3
                matches.append("screen_coding_pattern")
        
        # Check for "it's a DSA/coding question" type patterns
        if self.dsa_question_pattern.search(text):
            score += 0.5
            matches.append("dsa_question_pattern")
        
        # Check for imperative sentence structure with coding verb at start
        first_word = text.split()[0] if text.split() else ""
        if first_word in self.coding_verbs:
            score += 0.2
            matches.append(f"imperative_start: '{first_word}'")
        
        # Check for "please solve" or similar polite requests
        if re.search(r'\b(please|can you|could you|help me|i need you to)\b.*\b(solve|complete|fix|do|finish)\b', text):
            score += 0.3
            matches.append("polite_solve_request")
        
        return min(score, 1.0), matches
    
    def _apply_classification_logic(
        self,
        text: str,
        coding_score: float,
        non_coding_score: float,
        coding_matches: List[str],
        non_coding_matches: List[str]
    ) -> Tuple[str, float, str]:
        """
        Apply strict classification logic.
        
        CRITICAL: Only classify as CODING if:
        1. There is a coding verb present AND
        2. There is explicit implementation intent AND
        3. NON-CODING indicators don't dominate
        """
        
        # Rule 1: If no coding indicators at all → NON-CODING
        if coding_score == 0:
            return (
                "NON-CODING",
                0.95,
                "No coding indicators found"
            )
        
        # Rule 2: If only has programming language without coding verb → NON-CODING
        # e.g., "What is Python?" or "Tell me about JavaScript"
        has_coding_verb = any("coding_verb" in m or "imperative_start" in m 
                             for m in coding_matches)
        has_strong_phrase = any("coding_phrase" in m or "strong_coding_pattern" in m 
                               for m in coding_matches)
        has_solve_pattern = any("solve_coding_pattern" in m or "dsa_question_pattern" in m 
                               or "polite_solve_request" in m or "screen_coding_pattern" in m
                               for m in coding_matches)
        
        if not has_coding_verb and not has_strong_phrase and not has_solve_pattern:
            return (
                "NON-CODING",
                0.85,
                "No explicit coding intent verb found"
            )
        
        # Rule 3: If strong NON-CODING indicators present → analyze further
        if non_coding_score > 0.3:
            # Explanatory questions about code concepts → NON-CODING
            # e.g., "Explain how recursion works", "What is a function"
            if non_coding_score > coding_score:
                return (
                    "NON-CODING",
                    non_coding_score,
                    f"Non-coding indicators ({non_coding_score:.2f}) outweigh coding ({coding_score:.2f})"
                )
            
            # Mixed signals - be conservative, unless we have strong solve/DSA patterns
            if not has_strong_phrase and not has_solve_pattern:
                return (
                    "NON-CODING",
                    0.7,
                    "Ambiguous - defaulting to NON-CODING"
                )
        
        # Rule 4: Strong coding pattern or solve pattern present → CODING
        if has_strong_phrase or has_solve_pattern or (has_coding_verb and coding_score >= 0.5):
            confidence = min(coding_score + 0.2, 1.0)
            return (
                "CODING",
                confidence,
                f"Strong coding intent detected (score: {coding_score:.2f})"
            )
        
        # Rule 5: Default to NON-CODING for ambiguity
        return (
            "NON-CODING",
            0.6,
            "Insufficient coding intent - defaulting to NON-CODING"
        )


# Test cases from requirements
TEST_CASES = [
    # CODING cases
    ("Write a Python function to reverse a linked list", "CODING"),
    ("Give me Java code for Dijkstra's algorithm", "CODING"),
    ("Implement binary search in C++", "CODING"),
    ("Write SQL query to fetch duplicate rows", "CODING"),
    ("Create a REST API using Flask", "CODING"),
    ("Write a React component that displays a list", "CODING"),
    ("Write Python code for BFS", "CODING"),
    ("Implement recursion for factorial", "CODING"),
    ("Create a compiler in C", "CODING"),
    ("Implement HTTP server in Node.js", "CODING"),
    ("Write recursive factorial function", "CODING"),
    ("Write merge sort in python", "CODING"),
    
    # NON-CODING cases
    ("What is machine learning?", "NON-CODING"),
    ("Explain how blockchain works", "NON-CODING"),
    ("What is OOP?", "NON-CODING"),
    ("How does recursion work?", "NON-CODING"),
    ("What is the difference between stack and queue?", "NON-CODING"),
    ("Can you explain quicksort?", "NON-CODING"),
    ("Is Python better than Java?", "NON-CODING"),
    ("How do APIs work?", "NON-CODING"),
    ("What is overfitting?", "NON-CODING"),
    ("Tell me about AI", "NON-CODING"),
    ("How to prepare for coding interviews?", "NON-CODING"),
    ("Explain BFS algorithm", "NON-CODING"),
    ("How does HTTP work?", "NON-CODING"),
    ("What is recursion?", "NON-CODING"),
    ("What is a compiler?", "NON-CODING"),
    ("Explain merge sort", "NON-CODING"),
]


def run_tests(classifier: CodingClassifier) -> Dict[str, Any]:
    """Run test cases and return results."""
    results = {
        "passed": 0,
        "failed": 0,
        "failures": []
    }
    
    for text, expected in TEST_CASES:
        result = classifier.classify(text)
        
        if result.classification == expected:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "input": text,
                "expected": expected,
                "got": result.classification,
                "confidence": result.confidence,
                "reasoning": result.reasoning
            })
    
    return results


if __name__ == "__main__":
    # Test classifier
    print("🧪 Testing Coding Classifier")
    print("=" * 60)
    
    # Create classifier with default config
    config = {
        'classification': {
            'default_class': 'NON-CODING',
            'confidence_threshold': 0.7
        }
    }
    
    classifier = CodingClassifier(config)
    
    # Run tests
    results = run_tests(classifier)
    
    total = results["passed"] + results["failed"]
    accuracy = results["passed"] / total * 100 if total > 0 else 0
    
    print(f"\n📊 Results: {results['passed']}/{total} passed ({accuracy:.1f}%)")
    
    if results["failures"]:
        print(f"\n❌ Failures ({results['failed']}):")
        for f in results["failures"]:
            print(f"\n  Input: \"{f['input']}\"")
            print(f"  Expected: {f['expected']}, Got: {f['got']}")
            print(f"  Confidence: {f['confidence']:.2f}")
            print(f"  Reasoning: {f['reasoning']}")
    else:
        print("\n✅ All tests passed!")
    
    # Interactive testing
    print("\n" + "=" * 60)
    print("🎯 Interactive Testing (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            text = input("\nEnter text to classify: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            result = classifier.classify(text)
            print(f"\n  📋 Classification: {result.classification}")
            print(f"  📊 Confidence: {result.confidence:.2%}")
            print(f"  💭 Reasoning: {result.reasoning}")
            if result.matched_patterns:
                print(f"  🔍 Matched: {', '.join(result.matched_patterns[:5])}")
        
        except KeyboardInterrupt:
            break
    
    print("\n👋 Done!")
