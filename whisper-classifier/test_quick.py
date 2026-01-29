#!/usr/bin/env python3
"""Quick test script for the classifier."""

import sys
sys.path.insert(0, '.')

from src.classifier import CodingClassifier
from src.utils import load_config

# Load config and create classifier
config = load_config()
classifier = CodingClassifier(config)

# Test cases from requirements
TEST_CASES = [
    # CODING cases
    ("Write Python code for BFS", "CODING"),
    ("Write a Python function to reverse a linked list", "CODING"),
    ("Give me Java code for Dijkstra's algorithm", "CODING"),
    ("Implement binary search in C++", "CODING"),
    ("Write SQL query to fetch duplicate rows", "CODING"),
    ("Create a REST API using Flask", "CODING"),
    ("Write a React component that displays a list", "CODING"),
    ("Implement recursion for factorial", "CODING"),
    ("Create a compiler in C", "CODING"),
    ("Implement HTTP server in Node.js", "CODING"),
    ("Write merge sort in python", "CODING"),
    ("Write recursive factorial function", "CODING"),
    
    # NEW: Solve/DSA type questions (should be CODING)
    ("Can you solve these coding questions about making a DSA question?", "CODING"),
    ("Please solve this coding question on the screen.", "CODING"),
    ("Are you able to find the solution for the reporting question given on the screen? It's about DSA.", "CODING"),
    ("It's a DSA question, please solve it, I told you.", "CODING"),
    ("Solve this leetcode problem for me", "CODING"),
    ("Can you solve this algorithm problem?", "CODING"),
    ("Help me solve this coding problem", "CODING"),
    ("Find the solution for this DSA question", "CODING"),
    ("Solve the two sum problem in python", "CODING"),
    ("Can you complete this code for me?", "CODING"),
    ("Fix this bug in my code", "CODING"),
    ("Debug this function please", "CODING"),
    
    # NEW: User's specific examples - traversal and algorithm requests
    ("Give me pre-ordered traversal of a binary tree.", "CODING"),
    ("Give me post-order traversal of a binary tree.", "CODING"),
    ("Give me BFS approach of graph traversal.", "CODING"),
    ("Give me DFS approach of graph traversal.", "CODING"),
    ("Show me the inorder traversal", "CODING"),
    ("Show me level order traversal code", "CODING"),
    ("Provide the dijkstra algorithm", "CODING"),
    ("Give me merge sort implementation", "CODING"),
    ("Show me quick sort code", "CODING"),
    ("Give me binary search", "CODING"),
    ("Show me the two pointer approach", "CODING"),
    ("Give me sliding window solution", "CODING"),
    
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

print("=" * 70)
print("CLASSIFIER TEST RESULTS")
print("=" * 70)

passed = 0
failed = 0

for text, expected in TEST_CASES:
    result = classifier.classify(text)
    is_correct = result.classification == expected
    
    if is_correct:
        passed += 1
        status = "✓"
    else:
        failed += 1
        status = "✗"
    
    print(f"{status} [{expected:10}] {text[:50]}")
    if not is_correct:
        print(f"  → Got: {result.classification} (confidence: {result.confidence:.2f})")

print("=" * 70)
print(f"RESULTS: {passed}/{len(TEST_CASES)} passed ({100*passed/len(TEST_CASES):.1f}%)")
print(f"Target: >98% accuracy")
print("=" * 70)
