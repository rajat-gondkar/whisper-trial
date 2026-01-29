#!/usr/bin/env python3
"""Test all 165 questions from tests.txt"""

import sys
sys.path.insert(0, '.')

from src.classifier import CodingClassifier
from src.utils import load_config

# Load config and create classifier
config = load_config()
classifier = CodingClassifier(config)

# Read questions from tests.txt
with open('tests.txt', 'r') as f:
    lines = f.readlines()

# Parse questions (remove numbering)
questions = []
for line in lines:
    line = line.strip()
    if line and line[0].isdigit():
        # Remove the number prefix "1. ", "2. ", etc.
        question = line.split('. ', 1)[1] if '. ' in line else line
        questions.append(question)

print("=" * 80)
print(f"TESTING ALL {len(questions)} CODING QUESTIONS FROM tests.txt")
print("=" * 80)

passed = 0
failed = 0
failed_questions = []

for i, question in enumerate(questions, 1):
    result = classifier.classify(question)
    expected = "CODING"
    is_correct = result.classification == expected
    
    if is_correct:
        passed += 1
        status = "✓"
    else:
        failed += 1
        status = "✗"
        failed_questions.append((i, question, result))
    
    # Display progress every 25 questions
    if i % 25 == 0 or i == len(questions):
        print(f"Progress: {i}/{len(questions)} tested... ({passed} ✓, {failed} ✗)")

print("=" * 80)
accuracy = (100 * passed) // len(questions)
print(f"FINAL RESULTS: {passed}/{len(questions)} passed ({accuracy}%)")
print("=" * 80)

if failed_questions:
    print(f"\n❌ FAILED QUESTIONS ({len(failed_questions)}):\n")
    for idx, question, result in failed_questions:
        print(f"#{idx}: {question[:70]}...")
        print(f"   Got: {result.classification} (confidence: {result.confidence:.2f})")
        print(f"   Reasoning: {result.reasoning}\n")
else:
    print(f"\n✅ ALL {len(questions)} QUESTIONS CORRECTLY CLASSIFIED AS CODING!")
