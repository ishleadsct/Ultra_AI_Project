#!/bin/bash
echo "=== FILES WITH CONTENT ==="
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" -o -name "*.md" -o -name "*.sh" -o -name "*.json" \) -size +0c | sort

echo -e "\n=== EMPTY FILES ==="
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" -o -name "*.md" -o -name "*.sh" -o -name "*.json" \) -size 0c | sort

echo -e "\n=== FILE COUNT SUMMARY ==="
echo "Files with content: $(find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" -o -name "*.md" -o -name "*.sh" -o -name "*.json" \) -size +0c | wc -l)"
echo "Empty files: $(find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" -o -name "*.md" -o -name "*.sh" -o -name "*.json" \) -size 0c | wc -l)"
