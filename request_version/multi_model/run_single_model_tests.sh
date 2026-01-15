#!/bin/bash

# Script to run single model tests in order: reranker, embedding, chatbot, vision

set -e  # Exit on error

MODELS=("reranker" "embedding" "chatbot" "vision")

echo "=========================================="
echo "Starting Single Model Tests"
echo "=========================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing: $model"
    echo "=========================================="
    echo ""

    python benchmark.py --config "configs/single_${model}_model.yaml"

    echo ""
    echo "Completed: $model"
    echo ""
done

echo "=========================================="
echo "All single model tests completed!"
echo "=========================================="