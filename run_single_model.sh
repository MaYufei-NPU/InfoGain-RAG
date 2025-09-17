#!/bin/bash

model="qwen2.5-72b-instruct"

output_file="formal_answer/trivia/no_rag/${model}.jsonl"
mkdir -p "formal_answer/trivia/no_rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/trivia_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode no_rag

python judge_res.py \
    "qa_dataset/trivia_test_shuffle_2000.jsonl" \
    "$output_file"

output_file="formal_answer/trivia/rag/${model}.jsonl"
mkdir -p "formal_answer/trivia/rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/trivia_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
    "qa_dataset/trivia_test_shuffle_2000.jsonl" \
    "$output_file"


output_file="formal_answer/trivia/bge-rerank/${model}.jsonl"
mkdir -p "formal_answer/trivia/bge-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/trivia_test_shuffle_500_rerank_large.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/trivia_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="formal_answer/trivia/multi-rerank/${model}.jsonl"
mkdir -p "formal_answer/trivia/multi-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/trivia_test_shuffle_500_rerank_bert_multi_0.9weight_0.16.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/trivia_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="formal_answer/nq/no_rag/${model}.jsonl"
mkdir -p "formal_answer/nq/no_rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/nq_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode no_rag

python judge_res.py \
        "qa_dataset/nq_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="formal_answer/nq/rag/${model}.jsonl"
mkdir -p "formal_answer/nq/rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/nq_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/nq_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="formal_answer/nq/bge-rerank/${model}.jsonl"
mkdir -p "formal_answer/nq/bge-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/nq_test_shuffle_500_rerank_large.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/nq_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="formal_answer/nq/multi-rerank/${model}.jsonl"
mkdir -p "formal_answer/nq/multi-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/nq_test_shuffle_500_rerank_bert_multi_0.9weight_0.16.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/nq_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="formal_answer/pop/no_rag/${model}.jsonl"
mkdir -p "formal_answer/pop/no_rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/pop_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode no_rag

python judge_res.py \
        "qa_dataset/pop_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="formal_answer/pop/rag/${model}.jsonl"
mkdir -p "formal_answer/pop/rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/pop_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/pop_test_shuffle_2000.jsonl" \
        "$output_file"
        
output_file="formal_answer/pop/bge-rerank/${model}.jsonl"
mkdir -p "formal_answer/pop/bge-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/pop_test_shuffle_500_rerank_large.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/pop_test_shuffle_2000.jsonl" \
        "$output_file"
        
output_file="formal_answer/pop/multi-rerank/${model}.jsonl"
mkdir -p "formal_answer/pop/multi-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/pop_test_shuffle_500_rerank_bert_multi_0.9weight_0.16.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/pop_test_shuffle_2000.jsonl" \
        "$output_file"
        
echo "All models completed!"