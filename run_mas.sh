#!/bin/bash

if [ -f "./.env" ]; then
    export $(grep -v '^#' "./.env" | xargs)
fi

MODEL=${OPENAI_MODEL:-qwen/qwen-2.5-7b-instruct}

# Options:
# --mas_memory:    empty, chatdev, metagpt, voyager, generative, memorybank, g-memory
# --mas_type:      autogen, dylan, macnet
# --task:          alfworld, fever, pddl, sciworld

python3 tasks/run.py \
    --task alfworld \
    --reasoning io \
    --mas_memory g-memory \
    --max_trials 30 \
    --mas_type autogen \
    --model "$MODEL"
