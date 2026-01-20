#!/bin/bash
# Comprehensive evaluation script for FusedODModel
# Runs single-step, recursive forecasting, and multi-step evaluations

# Set paths (adjust these to your data locations)
ADJACENCY_PATH="/path/to/adjacency_matrix.csv"
DISTANCE_PATH="/path/to/distance_matrix.csv"
TRIPS_TENSOR_PATH="/path/to/trips_tensor.pt"
CHECKPOINT_PATH="/path/to/trained_model.pth"
OUTPUT_DIR="./evaluation_outputs"

# Evaluation parameters
BATCH_SIZE=64
HIDDEN_SIZE=64

# Window parameters (must match training)
W_LONG=144
W_SHORT=36
CHUNK_SIZE_SHORT=9
NUM_CHUNKS_SHORT=4

# Forecasting parameters
FORECAST_STEPS=144
PLOT_HORIZONS="1 6 18 72 144"

# Multi-step prediction horizons
PREDICTION_HORIZONS="1 36 144 432 1008"

# Single-Step Evaluation
echo "[1/3] Running single-step evaluation..."
python eval_main.py \
    --adjacency-path $ADJACENCY_PATH \
    --distance-path $DISTANCE_PATH \
    --trips-tensor-path $TRIPS_TENSOR_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --output-dir "${OUTPUT_DIR}/single_step" \
    --batch-size $BATCH_SIZE \
    --hidden-size $HIDDEN_SIZE \
    --w-long $W_LONG \
    --w-short $W_SHORT \
    --chunk-size-short $CHUNK_SIZE_SHORT \
    --num-chunks-short $NUM_CHUNKS_SHORT \
    --eval-train \
    --eval-test

if [ $? -eq 0 ]; then
    echo "✓ Single-step evaluation completed successfully!"
else
    echo "✗ Single-step evaluation failed!"
    exit 1
fi
echo ""

# Recursive Forecasting Evaluation
echo "[2/3] Running recursive forecasting evaluation..."
python eval_main.py \
    --adjacency-path $ADJACENCY_PATH \
    --distance-path $DISTANCE_PATH \
    --trips-tensor-path $TRIPS_TENSOR_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --output-dir "${OUTPUT_DIR}/recursive_forecast" \
    --batch-size $BATCH_SIZE \
    --hidden-size $HIDDEN_SIZE \
    --w-long $W_LONG \
    --w-short $W_SHORT \
    --chunk-size-short $CHUNK_SIZE_SHORT \
    --num-chunks-short $NUM_CHUNKS_SHORT \
    --recursive-forecast \
    --forecast-steps $FORECAST_STEPS \
    --plot-horizons $PLOT_HORIZONS

if [ $? -eq 0 ]; then
    echo "✓ Recursive forecasting completed successfully!"
else
    echo "✗ Recursive forecasting failed!"
    exit 1
fi
echo ""

# 3. Multi-Step Evaluation
echo "[3/3] Running multi-step evaluation..."
echo "--------------------------------------"
python eval_main.py \
    --adjacency-path $ADJACENCY_PATH \
    --distance-path $DISTANCE_PATH \
    --trips-tensor-path $TRIPS_TENSOR_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --output-dir "${OUTPUT_DIR}/multi_step" \
    --batch-size $BATCH_SIZE \
    --hidden-size $HIDDEN_SIZE \
    --w-long $W_LONG \
    --w-short $W_SHORT \
    --chunk-size-short $CHUNK_SIZE_SHORT \
    --num-chunks-short $NUM_CHUNKS_SHORT \
    --multi-step \
    --prediction-horizons $PREDICTION_HORIZONS \
    --eval-test \
    --recursive-forecast

if [ $? -eq 0 ]; then
    echo "✓ Multi-step evaluation completed successfully!"
else
    echo "✗ Multi-step evaluation failed!"
    exit 1
fi
echo ""

# Summary
echo "Results saved to:"
echo "  - Single-step:      ${OUTPUT_DIR}/single_step/"
echo "  - Recursive:        ${OUTPUT_DIR}/recursive_forecast/"
echo "  - Multi-step:       ${OUTPUT_DIR}/multi_step/"
