#!/bin/bash
# Safe Phase 2 Runner Script
# Includes safeguards to validate before committing to full GPU time

set -e  # Exit on error

echo "================================"
echo "RKCNN_SAE Phase 2 - Safe Runner"
echo "================================"
echo ""

# Navigate to project
cd /workspace/RKCNN_SAE

# ===== SAFEGUARD 1: Check Phase 1 passed =====
echo "[Safeguard 1] Checking Phase 1 results..."
if [ -f "/workspace/results/phase1/phase1_results.json" ]; then
    PHASE1_PASS=$(python -c "import json; print(json.load(open('/workspace/results/phase1/phase1_results.json'))['success_criteria']['overall_pass'])")
    if [ "$PHASE1_PASS" != "True" ]; then
        echo "ERROR: Phase 1 did not pass. Cannot proceed to Phase 2."
        echo "Please run Phase 1 first: python experiments/phase1_toy_model.py"
        exit 1
    fi
    echo "  Phase 1 passed ✓"
else
    echo "WARNING: No Phase 1 results found. Running Phase 1 first..."
    python experiments/phase1_toy_model.py --output-dir /workspace/results/phase1

    PHASE1_PASS=$(python -c "import json; print(json.load(open('/workspace/results/phase1/phase1_results.json'))['success_criteria']['overall_pass'])")
    if [ "$PHASE1_PASS" != "True" ]; then
        echo "ERROR: Phase 1 failed. Cannot proceed to Phase 2."
        exit 1
    fi
fi

# ===== SAFEGUARD 2: Dry Run =====
echo ""
echo "[Safeguard 2] Running dry run (minimal data)..."
python experiments/phase2_gpt2.py \
    --dry-run \
    --output-dir /workspace/results/phase2_dryrun \
    2>&1 | tee /workspace/results/phase2_dryrun.log

DRY_RUN_EXIT=$?
if [ $DRY_RUN_EXIT -ne 0 ]; then
    echo "ERROR: Dry run failed. Please fix issues before full run."
    exit 1
fi
echo "  Dry run passed ✓"

# ===== SAFEGUARD 3: User Confirmation =====
echo ""
echo "================================"
echo "Ready for full Phase 2 experiment"
echo "================================"
echo ""
echo "Estimated GPU time: 2-4 hours"
echo "This will:"
echo "  - Cache ~1M tokens from GPT-2"
echo "  - Train baseline SAE (10K steps)"
echo "  - Train RkCNN SAE (10K steps)"
echo "  - Save checkpoints and results"
echo ""
read -p "Proceed with full experiment? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 0
fi

# ===== FULL RUN =====
echo ""
echo "================================"
echo "Starting Full Phase 2 Experiment"
echo "================================"
echo ""

python experiments/phase2_gpt2.py \
    --config configs/phase2.yaml \
    --output-dir /workspace/results/phase2 \
    --checkpoint-dir /workspace/checkpoints \
    --checkpoint-every 1000 \
    2>&1 | tee /workspace/results/phase2_experiment.log

EXPERIMENT_EXIT=$?

echo ""
echo "================================"
echo "Experiment Complete"
echo "================================"
echo ""
echo "Results saved to: /workspace/results/phase2/"
echo "Checkpoints saved to: /workspace/checkpoints/"
echo ""

if [ $EXPERIMENT_EXIT -eq 0 ]; then
    echo "🎉 Phase 2 PASSED!"
else
    echo "❌ Phase 2 did not meet all criteria."
fi

# List results
echo ""
echo "Result files:"
ls -la /workspace/results/phase2/

exit $EXPERIMENT_EXIT
