#!/bin/bash
# Add missing paper-suite metrics to a completed result using its final model.
# This must only be used when last.ckpt is known to contain the final stage.

set -euo pipefail
cd "${FOMO_REPO_DIR:-$PWD}"
. jobs/rebuttal/load_cluster_environment.sh
. jobs/rebuttal/full_metric_suite.sh

experiment_name="${SOURCE_EXPERIMENT:?SOURCE_EXPERIMENT is required}"
dataset_config="${SOURCE_DATASET_CONFIG:-imagenet100_imbalanced}"
dataset_dir="${SOURCE_DATASET_DIR:-clane9_imagenet-100}"
model_config="${SOURCE_MODEL_CONFIG:-resnet50}"
ssl_config="${SOURCE_SSL_CONFIG:-simclr}"
seed="${SOURCE_SEED:?SOURCE_SEED is required}"
seed_dir="$CHECKPOINT_ROOT_DIR/$experiment_name/$dataset_dir/seed_$seed"
result="$seed_dir/result.json"
checkpoint="$seed_dir/last.ckpt"

test -s "$result"
test -s "$checkpoint"

if fomo_has_full_metric_suite "$result"; then
  echo "Full metric suite already exists at $result; skipping."
  exit 0
fi

declare -a benchmarks=()
add_if_missing() {
  local metric="$1"
  local benchmark="$2"
  if ! jq -e --arg metric "$metric" '.[$metric] | numbers' "$result" >/dev/null; then
    benchmarks+=("$benchmark")
  fi
}

add_if_missing cars_test_accuracy CarsFineTune
add_if_missing carsknn_knn_test_accuracy CarsKNNClassifier
add_if_missing aircraft_test_accuracy AircraftFineTune
add_if_missing aircraftknn_knn_test_accuracy AircraftKNNClassifier
add_if_missing flowers_test_accuracy FlowersFineTune
add_if_missing flowersknn_knn_test_accuracy FlowersKNNClassifier
add_if_missing pets_test_accuracy PetsFineTune
add_if_missing petsknn_knn_test_accuracy PetsKNNClassifier
add_if_missing cifar10r_test_accuracy CIFAR10FineTuner
add_if_missing cifar10knn_knn_test_accuracy CIFAR10KNNClassifier
add_if_missing cifar100r_test_accuracy CIFAR100FineTuner
add_if_missing cifar100knn_knn_test_accuracy CIFAR100KNNClassifier
add_if_missing imagenet100lt_test_accuracy ImageNet100LTFineTune
add_if_missing imagenet100ltknn_knn_test_accuracy ImageNet100LTKNNClassifier

benchmark_override="[$(IFS=,; echo "${benchmarks[*]}")]"
echo "Backfilling ${#benchmarks[@]} metrics for $experiment_name seed $seed"

python -m experiment \
  dataset="$dataset_config" model="$model_config" ssl="$ssl_config" \
  logger=false pretrain=false finetune=true \
  finetune_benchmarks="$benchmark_override" \
  result_benchmark_contract=paper_full merge_existing_result=true \
  num_runs=1 seed="$seed" checkpoint="$checkpoint" \
  experiment_name="$experiment_name"
