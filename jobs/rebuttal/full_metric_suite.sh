#!/bin/bash

# Canonical result contract for paper-facing experiments. Keep this in sync
# with FinetuningBenchmarks.benchmark_suites["paper_full"].
fomo_has_full_metric_suite() {
  local result_file="$1"
  [[ -s "$result_file" ]] && jq -e '
    [
      .cars_test_accuracy,
      .carsknn_knn_test_accuracy,
      .aircraft_test_accuracy,
      .aircraftknn_knn_test_accuracy,
      .flowers_test_accuracy,
      .flowersknn_knn_test_accuracy,
      .pets_test_accuracy,
      .petsknn_knn_test_accuracy,
      .cifar10r_test_accuracy,
      .cifar10knn_knn_test_accuracy,
      .cifar100r_test_accuracy,
      .cifar100knn_knn_test_accuracy,
      .imagenet100lt_test_accuracy,
      .imagenet100ltknn_knn_test_accuracy
    ] | all(.[]; type == "number")
  ' "$result_file" >/dev/null
}
