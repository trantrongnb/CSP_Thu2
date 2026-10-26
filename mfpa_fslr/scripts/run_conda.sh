#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_conda.sh [--env signclip_5090|sign5090|manta] COMMAND [ARGS...]

Commands: check, train, evaluate, episodes
Default command: check. Default environment: signclip_5090.
For WLASL training, pass --dataset-root PATH to specify the raw dataset directory.
Uses the conda interpreter directly and the existing CUDA MPS daemon.
CUDA checks have a 30-second timeout; override with MFPA_CUDA_CHECK_TIMEOUT (seconds).
EOF
}

env_name="${MFPA_CONDA_ENV:-signclip_5090}"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--env" ]]; then
  if [[ $# -lt 2 ]]; then usage >&2; exit 2; fi
  env_name="$2"
  shift 2
fi
case "$env_name" in
  sign5090) env_name="signclip_5090" ;;
  signclip_5090|manta) ;;
  *) echo "Supported environments: signclip_5090, sign5090, manta" >&2; exit 2 ;;
esac
action="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  check|train|evaluate|episodes) ;;
  *) usage >&2; exit 2 ;;
esac

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${MFPA_CONDA_ROOT:-/home/iec/disk2/cuongpd/miniconda3}"
python_path="$conda_root/envs/dung36838/bin/python"
if [[ ! -x "$python_path" ]]; then
  echo "Conda interpreter not found: $python_path" >&2
  exit 1
fi
export PYTHONPATH="$package_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
# This server has one GPU; the existing MPS daemon supplies the client device map.
unset CUDA_VISIBLE_DEVICES
cd -- "$package_dir"

check_seconds="${MFPA_CUDA_CHECK_TIMEOUT:-30}"
if [[ ! "$check_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "MFPA_CUDA_CHECK_TIMEOUT must be a positive integer" >&2
  exit 2
fi

# CUDA can block in native code, so the shell must remain available for Ctrl+C.
# setsid keeps each Python process in a group that can be stopped as a unit.
stop_child() {
  trap - INT TERM
  kill -TERM -- "-$child_pid" 2>/dev/null || true
  sleep 0.2
  kill -KILL -- "-$child_pid" 2>/dev/null || true
  wait "$child_pid" 2>/dev/null || true
  exit "$1"
}
run_supervised() {
  setsid "$@" &
  child_pid=$!
  trap 'stop_child 130' INT
  trap 'stop_child 143' TERM
  local status=0
  wait "$child_pid" || status=$?
  trap - INT TERM
  return "$status"
}
check_cuda() {
  local status=0
  run_supervised timeout --foreground --signal=KILL "${check_seconds}s" \
    "$python_path" "$package_dir/scripts/conda_entry.py" check || status=$?
  # GNU timeout exits 137 when its KILL deadline fires.
  if [[ "$status" -eq 137 ]]; then
    echo "CUDA check timed out after ${check_seconds}s; the MPS connection did not respond." >&2
    status=124
  fi
  return "$status"
}

if [[ "$action" == "check" ]]; then
  check_cuda
  exit $?
fi

if [[ "$action" == "train" || "$action" == "evaluate" ]]; then
  device=cuda
  show_help=false
  args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[i]}" in
      -h|--help) show_help=true ;;
      --device) device="${args[i+1]:-}" ;;
      --device=*) device="${args[i]#--device=}" ;;
    esac
  done
  if [[ "$show_help" == false && "$device" == cuda* ]]; then
    if ! check_cuda; then
      echo "CUDA is not responding; $action was not started." >&2
      exit 1
    fi
  fi
fi

if [[ "$action" == "train" && "${show_help:-false}" == false ]]; then
  config_path=""
  dataset_root=""
  filtered_args=()
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[i]}" in
      --config)
        config_path="${args[i+1]:-}"
        filtered_args+=("${args[i]}" "${args[i+1]:-}")
        ((i+=1))
        ;;
      --config=*)
        config_path="${args[i]#--config=}"
        filtered_args+=("${args[i]}")
        ;;
      --dataset-root)
        if [[ -n "$dataset_root" || -z "${args[i+1]:-}" || "${args[i+1]}" == --* ]]; then
          echo "--dataset-root requires one directory path and may be specified only once." >&2
          exit 2
        fi
        dataset_root="${args[i+1]}"
        ((i+=1))
        ;;
      --dataset-root=*)
        if [[ -n "$dataset_root" || -z "${args[i]#--dataset-root=}" ]]; then
          echo "--dataset-root requires one directory path and may be specified only once." >&2
          exit 2
        fi
        dataset_root="${args[i]#--dataset-root=}"
        ;;
      *)
        filtered_args+=("${args[i]}")
        ;;
    esac
  done
  args=("${filtered_args[@]}")
  if [[ "$(basename -- "$config_path")" == train_wlasl_*.yaml ]]; then
    prepare_args=(--config "$config_path")
    if [[ -n "$dataset_root" ]]; then
      prepare_args+=(--dataset-root "$dataset_root")
    fi
    run_supervised "$python_path" "$package_dir/scripts/prepare_dataset.py" "${prepare_args[@]}"
    prepared_config="$package_dir/outputs/.prepared_$(basename -- "${config_path%.*}").yaml"
    for ((i=0; i<${#args[@]}; i++)); do
      case "${args[i]}" in
        --config) args[i+1]="$prepared_config"; break ;;
        --config=*) args[i]="--config=$prepared_config"; break ;;
      esac
    done
    set -- "${args[@]}"
  elif [[ -n "$dataset_root" ]]; then
    echo "--dataset-root is only supported with configs/train_wlasl_*.yaml." >&2
    exit 2
  fi
fi
run_supervised "$python_path" "$package_dir/scripts/conda_entry.py" "$action" "$@"
