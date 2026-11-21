# RAPPORT pipeline entry points. Every stage except `probe` delegates to
# `python -m scripts.e2e --stage <name>` (data = acquire+clean+adapt).
# Use `make e2e` for the whole pipeline; `make e2e config=...` selects a config.
#
# `config=<path>` passes --config through, e.g.:
#   make train config=configs/experiments/starter.yaml
PYTHON ?= $(shell command -v python 2>/dev/null || command -v python3)
config ?=
STAGE_ARGS = $(if $(config),--config $(config),)

.PHONY: probe data split train calibrate report export bench e2e

# Inline probe, no extra module: re-verifies the runtime facts the T4 run
# depends on (capability, VRAM, SDPA backends, torch/CUDA builds), prints
# them, and records the resolved values in artifacts/run-manifest.json.
# Degrades gracefully to CPU when CUDA is absent (CI, local dev).
probe:
	@$(PYTHON) -c "import json, os, platform, datetime; import torch; cuda = torch.cuda.is_available(); p = torch.cuda.get_device_properties(0) if cuda else None; cap = torch.cuda.get_device_capability(0) if cuda else None; q = torch.backends.cuda; pick = lambda n, old: getattr(q, n, None) or getattr(q, old, None); sdpa = {k: (bool(f()) if f else None) for k, f in {'flash': pick('flash_sdp_enabled', 'sdp_flash_attention_enabled'), 'mem_efficient': pick('mem_efficient_sdp_enabled', 'sdp_mem_efficient_attention_enabled'), 'math': pick('math_sdp_enabled', 'sdp_math_attention_enabled'), 'cudnn': getattr(q, 'cudnn_sdp_enabled', None)}.items()}; print('cuda_available:', cuda); print('device_name:', torch.cuda.get_device_name(0) if cuda else 'cpu'); print('compute_capability:', cap); print('total_vram_gb:', round(p.total_memory / 1024**3, 2) if p else 'n/a'); [print('sdpa_%s_attention:' % k, v) for k, v in sdpa.items()]; print('torch_version:', torch.__version__); print('cuda_version:', torch.version.cuda); os.makedirs('artifacts', exist_ok=True); json.dump({'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cuda_available': cuda, 'device_name': torch.cuda.get_device_name(0) if cuda else None, 'compute_capability': list(cap) if cap else None, 'total_vram_bytes': p.total_memory if p else None, 'total_vram_gb': round(p.total_memory / 1024**3, 2) if p else None, 'sdpa_backends': sdpa, 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'python_version': platform.python_version()}, open('artifacts/run-manifest.json', 'w'), indent=2); print('run manifest written: artifacts/run-manifest.json')"

data:
	@$(PYTHON) -m scripts.e2e --stage data $(STAGE_ARGS)

split:
	@$(PYTHON) -m scripts.e2e --stage split $(STAGE_ARGS)

train:
	@$(PYTHON) -m scripts.e2e --stage train $(STAGE_ARGS)

calibrate:
	@$(PYTHON) -m scripts.e2e --stage calibrate $(STAGE_ARGS)

report:
	@$(PYTHON) -m scripts.e2e --stage report $(STAGE_ARGS)

export:
	@$(PYTHON) -m scripts.e2e --stage export $(STAGE_ARGS)

bench:
	@$(PYTHON) -m scripts.e2e --stage bench $(STAGE_ARGS)

e2e:
	@$(PYTHON) -m scripts.e2e --stage e2e $(STAGE_ARGS)
