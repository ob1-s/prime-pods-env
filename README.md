# DISCLAIMER

> ⚠️ This environment and the scripts used were made with a chronic level of vibe-coding, and its human co-author (me) is quite the noob about infra, bash, security and whatnot. Use at your own risk. You should review and own the code before considering using it for your own means.

This code provides a reference boilerplate of [`verifiers`](https://github.com/willccbb/verifiers) environment with auto-managed GPU pods to use for a boilerplate `modded-nanogpt` speedrun environment.

The included [boilerplate](https://github.com/ob1-s/prime-pods-env/tree/main/environments/boilerplate) environment acts as a reference on how to build an environment with the provisioning scripts.

### Usage

First, make sure you have [`prime-cli`](https://github.com/PrimeIntellect-ai/prime-cli) installed (`uv tool install prime`) and your [Prime Intellect](https://www.primeintellect.ai/) account configured (with ssh keys). Make sure your main ssh key doesn't have a passphrase, otherwise the provision script will not work.

1. Setup venv with uv

```bash
uv venv && source .venv/bin/activate
```

2. Install boilerplate env on interactive mode

```bash
uv pip install -e environments/boilerplate 
```

3. Have fun

```bash
uv run vf-eval boilerplate -n 1 -r 2 -a '{"num_pods": 2, "gpu_type": "H100_80GB", "socket_type": "SXM5", "on_demand": "all"}' -m "dummy-model" -b "http://localhost:9999/v1" -k "DUMMY_KEY" -s && vf-tui
```

The command above will provision and setup two pods of 8xH100 to run the async rollouts during the eval. The boilerplate env will wait for both pods to be ready, then send the packaged [`train_gpt.py`](https://github.com/ob1-s/prime-pods-env/blob/main/environments/boilerplate/train_gpt.py) script for training at each rollout step to an available provisioned pod.

The log is verbose, it'll show all the setup/training process for the first acquired pod (offset 0), and any critical messages from the others.

## TODOs

- [ ] make H100 the default
- [ ] terminate all pods if it fails to provision `min_pods`
- [ ] test with `-n` > 1
- [ ] add dependencies such as prime-cli

## Acknowledgement

### This repo would not exist if it wasn't for Prime Intellect infra and their generosity of giving me GPU credits (H100s go brr).

If you use this repo or it's of any help to you, consider mentioning it or my X account [@LatentLich](https://x.com/LatentLich/) (i'd also love to know it was helpful to you, so don't hesitate reaching me out there!)
