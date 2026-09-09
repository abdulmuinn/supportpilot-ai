# Model Artifact Setup

SupportPilot AI keeps the trained model artifact separate from the source-code repository.

This keeps the Git repository lightweight and allows the inference layer to load either:

- a local Hugging Face-compatible model directory
- a remote Hugging Face model repository ID

## Local Model Directory

A compatible local model directory typically contains files such as:

```text
model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Additional training artifacts may also be present, but the runtime model source must be loadable through Hugging Face `from_pretrained()`.

## Local Development

Set the model source before starting the API:

```bash
export SUPPORTPILOT_MODEL_ID=/absolute/path/to/model
```

Then run:

```bash
python -m uvicorn supportpilot.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
```

Expected structure:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "num_labels": 46
}
```

## Docker Compose

Docker Compose mounts the trained model into the API container as read-only storage.

Create the local environment file:

```bash
cp .env.example .env
```

Configure:

```text
SUPPORTPILOT_MODEL_HOST_PATH=/absolute/path/to/model
```

Then start the stack:

```bash
docker compose up -d --build
```

Inside the API container, the model is exposed at:

```text
/models/supportpilot
```

and configured through:

```text
SUPPORTPILOT_MODEL_ID=/models/supportpilot
```

## Remote Model Source

The released SupportPilot model is publicly available on Hugging Face:

`abdulmuinnn/supportpilot-distilbert`

For local development, set:

`SUPPORTPILOT_MODEL_ID=abdulmuinnn/supportpilot-distilbert`

The inference loader can download and cache the model automatically through Hugging Face Transformers.

The same model can also be loaded directly with `AutoTokenizer.from_pretrained()` and `AutoModelForSequenceClassification.from_pretrained()` using the model ID above.

The current Docker Compose configuration still uses a local read-only bind mount for the model artifact.

## Why the Model Is Not Stored in Git

The trained model is intentionally excluded from the repository because model binaries are large deployment artifacts rather than application source code.

Separating model artifacts from source code provides:

- smaller Git history
- faster cloning
- independent model versioning
- easier deployment configuration
- cleaner separation between application code and ML artifacts
