# Image 2.5 integration (opt-in)

This fork is an **optional backend** for [paperbanana-skill](https://github.com/PlutoLei/paperbanana-skill).
Codex native generation and the skill's editable PPTX renderer can work without it. The core
provides exact API controls and the retrieval/planning/Critic pipeline when those are requested.
Gemini routing, provider implementation, authentication, defaults, Polish regeneration and its
existing batch retry path are unchanged by this addition. No environment files are migrated.

## API entry without the full pipeline

Use a checkout containing this change and install its OpenAI extra in your chosen environment:
`python -m pip install -e '.[openai]'`. The tested SDK contract requires `openai>=2.30.0`.
This optional dependency change does not install or change the Gemini integration.

```bash
python -m paperbanana.cli image --input prompt.md --output candidate.png \
  --model gpt-image-2.5-flare --quality high --size 1536x864 --dry-run

python -m paperbanana.cli image --input edit.md --output edited.png \
  --model gpt-image-2.5-sunburst --quality xhigh --size 1536x864 \
  --reference source.png --mask mask.png --dry-run
```

`--dry-run` checks input files and controls without loading Settings/.env, constructing an API
client or generating images. Remove it only for authorized live execution with configured
`OPENAI_API_KEY` and, where applicable, `OPENAI_BASE_URL`. No API key is needed for the dry run.
The CLI is OpenAI-specific; Gemini requests continue through their existing commands.

Repeat `--reference` for composition. Editing sends real reference image bytes to `images.edit`;
`PolishAgent` uses this path for supported OpenAI image models. Masks need alpha and must match
the first reference's dimensions. Inputs are encoded as PNG, each under 50 MB. A new image has
no reference fields. Output extension selects PNG, JPEG or WebP; transparency requires PNG/WebP.
The native Codex tool has a different schema and does not expose these exact model controls.

Aliases: `gpt-image-2.5-sunburst`, `gpt-image-2.5-flare`; dated versions end in `-2026-09-08`.
Default `OpenAIImageGen` remains `gpt-image-2`; project provider defaults remain unchanged.
The new tiers `xhigh` and `max` are validated only for Image 2.5. Recognizing a model name does
not establish account access or validate the advertised quality/speed in this environment.

Sizes are `auto` or WIDTHxHEIGHT with both edges divisible by 16, edges <=3840, ratio 1:3..3:1,
and area 655360..8294400 pixels. Above 2560×1440 is experimental according to the official guide.
Precedence: call size > configured size > aspect ratio > dimensions. An explicit OpenAI image
model in Settings/CLI/YAML overrides an ambient `OPENAI_IMAGE_MODEL` default; an explicit
provider-specific model argument still takes precedence. Gemini's configuration rule is unchanged.

## Pipeline profiles

`configs/image25-sunburst.yaml` and `configs/image25-flare.yaml` are opt-in image profiles.
Commands which expose `--config` can load them; existing VLM access is still needed for a pipeline:

```bash
python -m paperbanana.cli slide --input slide.md --config configs/image25-sunburst.yaml
```

For `slide-batch`, select `--image-provider openai_imagen --image-model gpt-image-2.5-flare`.
It retains the existing batch flags; `IMAGE_QUALITY`, `IMAGE_SIZE` and `IMAGE_BACKGROUND` can
configure OpenAI controls. Do not pass a `--config` flag to a build which does not expose it.

## Recovery and response

The OpenAI provider owns retries: SDK retries off, at most three attempts by default, backoff for
429/5xx only. Quota/moderation/authentication/permission/invalid request failures stop. Cancellation
propagates. A timeout or connection failure becomes `result_unknown` and is not blindly retried;
the service may have accepted the request. Output decoding/shape/format failures are reported
without regenerating. No automatic provider or model switch is introduced.

Only OpenAI `slide-batch` uses `image-batch.json`. A matching request fingerprint plus verified
output hash and decode allows reuse. Changed prompts/recorded settings and corrupt outputs rebuild. Start a new output directory
after changing pipeline code, prompt templates or reference-library contents.
Running/ambiguous attempts remain skipped until an explicit `--retry-unknown` decision. One
process owns each batch directory; the receipt is not a distributed lock. Failures yield a
nonzero OpenAI batch exit code and preserve successful images. Gemini keeps its legacy behavior.

Generation returns safe metadata: requested model, reported model or null, request ID or null,
operation, attempts, elapsed time, dimensions, quality and `UNREVIEWED`. `save_image` preserves
this as an `.image.json` sidecar when available, and the direct CLI writes one beside its output.
Sidecars describe generation, not the final pipeline Critic verdict. Report model identity as
unreported when missing; cost as unknown without billing evidence. OpenAI pipeline Critic parsing rejects malformed JSON/scores instead of treating them as
acceptance. Gemini keeps its existing parsing behavior. Critic failure must not become review
acceptance. Inspect returned images before claiming their semantics match the request.

## Evaluation and optimization

See [the fixed evaluation protocol](../evaluations/image25/README.md). Engineering checks use
synthetic SDK responses, including a real SDK with an HTTP mock transport. They prove request
serialization, validation and recovery behavior, not provider access, real latency, cost or image
quality. The new default route/profile is not selected by synthetic test scores.

Sources checked 2026-09-09: [Sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst),
[Flare](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare),
[Image API](https://developers.openai.com/api/docs/guides/image-generation),
[Codex image generation](https://learn.chatgpt.com/docs/image-generation).
