import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from typer.testing import CliRunner

from paperbanana.core.image_batch import ImageBatch, fingerprint
from paperbanana.core.types import DiagramType
from paperbanana.providers.image_gen.openai_imagen import ImageGenerationError


def test_batch_receipts_reuse_only_current_verified_images(tmp_path):
    journal = ImageBatch(tmp_path)
    image = tmp_path / "01.png"
    Image.new("RGB", (8, 8)).save(image)
    digest = fingerprint({"prompt": "A", "model": "flare"})
    journal.record("01", digest, "complete", image)
    assert ImageBatch(tmp_path).decision("01", digest, image) == "reuse"
    assert journal.decision("01", fingerprint({"prompt": "B"}), image) == "run"
    image.write_bytes(b"broken")
    assert journal.decision("01", digest, image) == "run"
    journal.record("01", digest, "running")
    assert ImageBatch(tmp_path).decision("01", digest, image) == "result_unknown"
    assert journal.decision("01", digest, image, retry_unknown=True) == "run"


async def test_visualizer_and_polish_send_reference_image(tmp_path):
    from paperbanana.agents.polish import PolishAgent
    from paperbanana.agents.visualizer import VisualizerAgent

    original = Image.new("RGB", (32, 32), "blue")
    generated = Image.new("RGBA", (32, 32))
    generated.info["paperbanana_generation"] = {
        "requested_model": "gpt-image-2.5-sunburst",
        "review_status": "UNREVIEWED",
    }
    provider = SimpleNamespace(
        supports_edit=True, edit=AsyncMock(return_value=generated), name="test"
    )
    vlm = SimpleNamespace(generate=AsyncMock(return_value="Use thicker arrows"))
    visualizer = VisualizerAgent(provider, vlm, output_dir=str(tmp_path))
    path = await visualizer.run("Change only arrows", reference_images=[original])
    assert provider.edit.call_args.kwargs["images"][0] is original
    assert (
        json.loads(Path(path).with_suffix(".image.json").read_text())["review_status"]
        == "UNREVIEWED"
    )
    polish = PolishAgent(vlm, provider, output_dir=str(tmp_path))
    await polish._polish_image(original, "Thicker arrows", DiagramType.METHODOLOGY, None)
    assert provider.edit.call_args.kwargs["images"][0] is original
    provider.supports_edit = False
    provider.generate = AsyncMock(return_value=generated)
    await polish._polish_image(original, "Thicker arrows", DiagramType.METHODOLOGY, None)
    assert provider.generate.await_count == 1


def test_slide_batch_keeps_successes_and_does_not_repeat_unknown(tmp_path, monkeypatch):
    from paperbanana.cli import app
    from paperbanana.core.config import Settings
    from paperbanana.core.pipeline import PaperBananaPipeline

    prompts, out = tmp_path / "prompts", tmp_path / "out"
    prompts.mkdir()
    (prompts / "01-good.md").write_text("good")
    (prompts / "02-unknown.md").write_text("unknown")
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    settings = Settings(
        _env_file=None, image_provider="openai_imagen", openai_image_model="gpt-image-2.5-flare"
    )
    monkeypatch.setattr("paperbanana.cli.Settings", lambda **kwargs: settings)
    calls = []

    async def generate(self, request):
        calls.append(request.source_context)
        if request.source_context == "unknown":
            raise ImageGenerationError("result_unknown")
        return SimpleNamespace(image_path=str(source), iterations=[])

    monkeypatch.setattr(PaperBananaPipeline, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(PaperBananaPipeline, "generate", generate)
    args = [
        "slide-batch",
        "--prompts-dir",
        str(prompts),
        "--output-dir",
        str(out),
        "--concurrent",
        "1",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 1, first.output
    assert calls == ["good", "unknown"]
    assert (out / "01-good.png").exists()
    calls.clear()
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 1
    assert calls == []
    third = CliRunner().invoke(app, args + ["--retry-unknown"])
    assert third.exit_code == 1
    assert calls == ["unknown"]


def test_gemini_batch_retains_legacy_retry_path_without_openai_journal(tmp_path, monkeypatch):
    from paperbanana.cli import app
    from paperbanana.core.config import Settings
    from paperbanana.core.pipeline import PaperBananaPipeline

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "01.md").write_text("Gemini task")
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    settings = Settings(
        _env_file=None, image_provider="google_imagen", google_image_model="gemini-test"
    )
    monkeypatch.setattr("paperbanana.cli.Settings", lambda **kwargs: settings)
    monkeypatch.setattr(PaperBananaPipeline, "__init__", lambda *a, **kw: None)
    generate = AsyncMock(
        side_effect=[
            RuntimeError("synthetic transient"),
            RuntimeError("synthetic transient"),
            SimpleNamespace(image_path=str(source), iterations=[]),
        ]
    )
    monkeypatch.setattr(PaperBananaPipeline, "generate", generate)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app, ["slide-batch", "--prompts-dir", str(prompts), "--output-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert generate.await_count == 3
    assert not (out / "image-batch.json").exists()
    assert (out / "01.png").exists()


def test_gemini_provider_and_env_precedence_unchanged(monkeypatch):
    from paperbanana.core.config import Settings
    from paperbanana.providers.image_gen.google_imagen import GoogleImagenGen
    from paperbanana.providers.registry import ProviderRegistry

    monkeypatch.setenv("GOOGLE_IMAGE_MODEL", "gemini-existing-choice")
    settings = Settings(
        _env_file=None,
        image_provider="google_imagen",
        image_model="generic-choice",
        google_api_key="synthetic-not-a-key",
    )
    assert settings.effective_image_model == "gemini-existing-choice"
    original_init = GoogleImagenGen.__init__
    calls = []

    def capture(self, *args, **kwargs):
        calls.append(kwargs)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(GoogleImagenGen, "__init__", capture)
    provider = ProviderRegistry.create_image_gen(settings)
    assert isinstance(provider, GoogleImagenGen)
    assert provider.model_name == "generic-choice"  # Existing Google factory uses image_model.
    assert not {"quality", "size", "background", "output_format"} & calls[0].keys()


@pytest.mark.parametrize(
    "value",
    [
        "not json",
        "{}",
        "[]",
        '{"critic_suggestions":"ok"}',
        '{"critic_suggestions":[],"score":NaN}',
    ],
)
def test_openai_strict_critic_cannot_approve_malformed_output(value):
    from paperbanana.agents.critic import CriticAgent

    critic = CriticAgent(SimpleNamespace(), strict_response=True)
    with pytest.raises(ValueError, match="UNREVIEWED"):
        critic._parse_response(value)


def test_legacy_critic_default_and_valid_openai_review():
    from paperbanana.agents.critic import CriticAgent

    legacy = CriticAgent(SimpleNamespace())
    assert legacy._parse_response("not json").critic_suggestions == []
    strict = CriticAgent(SimpleNamespace(), strict_response=True)
    assert strict._parse_response('{"critic_suggestions":[],"score":9}').score == 9


async def test_unknown_later_openai_iteration_is_not_hidden_by_previous_candidate(tmp_path):
    from paperbanana.core.config import Settings
    from paperbanana.core.pipeline import PaperBananaPipeline
    from paperbanana.core.types import CritiqueResult, GenerationInput

    pipeline = object.__new__(PaperBananaPipeline)
    pipeline.settings = Settings(
        _env_file=None,
        image_provider="openai_imagen",
        max_critic_rounds=2,
        save_iterations=False,
        critic_score_threshold=0,
    )
    pipeline.visualizer = SimpleNamespace(
        run=AsyncMock(side_effect=["candidate.png", ImageGenerationError("result_unknown")])
    )
    pipeline.critic = SimpleNamespace(
        run=AsyncMock(
            return_value=CritiqueResult(
                critic_suggestions=["Fix title"], revised_description="Updated title", score=6
            )
        )
    )
    request = GenerationInput(
        source_context="source", communicative_intent="intent", diagram_type=DiagramType.SLIDE
    )
    with pytest.raises(ImageGenerationError) as error:
        await pipeline._critic_iteration_loop(request, "original title")
    assert error.value.code == "result_unknown"
    assert pipeline.visualizer.run.await_count == 2
    assert pipeline.critic.run.await_count == 1
