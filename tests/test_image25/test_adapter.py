import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from typer.testing import CliRunner

from paperbanana.providers.image_gen.openai_imagen import (
    IMAGE25,
    ImageGenerationError,
    OpenAIImageGen,
    _is_gpt_image_2,
    validate_size,
)


def response(size=(1024, 1024), mode="RGB", format="PNG"):
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, format=format)
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(buffer.getvalue()).decode())],
        _request_id="synthetic-request",
        model=None,
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in [
        "OPENAI_IMAGE_MODEL",
        "IMAGE_MODEL",
        "IMAGE_PROVIDER",
        "IMAGE_SIZE",
        "IMAGE_QUALITY",
        "IMAGE_BACKGROUND",
    ]:
        monkeypatch.delenv(key, raising=False)
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("No network in image adapter tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())


def provider(model="gpt-image-2.5-sunburst", result=None, **kwargs):
    gen = OpenAIImageGen(model=model, **kwargs)
    gen._client = SimpleNamespace(
        images=SimpleNamespace(
            generate=AsyncMock(return_value=result or response()),
            edit=AsyncMock(return_value=result or response()),
        )
    )
    return gen


@pytest.mark.parametrize(
    "model", ["gpt-image-2", "gpt-image-2-2026-04-21", *IMAGE25, "gpt-image-2.5-flare-2026-09-08"]
)
def test_model_families(model):
    assert _is_gpt_image_2(model)
    assert len(OpenAIImageGen(model=model).supported_ratios) == 8


@pytest.mark.parametrize("size", ["1536x864", "1024x1024", "3840x2160", "auto"])
def test_valid_sizes(size):
    assert validate_size(size) == size


@pytest.mark.parametrize(
    "size", ["1365x1024", "4096x2048", "128x128", "3840x3840", "2048x512", "0x1024", "wrong"]
)
def test_invalid_sizes(size):
    with pytest.raises(ValueError):
        OpenAIImageGen(model="gpt-image-2.5-flare", size=size)


@pytest.mark.parametrize("model", sorted(IMAGE25))
@pytest.mark.parametrize("quality", ["low", "medium", "high", "xhigh", "max", "auto"])
async def test_quality_reaches_request(model, quality):
    gen = provider(model, quality=quality)
    image = await gen.generate("Diagram")
    assert gen._client.images.generate.call_args.kwargs["quality"] == quality
    assert image.info["paperbanana_generation"]["requested_model"] == model
    assert image.info["paperbanana_generation"]["review_status"] == "UNREVIEWED"


@pytest.mark.parametrize("model", ["gpt-image-2", "gpt-image-1.5", "deployment-name"])
@pytest.mark.parametrize("quality", ["xhigh", "max"])
def test_older_models_reject_new_quality(model, quality):
    with pytest.raises(ValueError, match="quality"):
        OpenAIImageGen(model=model, quality=quality)


async def test_size_precedence_and_ratio():
    gen = provider(size="1024x1024", result=response((1536, 864)))
    await gen.generate("Diagram", size="1536x864", width=3840, height=2160, aspect_ratio="1:1")
    assert gen._client.images.generate.call_args.kwargs["size"] == "1536x864"
    gen._client.images.generate.return_value = response()
    await gen.generate("Diagram", aspect_ratio="16:9")
    assert gen._client.images.generate.call_args.kwargs["size"] == "1024x1024"
    gen.size = None
    gen._client.images.generate.return_value = response((1536, 1152))
    await gen.generate("Diagram", aspect_ratio="4:3")
    assert gen._client.images.generate.call_args.kwargs["size"] == "1536x1152"


async def test_references_and_mask_are_sent_to_edit():
    gen = provider()
    source = Image.new("RGB", (48, 48), "blue")
    style = Image.new("RGB", (32, 32), "red")
    mask = Image.new("RGBA", source.size, (0, 0, 0, 0))
    await gen.edit("Replace only the icon", [source, style], mask)
    gen._client.images.generate.assert_not_called()
    kwargs = gen._client.images.edit.call_args.kwargs
    assert len(kwargs["image"]) == 2
    assert Image.open(BytesIO(kwargs["image"][0][1])).getpixel((0, 0)) == (0, 0, 255)
    assert Image.open(BytesIO(kwargs["mask"][1])).mode == "RGBA"
    for bad in [Image.new("RGB", source.size), Image.new("RGBA", (2, 2))]:
        with pytest.raises(ValueError, match="Mask"):
            await gen.edit("Edit", [source], bad)
    assert gen._client.images.edit.await_count == 1


@pytest.mark.parametrize(
    "bad",
    [
        SimpleNamespace(data=[]),
        SimpleNamespace(data=[SimpleNamespace(b64_json="!bad!")]),
        response((32, 32)),
    ],
)
async def test_invalid_output_not_retried(bad):
    gen = provider(result=bad)
    with pytest.raises(ImageGenerationError, match="invalid_image_output"):
        await gen.generate("Diagram")
    assert gen._client.images.generate.await_count == 1


async def test_transparency_and_format_checks():
    gen = provider(background="transparent", result=response(mode="RGBA"))
    assert (await gen.generate("Icon")).mode == "RGBA"
    gen._client.images.generate.return_value = response(mode="RGB")
    with pytest.raises(ImageGenerationError):
        await gen.generate("Icon")
    with pytest.raises(ValueError):
        provider(background="transparent", output_format="jpeg")
    with pytest.raises(ValueError):
        provider("gpt-image-2", background="transparent")


@pytest.mark.parametrize(
    "status,code,attempts",
    [
        (401, None, 1),
        (403, None, 1),
        (400, "moderation_blocked", 1),
        (429, "insufficient_quota", 1),
        (429, None, 3),
        (503, None, 3),
    ],
)
async def test_failures_have_one_retry_owner(status, code, attempts):
    error = RuntimeError("private response must not leak")
    error.status_code, error.code = status, code
    gen = provider()
    gen._client.images.generate.side_effect = error
    with pytest.raises(ImageGenerationError) as caught:
        await gen.generate("Diagram")
    assert gen._client.images.generate.await_count == attempts
    assert caught.value.retryable is False
    assert "private" not in str(caught.value)


async def test_transient_error_recovers_without_switching_model():
    gen = provider()
    error = RuntimeError("unavailable")
    error.status_code = 503
    gen._client.images.generate.side_effect = [error, response()]
    image = await gen.generate("Diagram")
    assert image.info["paperbanana_generation"]["attempts"] == 2
    assert gen._client.images.generate.call_args.kwargs["model"] == gen.model_name


@pytest.mark.parametrize("error", [TimeoutError(), ConnectionError(), asyncio.CancelledError()])
async def test_unknown_results_and_cancellation_are_not_resent(error):
    gen = provider()
    gen._client.images.generate.side_effect = error
    expected = (
        asyncio.CancelledError
        if isinstance(error, asyncio.CancelledError)
        else ImageGenerationError
    )
    with pytest.raises(expected):
        await gen.generate("Diagram")
    assert gen._client.images.generate.await_count == 1


def test_sdk_retries_disabled(monkeypatch):
    import openai

    captured = {}
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: captured.update(kwargs) or object())
    OpenAIImageGen(api_key="synthetic")._get_client()
    assert captured["max_retries"] == 0


def test_config_registry_and_cli_dry_run(tmp_path, monkeypatch):
    from paperbanana.cli import app
    from paperbanana.core.config import Settings
    from paperbanana.providers.registry import ProviderRegistry

    settings = Settings(
        _env_file=None,
        openai_api_key="synthetic",
        image_provider="openai_imagen",
        image_model="gpt-image-2.5-flare",
        image_quality="max",
        image_size="1536x864",
    )
    gen = ProviderRegistry.create_image_gen(settings)
    assert (gen.quality, gen.size, gen.model_name) == ("max", "1536x864", "gpt-image-2.5-flare")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("A clean flowchart")
    monkeypatch.setattr(
        OpenAIImageGen, "_get_client", lambda _: pytest.fail("No client in dry run")
    )
    monkeypatch.setattr(
        Settings, "__init__", lambda *a, **k: pytest.fail("No settings/credentials in dry run")
    )
    result = CliRunner().invoke(
        app,
        [
            "image",
            "--input",
            str(prompt),
            "--model",
            gen.model_name,
            "--quality",
            "xhigh",
            "--size",
            "1536x864",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "validated-offline"


@pytest.mark.parametrize("name", ["sunburst", "flare"])
def test_profile_beats_ambient_model_default(monkeypatch, name):
    from paperbanana.core.config import Settings

    monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
    settings = Settings.from_yaml(f"configs/image25-{name}.yaml", _env_file=None)
    assert settings.effective_image_model == f"gpt-image-2.5-{name}"
    settings = Settings.from_yaml(
        f"configs/image25-{name}.yaml", image_model="gpt-image-2", _env_file=None
    )
    assert settings.effective_image_model == "gpt-image-2"
    assert (
        Settings(image_provider="openai_imagen", _env_file=None).effective_image_model
        == "gpt-image-2"
    )


async def test_real_sdk_serializes_new_quality_offline():
    import httpx
    from openai import AsyncOpenAI

    sent = []

    async def respond(request):
        sent.append(json.loads(request.content))
        value = response()
        return httpx.Response(
            200,
            json={"data": [{"b64_json": value.data[0].b64_json}]},
            headers={"x-request-id": "synthetic-http"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    gen = OpenAIImageGen(model="gpt-image-2.5-flare", quality="xhigh")
    gen._client = AsyncOpenAI(api_key="synthetic-not-a-key", http_client=client, max_retries=0)
    try:
        image = await gen.generate("A diagram")
        assert sent[0]["quality"] == "xhigh"
        assert sent[0]["model"] == gen.model_name
        assert image.info["paperbanana_generation"]["request_id"] == "synthetic-http"
    finally:
        await gen._client.close()


async def test_real_sdk_encodes_reference_and_mask_as_multipart_offline():
    import httpx
    from openai import AsyncOpenAI

    requests = []

    async def respond(request):
        requests.append((request.headers["content-type"], request.content))
        value = response()
        return httpx.Response(200, json={"data": [{"b64_json": value.data[0].b64_json}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    gen = OpenAIImageGen(model="gpt-image-2.5-sunburst", quality="xhigh")
    gen._client = AsyncOpenAI(api_key="synthetic-not-a-key", http_client=client, max_retries=0)
    try:
        await gen.edit("Change title", [Image.new("RGB", (64, 64))], Image.new("RGBA", (64, 64)))
        content_type, body = requests[0]
        assert "multipart/form-data" in content_type
        assert b'filename="reference-0.png"' in body
        assert b'filename="mask.png"' in body
        assert b"xhigh" in body and b"gpt-image-2.5-sunburst" in body
    finally:
        await gen._client.close()


@pytest.mark.parametrize("format", ["JPEG", "WEBP"])
async def test_output_formats_are_decoded_and_verified(format):
    gen = provider(result=response(format=format), output_format=format.lower())
    image = await gen.generate("A diagram")
    assert image.format == format
