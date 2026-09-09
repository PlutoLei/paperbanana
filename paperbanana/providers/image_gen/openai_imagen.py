"""OpenAI generation/editing with validated controls and one bounded retry owner."""

from __future__ import annotations

import asyncio
import base64
import re
import time
from io import BytesIO
from typing import Optional

import structlog
from PIL import Image

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()
RATIO_SIZES = {
    "1:1": (1024, 1024),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
    "16:9": (1536, 864),
    "9:16": (864, 1536),
    "4:3": (1536, 1152),
    "3:4": (1152, 1536),
    "21:9": (2016, 864),
}
IMAGE25 = {"gpt-image-2.5-sunburst", "gpt-image-2.5-flare"}


def model_family(model: str) -> str:
    """Recognize aliases and dated snapshots without matching lookalike names."""
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model.lower())


def _is_gpt_image_2(model: str) -> bool:
    return model_family(model) in {"gpt-image-2", *IMAGE25}


def validate_size(size: str) -> str:
    if size == "auto":
        return size
    match = re.fullmatch(r"([1-9]\d*)x([1-9]\d*)", size)
    if not match:
        raise ValueError("size must be auto or WIDTHxHEIGHT")
    w, h = map(int, match.groups())
    if (
        w % 16
        or h % 16
        or max(w, h) > 3840
        or not 1 / 3 <= w / h <= 3
        or not 655360 <= w * h <= 8294400
    ):
        raise ValueError(
            "Invalid image size: multiples of 16, ratio 1:3..3:1, "
            "edges <=3840 and 655360..8294400 pixels required"
        )
    return size


class ImageGenerationError(RuntimeError):
    """Safe failure: outer layers must not repeat an exhausted image request."""

    def __init__(self, code: str, *, request_id=None, attempts=1):
        super().__init__(f"OpenAI image request failed: {code}")
        self.code = code
        self.request_id = request_id if isinstance(request_id, str) else None
        self.attempts = attempts
        self.retryable = False


class OpenAIImageGen(ImageGenProvider):
    """The default remains GPT Image 2; a configured model does not prove access."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-image-2",
        base_url: str = "https://api.openai.com/v1",
        quality: str = "auto",
        size: Optional[str] = None,
        background: str = "auto",
        output_format: str = "png",
        max_attempts: int = 3,
    ):
        self._api_key, self._model, self._base_url = api_key, model, base_url
        self.quality, self.size = quality, size
        self.background, self.output_format = background, output_format
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be an integer from 1 to 5")
        self.max_attempts = max_attempts
        self._client = None
        self._options(quality, background, output_format)
        if size is not None:
            self._validate_model_size(size)

    @property
    def name(self) -> str:
        return "openai_imagen"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_edit(self) -> bool:
        return model_family(self._model) in {
            "gpt-image-1",
            "gpt-image-1.5",
            "gpt-image-1-mini",
            "gpt-image-2",
            *IMAGE25,
        }

    @property
    def supported_ratios(self) -> list[str]:
        return list(RATIO_SIZES) if _is_gpt_image_2(self._model) else ["1:1", "3:2", "2:3"]

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key, base_url=self._base_url, max_retries=0, timeout=300.0
            )
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _options(self, quality, background, output_format):
        qualities = {"low", "medium", "high", "auto"}
        if model_family(self._model) in IMAGE25:
            qualities |= {"xhigh", "max"}
        if quality not in qualities:
            raise ValueError(f"Unsupported quality for {self._model}: {quality}")
        if background not in {"auto", "opaque", "transparent"}:
            raise ValueError("background must be auto, opaque or transparent")
        if output_format not in {"png", "jpeg", "webp"}:
            raise ValueError("output_format must be png, jpeg or webp")
        if background == "transparent":
            if model_family(self._model) == "gpt-image-2":
                raise ValueError("GPT Image 2 does not support transparent backgrounds")
            if output_format == "jpeg":
                raise ValueError("Transparent backgrounds require png or webp")
        return {"quality": quality, "background": background, "output_format": output_format}

    def _validate_model_size(self, size):
        if _is_gpt_image_2(self._model):
            return validate_size(size)
        if size not in {"auto", "1024x1024", "1536x1024", "1024x1536"}:
            raise ValueError("This model does not support custom sizes")
        return size

    def _size_string(self, width, height):
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("width and height must be positive integers")
        if _is_gpt_image_2(self._model):
            return validate_size(f"{width}x{height}")
        ratio = width / height
        return "1536x1024" if ratio > 1.2 else "1024x1536" if ratio < 0.83 else "1024x1024"

    def _request(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        aspect_ratio,
        quality,
        size,
        background,
        output_format,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        # Explicit size > configured size > aspect ratio > dimensions.
        requested_size = size if size is not None else self.size
        if requested_size is None:
            if aspect_ratio is not None:
                if aspect_ratio not in RATIO_SIZES:
                    raise ValueError("Unsupported aspect ratio")
                width, height = RATIO_SIZES[aspect_ratio]
            requested_size = self._size_string(width, height)
        options = self._options(
            quality if quality is not None else self.quality,
            background if background is not None else self.background,
            output_format if output_format is not None else self.output_format,
        )
        full_prompt = prompt + (f"\n\nAvoid: {negative_prompt}" if negative_prompt else "")
        return {
            "model": self._model,
            "prompt": full_prompt,
            "n": 1,
            "size": self._validate_model_size(requested_size),
            **options,
        }

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        quality: Optional[str] = None,
        *,
        size=None,
        background=None,
        output_format=None,
    ) -> Image.Image:
        kwargs = self._request(
            prompt,
            negative_prompt,
            width,
            height,
            aspect_ratio,
            quality,
            size,
            background,
            output_format,
        )
        if seed is not None:
            logger.warning("OpenAI image API does not support a deterministic seed")
        return await self._execute("generate", kwargs)

    async def edit(
        self,
        prompt: str,
        images: list[Image.Image],
        mask: Optional[Image.Image] = None,
        width: int = 1024,
        height: int = 1024,
        *,
        size=None,
        quality=None,
        background=None,
        output_format=None,
    ) -> Image.Image:
        kwargs = self.prepare_edit(
            prompt,
            images,
            mask,
            width,
            height,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
        )
        return await self._execute("edit", kwargs)

    def prepare_edit(
        self,
        prompt,
        images,
        mask=None,
        width=1024,
        height=1024,
        *,
        size=None,
        quality=None,
        background=None,
        output_format=None,
    ):
        """Validate and encode an edit without loading credentials or starting a request."""
        if not self.supports_edit:
            raise ValueError("Configured model does not support image editing")
        if not images or any(not isinstance(image, Image.Image) for image in images):
            raise ValueError("Editing requires reference images")
        kwargs = self._request(
            prompt, None, width, height, None, quality, size, background, output_format
        )
        kwargs["image"] = [
            self._image_file(image, f"reference-{i}.png") for i, image in enumerate(images)
        ]
        if mask is not None:
            if mask.size != images[0].size or "A" not in mask.getbands():
                raise ValueError("Mask must match the first image and contain an alpha channel")
            kwargs["mask"] = self._image_file(mask, "mask.png")
        return kwargs

    @staticmethod
    def _image_file(image, name):
        image.load()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        if len(data) >= 50 * 1024 * 1024:
            raise ValueError("Each input image must be smaller than 50 MB")
        return (name, data, "image/png")

    async def _execute(self, operation, kwargs):
        client = self._get_client()
        started = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await getattr(client.images, operation)(**kwargs)
                break
            except Exception as error:
                status, code = getattr(error, "status_code", None), getattr(error, "code", None)
                unknown = isinstance(error, (TimeoutError, ConnectionError)) or type(
                    error
                ).__name__ in {"APITimeoutError", "APIConnectionError"}
                terminal = {
                    "insufficient_quota",
                    "moderation_blocked",
                    "billing_hard_limit_reached",
                }
                transient = status == 429 or isinstance(status, int) and 500 <= status < 600
                if unknown or not transient or code in terminal or attempt == self.max_attempts:
                    safe_code = (
                        "result_unknown"
                        if unknown
                        else {
                            400: "invalid_request",
                            401: "authentication",
                            403: "permission",
                            429: "rate_limit",
                        }.get(status, "provider_error")
                    )
                    if code in terminal:
                        safe_code = code
                    raise ImageGenerationError(
                        safe_code, request_id=getattr(error, "request_id", None), attempts=attempt
                    ) from None
                delay = min(2**attempt, 60)
                headers = getattr(getattr(error, "response", None), "headers", {})
                try:
                    delay = min(60, max(delay, float(headers.get("retry-after", 0))))
                except (TypeError, ValueError):
                    pass
                logger.warning("Retrying transient image request", status=status, attempt=attempt)
                await asyncio.sleep(delay)

        request_id = getattr(result, "_request_id", None)
        try:
            if not result.data or len(result.data) != 1:
                raise ValueError("Expected one image")
            raw = base64.b64decode(result.data[0].b64_json, validate=True)
            image = Image.open(BytesIO(raw))
            image.load()
            if _is_gpt_image_2(self._model) and kwargs["size"] != "auto":
                if image.size != tuple(map(int, kwargs["size"].split("x"))):
                    raise ValueError("Unexpected output dimensions")
            if image.format.lower() != kwargs["output_format"]:
                raise ValueError("Unexpected output format")
            if kwargs["background"] == "transparent" and "A" not in image.getbands():
                raise ValueError("Missing output alpha channel")
        except Exception:
            raise ImageGenerationError(
                "invalid_image_output", request_id=request_id, attempts=attempt
            ) from None
        response_model = getattr(result, "model", None)
        image.info["paperbanana_generation"] = {
            "provider": self.name,
            "requested_model": self._model,
            "response_model": response_model if isinstance(response_model, str) else None,
            "request_id": request_id if isinstance(request_id, str) else None,
            "operation": operation,
            "attempts": attempt,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "size": f"{image.width}x{image.height}",
            "quality": kwargs["quality"],
            "review_status": "UNREVIEWED",
        }
        return image
