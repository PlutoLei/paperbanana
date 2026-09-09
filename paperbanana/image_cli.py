"""Small API image entry: no retrieval or Critic initialization is required."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import typer
from PIL import Image

from paperbanana.providers.image_gen.openai_imagen import ImageGenerationError, OpenAIImageGen


def image_command(
    input: Path = typer.Option(..., "--input", "-i", exists=True, dir_okay=False),
    output: Path = typer.Option(Path("image.png"), "--output", "-o"),
    model: str = typer.Option("gpt-image-2", "--model"),
    quality: str = typer.Option("auto", "--quality"),
    size: str = typer.Option("auto", "--size"),
    background: str = typer.Option("auto", "--background"),
    reference: Optional[list[Path]] = typer.Option(
        None, "--reference", exists=True, dir_okay=False
    ),
    mask: Optional[Path] = typer.Option(None, "--mask", exists=True, dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Generate or edit one image with explicit OpenAI controls. Review is separate."""
    try:
        prompt = input.read_text(encoding="utf-8")
        output_format = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(
            output.suffix.lower()
        )
        if output_format is None:
            raise ValueError("Output must end in .png, .jpg, .jpeg or .webp")
        provider = OpenAIImageGen(
            model=model,
            quality=quality,
            size=size,
            background=background,
            output_format=output_format,
        )
        images = []
        for path in reference or []:
            with Image.open(path) as source:
                source.load()
                images.append(source.copy())
        mask_image = None
        if mask is not None:
            if not images:
                raise ValueError("--mask requires --reference")
            with Image.open(mask) as source:
                source.load()
                mask_image = source.copy()
        if images:
            provider.prepare_edit(prompt, images, mask_image)
        else:
            provider._request(prompt, None, 1024, 1024, None, None, None, None, None)
        if dry_run:
            typer.echo(
                json.dumps(
                    {
                        "status": "validated-offline",
                        "operation": "edit" if images else "generate",
                        "requested_model": model,
                        "size": size,
                        "quality": quality,
                        "references": len(images),
                        "mask": mask is not None,
                    }
                )
            )
            return

        # Only live execution loads configured credentials. Dry runs never read .env.
        from paperbanana.core.config import Settings

        settings = Settings()
        provider._api_key = settings.openai_api_key
        provider._base_url = settings.openai_base_url
        if not provider.is_available():
            raise ValueError("OPENAI_API_KEY is required for the API image route")
        operation = (
            provider.edit(prompt, images, mask_image) if images else provider.generate(prompt)
        )
        result = asyncio.run(operation)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, staged = tempfile.mkstemp(prefix=".image-", suffix=output.suffix, dir=output.parent)
        os.close(fd)
        try:
            result.save(staged, format=output_format.upper())
            os.replace(staged, output)
        finally:
            Path(staged).unlink(missing_ok=True)
        metadata = {**result.info["paperbanana_generation"], "output": str(output)}
        output.with_suffix(".image.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        typer.echo(json.dumps(metadata))
    except ImageGenerationError as error:
        typer.echo(
            json.dumps(
                {
                    "status": "failed",
                    "code": error.code,
                    "request_id": error.request_id,
                    "attempts": error.attempts,
                }
            ),
            err=True,
        )
        raise typer.Exit(1)
    except (ValueError, OSError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1)
