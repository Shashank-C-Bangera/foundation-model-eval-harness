from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token

DEFAULT_SPACE_NAME = "foundation-model-eval-dashboard"
DEFAULT_COMMIT_MESSAGE = "Update demo runs: baseline_models + rag_baseline"


def _optional_env(name: str) -> str:
    return os.getenv(name, "").strip()


def _resolve_token() -> str:
    token = _optional_env("HF_TOKEN")
    if token:
        return token

    cached = get_token()
    if cached:
        return cached

    raise RuntimeError(
        "Missing Hugging Face token. Set HF_TOKEN or login once via huggingface-cli login."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Space demo assets to Hugging Face Spaces.")
    parser.add_argument(
        "--space-assets-dir",
        default="space_assets",
        help="Folder to upload to the Space repo.",
    )
    parser.add_argument(
        "--space-name",
        default=DEFAULT_SPACE_NAME,
        help="Hugging Face Space repo name (without username).",
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help="Full Space repo id. If omitted, uses HF_USERNAME/space-name.",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message for upload.",
    )
    parser.add_argument(
        "--include-app-files",
        action="store_true",
        help="Also upload app source/config files used by the Space runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = _resolve_token()
    api = HfApi(token=hf_token)
    hf_username = _optional_env("HF_USERNAME")
    if not hf_username:
        whoami = api.whoami()
        hf_username = whoami.get("name", "").strip()
    if not hf_username:
        raise RuntimeError(
            "Unable to resolve HF username. Set HF_USERNAME or ensure token has account access."
        )

    repo_id = args.repo_id or f"{hf_username}/{args.space_name}"
    if not api.repo_exists(repo_id=repo_id, repo_type="space"):
        raise RuntimeError(
            f"Space repo does not exist: {repo_id}. Create it once in Hugging Face Spaces."
        )

    assets_dir = Path(args.space_assets_dir)
    if not assets_dir.exists():
        raise FileNotFoundError(f"Space assets directory not found: {assets_dir}")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(assets_dir),
        path_in_repo="space_assets",
        commit_message=args.commit_message,
    )
    print(f"Uploaded assets folder to {repo_id}: {assets_dir}")

    if args.include_app_files:
        app_dir = Path("app")
        streamlit_dir = Path(".streamlit")
        for p in [app_dir, streamlit_dir]:
            if p.exists():
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="space",
                    folder_path=str(p),
                    path_in_repo=str(p),
                    commit_message=args.commit_message,
                )
                print(f"Uploaded folder to {repo_id}: {p}")

        for p in [Path("app.py"), Path("README.md"), Path("requirements.txt"), Path("Dockerfile")]:
            if p.exists():
                api.upload_file(
                    repo_id=repo_id,
                    repo_type="space",
                    path_in_repo=str(p),
                    path_or_fileobj=str(p),
                    commit_message=args.commit_message,
                )
                print(f"Uploaded file to {repo_id}: {p}")


if __name__ == "__main__":
    main()
