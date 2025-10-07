import os
import re
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Union

import msgpack
import requests


def remove_dir_from_diff(patch: str, directory: str) -> str:
    def _norm_dir(d: str) -> str:
        d = d.replace("\\", "/")
        d = re.sub(r"^(?:a/|b/)+", "", d)  # strip leading a/ or b/ if user passed it
        d = d.lstrip("./")
        if d and not d.endswith("/"):
            d += "/"
        return d

    def _norm_file(p: str) -> str:
        p = p.replace("\\", "/")
        p = re.sub(r"^(?:a/|b/)+", "", p)
        p = p.lstrip("./")
        return p

    dir_norm = _norm_dir(directory)

    lines = patch.splitlines(keepends=True)
    out: list[str] = []

    i = 0
    n = len(lines)

    # Regex to capture per-file header: diff --git a/path b/path
    diff_header_re = re.compile(r"^diff --git a/(.*?) b/(.*?)\s*$")

    # Copy any preamble before the first "diff --git"
    while i < n and not lines[i].startswith("diff --git"):
        out.append(lines[i])
        i += 1

    # Iterate per-file sections
    while i < n:
        # Expect a "diff --git" line
        header_line = lines[i]
        m = diff_header_re.match(header_line)
        # If it's not a standard header, just include it and move on defensively
        if not m:
            out.append(header_line)
            i += 1
            continue

        a_path, b_path = m.groups()
        a_path = _norm_file(a_path)
        b_path = _norm_file(b_path)

        # Decide whether to drop this entire file section
        drop_section = (dir_norm and a_path.startswith(dir_norm)) or (
            dir_norm and b_path.startswith(dir_norm)
        )

        # Walk this section until the next "diff --git" or EOF
        section_start = i
        i += 1
        while i < n and not lines[i].startswith("diff --git"):
            i += 1
        section_end = i  # slice is [section_start:section_end]

        if not drop_section:
            out.extend(lines[section_start:section_end])
        # else: skip adding this section (effectively removing it)

    return "".join(out)


def keep_only_dir_from_diff(
    patch: str,
    directory: Union[str, Iterable[str]],
    keep_preamble: bool = True,
) -> str:
    def _norm_file(p: str) -> str:
        p = p.replace("\\", "/")
        p = re.sub(r"^(?:a/|b/)+", "", p)
        return p.lstrip("./")

    def _norm_dir(d: str) -> str:
        d = d.replace("\\", "/")
        d = re.sub(r"^(?:a/|b/)+", "", d).lstrip("./")
        if d and not d.endswith("/"):
            d += "/"
        return d

    if isinstance(directory, (list, tuple, set)):
        dir_prefixes: List[str] = [_norm_dir(d) for d in directory if str(d).strip()]
    else:
        dir_prefixes = [_norm_dir(str(directory))] if str(directory).strip() else []

    # If no valid directory provided, nothing matches; return only optional preamble
    lines = patch.splitlines(keepends=True)
    out: List[str] = []

    i, n = 0, len(lines)
    diff_header_re = re.compile(r"^diff --git a/(.*?) b/(.*?)\s*$")

    # Optional preamble (before first "diff --git")
    if keep_preamble:
        while i < n and not lines[i].startswith("diff --git"):
            out.append(lines[i])
            i += 1
    else:
        while i < n and not lines[i].startswith("diff --git"):
            i += 1

    # Walk per-file sections
    while i < n:
        header_line = lines[i]
        m = diff_header_re.match(header_line)
        if not m:
            # Defensive: if something weird shows up, drop it unless it's a diff header we understand.
            i += 1
            continue

        a_path, b_path = (_norm_file(m.group(1)), _norm_file(m.group(2)))

        def _matches_any_dir(path: str) -> bool:
            return any(path.startswith(dp) for dp in dir_prefixes if dp)

        keep_section = _matches_any_dir(a_path) or _matches_any_dir(b_path)

        # Locate end of this section (next "diff --git" or EOF)
        section_start = i
        i += 1
        while i < n and not lines[i].startswith("diff --git"):
            i += 1
        section_end = i

        if keep_section:
            out.extend(lines[section_start:section_end])

    return "".join(out)


def get_last_release_before_pr_merge(
    owner: str,
    repo: str,
    pr_number: int,
    include_prereleases: bool = False,
    include_drafts: bool = False,
) -> Optional[dict]:
    token = os.getenv("GITHUB_TOKEN")

    def _parse_iso8601_z(dt: str) -> datetime:
        return datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )

    def _headers(token: Optional[str]) -> dict:
        h = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "pr-release-finder",
        }
        if token:
            h["Authorization"] = f"Bearer {token}"
        return h

    def _get_pr_merged_at(
        owner: str, repo: str, pr_number: int, token: Optional[str]
    ) -> datetime:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        r = requests.get(url, headers=_headers(token), timeout=30)
        if r.status_code == 404:
            raise ValueError(f"PR #{pr_number} not found in {owner}/{repo}")
        r.raise_for_status()
        data = r.json()
        merged_at = data.get("merged_at")
        if not merged_at:
            state = data.get("state", "unknown")
            raise ValueError(f"PR #{pr_number} is not merged (state={state}).")
        return _parse_iso8601_z(merged_at)

    def _iter_releases(
        owner: str, repo: str, token: Optional[str], per_page: int = 100
    ):
        page = 1
        while True:
            url = f"https://api.github.com/repos/{owner}/{repo}/releases"
            r = requests.get(
                url,
                headers=_headers(token),
                params={"per_page": per_page, "page": page},
                timeout=30,
            )
            if r.status_code == 404:
                raise ValueError(
                    f"Repository {owner}/{repo} not found or releases are not accessible."
                )
            r.raise_for_status()
            items = r.json()
            if not items:
                break
            for rel in items:
                yield rel
            # Basic pagination: stop if fewer than per_page items
            if len(items) < per_page:
                break
            page += 1
            # Small courtesy sleep to avoid secondary rate limits
            time.sleep(0.05)

    merged_at = _get_pr_merged_at(owner, repo, pr_number, token)

    best_release = None
    best_pub_dt = None

    for rel in _iter_releases(owner, repo, token):
        # Filter by draft/prerelease visibility according to options
        if rel.get("draft", False) and not include_drafts:
            continue
        if rel.get("prerelease", False) and not include_prereleases:
            continue

        pub = rel.get("published_at")
        # Some releases might be drafts with null published_at
        if not pub:
            continue

        pub_dt = _parse_iso8601_z(pub)

        # We want the release published at/before the merge moment, with the latest possible time
        if pub_dt <= merged_at and (best_pub_dt is None or pub_dt > best_pub_dt):
            best_pub_dt = pub_dt
            best_release = {
                "id": rel.get("id"),
                "tag_name": rel.get("tag_name"),
                "name": rel.get("name"),
                "html_url": rel.get("html_url"),
                "draft": rel.get("draft", False),
                "prerelease": rel.get("prerelease", False),
                "created_at": rel.get("created_at"),
                "published_at": rel.get("published_at"),
                "body": rel.get("body"),
            }

    return best_release
