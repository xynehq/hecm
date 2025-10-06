import re
from typing import Iterable, List, Union


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
