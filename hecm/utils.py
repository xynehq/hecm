import re


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
