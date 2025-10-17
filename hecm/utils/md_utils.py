import re
from typing import Optional


def extract_markdown_section(markdown_text: str, heading_name: str) -> Optional[str]:
    """
    Extract the content under a specific heading/subheading from a markdown string.

    Args:
        markdown_text: The markdown string to parse
        heading_name: The name of the heading/subheading to find (without # symbols)

    Returns:
        The content under the specified heading, or None if the heading is not found.
        The content includes everything until the next heading of equal or higher level.
    """
    lines = markdown_text.split("\n")

    # Find the heading
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    target_heading_level = None
    start_idx = None

    for idx, line in enumerate(lines):
        match = heading_pattern.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            # Check if this is our target heading
            if title == heading_name.strip():
                target_heading_level = level
                start_idx = idx + 1  # Start from the line after the heading
                break

    # If heading not found, return None
    if start_idx is None:
        return None

    # Find the end of this section (next heading of equal or higher level)
    end_idx = len(lines)
    for idx in range(start_idx, len(lines)):
        match = heading_pattern.match(lines[idx].strip())
        if match:
            level = len(match.group(1))
            # If we found a heading of equal or higher level (lower number), stop here
            if level <= target_heading_level:
                end_idx = idx
                break

    # Extract the content
    content_lines = lines[start_idx:end_idx]

    # Strip leading and trailing empty lines
    while content_lines and not content_lines[0].strip():
        content_lines.pop(0)
    while content_lines and not content_lines[-1].strip():
        content_lines.pop()

    return "\n".join(content_lines) if content_lines else ""


def remove_markdown_comments(markdown_text: str) -> str:
    """
    Remove all HTML-style comments from a markdown string.

    Markdown uses HTML comment syntax: <!-- comment -->
    Comments can span single or multiple lines.

    Args:
        markdown_text: The markdown string to clean

    Returns:
        The markdown string with all comments removed
    """
    # Remove HTML comments (single or multi-line)
    # The pattern matches <!-- followed by any characters (including newlines) until -->
    comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)
    cleaned_text = comment_pattern.sub("", markdown_text)

    return cleaned_text
