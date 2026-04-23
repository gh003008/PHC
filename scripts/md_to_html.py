"""Convert a markdown file to a self-contained styled HTML document.

Usage:
    python scripts/md_to_html.py <input.md> <output.html>
"""
import sys
from pathlib import Path

import markdown

CSS = """
:root {
  --fg: #1f2328;
  --fg-muted: #656d76;
  --bg: #ffffff;
  --bg-code: #f6f8fa;
  --bg-quote: #f6f8fa;
  --border: #d0d7de;
  --accent: #0969da;
  --accent-bg: #ddf4ff;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; font-size: 15px; }
body {
  margin: 0 auto; max-width: 980px; padding: 2rem 2.5rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
               Arial, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
  color: var(--fg); background: var(--bg); line-height: 1.6;
}

h1, h2, h3, h4 {
  margin-top: 2rem; margin-bottom: 0.6rem;
  font-weight: 600; line-height: 1.25;
}
h1 { font-size: 2rem; padding-bottom: 0.35em; border-bottom: 1px solid var(--border); }
h2 { font-size: 1.5rem; padding-bottom: 0.3em; border-bottom: 1px solid var(--border); }
h3 { font-size: 1.2rem; }
h4 { font-size: 1rem; color: var(--fg-muted); }

p { margin: 0.6rem 0; }
hr { border: 0; border-top: 1px solid var(--border); margin: 2rem 0; }

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

code {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 0.88em;
  background: var(--bg-code);
  padding: 0.14em 0.36em;
  border-radius: 4px;
}
pre {
  background: var(--bg-code);
  padding: 1rem; border-radius: 6px;
  overflow-x: auto;
  line-height: 1.45;
  font-size: 0.85rem;
}
pre code { background: transparent; padding: 0; border-radius: 0; font-size: inherit; }

blockquote {
  margin: 0.8rem 0;
  padding: 0.5rem 1rem;
  color: var(--fg-muted);
  background: var(--bg-quote);
  border-left: 4px solid var(--border);
  border-radius: 0 4px 4px 0;
}
blockquote p:first-child { margin-top: 0; }
blockquote p:last-child { margin-bottom: 0; }

table { border-collapse: collapse; margin: 1rem 0; width: 100%; font-size: 0.93em; }
th, td { border: 1px solid var(--border); padding: 6px 13px; text-align: left; vertical-align: top; }
th { background: var(--bg-code); font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }

ul, ol { padding-left: 1.6em; margin: 0.5rem 0; }
li { margin: 0.15rem 0; }
li > p { margin: 0.3rem 0; }

/* Task list (GitHub-flavored markdown) */
li input[type="checkbox"] {
  margin-right: 0.45em; vertical-align: middle;
  transform: translateY(-1px);
}
ul.task-list, ol.task-list { list-style-type: none; padding-left: 0.5em; }
li.task-list-item { list-style-type: none; }

/* Layout tweaks for long code lines in tables */
td pre, td code { white-space: pre-wrap; word-break: break-word; }

/* Strong / emphasis refinement */
strong { font-weight: 600; }

/* Print-friendly */
@media print {
  body { max-width: none; padding: 1rem; }
  pre, blockquote { page-break-inside: avoid; }
  h1, h2, h3 { page-break-after: avoid; }
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def convert(md_path: Path, out_path: Path):
    text = md_path.read_text(encoding="utf-8")

    # Strip the leading H1 title for use as <title>, keep it in the body too.
    title = md_path.stem
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    body_html = markdown.markdown(
        text,
        extensions=[
            "tables",
            "fenced_code",
            "codehilite",
            "toc",
            "sane_lists",
            "nl2br",
        ],
        extension_configs={
            "codehilite": {"use_pygments": False, "css_class": "codehilite"},
        },
    )

    # Convert GFM-style task list markers "[ ]" / "[x]" into real checkboxes.
    # python-markdown's default doesn't handle task lists, so do a small
    # post-process pass.
    body_html = _convert_task_list_items(body_html)

    html = HTML_TEMPLATE.format(title=title, css=CSS, content=body_html)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}  ({len(html)} chars)")


def _convert_task_list_items(html: str) -> str:
    """Replace leading '[ ]' / '[x]' inside <li> with <input type=checkbox>."""
    import re

    def repl_unchecked(m):
        inner = m.group(1)
        return (
            f'<li class="task-list-item">'
            f'<input type="checkbox" disabled> {inner}</li>'
        )

    def repl_checked(m):
        inner = m.group(1)
        return (
            f'<li class="task-list-item">'
            f'<input type="checkbox" disabled checked> {inner}</li>'
        )

    html = re.sub(r"<li>\s*\[\s\]\s*(.*?)</li>", repl_unchecked, html, flags=re.DOTALL)
    html = re.sub(
        r"<li>\s*\[\s*[xX]\s*\]\s*(.*?)</li>", repl_checked, html, flags=re.DOTALL
    )
    # Also mark parent <ul> as task-list if it contains any task-list-item
    html = re.sub(
        r"<ul>(\s*<li class=\"task-list-item\")",
        r'<ul class="task-list">\1',
        html,
    )
    return html


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/md_to_html.py <input.md> <output.html>")
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
