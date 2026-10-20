"""Turn the README's relative links and images into absolute URLs for PyPI.

Usage: python .github/scripts/absolute_readme_links.py README.md TAG

GitHub resolves relative paths in the README on any branch, but PyPI cannot.
Run at release time, before building, this pins every relative path to TAG:
images and <source> elements to raw.githubusercontent.com, and other links to
GitHub's blob view for files or tree view for directories. Each release's
PyPI page then shows that release's figures. A path that does not exist in
the repository is an error, so nothing broken gets published.
"""

import re
import sys
from pathlib import Path

REPO = "craabreu/openmm-opes"
ATTRIBUTE = re.compile(r'\b(src|srcset)="([^"]+)"')
TARGET = re.compile(r"\]\(([^)\s]+)\)")


def is_relative(target):
    return not re.match(r"^([a-z]+:|#|/)", target)


def checked(target):
    path = Path(target.split("#")[0])
    if not path.exists():
        sys.exit(f"README links to {target}, which does not exist")
    return path


def raw(target, tag):
    checked(target)
    return f"https://raw.githubusercontent.com/{REPO}/{tag}/{target}"


def page(target, tag):
    view = "tree" if checked(target).is_dir() else "blob"
    return f"https://github.com/{REPO}/{view}/{tag}/{target}"


def main():
    readme, tag = Path(sys.argv[1]), sys.argv[2]
    text = readme.read_text()

    def attribute(match):
        name, target = match.groups()
        if is_relative(target):
            target = raw(target, tag)
        return f'{name}="{target}"'

    def is_image(text, close):
        """Whether the ] at index close ends an image's alt text, ![...]."""
        depth = 0
        for i in range(close, -1, -1):
            depth += {"]": 1, "[": -1}.get(text[i], 0)
            if depth == 0:
                return i > 0 and text[i - 1] == "!"
        return False

    def link(match, text):
        target = match.group(1)
        if is_relative(target):
            if is_image(text, match.start()):
                target = raw(target, tag)
            else:
                target = page(target, tag)
        return f"]({target})"

    text = ATTRIBUTE.sub(attribute, text)
    text = TARGET.sub(lambda match: link(match, text), text)
    readme.write_text(text)


if __name__ == "__main__":
    main()
