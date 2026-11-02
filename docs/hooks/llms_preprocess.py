"""HTML preprocessing for the llms.txt Markdown pages.

The HTML-to-Markdown conversion escapes backslashes in text, so TeX written
for KaTeX (`\\sigma`) would reach an agent as `\\\\sigma`. Code is copied
verbatim, so math is handed over as code: inline math becomes `$...$` in a
code span, display math a fenced `latex` block.
"""

from bs4 import BeautifulSoup


def _tex(text: str) -> str:
    text = text.strip()
    for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$")):
        if text.startswith(left) and text.endswith(right):
            return text[len(left):-len(right)].strip()
    return text


def preprocess(soup: BeautifulSoup, output: str) -> None:
    for node in soup.select(".arithmatex"):
        tex = _tex(node.get_text())
        if node.name == "div":
            pre = soup.new_tag("pre")
            code = soup.new_tag("code", attrs={"class": "language-latex"})
            code.string = tex
            pre.append(code)
            node.replace_with(pre)
        else:
            code = soup.new_tag("code")
            code.string = "$" + tex + "$"
            node.replace_with(code)
