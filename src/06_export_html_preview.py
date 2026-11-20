"""
Export TOWARDS_DATA_SCIENCE_ARTICLE.md to a clean, Medium/TDS-styled HTML
preview (preview_tds_article.html) for easy reading and review in any browser.
"""

from pathlib import Path
import html
import re


def inline_fmt(s: str) -> str:
    codes = []

    def _code(m):
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"@@CODE{len(codes) - 1}@@"

    s = re.sub(r"`([^`]+)`", _code, s)

    math_map = [
        (r"\\sigma\^0_\{\\text\{dB\}\}", "&sigma;<sup>0</sup><sub>dB</sub>"),
        (r"\\sigma\^0", "&sigma;<sup>0</sup>"),
        (r"\\log_\{10\}", "log<sub>10</sub>"),
        (r"\\alpha_\{\\text\{SAR\}\}", "&alpha;<sub>SAR</sub>"),
        (r"\\gamma", "&gamma;"),
        (r"\\approx", "&approx;"),
        (r"\\le", "&le;"),
        (r"\\pm", "&plusmn;"),
        (r"\\cdot", "&middot;"),
        (r"\\text\{ Mm\}\^3", " Mm<sup>3</sup>"),
        (r"R\^2", "R<sup>2</sup>"),
        (r"A\^\{1\.46\}", "A<sup>1.46</sup>"),
        (r"A\^\\gamma", "A<sup>&gamma;</sup>"),
        (r"c_0", "c<sub>0</sub>"),
        (r"c_1", "c<sub>1</sub>"),
    ]

    def _math(m):
        expr = m.group(1)
        for pat, rep in math_map:
            expr = re.sub(pat, rep, expr)
        return f'<span class="math">{expr}</span>'

    s = re.sub(r"\$\$([^$]+)\$\$", lambda m: f'<div class="math-block">{_math(m)}</div>', s)
    s = re.sub(r"\$([^$]+)\$", _math, s)

    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    s = s.replace(" -> ", " &rarr; ").replace("—", "&mdash;")

    for idx, c in enumerate(codes):
        s = s.replace(f"@@CODE{idx}@@", c)
    return s


def main() -> None:
    md = Path("TOWARDS_DATA_SCIENCE_ARTICLE.md").read_text(encoding="utf-8")
    lines = md.splitlines()
    out = []
    i = 0
    in_code = False
    code_buf = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            close_lists()
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        s = line.strip()
        if not s:
            close_lists()
            i += 1
            continue
        if s == "---":
            close_lists()
            out.append("<hr/>")
            i += 1
            continue
        if s.startswith("# "):
            close_lists()
            out.append(f"<h1>{inline_fmt(s[2:])}</h1>")
            i += 1
            continue
        if s.startswith("## ") and len(out) == 1:
            close_lists()
            out.append(
                f'<p class="subtitle">{inline_fmt(s[3:])}</p>'
                '<div class="byline">By <strong>Davide Parisi</strong> &middot; '
                "Towards Data Science Article Preview</div>"
            )
            i += 1
            continue
        if s.startswith("## "):
            close_lists()
            out.append(f"<h2>{inline_fmt(s[3:])}</h2>")
            i += 1
            continue
        if s.startswith("### "):
            close_lists()
            out.append(f"<h3>{inline_fmt(s[4:])}</h3>")
            i += 1
            continue
        if s.startswith("![") and "](" in s:
            close_lists()
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
            if m:
                alt, src = m.group(1), m.group(2)
                cap = ""
                if (
                    i + 1 < len(lines)
                    and lines[i + 1].strip().startswith("*")
                    and lines[i + 1].strip().endswith("*")
                ):
                    cap = inline_fmt(lines[i + 1].strip()[1:-1])
                    i += 1
                out.append(
                    f'<figure><img src="{src}" alt="{html.escape(alt)}"/>'
                    f"<figcaption>{cap}</figcaption></figure>"
                )
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            close_lists()
            headers = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            tbl = [
                "<table><thead><tr>"
                + "".join(f"<th>{inline_fmt(h)}</th>" for h in headers)
                + "</tr></thead><tbody>"
            ]
            for r in rows:
                tbl.append("<tr>" + "".join(f"<td>{inline_fmt(c)}</td>" for c in r) + "</tr>")
            tbl.append("</tbody></table>")
            out.append("\n".join(tbl))
            continue
        if s.startswith("- "):
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline_fmt(s[2:])}</li>")
            i += 1
            continue
        m_ol = re.match(r"^(\d+)\.\s+(.*)", s)
        if m_ol:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{inline_fmt(m_ol.group(2))}</li>")
            i += 1
            continue
        if s.startswith("$$") and s.endswith("$$"):
            close_lists()
            out.append(inline_fmt(s))
            i += 1
            continue

        close_lists()
        out.append(f"<p>{inline_fmt(s)}</p>")
        i += 1

    close_lists()

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Tracking Drought Reservoirs Through Clouds With Sentinel-1 SAR and Python</title>
<style>
  body {{
    font-family: Georgia, "Charter", "Times New Roman", serif;
    color: #242424;
    background: #ffffff;
    line-height: 1.72;
    font-size: 19px;
    margin: 0;
    padding: 44px 20px 90px;
  }}
  article {{
    max-width: 740px;
    margin: 0 auto;
  }}
  h1, h2, h3, .subtitle, .byline, table, figcaption {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  h1 {{
    font-size: 2.25rem;
    line-height: 1.18;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 12px;
    color: #111827;
  }}
  .subtitle {{
    font-size: 1.24rem;
    line-height: 1.45;
    color: #6B7280;
    font-weight: 400;
    margin: 0 0 22px;
  }}
  .byline {{
    font-size: 0.92rem;
    color: #6B7280;
    padding-bottom: 24px;
    margin-bottom: 28px;
    border-bottom: 1px solid #E5E7EB;
  }}
  h2 {{
    font-size: 1.5rem;
    font-weight: 700;
    margin: 42px 0 14px;
    color: #111827;
    letter-spacing: -0.01em;
  }}
  h3 {{
    font-size: 1.18rem;
    font-weight: 700;
    margin: 28px 0 10px;
    color: #1F2937;
  }}
  p {{ margin: 0 0 20px; }}
  ul, ol {{ margin: 0 0 22px; padding-left: 26px; }}
  li {{ margin-bottom: 10px; }}
  a {{ color: #0284C7; text-decoration: underline; text-underline-offset: 2px; }}
  code {{
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 0.84em;
    background: #F3F4F6;
    padding: 2px 6px;
    border-radius: 4px;
    color: #111827;
  }}
  pre {{
    background: #0F172A;
    color: #E2E8F0;
    padding: 18px 20px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 0.82rem;
    line-height: 1.55;
    margin: 24px 0 28px;
  }}
  pre code {{
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
  }}
  figure {{
    margin: 34px -50px;
    text-align: center;
  }}
  figure img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
  }}
  figcaption {{
    font-size: 0.88rem;
    color: #6B7280;
    line-height: 1.5;
    margin-top: 10px;
    padding: 0 50px;
    font-style: italic;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
    margin: 26px 0 30px;
  }}
  th, td {{
    padding: 10px 10px;
    border-bottom: 1px solid #E5E7EB;
    text-align: left;
  }}
  th {{
    background: #F8FAFC;
    font-weight: 700;
    color: #1E293B;
    border-bottom: 2px solid #CBD5E1;
  }}
  tr:hover td {{ background: #F8FAFC; }}
  hr {{
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 38px 0;
  }}
  .math {{ font-style: italic; }}
  .math-block {{
    text-align: center;
    font-size: 1.25rem;
    margin: 22px 0;
    font-style: italic;
  }}
  @media (max-width: 860px) {{
    figure {{ margin: 28px 0; }}
    figcaption {{ padding: 0; }}
  }}
</style>
</head>
<body>
<article>
{"".join(out)}
</article>
</body>
</html>
"""
    Path("preview_tds_article.html").write_text(html_doc, encoding="utf-8")
    print("Generated preview_tds_article.html successfully!")


if __name__ == "__main__":
    main()
