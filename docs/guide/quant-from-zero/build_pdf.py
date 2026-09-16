#!/usr/bin/env python3
"""Render QUANT_FROM_ZERO.md as a PDF.

    pip install reportlab
    python3 docs/guide/quant-from-zero/build_pdf.py

The Markdown file is the source of truth; this script only typesets it. It understands the subset of
Markdown the guide uses: headings, paragraphs, bullet and numbered lists, fenced code blocks, pipe
tables, horizontal rules, and inline bold / code / links.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak, Paragraph,
                                PageTemplate, Preformatted, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#5a6472")
ACCENT = colors.HexColor("#1d4ed8")
RULE = colors.HexColor("#d5dae1")
CODE_BG = colors.HexColor("#f5f7f9")


def styles() -> dict:
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.6,
                          leading=14.2, textColor=INK, spaceAfter=7)
    return {
        "title": ParagraphStyle("TitleBig", parent=body, fontName="Helvetica-Bold", fontSize=25,
                                leading=30, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", parent=body, fontSize=11.5, leading=16, textColor=MUTED),
        "h1": ParagraphStyle("H1", parent=body, fontName="Helvetica-Bold", fontSize=17, leading=22,
                             spaceBefore=16, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=17,
                             spaceBefore=13, spaceAfter=6),
        "h3": ParagraphStyle("H3", parent=body, fontName="Helvetica-Bold", fontSize=10.8, leading=15,
                             spaceBefore=10, spaceAfter=4),
        "body": body,
        "bullet": ParagraphStyle("Bullet", parent=body, leftIndent=12, bulletIndent=3, spaceAfter=3),
        "code": ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=8, leading=10.6,
                               textColor=INK),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8.3, leading=11, spaceAfter=0),
        "cellhead": ParagraphStyle("CellHead", parent=body, fontName="Helvetica-Bold", fontSize=8.3,
                                   leading=11, spaceAfter=0),
        "caption": ParagraphStyle("Caption", parent=body, fontSize=8.6, textColor=MUTED,
                                  alignment=TA_CENTER),
    }


def inline(text: str) -> str:
    """Markdown inline formatting to ReportLab markup."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<font color="#1d4ed8">\1</font>', out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\w`])`([^`]+)`(?![\w`])",
                 r'<font face="Courier" size="8.6">\1</font>', out)
    out = out.replace("---", "&#8212;").replace("--", "&#8211;")
    return out


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build_table(rows: list[list[str]], width: float, st: dict) -> Table:
    header, *body = rows
    columns = max(len(r) for r in rows)
    data = []
    for index, row in enumerate(rows):
        padded = row + [""] * (columns - len(row))
        style = st["cellhead"] if index == 0 else st["cell"]
        data.append([Paragraph(inline(c), style) for c in padded])

    # Give the first column more room; it usually holds the label.
    first = min(0.34, max(0.16, 1.6 / columns))
    widths = [width * first] + [width * (1 - first) / (columns - 1)] * (columns - 1) if columns > 1 \
        else [width]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 0), (-1, 0), CODE_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]))
    _ = header, body
    return table


def code_block(lines: list[str], width: float, st: dict) -> Table:
    text = "\n".join(line.rstrip() for line in lines) or " "
    inner = Preformatted(text, st["code"], maxLineLength=104)
    table = Table([[inner]], colWidths=[width], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def parse(markdown: str, width: float, st: dict) -> list:
    story: list = []
    lines = markdown.replace("\r\n", "\n").split("\n")
    i = 0
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), st["body"]))
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(Spacer(1, 3))
            story.append(code_block(block, width, st))
            story.append(Spacer(1, 7))
        elif stripped.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            flush()
            rows = [split_row(stripped)]
            i += 2   # skip the |---| separator
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i].strip()))
                i += 1
            story.append(Spacer(1, 3))
            story.append(build_table(rows, width, st))
            story.append(Spacer(1, 8))
            continue
        elif stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            story.append(Paragraph(inline(text), st[f"h{min(level, 3)}"]))
        elif stripped in ("---", "***", "___"):
            flush()
            story.append(Spacer(1, 4))
            rule = Table([[""]], colWidths=[width], rowHeights=[0.7])
            rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), RULE)]))
            story.append(rule)
            story.append(Spacer(1, 8))
        elif re.match(r"^[-*] ", stripped) or re.match(r"^\d+\. ", stripped):
            flush()
            marker = "&#8226;" if stripped[0] in "-*" else stripped.split(".", 1)[0] + "."
            text = re.sub(r"^([-*]|\d+\.) ", "", stripped)
            story.append(Paragraph(inline(text), st["bullet"], bulletText=marker))
        elif not stripped:
            flush()
        else:
            paragraph.append(stripped)
        i += 1

    flush()
    return story


def title_page(st: dict, width: float) -> list:
    return [
        Spacer(1, 46 * mm),
        Paragraph("Quant From Zero", st["title"]),
        Paragraph("Learning quantitative finance by dissecting a working engine", st["subtitle"]),
        Spacer(1, 10),
        code_block(["git clone https://github.com/Lukey-7/AxiomQuant",
                    "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel",
                    "./build/bin/axiomquant --data sample_data --no-db"], width, st),
        Spacer(1, 14),
        Paragraph("A companion to the AxiomQuant repository: what quantitative finance is, how a "
                  "backtest lies to you, what every module of this engine does, how to run it, and "
                  "eight experiments that show each idea in action.", st["body"]),
        PageBreak(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(HERE / "QUANT_FROM_ZERO.md"))
    parser.add_argument("--output", default=str(HERE / "Quant-From-Zero.pdf"))
    args = parser.parse_args()

    st = styles()
    margin = 19 * mm
    width = A4[0] - 2 * margin

    doc = BaseDocTemplate(args.output, pagesize=A4, leftMargin=margin, rightMargin=margin,
                          topMargin=17 * mm, bottomMargin=17 * mm,
                          title="Quant From Zero", author="AxiomQuant")

    def decorate(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        if document.page > 1:
            canvas.drawString(margin, 10 * mm, "Quant From Zero - AxiomQuant")
            canvas.drawRightString(A4[0] - margin, 10 * mm, str(document.page - 1))
        canvas.restoreState()

    frame = Frame(margin, 17 * mm, width, A4[1] - 34 * mm, id="body")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=decorate)])

    markdown = Path(args.input).read_text(encoding="utf-8")
    # The Markdown title and contents table are replaced by the PDF's own title page.
    body = markdown.split("\n---\n", 1)[1] if "\n---\n" in markdown else markdown
    story = title_page(st, width) + parse(body, width, st)
    doc.build(story)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
