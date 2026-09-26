"""
Module: pdf_export.py
Description: Generates publication-grade, professionally styled PDF reports
from ResearchMind AI collaborative multi-agent synthesis results.
"""
from io import BytesIO
from typing import Dict, Any, List
import re
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "ResearchMind AI — Collaborative Multi-Agent Synthesis Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running footer
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)

        self.drawString(54, 32, "Confidential & Generated for Academic Research • ResearchMind AI")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


def _clean_for_xml(text: str) -> str:
    """Escapes XML entities while safely formatting bold, italics, and code tags."""
    if not text:
        return ""
    # 1. Escape literal ampersands not already part of an entity
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|bull;)", "&amp;", text)
    # 2. Escape literal angle brackets
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # 3. Convert markdown bold **...**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # 4. Convert markdown italics *...*
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    # 5. Convert markdown code `...`
    text = re.sub(r"`([^`]+?)`", r'<font face="Courier" color="#2563eb">\1</font>', text)
    return text


def _markdown_to_flowables(md_text: str, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    """Converts common markdown constructs (headings, bullets, tables) into ReportLab flowables."""
    flowables = []
    lines = md_text.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Check for Markdown Table (| col1 | col2 |)
        if line.startswith("|") and line.endswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                row_str = lines[i].strip()
                # Skip separator lines like |---|---|
                if not re.match(r"^\|(\s*[-:]+\s*\|)+$", row_str):
                    raw_cells = [c.strip() for c in row_str.strip("|").split("|")]
                    table_lines.append(raw_cells)
                i += 1

            if table_lines:
                table_data = []
                for row_idx, row in enumerate(table_lines):
                    formatted_row = []
                    style_to_use = styles["TableHead"] if row_idx == 0 else styles["TableBody"]
                    for cell in row:
                        formatted_row.append(Paragraph(_clean_for_xml(cell), style_to_use))
                    table_data.append(formatted_row)

                # Column widths allocation
                col_count = len(table_data[0]) if table_data else 1
                col_width = 504 / col_count
                tbl = Table(table_data, colWidths=[col_width] * col_count)
                tbl.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ])
                )
                flowables.append(tbl)
                flowables.append(Spacer(1, 8))
            continue

        # Headings
        if line.startswith("### "):
            title = _clean_for_xml(line[4:].strip())
            flowables.append(Paragraph(title, styles["H3"]))
            flowables.append(Spacer(1, 4))
        elif line.startswith("## "):
            title = _clean_for_xml(line[3:].strip())
            flowables.append(Paragraph(title, styles["H2"]))
            flowables.append(Spacer(1, 6))
        elif line.startswith("# "):
            title = _clean_for_xml(line[2:].strip())
            flowables.append(Paragraph(title, styles["H1"]))
            flowables.append(Spacer(1, 8))
        # Bullet list items
        elif line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            bullet_text = _clean_for_xml(line[2:].strip())
            p = Paragraph(f"&bull; {bullet_text}", styles["BulletText"])
            flowables.append(p)
            flowables.append(Spacer(1, 3))
        # Numbered list items
        elif re.match(r"^\d+\.\s+", line):
            m = re.match(r"^(\d+\.)\s+(.+)$", line)
            if m:
                num, num_text = m.groups()
                p = Paragraph(f"<b>{num}</b> {_clean_for_xml(num_text)}", styles["NumberedText"])
                flowables.append(p)
                flowables.append(Spacer(1, 3))
        # Regular paragraph
        else:
            p = Paragraph(_clean_for_xml(line), styles["BodyText"])
            flowables.append(p)
            flowables.append(Spacer(1, 5))

        i += 1

    return flowables


def generate_synthesis_pdf(
    paper_id: str,
    engine_label: str,
    search_data: Dict[str, Any],
    reader_data: Dict[str, Any],
    comp_data: Dict[str, Any],
    gap_data: Dict[str, Any],
    rev_data: Dict[str, Any],
) -> bytes:
    """Builds a complete, formatted PDF document in memory and returns raw PDF bytes."""
    buffer = BytesIO()

    # Document Geometry
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    # Styles
    base_styles = getSampleStyleSheet()

    styles = {
        "DocTitle": ParagraphStyle(
            "DocTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f172a"),
            alignment=0,
            spaceAfter=4,
        ),
        "DocSubtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=12,
        ),
        "SectionBanner": ParagraphStyle(
            "SectionBanner",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=12,
            spaceAfter=6,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceBefore=5,
            spaceAfter=3,
        ),
        "H3": ParagraphStyle(
            "H3",
            parent=base_styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#3b82f6"),
            spaceBefore=4,
            spaceAfter=2,
        ),
        "BodyText": ParagraphStyle(
            "BodyText",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1e293b"),
            alignment=4,  # Justified
        ),
        "BulletText": ParagraphStyle(
            "BulletText",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1e293b"),
            leftIndent=12,
        ),
        "NumberedText": ParagraphStyle(
            "NumberedText",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1e293b"),
            leftIndent=12,
        ),
        "TableHead": ParagraphStyle(
            "TableHead",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#0f172a"),
        ),
        "TableBody": ParagraphStyle(
            "TableBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#334155"),
        ),
        "CalloutBox": ParagraphStyle(
            "CalloutBox",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1e3a8a"),
        ),
        "CitationTag": ParagraphStyle(
            "CitationTag",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#2563eb"),
        ),
    }

    story: List[Any] = []

    # Title & Metadata Banner
    story.append(Paragraph("ResearchMind AI — Synthesis Report", styles["DocTitle"]))
    story.append(Paragraph("Collaborative Multi-Agent Academic Analysis & Peer Review", styles["DocSubtitle"]))

    now_str = datetime.now().strftime("%B %d, %Y • %H:%M UTC")
    meta_data = [
        [
            Paragraph("<b>Target Document:</b>", styles["TableHead"]),
            Paragraph(f"<font color='#2563eb'><b>{paper_id}</b></font>", styles["TableBody"]),
            Paragraph("<b>Generated On:</b>", styles["TableHead"]),
            Paragraph(now_str, styles["TableBody"]),
        ],
        [
            Paragraph("<b>Inference Engine:</b>", styles["TableHead"]),
            Paragraph(engine_label, styles["TableBody"]),
            Paragraph("<b>Review Rubric:</b>", styles["TableHead"]),
            Paragraph("NeurIPS / ICLR Standards", styles["TableBody"]),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[100, 152, 92, 160])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # SECTION 1: EXECUTIVE OVERVIEW
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceBefore=4, spaceAfter=8))
    story.append(Paragraph("1. Literature Search & Executive Overview", styles["SectionBanner"]))
    summary_text = search_data.get("summary", "No executive summary produced.")
    summary_flowables = _markdown_to_flowables(summary_text, styles)
    story.extend(summary_flowables)
    story.append(Spacer(1, 12))

    # SECTION 2: STRUCTURED PAPER READER MATRIX
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceBefore=4, spaceAfter=8))
    story.append(Paragraph("2. Structured Paper Reader Matrix", styles["SectionBanner"]))

    table_dict = reader_data.get("table", {})
    if table_dict:
        matrix_rows = [
            [
                Paragraph("<b>Academic Dimension</b>", styles["TableHead"]),
                Paragraph("<b>Extracted Findings & Grounded Synthesis</b>", styles["TableHead"]),
            ]
        ]
        for dim, findings in table_dict.items():
            dim_p = Paragraph(f"<b>{dim}</b>", styles["TableHead"])
            clean_f = _clean_for_xml(str(findings))
            find_p = Paragraph(clean_f, styles["TableBody"])
            matrix_rows.append([dim_p, find_p])

        matrix_table = Table(matrix_rows, colWidths=[140, 364])
        matrix_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ])
        )
        story.append(matrix_table)

    citations = reader_data.get("citations")
    if citations:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Grounded Citations:</b> {citations}", styles["CitationTag"]))

    story.append(Spacer(1, 12))

    # SECTION 3: COMPARATIVE BASELINE ANALYSIS
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceBefore=4, spaceAfter=8))
    story.append(Paragraph("3. Cross-Paper & Baseline Comparison", styles["SectionBanner"]))
    comp_text = comp_data.get("comparison", "No comparison produced.")
    story.extend(_markdown_to_flowables(comp_text, styles))
    story.append(Spacer(1, 12))

    # SECTION 4: RESEARCH GAPS & HYPOTHESES
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceBefore=4, spaceAfter=8))
    story.append(Paragraph("4. Discovered Research Gaps & Hypotheses", styles["SectionBanner"]))
    gap_text = gap_data.get("analysis", "No gap analysis produced.")
    story.extend(_markdown_to_flowables(gap_text, styles))
    story.append(Spacer(1, 12))

    # SECTION 5: PEER REVIEW CRITIQUE
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceBefore=4, spaceAfter=8))
    story.append(Paragraph("5. Academic Conference Peer Review", styles["SectionBanner"]))
    rev_text = rev_data.get("critique", "No review produced.")
    story.extend(_markdown_to_flowables(rev_text, styles))

    # Build PDF with dynamic 2-pass page numbering
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
