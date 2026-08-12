import io
from datetime import datetime

# builds a before and after cleaning summary as a pdf, returned as bytes
def build_report_pdf(original_df, cleaned_df, history, filename="dataset"):
    """Uses only reportlab, which ships with most python environments."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib.enums import TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    style_h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1f77b4"),
        spaceBefore=16,
        spaceAfter=4,
    )
    style_body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#333333"),
    )
    style_caption = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#666666"),
        spaceAfter=4,
    )

    BLUE = colors.HexColor("#1f77b4")
    LIGHT_GRAY = colors.HexColor("#f3f4f6")
    DIVIDER = colors.HexColor("#e5e7eb")

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=DIVIDER, spaceAfter=8, spaceBefore=4)

    def section(title):
        return [Spacer(1, 0.2 * cm), Paragraph(title, style_h2), hr()]

    def stat_table(rows, col_widths=None):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.3, DIVIDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    story = []

    # title block
    generated_at = datetime.now().strftime("%d %B %Y at %H:%M")
    story.append(Paragraph("Data Cleaning Report", style_title))
    story.append(Paragraph(f"Dataset: {filename}", style_body))
    story.append(Paragraph(f"Generated: {generated_at}", style_caption))
    story.append(Spacer(1, 0.3 * cm))
    story.append(hr())

    # summary metrics
    orig_rows, orig_cols = original_df.shape
    clean_rows, clean_cols = cleaned_df.shape
    orig_missing = int(original_df.isna().sum().sum())
    clean_missing = int(cleaned_df.isna().sum().sum())
    orig_dups = int(original_df.duplicated().sum())
    clean_dups = int(cleaned_df.duplicated().sum())
    n_steps = len(history)

    story += section("Summary")
    summary_rows = [
        ["Metric", "Before", "After", "Change"],
        ["Rows", str(orig_rows), str(clean_rows), _delta(orig_rows, clean_rows)],
        ["Columns", str(orig_cols), str(clean_cols), _delta(orig_cols, clean_cols)],
        ["Missing cells", str(orig_missing), str(clean_missing), _delta(orig_missing, clean_missing)],
        ["Duplicate rows", str(orig_dups), str(clean_dups), _delta(orig_dups, clean_dups)],
        ["Cleaning steps applied", "", str(n_steps), ""],
    ]
    story.append(stat_table(summary_rows, col_widths=[6 * cm, 3 * cm, 3 * cm, 3 * cm]))

    # cleaning steps
    story += section("Cleaning Steps Performed")
    if not history:
        story.append(Paragraph("No cleaning operations were recorded.", style_body))
    else:
        step_rows = [["Step", "Operation", "Shape After"]]
        for i, step in enumerate(history):
            df_snap = step["df"]
            step_rows.append([
                str(i + 1),
                step["label"],
                f"{df_snap.shape[0]} rows x {df_snap.shape[1]} cols",
            ])
        story.append(stat_table(step_rows, col_widths=[1.5 * cm, 10 * cm, 4 * cm]))

    # before column profile
    story += section("Column Profile: Before")
    story.append(Paragraph(
        "Shows the state of each column in the original uploaded file.",
        style_caption,
    ))
    story.append(_column_profile_table(original_df, BLUE, LIGHT_GRAY, DIVIDER))

    # after column profile
    story += section("Column Profile: After")
    story.append(Paragraph(
        "Shows the state of each column after all cleaning operations.",
        style_caption,
    ))
    story.append(_column_profile_table(cleaned_df, BLUE, LIGHT_GRAY, DIVIDER))

    # missing value summary
    story += section("Missing Value Detail")
    missing_before = original_df.isna().sum()
    missing_after = cleaned_df.isna().sum()
    shared = [c for c in original_df.columns if c in cleaned_df.columns]
    mv_rows = [["Column", "Missing Before", "Missing After", "Resolved"]]
    any_missing = False
    for col in shared:
        mb = int(missing_before.get(col, 0))
        ma = int(missing_after.get(col, 0))
        if mb > 0 or ma > 0:
            any_missing = True
            mv_rows.append([col, str(mb), str(ma), "yes" if mb > 0 and ma == 0 else "no"])
    if any_missing:
        story.append(stat_table(mv_rows, col_widths=[6 * cm, 3.5 * cm, 3.5 * cm, 2.5 * cm]))
    else:
        story.append(Paragraph("No missing values found in either version.", style_body))

    # data sample
    story += section("Data Sample: After Cleaning")
    story.append(Paragraph("First 10 rows of the cleaned dataset.", style_caption))
    sample = cleaned_df.head(10)

    # truncate values for display
    sample_display = sample.copy()
    for col in sample_display.columns:
        sample_display[col] = sample_display[col].astype(str).str[:18]
    sample_rows = [list(sample_display.columns)]
    for _, row in sample_display.iterrows():
        sample_rows.append(row.tolist())

    # content aware column widths, measured from the longest value per column
    available_width = 25 * cm
    char_widths = []
    for i, col in enumerate(sample_display.columns):
        max_len = len(str(col))
        for _, row in sample_display.iterrows():
            max_len = max(max_len, len(str(row.iloc[i])))
        char_widths.append(max(max_len, 4))
    total_chars = sum(char_widths)
    col_widths = [max(1.5 * cm, (w / total_chars) * available_width) for w in char_widths]

    # if total exceeds page width, scale down proportionally
    total_w = sum(col_widths)
    if total_w > available_width:
        scale = available_width / total_w
        col_widths = [w * scale for w in col_widths]

    sample_table = Table(sample_rows, colWidths=col_widths, repeatRows=1)
    sample_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.3, DIVIDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    story.append(sample_table)

    doc.build(story)
    return buf.getvalue()

# formats a before/after change as a signed string
def _delta(before, after):
    diff = after - before
    if diff == 0:
        return "no change"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff}"

# builds the column type/null/unique summary table used for before and after
def _column_profile_table(df, header_color, alt_color, grid_color):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    rows = [["Column", "Type", "Non-Null", "Null", "Null %", "Unique"]]
    for col in df.columns:
        s = df[col]
        rows.append([
            str(col)[:30],
            str(s.dtype),
            str(int(s.notna().sum())),
            str(int(s.isna().sum())),
            f"{s.isna().mean() * 100:.1f}%",
            str(int(s.nunique())),
        ])
    t = Table(rows, colWidths=[5 * cm, 2.5 * cm, 2 * cm, 1.8 * cm, 1.8 * cm, 2 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, alt_color]),
        ("GRID", (0, 0), (-1, -1), 0.3, grid_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t
