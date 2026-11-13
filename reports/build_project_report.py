from __future__ import annotations

from io import BytesIO
from pathlib import Path
from copy import deepcopy

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(r"C:\Users\Aayush\OneDrive\Desktop\FINAL MINOR.docx")
CONTENT = ROOT / "reports" / "project_report_content.docx"
OUTPUT = ROOT / "Rainfall_Predictive_Project_Submission_Report_Reference_Format.docx"
BLACK = RGBColor(0, 0, 0)


def set_para_text(paragraph: Paragraph, text: str) -> None:
    """Replace paragraph text while retaining its paragraph properties and first-run formatting."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for run in runs[1:]:
        run.text = ""


def set_cell_text(cell, text: str, *, bold=False) -> None:
    p = cell.paragraphs[0]
    set_para_text(p, text)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = BLACK
        if bold:
            run.bold = True
        if run.font.size is None:
            run.font.size = Pt(12)


def add_field(paragraph: Paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    value = OxmlElement("w:t")
    value.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instr, separate, value, end))


def add_hyperlink(paragraph: Paragraph, text: str, url: str) -> None:
    rid = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "000000")
    props.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)
    run.append(props)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    link.append(run)
    paragraph._p.append(link)


def set_page_ref(cell, bookmark: str) -> None:
    p = cell.paragraphs[0]
    set_para_text(p, "")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(p, f" PAGEREF {bookmark} \\h ")


def add_bookmark(paragraph: Paragraph, name: str, bookmark_id: int) -> None:
    first_run = paragraph.runs[0] if paragraph.runs else paragraph.add_run("")
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(bookmark_id))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(bookmark_id))
    first_run._r.addprevious(start)
    first_run._r.addnext(end)


def set_run_style(run, *, size=12, bold=None, italic=None, name="Times New Roman") -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_body_paragraph(doc: Document, text: str, style="Body Text", *, alignment=None,
                       bold=False, italic=False, size=12, bookmark=None, bookmark_id=200):
    p = doc.add_paragraph(style=style)
    if alignment is not None:
        p.alignment = alignment
    run = p.add_run(text)
    set_run_style(run, size=size, bold=bold, italic=italic)
    if bookmark:
        add_bookmark(p, bookmark, bookmark_id)
    return p


def add_reference_heading(doc: Document, text: str, level: int, bookmark=None, bookmark_id=200):
    # The reference uses plain, black Times New Roman headings rather than a colored theme.
    p = doc.add_paragraph(style="Body Text" if level == 1 else "Heading 2")
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.page_break_before = True
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(9)
        run = p.add_run(text.upper())
        set_run_style(run, size=18, bold=True)
    else:
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text.upper())
        set_run_style(run, size=12, bold=True)
    if bookmark:
        add_bookmark(p, bookmark, bookmark_id)
    return p


def add_reference_table(doc: Document, source_table: Table) -> None:
    rows, cols = len(source_table.rows), len(source_table.columns)
    table = doc.add_table(rows=rows, cols=cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(source_table.rows):
        for ci, cell in enumerate(row.cells):
            target = table.cell(ri, ci)
            target.text = ""
            p = target.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(cell.text)
            set_run_style(run, size=10.5, bold=ri == 0)
    doc.add_paragraph()


def copy_image_from_paragraph(source_doc: Document, target_doc: Document, source_para: Paragraph) -> bool:
    blips = source_para._p.xpath(".//a:blip")
    if not blips:
        return False
    target = target_doc.add_paragraph()
    target.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for blip in blips:
        rel_id = blip.get(qn("r:embed"))
        part = source_doc.part.related_parts[rel_id]
        target.add_run().add_picture(BytesIO(part.blob), width=Inches(7.15))
    return True


def add_copied_paragraph(source_doc: Document, target_doc: Document, source_para: Paragraph,
                         bookmarks: dict[str, tuple[str, int]], figure_bookmarks: dict[str, tuple[str, int]],
                         next_id: list[int], last_list: list[str]) -> None:
    text = source_para.text.strip()
    if not text:
        if source_para._p.xpath('.//w:br[@w:type="page"]'):
            target_doc.add_page_break()
        elif source_para._p.xpath(".//w:drawing"):
            copy_image_from_paragraph(source_doc, target_doc, source_para)
        return
    if source_para._p.xpath('.//w:br[@w:type="page"]'):
        target_doc.add_page_break()
        return

    source_style = source_para.style.name if source_para.style else "Normal"
    is_chapter = source_style == "Heading 1"
    is_subheading = source_style in {"Heading 2", "Heading 3"}
    is_list = source_style in {"List Bullet", "List Number"}
    is_caption = text.startswith(("Figure ", "Table "))

    if is_chapter:
        chapter_key = None
        for number, key in (("1", "ch1"), ("2", "ch2"), ("3", "ch3"), ("4", "ch4"), ("5", "ch5")):
            if text.startswith(f"CHAPTER {number} "):
                chapter_key = key
                break
        if chapter_key:
            bookmark, bid = bookmarks[chapter_key]
            add_reference_heading(target_doc, text, 1, bookmark, bid)
            if text == "CHAPTER 1 — INTRODUCTION":
                add_body_paragraph(target_doc, "TOPIC - RAINFALL PREDICTIVE ANALYSIS", "Body Text",
                                   alignment=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
        else:
            key = ""
            if text.startswith("RESULTS AND CONCLUSION"):
                key = "results"
            elif text.startswith("FUTURE SCOPE"):
                key = "future"
            elif text.startswith("REFERENCES"):
                key = "references"
            bookmark = None
            if key:
                bookmark = bookmarks[key][0]
                next_id[0] += 1
                bookmarks[key] = (bookmark, next_id[0])
            add_reference_heading(target_doc, text, 1, bookmark, next_id[0])
        last_list[0] = ""
    elif is_subheading:
        # Preserve the template's uppercase, bold subsection heading treatment.
        add_reference_heading(target_doc, text, 2)
        last_list[0] = ""
    elif is_list:
        marker = "• "
        if source_style == "List Number":
            if last_list[0] != "numbered":
                last_list[1] = 0
            last_list[1] += 1
            marker = f"{last_list[1]}. "
            last_list[0] = "numbered"
        else:
            last_list[0] = "bullet"
        p = add_body_paragraph(target_doc, marker + text, "List Paragraph", size=12)
    else:
        last_list[0] = ""
        if is_caption:
            p = add_body_paragraph(target_doc, text, "Body Text", alignment=WD_ALIGN_PARAGRAPH.CENTER,
                                   italic=True, size=10.5)
            import re
            match = re.match(r"Figure\s+(\d+\.\d+)", text)
            if match:
                key = "fig" + match.group(1).replace(".", "")
                if key in figure_bookmarks:
                    bookmark, bid = figure_bookmarks[key]
                    add_bookmark(p, bookmark, bid)
        else:
            is_code = any(run.font.name and "Consolas" in run.font.name for run in source_para.runs)
            p = add_body_paragraph(target_doc, text, "Body Text" if not is_code else "Normal",
                                   size=9.5 if is_code else 12)
            if is_code:
                for run in p.runs:
                    set_run_style(run, size=9.5, name="Consolas")
            source_links = list(getattr(source_para, "hyperlinks", []))
            if source_links:
                p.runs[0].text = p.runs[0].text.removesuffix(" Source").rstrip() + " "
                for link in source_links:
                    add_hyperlink(p, link.text, link.address)

    if source_para._p.xpath(".//w:drawing"):
        copy_image_from_paragraph(source_doc, target_doc, source_para)


def update_front_matter(doc: Document) -> None:
    paragraphs = doc.paragraphs
    replacements = {
        "LEAVE MANAGEMENT SYSTEM ": "RAINFALL PREDICTIVE PROJECT",
        "TOPIC- SIGN LANGUAGE RECOGNITION": "TOPIC - RAINFALL PREDICTIVE ANALYSIS",
        "ACKNOWLEGEMENT": "ACKNOWLEDGEMENT",
    }
    for p in paragraphs:
        for old, new in replacements.items():
            if old in p.text:
                set_para_text(p, p.text.replace(old, new))

    # Keep the original split runs and tabs on the cover while updating the guide label.
    p = paragraphs[37]
    for run in p.runs:
        if run.text == "Dr":
            run.text = "Suman"
        elif run.text in {".", "Archana"}:
            run.text = ""
    p = paragraphs[38]
    for run in p.runs:
        if run.text == "Associate ":
            run.text = "Project "
        elif run.text == "Professor":
            run.text = "Guide"

    set_para_text(paragraphs[47], "I would like to thank Guru Gobind Singh Indraprastha University for including project work in the MCA curriculum. The opportunity to study rainfall data has helped me connect software development, database design, data science, and machine learning with a practical environmental problem. I am grateful to everyone who supported this academic work.")
    set_para_text(paragraphs[49], "I express my sincere gratitude to Suman for guiding me through the planning and development of the Rainfall Predictive Project. The guidance, encouragement, and constructive suggestions helped shape the project and this report.")
    set_para_text(paragraphs[51], "I also thank the faculty members of Jagan Institute of Management Studies, my family, and my friends for their encouragement and support throughout the project. Their help made it possible to complete the data analysis, model evaluation, and dashboard work.")
    set_para_text(paragraphs[84], "CERTIFICATE")
    cert = ("This is to certify that KHUSHI KAPOOR, student of MCA 2nd Shift, Semester II, has undertaken the Minor Project-II titled “RAINFALL PREDICTIVE PROJECT” under the guidance of Suman. The work is submitted in partial fulfilment of the requirements for the award of the degree of Master of Computer Applications (MCA) of Guru Gobind Singh Indraprastha University, New Delhi-110078. To the best of my knowledge, this report presents the project work carried out by the student and has not been submitted earlier for another award.")
    set_para_text(paragraphs[87], cert)
    set_para_text(paragraphs[122], "Suman")
    set_para_text(paragraphs[123], "Project Guide")

    abstract = [
        "The Rainfall Predictive Project is a data-science application for exploring historical rainfall, viewing available forecast guidance, and estimating next-calendar-day rainfall at selected grid cells. It combines a PostgreSQL data layer, Python ingestion and validation modules, a Streamlit dashboard, interactive geographic and time-series charts, and machine-learning evaluation. The dashboard presents six views: overview, rainfall analytics, forecasts and warnings, live weather, historical grid, and model performance.",
        "The historical database currently contains 2,542,458 daily grid-cell records across 4,964 cells from 1 January 2015 to 31 December 2025. The model training export uses 200 consistently sampled cells and contains 800,770 next-day examples. A chronological holdout compares a previous-day persistence baseline, Random Forest, and gradient-boosted trees. The selected model achieved a holdout MAE of 3.447 mm and RMSE of 9.385 mm, improving MAE by 7.1% and RMSE by 19.4% over the baseline. Its R-squared was 0.248, indicating useful but limited explanatory performance.",
        "The project emphasizes readable outputs and honest uncertainty. Grid-cell estimates are not district-level warnings, and the one-day-ahead model is experimental. District and state observation tables and official warning tables are empty in the current database, so the dashboard reports that no records are loaded. Future work should add verified administrative-area mappings, more recent observations, probability calibration, and broader spatial and seasonal evaluation.",
        "Keywords: rainfall analysis, gridded observations, time series, machine learning, PostgreSQL, Streamlit.",
    ]
    # Keep the reference's abstract page and its four original body paragraphs.
    for idx, text in zip((165, 166, 167, 168), abstract):
        set_para_text(paragraphs[idx], text)

    # Refresh the existing index table without changing its grid layout or column widths.
    contents = [
        ("1.", "Chapter 1. INTRODUCTION\nOverview\nScope of the Project\nObjectives\nProblem Statement\nMotivation", "ch1"),
        ("2.", "Chapter 2. LITERATURE REVIEW", "ch2"),
        ("3.", "Chapter 3. METHODOLOGY\nSystem Development Model\nData Processing\nData Management\nSystem Operation", "ch3"),
        ("4.", "Chapter 4. SYSTEM ANALYSIS\nProposed System\nRequirements\nSystem Design\nMachine-Learning Method", "ch4"),
        ("5.", "Chapter 5. SOFTWARE DESIGN\nImplementation\nDashboard Pages\nTesting and Validation", "ch5"),
        ("6.", "RESULTS AND CONCLUSION", "results"),
        ("7.", "FUTURE SCOPE OF THE PROJECT", "future"),
        ("8.", "REFERENCES", "references"),
    ]
    index = doc.tables[0]
    for i, (serial, topic, bookmark) in enumerate(contents, start=1):
        set_cell_text(index.cell(i, 0), serial)
        set_cell_text(index.cell(i, 1), topic)
        set_page_ref(index.cell(i, 2), f"b_{bookmark}")

    figure_items = [
        ("Figure 3.1  Rainfall data and prediction workflow", "fig31"),
        ("Figure 4.1  System architecture", "fig41"),
        ("Figure 4.2  Development life cycle", "fig42"),
        ("Figure 4.3  Level-0 data-flow diagram", "fig43"),
        ("Figure 4.4  User and analyst use cases", "fig44"),
        ("Figure 4.5  Core data entities", "fig45"),
        ("Figure 4.6  Next-day model feature pipeline", "fig46"),
        ("Figure 4.7  Chronological holdout error comparison", "fig47"),
        ("Figure 5.1  Six-page dashboard map", "fig51"),
    ]
    figures = doc.tables[1]
    while len(figures.rows) < len(figure_items) + 1:
        figures.add_row()
    for i, (label, bookmark) in enumerate(figure_items, start=1):
        set_cell_text(figures.cell(i, 0), str(i))
        set_cell_text(figures.cell(i, 1), label)
        set_page_ref(figures.cell(i, 2), f"b_{bookmark}")


def build_report() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Reference document is missing: {TEMPLATE}")
    if not CONTENT.exists():
        raise FileNotFoundError(f"Project report content file is missing: {CONTENT}")
    template = Document(TEMPLATE)
    content = Document(CONTENT)
    reference_section_break = next(
        deepcopy(p._p.pPr.sectPr) for p in template.paragraphs
        if p._p.pPr is not None and p._p.pPr.sectPr is not None
    )
    update_front_matter(template)

    # Retain the original cover, section break, front matter, logo images, tables,
    # headers, footers, margins and page styles; replace the old project body only.
    template_body = template.element.body
    chapter_node = next(
        child for child in template_body.iterchildren()
        if child.tag == qn("w:p") and "CHAPTER-1" in "".join(t.text or "" for t in child.findall(".//" + qn("w:t")))
    )
    for child in list(template_body.iterchildren()):
        if child is chapter_node:
            break
    started = False
    for child in list(template_body.iterchildren()):
        if child is chapter_node:
            started = True
        if started and child.tag != qn("w:sectPr"):
            template_body.remove(child)

    # Keep the template's plain black Times New Roman chapter-page treatment.
    bookmarks = {
        "ch1": ("b_ch1", 201), "ch2": ("b_ch2", 202), "ch3": ("b_ch3", 203),
        "ch4": ("b_ch4", 204), "ch5": ("b_ch5", 205), "results": ("b_results", 206),
        "future": ("b_future", 207), "references": ("b_references", 208),
    }
    figure_bookmarks = {key: ("b_" + key, 300 + i) for i, key in enumerate(
        ("fig31", "fig41", "fig42", "fig43", "fig44", "fig45", "fig46", "fig47", "fig51"))}
    next_id = [400]
    last_list = ["", 0]
    content_body = content.element.body
    began = False
    for child in content_body.iterchildren():
        if child.tag == qn("w:sectPr"):
            continue
        if child.tag == qn("w:p"):
            para = Paragraph(child, content)
            if para.text.strip().startswith("APPENDIX A"):
                break
            if para.text.strip().startswith("CHAPTER 1 — INTRODUCTION"):
                began = True
            if not began:
                continue
            add_copied_paragraph(content, template, para, bookmarks, figure_bookmarks, next_id, last_list)
        elif child.tag == qn("w:tbl") and began:
            add_reference_table(template, Table(child, content))

    # The source uses a paragraph section-break after the report body. Restore it
    # so the reference's page size, margins, header/footer and numbering remain intact.
    break_paragraph = template.add_paragraph()
    break_ppr = break_paragraph._p.get_or_add_pPr()
    break_ppr.append(reference_section_break)

    # Ask Word to refresh the page references and the existing footer page field.
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    template.settings.element.append(update)
    template.core_properties.title = "Rainfall Predictive Project — Minor Project-II"
    template.core_properties.subject = "MCA Minor Project Submission Report"
    template.core_properties.author = "Khushi Kapoor"
    template.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
