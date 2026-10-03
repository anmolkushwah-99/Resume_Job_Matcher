"""
Test Utilities for Synthetic PDF Generation.
Generates valid multi-page and single-page PDF byte streams for automated testing.
"""

import io
from typing import List


def generate_synthetic_pdf_bytes(pages_lines: List[List[str]]) -> bytes:
    """
    Constructs a synthetically valid multi-page PDF byte stream without external binary tools.

    :param pages_lines: A list of pages, where each page is a list of text lines.
    :return: Valid PDF bytes.
    """
    if not pages_lines:
        pages_lines = [[]]

    page_objs = []
    content_objs = []
    font_obj_num = 3 + len(pages_lines) * 2

    for i, lines in enumerate(pages_lines):
        if lines:
            # Build PDF content stream for text lines
            stream_commands = ["BT", "/F1 12 Tf", "50 750 Td", "14 TL"]
            for idx, line in enumerate(lines):
                # Escape parentheses and backslashes for PDF string literal
                safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                if idx == 0:
                    stream_commands.append(f"({safe_line}) Tj")
                else:
                    stream_commands.append(f"T* ({safe_line}) Tj")
            stream_commands.append("ET")
            content_str = "\n".join(stream_commands)
            content_bytes = content_str.encode("latin-1", errors="replace")
        else:
            content_bytes = b""

        content_obj = (
            f"{4 + i * 2} 0 obj\n<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1")
            + content_bytes
            + b"\nendstream\nendobj\n"
        )
        content_objs.append(content_obj)

        page_obj = (
            f"{3 + i * 2} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {4 + i * 2} 0 R /Resources << /Font << /F1 {font_obj_num} 0 R >> >> >>\n"
            f"endobj\n"
        ).encode("latin-1")
        page_objs.append(page_obj)

    # Build catalog and page tree
    kids = " ".join([f"{3 + i * 2} 0 R" for i in range(len(pages_lines))])
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(pages_lines)} >>\nendobj\n".encode("latin-1")
    font_obj = f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin-1")

    all_objs = [obj1, obj2]
    for p, c in zip(page_objs, content_objs):
        all_objs.extend([p, c])
    all_objs.append(font_obj)

    buf = b"%PDF-1.4\n"
    xref = [0]
    for obj in all_objs:
        xref.append(len(buf))
        buf += obj

    xref_offset = len(buf)
    buf += f"xref\n0 {len(xref)}\n0000000000 65535 f \n".encode("latin-1")
    for offset in xref[1:]:
        buf += f"{offset:010d} 00000 n \n".encode("latin-1")
    buf += f"trailer\n<< /Size {len(xref)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")

    return buf


def get_standard_synthetic_resume_pdf() -> bytes:
    """
    Returns standard synthetic resume PDF bytes with technical terms and sections.
    """
    page1 = [
        "John Doe",
        "Python Developer",
        "",
        "Education:",
        "B.Tech in Computer Science",
        "",
        "Skills:",
        "Python, C++, C#, Java, .NET, Node.js, React.js",
        "SQL, MongoDB, MySQL, TensorFlow, PyTorch, Scikit-learn",
        "Flask, FastAPI, AWS, REST API, Git",
    ]
    page2 = [
        "Experience:",
        "Software Engineer - 2 years of software development experience.",
        "Built REST APIs using Flask and FastAPI.",
        "Created ML pipelines using Scikit-learn and Pandas.",
    ]
    return generate_synthetic_pdf_bytes([page1, page2])
