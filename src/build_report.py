"""Build the report PDFs from the markdown sources in docs/.

Two jobs:
  1. figures: draws the ensemble diagram and the per-variant robustness plot
     (from outputs/eval/*.csv) into --fig_dir.
  2. pdf: renders a small markdown subset (headings, paragraphs, bullet and
     numbered lists, pipe tables, fenced code, images) with reportlab.

Image paths inside the markdown are resolved against the repo root, so run
this from the repo root.

Usage:
    python src/build_report.py --figures --fig_dir outputs/figures
    python src/build_report.py --md docs/technical_report.md --out technical_report.pdf \
        --title "Deepfake detection: technical report" --body_size 9
"""

import argparse
import html
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

MEMBERS = [("final_resnet50", "ResNet-50"), ("e_cnxt_small", "ConvNeXt-Small"),
           ("final_swin_t", "Swin-T")]
COLORS = ["#2b6cb0", "#c05621", "#2f855a"]


def draw_architecture(path):
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#edf2f7", ec="#4a5568", size=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.0))

    mid = 4.8
    box(0.2, mid - 0.8, 2.6, 1.6, "224x224 face\nJPEG\nResize, ToTensor,\nImageNet normalise")
    box(3.7, mid - 0.8, 2.2, 1.6, "Original +\nhorizontal flip\n(2 passes)")
    arrow(2.9, mid, 3.65, mid)
    names = ["ResNet-50 (23.5M)", "ConvNeXt-Small (49.5M)", "Swin-T (27.5M)"]
    ys = [8.0 - 1.55 * i for i in range(6)]
    for i, y in enumerate(ys):
        col = COLORS[i // 2]
        seed = (42, 7)[i % 2]
        box(7.0, y, 3.2, 1.1, f"{names[i // 2]}\nseed {seed}", fc="#ffffff", ec=col, size=8)
        arrow(6.0, mid, 6.95, y + 0.55)
        box(10.9, y + 0.1, 2.2, 0.9, "1 logit\n(Dropout + Linear)", size=7.5)
        arrow(10.25, y + 0.55, 10.85, y + 0.55)
        arrow(13.15, y + 0.55, 13.75, mid)
    box(13.8, mid - 0.8, 2.2, 1.6, "Mean of\nlogits over\n6 members x\n2 views")
    arrow(16.05, mid, 16.6, mid)
    box(16.6, mid - 0.8, 1.3, 1.6, "sigmoid\nP(fake)\n>= 0.5", size=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_robustness(path, eval_dir):
    frames = {}
    for run, label in MEMBERS:
        d = pd.read_csv(Path(eval_dir) / f"{run}.csv").set_index("variant")
        frames[label] = d["auc"].drop("summary")
    df = pd.DataFrame(frames)
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    x = range(len(df))
    for (label, col), c in zip(df.items(), COLORS):
        ax.plot(x, col.values, marker="o", ms=3.5, lw=1.2, color=c, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("validation AUC")
    ax.set_ylim(0.885, 0.912)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="lower left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_figures(fig_dir, eval_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    draw_architecture(fig_dir / "report_architecture.png")
    draw_robustness(fig_dir / "report_robust_members.png", eval_dir)
    print("wrote figures to", fig_dir)


# ---------------------------------------------------------------- markdown -> pdf

def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="-1">\1</font>', text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def parse_blocks(lines):
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            blocks.append(("code", code))
        elif re.match(r"#{1,3} ", line):
            level = len(line.split(" ")[0])
            blocks.append((f"h{level}", line[level + 1:].strip()))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            blocks.append(("image", (m.group(1), m.group(2))))
            i += 1
        elif re.match(r"\s*[-*] ", line) or re.match(r"\s*\d+\. ", line):
            items = []
            ordered = bool(re.match(r"\s*\d+\. ", line))
            while i < len(lines) and (re.match(r"\s*[-*] ", lines[i]) or re.match(r"\s*\d+\. ", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip())):
                cur = lines[i].rstrip("\n")
                if re.match(r"\s*([-*]|\d+\.) ", cur):
                    items.append(re.sub(r"^\s*([-*]|\d+\.) ", "", cur))
                else:
                    items[-1] += " " + cur.strip()
                i += 1
            blocks.append(("ol" if ordered else "ul", items))
        else:
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"(#{1,3} |\||```|!\[|\s*[-*] |\s*\d+\. )", lines[i]):
                para.append(lines[i].strip())
                i += 1
            blocks.append(("p", " ".join(para)))
    return blocks


def fill_gaps(story, kinds, frame_w, frame_h):
    """Reorder flowables so a figure that does not fit the rest of a page does not
    leave a blank band: text blocks of the same section that do fit are pulled
    ahead of the figure. Page fill is tracked with a running height estimate."""
    textual = ("p", "ul", "ol", "table", "code")
    from reportlab.pdfgen.canvas import Canvas
    scratch = Canvas(io.BytesIO())
    items = list(zip(kinds, story))
    out, pos, i = [], 0.0, 0

    def size(fl):
        _, h = fl.wrapOn(scratch, frame_w, frame_h)
        return h + fl.getSpaceBefore() + fl.getSpaceAfter()

    def advance(pos, h):
        pos += h
        while pos > frame_h:
            pos -= frame_h
        return pos

    while i < len(items):
        kind, fl = items[i]
        h = size(fl)
        if kind == "image" and pos + h > frame_h and pos > 0:
            j, room, pulled = i + 1, frame_h - pos, []
            while j < len(items) and items[j][0] in textual:
                hj = size(items[j][1])
                if hj > room:
                    break
                pulled.append(items[j])
                room -= hj
                j += 1
            if pulled:
                for k, f in pulled:
                    out.append(f)
                    pos += size(f)
                items[i + 1:j] = []
        if kind == "image" and pos + h > frame_h:
            pos = 0.0
        out.append(fl)
        pos = advance(pos, h) if kind != "image" else pos + h
        i += 1
    return out


def build_pdf(md_path, out_path, title, author, body_size, root, extra_root=None):
    from PIL import Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, KeepTogether, ListFlowable, ListItem,
                                    Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=body_size,
                          leading=body_size * 1.32, alignment=TA_LEFT, spaceAfter=body_size * 0.5)
    styles = {
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=body_size * 1.9,
                             leading=body_size * 2.3, spaceAfter=body_size * 0.6, keepWithNext=1),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=body_size * 1.3,
                             leading=body_size * 1.6, spaceBefore=body_size * 0.9,
                             spaceAfter=body_size * 0.4, textColor=colors.HexColor("#1a365d"),
                             keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=body_size * 1.08,
                             spaceBefore=body_size * 0.5, spaceAfter=body_size * 0.25,
                             keepWithNext=1),
        "cap": ParagraphStyle("cap", parent=base, fontName="Helvetica-Oblique",
                              fontSize=body_size * 0.88, leading=body_size * 1.1,
                              textColor=colors.HexColor("#4a5568")),
        "cell": ParagraphStyle("cell", parent=base, fontSize=body_size * 0.86,
                               leading=body_size * 1.04, spaceAfter=0),
        "cellh": ParagraphStyle("cellh", parent=base, fontName="Helvetica-Bold",
                                fontSize=body_size * 0.86, leading=body_size * 1.04, spaceAfter=0),
        "code": ParagraphStyle("code", parent=base, fontName="Courier", fontSize=body_size * 0.82,
                               leading=body_size * 1.0),
    }
    page_w, page_h = A4
    margin = 17 * mm
    avail = page_w - 2 * margin
    story = []
    kinds = []
    blocks = parse_blocks(Path(md_path).read_text(encoding="utf-8").splitlines(True))

    for kind, val in blocks:
        n_before = len(story)
        if kind in ("h1", "h2", "h3"):
            story.append(Paragraph(inline(val), styles[kind]))
        elif kind == "p":
            story.append(Paragraph(inline(val), base))
        elif kind in ("ul", "ol"):
            items = [ListItem(Paragraph(inline(t), base), leftIndent=12) for t in val]
            story.append(ListFlowable(items, bulletType="bullet" if kind == "ul" else "1",
                                      start="circle" if kind == "ul" else None, leftIndent=12,
                                      bulletFontSize=body_size * 0.8 if kind == "ul" else body_size))
        elif kind == "code":
            story.append(Preformatted("\n".join(val), styles["code"], maxLineLength=110))
            story.append(Spacer(1, body_size * 0.5))
        elif kind == "table":
            ncol = max(len(r) for r in val)
            rows = [r + [""] * (ncol - len(r)) for r in val]
            data = [[Paragraph(inline(c), styles["cellh" if ri == 0 else "cell"]) for c in r]
                    for ri, r in enumerate(rows)]
            lens = [max(len(re.sub(r"[`*]", "", r[ci])) for r in rows) for ci in range(ncol)]
            weights = [max(4, min(l, 40)) ** 0.8 for l in lens]
            widths = [avail * w / sum(weights) for w in weights]
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a0aec0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t)
            story.append(Spacer(1, body_size * 0.6))
        elif kind == "image":
            cap, spec = val
            path, _, frac = spec.partition("|")
            frac = float(frac) if frac else 1.0
            src = Path(root) / path
            if not src.exists() and extra_root:
                src = Path(extra_root) / Path(path).name
            img = Image.open(src).convert("RGB")
            w_pt = avail * frac
            target_px = int(w_pt / 72 * 150)
            if img.width > target_px:
                img = img.resize((target_px, int(img.height * target_px / img.width)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            buf.seek(0)
            h_pt = w_pt * img.height / img.width
            story.append(KeepTogether([RLImage(buf, width=w_pt, height=h_pt),
                                       Paragraph(inline(cap), styles["cap"])]))
        kinds.extend([kind] * (len(story) - n_before))

    frame_w = avail - 12
    frame_h = page_h - 15 * mm - 17 * mm - 12
    story = fill_gaps(story, kinds, frame_w, frame_h)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawString(margin, 9 * mm, f"{author} - {title}")
        canvas.drawRightString(page_w - margin, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=margin, rightMargin=margin,
                            topMargin=15 * mm, bottomMargin=17 * mm, title=title, author=author,
                            creator=author)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figures", action="store_true", help="draw the report figures")
    p.add_argument("--fig_dir", default="outputs/figures")
    p.add_argument("--eval_dir", default="outputs/eval")
    p.add_argument("--md", help="markdown source")
    p.add_argument("--out", help="output PDF")
    p.add_argument("--title", default="Report")
    p.add_argument("--author", default="lowfps")
    p.add_argument("--body_size", type=float, default=9.0)
    p.add_argument("--root", default=".", help="base dir for image paths in the markdown")
    p.add_argument("--extra_root", default=None, help="fallback dir searched by file name")
    args = p.parse_args()
    if args.figures:
        make_figures(args.fig_dir, args.eval_dir)
    if args.md:
        build_pdf(args.md, args.out, args.title, args.author, args.body_size, args.root, args.extra_root)


if __name__ == "__main__":
    main()
