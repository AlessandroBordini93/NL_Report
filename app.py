# app.py — Concrete Rebar Resisto (PDF+ZIP export for n8n)
#
# Obiettivo:
# - FastAPI endpoint /export che restituisce uno ZIP BINARIO (application/zip)
# - Nome file: <job_id>_allegati.zip (es: progetto_sisma__...__01_allegati.zip)
# - Dentro lo ZIP: Report_resisto59_completo.pdf
#
# PDF layout richiesto:
# 1) Prima pagina: SOLO immagine schema + titolo "Schema di posa Resisto 5.9"
# 2) Dalla seconda pagina in poi: 2 “tamponamenti” per pagina:
#    - titolo: "Tamponamento: {pid} | B: {B_cm:.0f} cm - H: {H_cm:.0f} cm | Curve non lineari equivalenti"
#    - sotto: immagine (curva) + tabella (come immagini che già create)
#
# Nota importante:
# - Questo script NON calcola curve o tabelle: le riceve dal JSON (da n8n) come PNG base64 + tabella.
# - Quindi puoi usare il tuo nodo precedente per generare immagini e dati e passarli qui.
#
# JSON atteso (schema minimo):
# {
#   "meta": {
#     "project_name": "...",
#     "location_name": "...",
#     "wall_orientation": "Nord|Sud|Est|Ovest|...",
#     "suffix": "01"
#   },
#   "schema": { "png_base64": "...." },
#   "tamponamenti": [
#     {
#       "pid": "0,0",
#       "B_cm": 300,
#       "H_cm": 270,
#       "plot_png_base64": "....",          # immagine curva (PNG) già generata da voi
#       "table": {                          # tabella “sotto” (come immagini): la renderizziamo noi in PDF
#         "columns": ["col1","col2",...],
#         "rows": [
#           ["...", "...", ...],
#           ...
#         ]
#       }
#     },
#     ...
#   ]
# }
#
# n8n:
# - HTTP Request: POST /export?schema_scale=1.5
# - Response: File
# - Binary property: data

from __future__ import annotations

import base64
import re
import textwrap
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import RootModel

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(title="Concrete Rebar Resisto Export", version="1.0.0")


class Payload(RootModel[Dict[str, Any]]):
    pass


# ============================================================
# CONFIG / FONTS
# ============================================================
FONT_REG, FONT_BOLD = "Helvetica", "Helvetica-Bold"


# ============================================================
# UTILS — naming
# ============================================================
def _must(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def slugify(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "x"


def safe_suffix(s: str) -> str:
    s = (s or "").strip()
    _must(len(s) > 0, "meta.suffix mancante o vuoto")
    _must(len(s) <= 64, "meta.suffix troppo lungo (max 64)")
    _must(
        re.fullmatch(r"[A-Za-z0-9._-]+", s) is not None,
        "meta.suffix non valido (usa solo A-Z a-z 0-9 . _ -)",
    )
    return s


ORIENT_MAP = {
    "n": "Nord", "nord": "Nord",
    "s": "Sud", "sud": "Sud",
    "e": "Est", "est": "Est",
    "o": "Ovest", "ovest": "Ovest", "w": "Ovest",
    "ne": "Nord-Est", "nordest": "Nord-Est", "nord-est": "Nord-Est",
    "no": "Nord-Ovest", "nordovest": "Nord-Ovest", "nord-ovest": "Nord-Ovest",
    "se": "Sud-Est", "sudest": "Sud-Est", "sud-est": "Sud-Est",
    "so": "Sud-Ovest", "sudovest": "Sud-Ovest", "sud-ovest": "Sud-Ovest",
}
ALLOWED_ORIENTATIONS = {"Nord", "Sud", "Est", "Ovest", "Nord-Est", "Nord-Ovest", "Sud-Est", "Sud-Ovest"}


def normalize_orientation(s: str) -> str:
    raw = (s or "").strip().lower().replace(" ", "")
    raw = raw.replace("–", "-").replace("—", "-").replace("_", "-")
    raw_no_dash = raw.replace("-", "")
    norm = ORIENT_MAP.get(raw_no_dash)
    if norm is not None:
        return norm
    s2 = (s or "").strip().replace("–", "-").replace("—", "-")
    if s2 in ALLOWED_ORIENTATIONS:
        return s2
    raise ValueError(f"meta.wall_orientation non valida. Valori ammessi: {sorted(ALLOWED_ORIENTATIONS)}")


def make_job_id(project_name: str, location_name: str, wall_orientation: str, suffix: str) -> Tuple[str, str]:
    o = normalize_orientation(wall_orientation)
    job_id = f"{slugify(project_name)}__{slugify(location_name)}__{slugify(o)}__{safe_suffix(suffix)}"
    return job_id, o


def _wrap(txt: str, width=95) -> str:
    return "\n".join(textwrap.fill(p.strip(), width) for p in (txt or "").strip().splitlines() if p.strip())


# ============================================================
# UTILS — n8n body handling
# ============================================================
def _load_body_from_any(obj: Any) -> Dict[str, Any]:
    """
    n8n a volte passa un array tipo: [{"json": {...}}] oppure {"json": {...}}
    Qui normalizziamo a dict “pulito”.
    """
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict) and "json" in first and isinstance(first["json"], dict):
            return first["json"]
        if isinstance(first, dict):
            return first
    if isinstance(obj, dict) and "json" in obj and isinstance(obj["json"], dict):
        return obj["json"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Body JSON non valido (atteso dict oppure array n8n-style).")


# ============================================================
# UTILS — images base64
# ============================================================
def _b64_to_bytes(b64: str) -> bytes:
    if not b64:
        raise ValueError("png_base64 mancante/vuoto")
    # supporta anche "data:image/png;base64,...."
    if "," in b64 and b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64, validate=True)
    except Exception:
        # fallback senza validate
        return base64.b64decode(b64)


# ============================================================
# PDF helpers (header/footer come esempio tuo)
# ============================================================
def _footer(c: canvas.Canvas, W: float, H: float):
    h5 = H / 15
    c.setFont(FONT_REG, 11)
    c.drawCentredString(W / 2, 0.75 * h5, "Ing. Alessandro Bordini")
    c.drawCentredString(W / 2, 0.35 * h5, "Phone: 3451604706 - ✉: alessandro_bordini@outlook.com")


def _draw_header_lines(c: canvas.Canvas, W: float, H: float, header_lines: List[str]) -> float:
    """
    Header leggero (centrato) come nel tuo script.
    Ritorna la y “sotto header” utile per piazzare contenuti.
    """
    c.setFont(FONT_REG, 11)
    y = H - 1.45 * cm
    for ln in header_lines:
        c.drawCentredString(W / 2, y, ln)
        y -= 0.50 * cm
    return y


def _fit_image_box(iw: float, ih: float, box_w: float, box_h: float) -> Tuple[float, float]:
    if iw <= 0 or ih <= 0:
        return (0.0, 0.0)
    scale = min(box_w / iw, box_h / ih)
    return (iw * scale, ih * scale)


def _draw_png_centered(
    c: canvas.Canvas,
    img_bytes: bytes,
    x: float,
    y: float,
    box_w: float,
    box_h: float,
):
    img = ImageReader(BytesIO(img_bytes))
    iw, ih = img.getSize()
    w_img, h_img = _fit_image_box(iw, ih, box_w, box_h)
    x_img = x + (box_w - w_img) / 2
    y_img = y + (box_h - h_img) / 2
    c.drawImage(img, x_img, y_img, w_img, h_img, mask="auto")


def _draw_table(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    columns: List[str],
    rows: List[List[Any]],
    *,
    font_size: int = 8,
):
    """
    Tabella semplice in PDF (grid + testo).
    y è il bottom della tabella (stile reportlab).
    """
    # normalizzazione
    columns = [str(v) for v in (columns or [])]
    rows = rows or []
    rows_s = [[("" if v is None else str(v)) for v in r] for r in rows]

    ncol = max(1, len(columns))
    nrow = 1 + len(rows_s)  # header + righe

    col_w = w / ncol
    row_h = h / max(1, nrow)

    # box + griglia
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(x, y, w, h, stroke=1, fill=0)

    for i in range(1, ncol):
        c.line(x + i * col_w, y, x + i * col_w, y + h)
    for j in range(1, nrow):
        c.line(x, y + j * row_h, x + w, y + j * row_h)

    # header background (leggero)
    c.setFillColor(colors.whitesmoke)
    c.rect(x, y + h - row_h, w, row_h, stroke=0, fill=1)
    c.setFillColor(colors.black)

    # testo
    c.setFont(FONT_BOLD, font_size)
    for ci in range(ncol):
        txt = columns[ci] if ci < len(columns) else ""
        c.drawString(x + ci * col_w + 2, y + h - row_h + 2, txt[:80])

    c.setFont(FONT_REG, font_size)
    for ri, r in enumerate(rows_s):
        yy = y + h - (ri + 2) * row_h + 2
        for ci in range(ncol):
            txt = r[ci] if ci < len(r) else ""
            c.drawString(x + ci * col_w + 2, yy, txt[:80])


# ============================================================
# PDF generation (layout richiesto)
# ============================================================
@dataclass
class TamponamentoItem:
    pid: str
    B_cm: float
    H_cm: float
    plot_png: bytes
    table_columns: List[str]
    table_rows: List[List[Any]]


def build_report_pdf_bytes(
    *,
    header_lines: List[str],
    schema_png: bytes,
    tamponamenti: List[TamponamentoItem],
    title_first_page: str = "Schema di posa Resisto 5.9",
) -> bytes:
    W, H = A4
    mem = BytesIO()
    c = canvas.Canvas(mem, pagesize=A4)

    # ====== PAGINA 1: SOLO schema + titolo ======
    c.setFont(FONT_BOLD, 14)
    c.drawCentredString(W / 2, H - 0.80 * cm, title_first_page)

    y_after_header = _draw_header_lines(c, W, H, header_lines=header_lines)

    footer_space = 2.1 * cm
    bottom_limit = footer_space + 0.55 * cm
    top_limit = y_after_header - 0.50 * cm

    avail_w = W - 1.4 * cm
    avail_h = max(10.0, top_limit - bottom_limit)

    _draw_png_centered(
        c,
        schema_png,
        x=(W - avail_w) / 2,
        y=bottom_limit,
        box_w=avail_w,
        box_h=avail_h,
    )

    _footer(c, W, H)
    c.showPage()

    # ====== PAGINE SUCCESSIVE: 2 per pagina ======
    # Layout: due slot verticali, ciascuno:
    # - titolo
    # - immagine curva
    # - tabella sotto
    slot_gap = 0.6 * cm
    top_margin = 1.2 * cm
    bottom_margin = 2.2 * cm  # include footer area
    side_margin = 1.6 * cm

    usable_h = H - top_margin - bottom_margin
    slot_h = (usable_h - slot_gap) / 2.0
    slot_w = W - 2 * side_margin

    # proporzioni interne slot
    title_h = 0.85 * cm
    table_h = 4.3 * cm
    img_h = max(2.0 * cm, slot_h - title_h - table_h - 0.3 * cm)

    def draw_slot(item: TamponamentoItem, x0: float, y0: float):
        # y0 = bottom dello slot
        # Titolo
        c.setFont(FONT_BOLD, 11)
        title = f"Tamponamento: {item.pid} | B: {item.B_cm:.0f} cm - H: {item.H_cm:.0f} cm | Curve non lineari equivalenti"
        c.drawString(x0, y0 + slot_h - title_h + 0.15 * cm, title[:200])

        # Immagine curva
        img_y = y0 + table_h + 0.10 * cm
        _draw_png_centered(
            c,
            item.plot_png,
            x=x0,
            y=img_y,
            box_w=slot_w,
            box_h=img_h,
        )

        # Tabella
        table_y = y0 + 0.10 * cm
        _draw_table(
            c,
            x=x0,
            y=table_y,
            w=slot_w,
            h=table_h - 0.15 * cm,
            columns=item.table_columns,
            rows=item.table_rows,
            font_size=8,
        )

    for idx, t in enumerate(tamponamenti):
        # nuova pagina ogni 2
        if idx % 2 == 0:
            # header pagina
            y_after_header = _draw_header_lines(c, W, H, header_lines=header_lines)

        slot_index = idx % 2
        # slot 0 = alto, slot 1 = basso
        if slot_index == 0:
            y_slot_bottom = bottom_margin + slot_h + slot_gap
        else:
            y_slot_bottom = bottom_margin

        draw_slot(t, x0=side_margin, y0=y_slot_bottom)

        if slot_index == 1 or idx == len(tamponamenti) - 1:
            _footer(c, W, H)
            c.showPage()

    c.save()
    mem.seek(0)
    return mem.getvalue()


# ============================================================
# MAIN endpoint
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/export")
def export(
    payload: Payload,
    schema_scale: float = Query(default=1.0, ge=0.2, le=5.0),  # tenuto per compatibilità URL (non usato qui)
):
    """
    Ritorna uno ZIP binario scaricabile (perfetto per n8n HTTP Request -> Response: File).

    Nome file zip:
      <job_id>_allegati.zip

    Dentro:
      Report_resisto59_completo.pdf
    """
    try:
        raw = dict(payload.root)
        data = _load_body_from_any(raw)

        meta = data.get("meta") or {}
        project_name = meta.get("project_name", "")
        location_name = meta.get("location_name", "")
        wall_orientation_raw = meta.get("wall_orientation", "")
        suffix = meta.get("suffix", "")

        _must(project_name.strip() != "", "meta.project_name mancante o vuoto")
        _must(location_name.strip() != "", "meta.location_name mancante o vuoto")
        _must(wall_orientation_raw.strip() != "", "meta.wall_orientation mancante o vuoto")
        _must(suffix.strip() != "", "meta.suffix mancante o vuoto")

        job_id, wall_orientation = make_job_id(project_name, location_name, wall_orientation_raw, suffix)

        header_lines = [
            f"Progetto: {project_name}",
            f"Posizione: {location_name}",
            f"Parete: {wall_orientation} | Revisione: {suffix}",
        ]

        schema_block = data.get("schema") or {}
        schema_png_b64 = schema_block.get("png_base64", "")
        schema_png_bytes = _b64_to_bytes(schema_png_b64)

        t_list = data.get("tamponamenti") or []
        tamponamenti: List[TamponamentoItem] = []
        for t in t_list:
            pid = str(t.get("pid", "")).strip()
            _must(pid != "", "tamponamenti[].pid mancante/vuoto")

            B_cm = float(t.get("B_cm", 0.0))
            H_cm = float(t.get("H_cm", 0.0))
            _must(B_cm > 0 and H_cm > 0, f"tamponamenti[{pid}]: B_cm/H_cm non validi")

            plot_b64 = t.get("plot_png_base64", "")
            plot_bytes = _b64_to_bytes(plot_b64)

            table = t.get("table") or {}
            columns = table.get("columns") or []
            rows = table.get("rows") or []

            tamponamenti.append(
                TamponamentoItem(
                    pid=pid,
                    B_cm=B_cm,
                    H_cm=H_cm,
                    plot_png=plot_bytes,
                    table_columns=list(columns),
                    table_rows=list(rows),
                )
            )

        _must(len(tamponamenti) > 0, "tamponamenti vuoto: niente pagine da generare dopo lo schema")

        pdf_bytes = build_report_pdf_bytes(
            header_lines=header_lines,
            schema_png=schema_png_bytes,
            tamponamenti=tamponamenti,
            title_first_page="Schema di posa Resisto 5.9",
        )

        # ZIP in memoria
        zip_filename = f"{job_id}_allegati.zip"
        pdf_name_in_zip = "Report_resisto59_completo.pdf"

        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(pdf_name_in_zip, pdf_bytes)

        mem_zip.seek(0)

        return StreamingResponse(
            mem_zip,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
