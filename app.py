# app.py — Resisto 5.9 PDF+ZIP export (n8n-ready, immagini generate da Python)
#
# Input: JSON come quello che mi hai incollato (overlays + panels + nl.curves)
# Output: ZIP binario con Report_resisto59_completo.pdf
#
# FastAPI:
#   GET  /health
#   POST /export?overlay_id=grid
#
from __future__ import annotations

import re
import base64
import zipfile
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import RootModel

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# Plots (server-side)
import matplotlib
matplotlib.use("Agg")  # importantissimo in server/headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(title="Resisto 5.9 Export", version="2.0.0")


class Payload(RootModel[Dict[str, Any]]):
    pass


# ============================================================
# CONFIG
# ============================================================
FONT_REG, FONT_BOLD = "Helvetica", "Helvetica-Bold"


# ============================================================
# UTILS — n8n body handling
# ============================================================
def _load_body_from_any(obj: Any) -> Dict[str, Any]:
    """
    Accetta:
      - {"body": {...}}
      - {"json": {...}}
      - [{"body": {...}}]
      - [{"json": {...}}]
      - direttamente il dict "body"
    """
    if isinstance(obj, list) and obj:
        obj = obj[0]

    if isinstance(obj, dict) and "body" in obj and isinstance(obj["body"], dict):
        return obj["body"]

    if isinstance(obj, dict) and "json" in obj and isinstance(obj["json"], dict):
        return obj["json"]

    if isinstance(obj, dict):
        return obj

    raise ValueError("Body JSON non valido (atteso dict oppure array n8n-style).")


# ============================================================
# UTILS — naming (job_id)
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
    _must(re.fullmatch(r"[A-Za-z0-9._-]+", s) is not None,
          "meta.suffix non valido (usa solo A-Z a-z 0-9 . _ -)")
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


# ============================================================
# PLOT — schema da overlays
# ============================================================
def _get_overlay(body: dict, overlay_id: str) -> dict:
    for o in body.get("overlays", []) or []:
        if o.get("id") == overlay_id:
            return o
    raise ValueError(f"Overlay '{overlay_id}' non trovato. Disponibili: {[o.get('id') for o in body.get('overlays',[])]}")


def plot_schema_to_png_bytes(body: dict, overlay_id: str = "grid", figsize=(11, 8)) -> bytes:
    overlay = _get_overlay(body, overlay_id)
    entities = overlay.get("entities", []) or []

    bbox = body.get("wall_bbox") or {}
    xmin = float(bbox.get("xmin", 0.0))
    ymin = float(bbox.get("ymin", 0.0))
    xmax = float(bbox.get("xmax", 1.0))
    ymax = float(bbox.get("ymax", 1.0))

    fig, ax = plt.subplots(figsize=figsize)

    # Disegno entities (line + text)
    for e in entities:
        t = e.get("type")
        if t == "line":
            a = e.get("a", [0, 0])
            b = e.get("b", [0, 0])
            style = e.get("style") or {}
            ax.plot([float(a[0]), float(b[0])], [float(a[1]), float(b[1])],
                    linewidth=float(style.get("width", 1.0)))
        elif t == "text":
            pos = e.get("pos", [0, 0])
            txt = str(e.get("text", ""))
            style = e.get("style") or {}
            ax.text(float(pos[0]), float(pos[1]), txt, fontsize=float(style.get("size", 10)))

    # Label panel_id al centro pannello (se presenti)
    for p in body.get("panels", []) or []:
        b = p.get("bounds") or {}
        cx = 0.5 * (float(b.get("xmin", 0)) + float(b.get("xmax", 0)))
        cy = 0.5 * (float(b.get("ymin", 0)) + float(b.get("ymax", 0)))
        pid = p.get("panel_id", f"{p.get('i','?')},{p.get('j','?')}")
        ax.text(cx, cy, str(pid), ha="center", va="center",
                fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65))

    ax.set_xlim(xmin - 10, xmax + 10)
    ax.set_ylim(ymin - 10, ymax + 10)
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.grid(True, alpha=0.25)
    ax.set_title("Schema di posa Resisto 5.9")
    fig.tight_layout()

    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=300)
    plt.close(fig)
    return bio.getvalue()


# ============================================================
# PLOT — curve NL per panel (immagine) + tabella (dati)
# ============================================================
def _get_points(panel: dict, curve_name: str) -> Tuple[List[float], List[float]]:
    curves = (((panel.get("nl") or {}).get("curves")) or {})
    c = curves.get(curve_name, {}) or {}
    pts = c.get("points", []) or []
    xs, ys = [], []
    for p in pts:
        if "x" in p and "y" in p:
            xs.append(float(p["x"]))
            ys.append(float(p["y"]) / 1000.0)  # N -> kN
        elif "d" in p and "F" in p:
            xs.append(float(p["d"]))
            ys.append(float(p["F"]) / 1000.0)  # N -> kN
    return xs, ys


def _panel_BH_cm(panel: dict) -> Tuple[float, float]:
    if panel.get("width_cm") is not None and panel.get("height_cm") is not None:
        return float(panel["width_cm"]), float(panel["height_cm"])
    b = panel.get("bounds", {}) or {}
    return float(b.get("xmax", 0) - b.get("xmin", 0)), float(b.get("ymax", 0) - b.get("ymin", 0))


def _fmt(v: Any, nd: int = 1) -> str:
    try:
        return f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return ""


def _kn(v: Any) -> Optional[float]:
    try:
        return float(v) / 1000.0
    except Exception:
        return None


def nl_table_from_final_reduced(panel: dict) -> Tuple[List[str], List[List[Any]]]:
    fr = ((((panel.get("nl") or {}).get("curves")) or {}).get("final_reduced") or {})

    Fy_T = fr.get("Fy_T"); dy_T = fr.get("dy_T")
    Fu_T = fr.get("Fu_T"); du_T = fr.get("du_T")
    Fy_C = fr.get("Fy_C"); dy_C = fr.get("dy_C")
    Fu_C = fr.get("Fu_C"); du_C = fr.get("du_C")

    # Nota: in compressione li metto negativi per coerenza con la tabella del tuo jupyter
    rows = [
        ["Fu_T [kN]", _fmt(_kn(Fu_T), 1), "du_T [mm]", _fmt(du_T, 1)],
        ["Fy_T [kN]", _fmt(_kn(Fy_T), 1), "dy_T [mm]", _fmt(dy_T, 1)],
        ["0 [kN]",    "0",              "0 [mm]",    "0"],
        ["Fy_C [kN]", _fmt(_kn(-Fy_C), 1), "dy_C [mm]", _fmt(-dy_C, 1)],
        ["Fu_C [kN]", _fmt(_kn(-Fu_C), 1), "du_C [mm]", _fmt(-du_C, 1)],
    ]
    columns = ["", "Forza [kN]", "", "Spostamento [mm]"]
    return columns, rows


def plot_nl_panel_to_png_bytes(panel: dict) -> bytes:
    pid = panel.get("panel_id", f"{panel.get('i','?')},{panel.get('j','?')}")
    B_cm, H_cm = _panel_BH_cm(panel)

    # 4 curve
    x_low, y_low = _get_points(panel, "abaco_low")
    x_high, y_high = _get_points(panel, "abaco_high")
    x_interp, y_interp = _get_points(panel, "interp_step")
    x_final, y_final = _get_points(panel, "final_reduced")

    if not x_final:
        # niente curve => png vuoto (ma meglio bloccare prima)
        raise ValueError(f"panel {pid}: final_reduced points mancanti")

    # limiti
    all_x = x_low + x_high + x_interp + x_final
    all_y = y_low + y_high + y_interp + y_final
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    dx = (xmax - xmin) if xmax > xmin else 1.0
    dy = (ymax - ymin) if ymax > ymin else 1.0
    xmin -= 0.08 * dx
    xmax += 0.08 * dx
    ymin -= 0.10 * dy
    ymax += 0.10 * dy

    fig, ax = plt.subplots(figsize=(10, 6.0))

    if x_low:   ax.plot(x_low,   y_low,   label="abaco_low")
    if x_high:  ax.plot(x_high,  y_high,  label="abaco_high")
    if x_interp:ax.plot(x_interp,y_interp,label="interp_step")
    ax.plot(x_final, y_final, linewidth=2.2, label="final_reduced")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Spostamento d [mm]")
    ax.set_ylabel("Forza F [kN]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.95)

    ax.set_title(f"Tamponamento: {pid} | B: {B_cm:.0f} cm - H: {H_cm:.0f} cm | Curve non lineari equivalenti")
    fig.tight_layout()

    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=300)
    plt.close(fig)
    return bio.getvalue()


# ============================================================
# PDF helpers (header/footer + image fit + table)
# ============================================================
def _footer(c: canvas.Canvas, W: float, H: float):
    h5 = H / 15
    c.setFont(FONT_REG, 11)
    c.drawCentredString(W / 2, 0.75 * h5, "Ing. Alessandro Bordini")
    c.drawCentredString(W / 2, 0.35 * h5, "Phone: 3451604706 - ✉: alessandro_bordini@outlook.com")


def _draw_header_lines(c: canvas.Canvas, W: float, H: float, header_lines: List[str]) -> float:
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


def _draw_png_centered(c: canvas.Canvas, img_bytes: bytes, x: float, y: float, box_w: float, box_h: float):
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
    columns = [str(v) for v in (columns or [])]
    rows = rows or []
    rows_s = [[("" if v is None else str(v)) for v in r] for r in rows]

    ncol = max(1, len(columns))
    nrow = 1 + len(rows_s)

    col_w = w / ncol
    row_h = h / max(1, nrow)

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(x, y, w, h, stroke=1, fill=0)

    for i in range(1, ncol):
        c.line(x + i * col_w, y, x + i * col_w, y + h)
    for j in range(1, nrow):
        c.line(x, y + j * row_h, x + w, y + j * row_h)

    c.setFillColor(colors.whitesmoke)
    c.rect(x, y + h - row_h, w, row_h, stroke=0, fill=1)
    c.setFillColor(colors.black)

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


def build_report_pdf_bytes(*, header_lines: List[str], schema_png: bytes, tamponamenti: List[TamponamentoItem]) -> bytes:
    W, H = A4
    mem = BytesIO()
    c = canvas.Canvas(mem, pagesize=A4)

    # ====== PAGINA 1: SOLO schema + titolo ======
    c.setFont(FONT_BOLD, 14)
    c.drawCentredString(W / 2, H - 0.80 * cm, "Schema di posa Resisto 5.9")

    y_after_header = _draw_header_lines(c, W, H, header_lines=header_lines)

    footer_space = 2.1 * cm
    bottom_limit = footer_space + 0.55 * cm
    top_limit = y_after_header - 0.50 * cm

    avail_w = W - 1.4 * cm
    avail_h = max(10.0, top_limit - bottom_limit)

    _draw_png_centered(c, schema_png, x=(W - avail_w) / 2, y=bottom_limit, box_w=avail_w, box_h=avail_h)

    _footer(c, W, H)
    c.showPage()

    # ====== PAGINE SUCCESSIVE: 2 tamponamenti per pagina ======
    slot_gap = 0.6 * cm
    top_margin = 1.2 * cm
    bottom_margin = 2.2 * cm
    side_margin = 1.6 * cm

    usable_h = H - top_margin - bottom_margin
    slot_h = (usable_h - slot_gap) / 2.0
    slot_w = W - 2 * side_margin

    title_h = 0.85 * cm
    table_h = 4.3 * cm
    img_h = max(2.0 * cm, slot_h - title_h - table_h - 0.3 * cm)

    def draw_slot(item: TamponamentoItem, x0: float, y0: float):
        c.setFont(FONT_BOLD, 11)
        title = f"Tamponamento: {item.pid} | B: {item.B_cm:.0f} cm - H: {item.H_cm:.0f} cm | Curve non lineari equivalenti"
        c.drawString(x0, y0 + slot_h - title_h + 0.15 * cm, title[:200])

        img_y = y0 + table_h + 0.10 * cm
        _draw_png_centered(c, item.plot_png, x=x0, y=img_y, box_w=slot_w, box_h=img_h)

        table_y = y0 + 0.10 * cm
        _draw_table(c, x=x0, y=table_y, w=slot_w, h=table_h - 0.15 * cm,
                    columns=item.table_columns, rows=item.table_rows, font_size=8)

    for idx, t in enumerate(tamponamenti):
        if idx % 2 == 0:
            _draw_header_lines(c, W, H, header_lines=header_lines)

        slot_index = idx % 2
        y_slot_bottom = (bottom_margin + slot_h + slot_gap) if slot_index == 0 else bottom_margin
        draw_slot(t, x0=side_margin, y0=y_slot_bottom)

        if slot_index == 1 or idx == len(tamponamenti) - 1:
            _footer(c, W, H)
            c.showPage()

    c.save()
    mem.seek(0)
    return mem.getvalue()


# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/export")
def export(
    payload: Payload,
    overlay_id: str = Query(default="grid"),
):
    """
    Ritorna uno ZIP binario scaricabile (n8n HTTP Request -> Response: File).

    Dentro lo ZIP:
      Report_resisto59_completo.pdf
    """
    try:
        raw = dict(payload.root)
        body = _load_body_from_any(raw)

        meta = body.get("meta") or {}
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

        # 1) schema png bytes (GENERATO)
        schema_png = plot_schema_to_png_bytes(body, overlay_id=overlay_id, figsize=(11, 8))

        # 2) tamponamenti: png curve + tabella (GENERATI)
        panels = body.get("panels") or []
        _must(len(panels) > 0, "panels vuoto: niente tamponamenti da inserire nel PDF")

        tamponamenti: List[TamponamentoItem] = []
        for p in panels:
            pid = p.get("panel_id", f"{p.get('i','?')},{p.get('j','?')}")
            B_cm, H_cm = _panel_BH_cm(p)

            # curva
            plot_png = plot_nl_panel_to_png_bytes(p)
            # tabella
            tcols, trows = nl_table_from_final_reduced(p)

            tamponamenti.append(
                TamponamentoItem(
                    pid=str(pid),
                    B_cm=float(B_cm),
                    H_cm=float(H_cm),
                    plot_png=plot_png,
                    table_columns=tcols,
                    table_rows=trows,
                )
            )

        # 3) PDF bytes
        pdf_bytes = build_report_pdf_bytes(
            header_lines=header_lines,
            schema_png=schema_png,
            tamponamenti=tamponamenti,
        )

        # 4) ZIP in memoria
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
