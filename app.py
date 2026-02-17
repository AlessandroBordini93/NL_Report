# app.py — Resisto 5.9 PDF+ZIP export (n8n-ready)
#
# ✅ Aggiornamenti richiesti:
# 1) Schema: IDENTICO al tuo Jupyter:
#    - travi grigio chiaro (#d0d0d0)
#    - pilastri grigio scuro (#a0a0a0)
#    - aperture blu (rettangolo, lw=1.4)
#    - rinforzo/overlay linee NERE (forzato per grid/grid_full)
#    - label panel_id al centro
# 2) PDF: header SOLO in prima pagina; dalla seconda in poi niente header.
# 3) ZIP: contiene:
#      - Report_resisto59_completo.pdf
#      - schema_posa_resisto59.dxf
#
# Endpoint:
#   GET  /health
#   POST /export?overlay_id=grid&schema_scale=1.0
#
# NOTE:
# - Richiede "ezdxf" installato per esportare il DXF.
# - I grafici NL restano identici al tuo Jupyter (assi + tabella + Ke_T/Ke_C + ticks blu).

from __future__ import annotations

import re
import zipfile
from io import BytesIO, StringIO
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import RootModel

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# Plots (server-side)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# DXF (optional)
try:
    import ezdxf
except ImportError:
    ezdxf = None


# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(title="Resisto 5.9 Export", version="2.2.1")


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


# ============================================================
# SCHEMA — IDENTICO JUPYTER (travi/pilastri grigi + aperture blu + rinforzo nero + panel_id)
# ============================================================
def _cum_coords(interassi_cm: List[float]) -> List[float]:
    out = [0.0]
    for v in interassi_cm or []:
        out.append(out[-1] + float(v))
    return out


def _grid_axes_from_body(body: dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    g = body.get("grid", {}) or {}
    beams = g.get("beams", {}) or {}
    cols = g.get("columns", {}) or {}

    x = _cum_coords(cols.get("interassi_cm", []) or [])
    y = _cum_coords(beams.get("interassi_cm", []) or [])

    spB = list(map(float, beams.get("spessori_cm", []) or []))  # travi (orizzontali) per ogni asse y
    spC = list(map(float, cols.get("spessori_cm", []) or []))   # pilastri (verticali) per ogni asse x

    if not spB:
        spB = [30.0] * max(1, len(y))
    if not spC:
        spC = [30.0] * max(1, len(x))

    if len(spB) != len(y):
        spB = (spB[:len(y)] + [spB[-1]] * max(0, len(y) - len(spB)))
    if len(spC) != len(x):
        spC = (spC[:len(x)] + [spC[-1]] * max(0, len(x) - len(spC)))

    return x, y, spB, spC


def _get_overlay(body: dict, overlay_id: str) -> dict:
    for o in body.get("overlays", []) or []:
        if o.get("id") == overlay_id:
            return o
    raise ValueError(
        f"Overlay '{overlay_id}' non trovato. Disponibili: {[o.get('id') for o in body.get('overlays',[])]}"
    )


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def plot_schema_to_png_bytes(body: dict, overlay_id: str = "grid", figsize=(11, 8)) -> bytes:
    x, y, spB, spC = _grid_axes_from_body(body)
    ov = _get_overlay(body, overlay_id)

    wall = body.get("wall_bbox", {}) or {}
    xmin = _safe_float(wall.get("xmin", min(x) if x else 0.0))
    ymin = _safe_float(wall.get("ymin", min(y) if y else 0.0))
    xmax = _safe_float(wall.get("xmax", max(x) if x else 0.0))
    ymax = _safe_float(wall.get("ymax", max(y) if y else 0.0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # struttura grigia
    if x and y:
        x_min = x[0] - spC[0] / 2
        x_max = x[-1] + spC[-1] / 2
        for yy, sp in zip(y, spB):
            ax.add_patch(Rectangle((x_min, yy - sp / 2), x_max - x_min, sp,
                                   facecolor="#d0d0d0", edgecolor="none", zorder=0))

        y_min = y[0] - spB[0] / 2
        y_max = y[-1] + spB[-1] / 2
        for xx, sp in zip(x, spC):
            ax.add_patch(Rectangle((xx - sp / 2, y_min), sp, y_max - y_min,
                                   facecolor="#a0a0a0", edgecolor="none", zorder=0))

    # aperture blu
    for p in body.get("panels", []) or []:
        for w in (p.get("openings") or []):
            ax.add_patch(Rectangle((float(w["x"]), float(w["y"])),
                                   float(w["w"]), float(w["h"]),
                                   fill=False, edgecolor="blue", linewidth=1.4, zorder=1))

    # overlay (rinforzo nero forzato per grid/grid_full)
    force_black = overlay_id in {"grid", "grid_full"}
    for e in ov.get("entities", []) or []:
        if e.get("type") != "line":
            continue
        (x1, y1) = e.get("a", [0, 0])
        (x2, y2) = e.get("b", [0, 0])
        st = e.get("style", {}) or {}
        color = "black" if force_black else st.get("stroke", "#111")
        lw = float(st.get("width", 1.0))
        dash = st.get("dash", []) or []
        z = 3 if force_black else 4

        ln, = ax.plot([float(x1), float(x2)], [float(y1), float(y2)],
                      color=color, linewidth=lw, zorder=z)
        if dash:
            ln.set_dashes(dash)

    # label panel_id
    for p in body.get("panels", []) or []:
        b = p.get("bounds", {}) or {}
        cx = (float(b.get("xmin", 0)) + float(b.get("xmax", 0))) / 2.0
        cy = (float(b.get("ymin", 0)) + float(b.get("ymax", 0))) / 2.0
        pid = p.get("panel_id", f"{p.get('i','?')},{p.get('j','?')}")
        ax.text(cx, cy, str(pid), ha="center", va="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.65),
                zorder=10)

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
    bio.seek(0)
    return bio.getvalue()


# ============================================================
# CURVE NL — IDENTICHE AL TUO JUPYTER (in PNG bytes)
# ============================================================
def _get_points(panel: dict, curve_name: str) -> Tuple[List[float], List[float]]:
    curves = (((panel.get("nl") or {}).get("curves")) or {})
    c = curves.get(curve_name, {}) or {}
    pts = c.get("points", []) or []
    xs, ys = [], []
    for p in pts:
        if "x" in p and "y" in p:
            x = float(p["x"])
            y = float(p["y"]) / 1000.0
        elif "d" in p and "F" in p:
            x = float(p["d"])
            y = float(p["F"]) / 1000.0
        else:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _panel_BH_cm(panel: dict) -> Tuple[float, float]:
    if panel.get("width_cm") is not None and panel.get("height_cm") is not None:
        return float(panel["width_cm"]), float(panel["height_cm"])
    b = panel.get("bounds", {}) or {}
    return float(b.get("xmax", 0) - b.get("xmin", 0)), float(b.get("ymax", 0) - b.get("ymin", 0))


def _set_ticks_only_final(ax, x_final: List[float], y_final: List[float]) -> None:
    xt = sorted(set([round(float(v), 6) for v in x_final]))
    yt = sorted(set([round(float(v), 6) for v in y_final]))
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) >= 1e-12 else "0" for v in xt],
                       color="blue", fontweight="bold")
    ax.set_yticklabels([f"{v:.1f}".rstrip("0").rstrip(".") if abs(v) >= 1e-12 else "0" for v in yt],
                       color="blue", fontweight="bold")


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return ""


def _kn(v: Any) -> Optional[float]:
    try:
        return float(v) / 1000.0
    except Exception:
        return None


def plot_nl_panel_to_png_bytes(panel: dict) -> bytes:
    pid = panel.get("panel_id", f"{panel.get('i','?')},{panel.get('j','?')}")
    B_cm, H_cm = _panel_BH_cm(panel)

    nl = panel.get("nl", {}) or {}
    passo_low = nl.get("passo_low", None)
    passo_high = nl.get("passo_high", None)
    avg_passo_mm = nl.get("avg_passo_mm", None)
    ratio = nl.get("ratio", None)

    x_low, y_low = _get_points(panel, "abaco_low")
    x_high, y_high = _get_points(panel, "abaco_high")
    x_interp, y_interp = _get_points(panel, "interp_step")
    x_final, y_final = _get_points(panel, "final_reduced")

    if not x_final:
        raise ValueError(f"panel {pid}: final_reduced points mancanti")

    all_x, all_y = [], []
    for xs, ys in [(x_low, y_low), (x_high, y_high), (x_interp, y_interp), (x_final, y_final)]:
        all_x += xs
        all_y += ys

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    dx = (xmax - xmin) if xmax > xmin else 1.0
    dy = (ymax - ymin) if ymax > ymin else 1.0
    xmin -= 0.08 * dx
    xmax += 0.08 * dx
    ymin -= 0.10 * dy
    ymax += 0.10 * dy

    fig = plt.figure(figsize=(10, 8.2))
    ax = fig.add_axes([0.08, 0.36, 0.88, 0.60])
    ax_tbl = fig.add_axes([0.08, 0.08, 0.88, 0.22])
    ax_tbl.axis("off")

    ax.grid(True, alpha=0.25)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if x_low:
        ln_low, = ax.plot(x_low, y_low, color="black", linewidth=1.0, linestyle="--",
                          marker="o", markersize=3, label=f"Abaco precedente {passo_low} mm",
                          zorder=1, alpha=0.80)
        ln_low.set_dashes([6, 4])

    if x_high:
        ln_high, = ax.plot(x_high, y_high, color="red", linewidth=1.0, linestyle="--",
                           marker="o", markersize=3, label=f"Abaco successivo {passo_high} mm",
                           zorder=2, alpha=0.85)
        ln_high.set_dashes([6, 4])

    if x_interp:
        ln_int, = ax.plot(x_interp, y_interp, color="green", linewidth=1.8, linestyle="--",
                          marker="o", markersize=3, label=f"Interpolazione al passo {avg_passo_mm} mm",
                          zorder=3, alpha=1.0)
        ln_int.set_dashes([10, 2])

    ax.plot(x_final, y_final, linestyle="-", color="blue", linewidth=6.0,
            marker="o", markersize=5, label=f"Curva da calcolo (ratio = {ratio})",
            zorder=4)

    _set_ticks_only_final(ax, x_final, y_final)

    ax.set_xlabel("Spostamento [mm]")
    ax.set_ylabel("Forza [kN]")
    ax.set_title(f"Tamponamento: {pid} | B: {B_cm:.0f} cm - H: {H_cm:.0f} cm | Curve non lineari equivalenti")

    fr = (((nl.get("curves") or {}).get("final_reduced")) or {})
    ke_t = fr.get("Ke_T", None)
    ke_c = fr.get("Ke_C", None)

    bbox_kw = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black",
                   linewidth=0.8, alpha=0.95)

    x_pos = 0.06 * (xmax - xmin)
    y_pos = 0.06 * (ymax - ymin)
    x_neg = -0.28 * (xmax - xmin)
    y_neg = -0.10 * (ymax - ymin)

    if ke_t is not None:
        ke_t_kn = float(ke_t) / 1000.0
        ax.text(0 + x_pos, 0 + y_pos, f"Ke_T = {ke_t_kn:.1f} kN/mm",
                ha="left", va="bottom", fontsize=10, fontweight="bold",
                color="blue", bbox=bbox_kw, zorder=20)

    if ke_c is not None:
        ke_c_kn = float(ke_c) / 1000.0
        ax.text(0 + x_neg, 0 + y_neg, f"Ke_C = {ke_c_kn:.1f} kN/mm",
                ha="left", va="top", fontsize=10, fontweight="bold",
                color="blue", bbox=bbox_kw, zorder=20)

    ax.legend(loc="upper left", framealpha=0.95)

    Fy_T = fr.get("Fy_T", None); dy_T = fr.get("dy_T", None)
    Fu_T = fr.get("Fu_T", None); du_T = fr.get("du_T", None)
    Fy_C = fr.get("Fy_C", None); dy_C = fr.get("dy_C", None)
    Fu_C = fr.get("Fu_C", None); du_C = fr.get("du_C", None)

    table_rows = [
        ["Fu_T [kN]", _fmt(_kn(Fu_T), 1), "du_T [mm]", _fmt(du_T, 1)],
        ["Fy_T [kN]", _fmt(_kn(Fy_T), 1), "dy_T [mm]", _fmt(dy_T, 1)],
        ["0 [kN]",    "0",                "0 [mm]",    "0"],
        ["Fy_C [kN]", _fmt(_kn(-Fy_C), 1), "dy_C [mm]", _fmt(-dy_C, 1)],
        ["Fu_C [kN]", _fmt(_kn(-Fu_C), 1), "du_C [mm]", _fmt(-du_C, 1)],
    ]

    if any(r[1] == "" or r[3] == "" for r in table_rows):
        if len(x_final) == 5 and len(y_final) == 5:
            table_rows = [
                ["Fu_T [kN]", _fmt(y_final[4], 1), "du_T [mm]", _fmt(x_final[4], 1)],
                ["Fy_T [kN]", _fmt(y_final[3], 1), "dy_T [mm]", _fmt(x_final[3], 1)],
                ["0 [kN]",    "0",                "0 [mm]",    "0"],
                ["Fy_C [kN]", _fmt(y_final[1], 1), "dy_C [mm]", _fmt(x_final[1], 1)],
                ["Fu_C [kN]", _fmt(y_final[0], 1), "du_C [mm]", _fmt(x_final[0], 1)],
            ]

    tbl = ax_tbl.table(
        cellText=table_rows,
        colLabels=["", "Forza [kN]", "", "Spostamento [mm]"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f2f2f2")
        else:
            if c in (0, 2):
                cell.set_text_props(fontweight="bold")

    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=300)
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()


# ============================================================
# DXF EXPORT (schema_posa_resisto59.dxf) — FIX: stream testuale -> bytes UTF-8
# ============================================================
def export_schema_dxf_bytes(body: dict, overlay_id: str = "grid") -> bytes:
    if ezdxf is None:
        raise ValueError("DXF export richiede 'ezdxf' installato (pip install ezdxf).")

    x, y, spB, spC = _grid_axes_from_body(body)
    _must(x and y, "grid mancante o incompleto: impossibile costruire struttura DXF")

    ov = _get_overlay(body, overlay_id)

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # layers
    if "Struttura" not in doc.layers:
        doc.layers.new("Struttura", dxfattribs={"color": 1})
    if "Resisto" not in doc.layers:
        doc.layers.new("Resisto", dxfattribs={"color": 7})

    def rect(x1, y1, x2, y2, layer="Struttura"):
        msp.add_lwpolyline(
            [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)],
            dxfattribs={"layer": layer},
        )

    # pilastri verticali
    y_min = y[0] - spB[0] / 2
    y_max = y[-1] + spB[-1] / 2
    for xx, sp in zip(x, spC):
        rect(xx - sp / 2, y_min, xx + sp / 2, y_max, layer="Struttura")

    # travi orizzontali
    x_min = x[0] - spC[0] / 2
    x_max = x[-1] + spC[-1] / 2
    for yy, sp in zip(y, spB):
        rect(x_min, yy - sp / 2, x_max, yy + sp / 2, layer="Struttura")

    # aperture
    for p in body.get("panels", []) or []:
        for w in (p.get("openings") or []):
            rect(
                float(w["x"]),
                float(w["y"]),
                float(w["x"]) + float(w["w"]),
                float(w["y"]) + float(w["h"]),
                layer="Struttura",
            )

    # rinforzo overlay (linee) su layer Resisto
    for e in ov.get("entities", []) or []:
        if e.get("type") != "line":
            continue
        a = e.get("a", [0, 0])
        b = e.get("b", [0, 0])
        msp.add_line(
            (float(a[0]), float(a[1])),
            (float(b[0]), float(b[1])),
            dxfattribs={"layer": "Resisto"},
        )

    # ✅ write DXF as TEXT, then encode -> bytes (compatibile con writestr)
    sio = StringIO()
    doc.write(sio)
    txt = sio.getvalue()
    return txt.encode("utf-8")


# ============================================================
# PDF helpers
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


# ============================================================
# PDF generation
# ============================================================
@dataclass
class TamponamentoItem:
    pid: str
    B_cm: float
    H_cm: float
    plot_png: bytes


def build_report_pdf_bytes(*, header_lines: List[str], schema_png: bytes, tamponamenti: List[TamponamentoItem]) -> bytes:
    W, H = A4
    mem = BytesIO()
    c = canvas.Canvas(mem, pagesize=A4)

    # PAGINA 1: schema + header + footer
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

    # PAGINE SUCCESSIVE: 2 tamponamenti per pagina, NO HEADER
    slot_gap = 0.6 * cm
    top_margin = 1.2 * cm
    bottom_margin = 2.2 * cm
    side_margin = 1.6 * cm

    usable_h = H - top_margin - bottom_margin
    slot_h = (usable_h - slot_gap) / 2.0
    slot_w = W - 2 * side_margin

    img_h = max(2.0 * cm, slot_h)

    def draw_slot(item: TamponamentoItem, x0: float, y0: float):
        _draw_png_centered(c, item.plot_png, x=x0, y=y0, box_w=slot_w, box_h=img_h)

    for idx, t in enumerate(tamponamenti):
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
    schema_scale: float = Query(default=1.0, ge=0.2, le=5.0),
):
    """
    Ritorna uno ZIP binario scaricabile (n8n HTTP Request -> Response: File).

    Dentro lo ZIP:
      - Report_resisto59_completo.pdf
      - schema_posa_resisto59.dxf
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

        # 1) schema png (GENERATO, stile Jupyter)
        schema_png = plot_schema_to_png_bytes(
            body,
            overlay_id=overlay_id,
            figsize=(11 * float(schema_scale), 8 * float(schema_scale)),
        )

        # 2) DXF bytes (GENERATO, robusto)
        dxf_bytes = export_schema_dxf_bytes(body, overlay_id=overlay_id)

        # 3) tamponamenti: grafico NL identico Jupyter
        panels = body.get("panels") or []
        _must(len(panels) > 0, "panels vuoto: niente tamponamenti da inserire nel PDF")

        tamponamenti: List[TamponamentoItem] = []
        for p in panels:
            pid = p.get("panel_id", f"{p.get('i','?')},{p.get('j','?')}")
            B_cm, H_cm = _panel_BH_cm(p)
            plot_png = plot_nl_panel_to_png_bytes(p)

            tamponamenti.append(
                TamponamentoItem(
                    pid=str(pid),
                    B_cm=float(B_cm),
                    H_cm=float(H_cm),
                    plot_png=plot_png,
                )
            )

        # 4) PDF bytes
        pdf_bytes = build_report_pdf_bytes(
            header_lines=header_lines,
            schema_png=schema_png,
            tamponamenti=tamponamenti,
        )

        # 5) ZIP in memoria
        zip_filename = f"{job_id}_allegati.zip"
        pdf_name_in_zip = "Report_resisto59_completo.pdf"
        dxf_name_in_zip = "schema_posa_resisto59.dxf"

        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(pdf_name_in_zip, pdf_bytes)      # bytes
            z.writestr(dxf_name_in_zip, dxf_bytes)      # bytes

        mem_zip.seek(0)
        return StreamingResponse(
            mem_zip,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
