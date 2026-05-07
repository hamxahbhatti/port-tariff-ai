#!/usr/bin/env python3
"""
Port Tariff AI — Architecture Document Generator
Produces a professional PDF with embedded architecture diagrams.
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, NextPageTemplate,
    Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY        = colors.HexColor('#0F2744')
STEEL       = colors.HexColor('#1A3A5C')
ACCENT      = colors.HexColor('#2563EB')
ACCENT_LIGHT= colors.HexColor('#EFF6FF')
TEAL        = colors.HexColor('#0E7490')
TEAL_LIGHT  = colors.HexColor('#ECFEFF')
SLATE       = colors.HexColor('#334155')
MID_GRAY    = colors.HexColor('#64748B')
BORDER      = colors.HexColor('#CBD5E1')
RULE        = colors.HexColor('#E2E8F0')
WHITE       = colors.white
PAGE_BG     = colors.HexColor('#FAFBFC')

W, H = A4   # 595.27 x 841.89 pt

OUT_PATH = Path.home() / "Desktop" / "Port_Tariff_AI_Architecture.pdf"


# ── Styles ────────────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        'cover_title': s('cover_title',
            fontName='Helvetica-Bold', fontSize=36, textColor=WHITE,
            leading=44, spaceAfter=8, alignment=TA_LEFT),

        'cover_sub': s('cover_sub',
            fontName='Helvetica', fontSize=14, textColor=colors.HexColor('#BFDBFE'),
            leading=20, spaceAfter=4, alignment=TA_LEFT),

        'cover_meta': s('cover_meta',
            fontName='Helvetica', fontSize=11, textColor=colors.HexColor('#93C5FD'),
            leading=16, spaceAfter=2, alignment=TA_LEFT),

        'cover_author': s('cover_author',
            fontName='Helvetica-Bold', fontSize=13, textColor=WHITE,
            leading=18, spaceAfter=2, alignment=TA_LEFT),

        'section_h1': s('section_h1',
            fontName='Helvetica-Bold', fontSize=16, textColor=NAVY,
            leading=22, spaceBefore=18, spaceAfter=6, alignment=TA_LEFT),

        'section_h2': s('section_h2',
            fontName='Helvetica-Bold', fontSize=12, textColor=STEEL,
            leading=17, spaceBefore=12, spaceAfter=4, alignment=TA_LEFT),

        'body': s('body',
            fontName='Helvetica', fontSize=9.5, textColor=SLATE,
            leading=15, spaceBefore=0, spaceAfter=6, alignment=TA_JUSTIFY),

        'body_bold': s('body_bold',
            fontName='Helvetica-Bold', fontSize=9.5, textColor=SLATE,
            leading=15, spaceBefore=0, spaceAfter=4, alignment=TA_JUSTIFY),

        'caption': s('caption',
            fontName='Helvetica-Oblique', fontSize=8.5, textColor=MID_GRAY,
            leading=12, spaceBefore=4, spaceAfter=10, alignment=TA_CENTER),

        'table_head': s('table_head',
            fontName='Helvetica-Bold', fontSize=8.5, textColor=WHITE,
            leading=12, alignment=TA_LEFT),

        'table_cell': s('table_cell',
            fontName='Helvetica', fontSize=8.5, textColor=SLATE,
            leading=12, alignment=TA_LEFT),

        'badge': s('badge',
            fontName='Helvetica-Bold', fontSize=7.5, textColor=ACCENT,
            leading=10, alignment=TA_LEFT),

        'page_num': s('page_num',
            fontName='Helvetica', fontSize=8, textColor=MID_GRAY,
            leading=10, alignment=TA_RIGHT),
    }


# ── Cover page painter ────────────────────────────────────────────────────────
class CoverPage(Flowable):
    def __init__(self, width, height):
        super().__init__()
        self.width  = width
        self.height = height

    def draw(self):
        c = self.canv

        # Navy background
        c.setFillColor(NAVY)
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)

        # Subtle deep-blue accent strip at top
        c.setFillColor(colors.HexColor('#162E4D'))
        c.rect(0, self.height - 8*mm, self.width, 8*mm, fill=1, stroke=0)

        # Left accent bar
        c.setFillColor(ACCENT)
        c.rect(0, 0, 5*mm, self.height, fill=1, stroke=0)

        # Bottom info strip
        c.setFillColor(colors.HexColor('#0A1E33'))
        c.rect(0, 0, self.width, 28*mm, fill=1, stroke=0)

        # Logo / icon area — stylised anchor
        c.setFillColor(colors.HexColor('#1D4ED8'))
        c.circle(self.width - 48*mm, self.height - 52*mm, 28*mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#2563EB'))
        c.circle(self.width - 48*mm, self.height - 52*mm, 22*mm, fill=1, stroke=0)

        # Port icon text inside circle
        c.setFont('Helvetica-Bold', 22)
        c.setFillColor(WHITE)
        c.drawCentredString(self.width - 48*mm, self.height - 56*mm, "⚓")

        # Main title
        c.setFont('Helvetica-Bold', 38)
        c.setFillColor(WHITE)
        c.drawString(14*mm, self.height - 88*mm, "Port Tariff AI")

        # Subtitle rule
        c.setStrokeColor(ACCENT)
        c.setLineWidth(2)
        c.line(14*mm, self.height - 95*mm, 100*mm, self.height - 95*mm)

        # Subtitle
        c.setFont('Helvetica', 13)
        c.setFillColor(colors.HexColor('#93C5FD'))
        c.drawString(14*mm, self.height - 106*mm, "Generative AI Solutions Developer Assessment")

        # Desc
        c.setFont('Helvetica', 11)
        c.setFillColor(colors.HexColor('#BFDBFE'))
        c.drawString(14*mm, self.height - 119*mm, "Architecture and Technical Approach")

        # Divider
        c.setStrokeColor(colors.HexColor('#1E3A5F'))
        c.setLineWidth(0.5)
        c.line(14*mm, self.height - 145*mm, self.width - 14*mm, self.height - 145*mm)

        # Author block
        c.setFont('Helvetica-Bold', 14)
        c.setFillColor(WHITE)
        c.drawString(14*mm, self.height - 162*mm, "Hamza Shabbir")

        c.setFont('Helvetica', 11)
        c.setFillColor(colors.HexColor('#93C5FD'))
        c.drawString(14*mm, self.height - 174*mm, "Principal AI Solution Architect")

        c.setFont('Helvetica', 10)
        c.setFillColor(colors.HexColor('#64748B'))
        c.drawString(14*mm, self.height - 185*mm, "Systems Limited (Techvista)  ·  Dubai, UAE")

        # Tags
        tags = ["Agentic AI", "RAG Pipelines", "LLMOps", "FastMCP", "GCC Enterprise"]
        x = 14*mm
        y = self.height - 210*mm
        c.setFont('Helvetica', 8)
        for tag in tags:
            tw = c.stringWidth(tag, 'Helvetica', 8) + 8*mm
            c.setFillColor(colors.HexColor('#1E3A5F'))
            c.roundRect(x, y - 3*mm, tw, 6*mm, 2*mm, fill=1, stroke=0)
            c.setFillColor(colors.HexColor('#60A5FA'))
            c.drawString(x + 4*mm, y + 0.5*mm, tag)
            x += tw + 3*mm

        # Bottom strip info
        c.setFont('Helvetica', 9)
        c.setFillColor(colors.HexColor('#475569'))
        c.drawString(14*mm, 14*mm, "Transnet National Ports Authority  ·  Durban 2024/25 Tariff Schedule")

        c.setFont('Helvetica', 9)
        c.setFillColor(colors.HexColor('#475569'))
        c.drawRightString(self.width - 14*mm, 14*mm, "May 2026")

    def wrap(self, availW, availH):
        return (self.width, self.height)


# ── Section heading with accent bar ──────────────────────────────────────────
class SectionBar(Flowable):
    def __init__(self, text, width, num=''):
        super().__init__()
        self._text  = text
        self._num   = num
        self._width = width
        self.height = 12*mm

    def draw(self):
        c = self.canv
        c.setFillColor(ACCENT_LIGHT)
        c.rect(0, 0, self._width, 10*mm, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.rect(0, 0, 3*mm, 10*mm, fill=1, stroke=0)
        c.setFont('Helvetica-Bold', 12)
        c.setFillColor(NAVY)
        label = f"{self._num}  {self._text}" if self._num else self._text
        c.drawString(7*mm, 3*mm, label)

    def wrap(self, *args):
        return (self._width, self.height)


# ── Inline key-value info box ─────────────────────────────────────────────────
class InfoBox(Flowable):
    def __init__(self, items, width, bg=ACCENT_LIGHT, border=ACCENT):
        super().__init__()
        self._items  = items
        self._width  = width
        self._bg     = bg
        self._border = border
        self.height  = max(len(items) * 5*mm + 8*mm, 18*mm)

    def draw(self):
        c = self.canv
        c.setFillColor(self._bg)
        c.roundRect(0, 0, self._width, self.height, 2*mm, fill=1, stroke=0)
        c.setStrokeColor(self._border)
        c.setLineWidth(0.5)
        c.roundRect(0, 0, self._width, self.height, 2*mm, fill=0, stroke=1)
        c.setFont('Helvetica', 8.5)
        y = self.height - 7*mm
        for k, v in self._items:
            c.setFillColor(SLATE)
            c.setFont('Helvetica-Bold', 8.5)
            c.drawString(4*mm, y, k + ':')
            c.setFont('Helvetica', 8.5)
            c.drawString(4*mm + c.stringWidth(k + ': ', 'Helvetica-Bold', 8.5), y, v)
            y -= 5*mm

    def wrap(self, *args):
        return (self._width, self.height)


# ── Architecture diagram — Development ───────────────────────────────────────
def make_dev_diagram(width_px=1600, height_px=1100):
    fig, ax = plt.subplots(figsize=(width_px/100, height_px/100), dpi=100)
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FAFBFC')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    C = {
        'navy':    '#0F2744',
        'steel':   '#1A3A5C',
        'accent':  '#2563EB',
        'teal':    '#0E7490',
        'green':   '#059669',
        'amber':   '#D97706',
        'slate':   '#334155',
        'gray':    '#64748B',
        'light':   '#EFF6FF',
        'teal_l':  '#ECFEFF',
        'green_l': '#ECFDF5',
        'amber_l': '#FFFBEB',
        'white':   '#FFFFFF',
        'rule':    '#E2E8F0',
    }

    def box(ax, x, y, w, h, label, sublabel='', bg='#EFF6FF', fg='#0F2744',
            border='#2563EB', bw=1.5, fs=9, sfs=7.5, bold=True):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.05",
                              facecolor=bg, edgecolor=border, linewidth=bw)
        ax.add_patch(rect)
        cx, cy = x + w/2, y + h/2
        dy = 0.15 if sublabel else 0
        ax.text(cx, cy + dy, label,
                ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', color=fg)
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel,
                    ha='center', va='center', fontsize=sfs,
                    fontweight='normal', color=C['gray'])

    def arrow(ax, x1, y1, x2, y2, color='#2563EB', lw=1.5, style='->', shrink=3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                   lw=lw, shrinkA=shrink, shrinkB=shrink))

    def layer_bg(ax, x, y, w, h, label, bg, border, lc):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=bg, edgecolor=border,
                              linewidth=1.5, linestyle='--')
        ax.add_patch(rect)
        ax.text(x + 0.18, y + h - 0.22, label,
                ha='left', va='top', fontsize=8,
                fontweight='bold', color=lc)

    # ── Layer backgrounds ──────────────────────────────────────────────────
    # Ingestion layer (top)
    layer_bg(ax, 0.3, 7.8, 15.4, 2.9,
             'INGESTION LAYER  (Local Only — Not Deployed)',
             '#FEF9F0', '#D97706', '#92400E')

    # Knowledge store
    layer_bg(ax, 0.3, 5.6, 15.4, 1.9,
             'KNOWLEDGE STORE',
             '#F0FDF4', '#059669', '#065F46')

    # MCP servers
    layer_bg(ax, 0.3, 3.4, 15.4, 1.9,
             'MCP SERVERS  (Model Context Protocol)',
             '#EFF6FF', '#2563EB', '#1E40AF')

    # Agent & API
    layer_bg(ax, 0.3, 1.1, 15.4, 2.0,
             'AGENT + API LAYER',
             '#F8F4FF', '#7C3AED', '#4C1D95')

    # ── Source document ────────────────────────────────────────────────────
    box(ax, 0.6, 9.2, 1.7, 1.0,
        'TNPA PDF', 'Tariff Book 2024/25',
        bg='#0F2744', fg='#FFFFFF', border='#0F2744', fs=8, sfs=6.5)

    # ── Docling ────────────────────────────────────────────────────────────
    box(ax, 2.8, 9.0, 3.8, 1.4,
        'Docling', 'DocLayNet + TableFormer\nDirect PDF parse — no images',
        bg=C['amber_l'], fg=C['navy'], border=C['amber'], fs=9, sfs=7)

    # Arrow: PDF → Docling
    arrow(ax, 2.3, 9.7, 2.8, 9.7, color=C['amber'])

    # ── Quality Assessment ─────────────────────────────────────────────────
    box(ax, 7.2, 9.1, 2.8, 1.2,
        'Quality Assessment', 'Markdown scoring per page',
        bg='#FFF7ED', fg=C['navy'], border='#F97316', fs=8.5, sfs=7)

    arrow(ax, 6.6, 9.7, 7.2, 9.7, color='#F97316')

    # ── Gemini Text ────────────────────────────────────────────────────────
    box(ax, 5.5, 7.9, 2.6, 1.0,
        'Gemini Text API', 'Clean pages — batch call',
        bg=C['green_l'], fg=C['navy'], border=C['green'], fs=8, sfs=6.8)

    # ── Gemini Vision ──────────────────────────────────────────────────────
    box(ax, 8.4, 7.9, 2.8, 1.0,
        'Gemini Vision', 'Complex pages — double pass\nPyMuPDF renders image',
        bg=C['teal_l'], fg=C['navy'], border=C['teal'], fs=8, sfs=6.5)

    # ── MinerU ────────────────────────────────────────────────────────────
    box(ax, 11.5, 7.9, 2.2, 1.0,
        'MinerU', 'Flagged page fallback',
        bg='#FFF1F2', fg=C['navy'], border='#E11D48', fs=8, sfs=6.8)

    # Arrows from Quality Assessment
    arrow(ax, 8.6,  9.1, 6.8, 8.9,  color=C['green'])     # → Text
    arrow(ax, 8.6,  9.1, 9.8, 8.9,  color=C['teal'])      # → Vision
    arrow(ax, 9.8,  9.1, 12.6,8.9,  color='#E11D48', style='->')  # → MinerU

    # Route labels
    ax.text(7.0, 9.2, 'Clean', fontsize=6.5, color=C['green'], ha='center')
    ax.text(9.5, 9.2, 'Complex', fontsize=6.5, color=C['teal'], ha='center')
    ax.text(11.3, 9.05, 'Flagged', fontsize=6.5, color='#E11D48', ha='center')

    # ── JSON Store ────────────────────────────────────────────────────────
    box(ax, 0.6, 5.75, 3.6, 1.5,
        'JSON Rate Store', '39 files  ·  Exact rate rows\nDeterministic key-value reads',
        bg=C['green_l'], fg=C['navy'], border=C['green'], fs=9, sfs=7)

    # ── ChromaDB ──────────────────────────────────────────────────────────
    box(ax, 4.8, 5.75, 4.2, 1.5,
        'ChromaDB', 'Prose chunks + table descriptions\nCosine similarity  ·  text-embedding-004',
        bg=C['teal_l'], fg=C['navy'], border=C['teal'], fs=9, sfs=7)

    # Arrows → Knowledge Store
    arrow(ax, 6.8, 8.9, 2.4,  7.25, color=C['green'])
    arrow(ax, 6.8, 8.9, 6.9,  7.25, color=C['teal'])
    arrow(ax, 9.8, 8.9, 6.9,  7.25, color=C['teal'])
    arrow(ax, 12.6,8.9, 6.9,  7.25, color=C['teal'], lw=1)

    # ── MCP: Rules Engine ─────────────────────────────────────────────────
    box(ax, 0.6, 3.55, 3.0, 1.5,
        'Rules Engine', 'Charge applicability\nExemption logic  ·  GT thresholds',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=8.5, sfs=7)

    # ── MCP: Calculator ───────────────────────────────────────────────────
    box(ax, 4.0, 3.55, 3.0, 1.5,
        'Calculator', '8 pure Python functions\nNo LLM  ·  Fully auditable',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=8.5, sfs=7)

    # ── MCP: Tariff RAG ───────────────────────────────────────────────────
    box(ax, 7.4, 3.55, 3.2, 1.5,
        'Tariff RAG', 'get_tariff_table (JSON)\nsearch_rules (ChromaDB)',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=8.5, sfs=7)

    # ── MCP: Vessel ───────────────────────────────────────────────────────
    box(ax, 11.0, 3.55, 2.6, 1.5,
        'Vessel Server', 'Profile validation\nNormalisation  ·  GT classify',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=8.5, sfs=7)

    # Arrows → MCP
    arrow(ax, 2.4,  5.75, 2.1, 5.05, color=C['accent'])
    arrow(ax, 6.9,  5.75, 5.5, 5.05, color=C['accent'])
    arrow(ax, 6.9,  5.75, 9.0, 5.05, color=C['accent'])

    # ── Agent: Chat Agent ─────────────────────────────────────────────────
    box(ax, 0.6, 1.25, 4.5, 1.6,
        'Chat Agent', 'LangChain tool-calling\n2 tools  ·  Session memory\nGemini 2.5 Flash Lite  ·  T=0',
        bg='#F5F3FF', fg=C['navy'], border='#7C3AED', fs=9, sfs=7)

    # ── Agent: LangGraph ─────────────────────────────────────────────────
    box(ax, 5.5, 1.25, 4.0, 1.6,
        'LangGraph Agent', 'StateGraph ReAct\n17 tools  ·  Gemini Flash\nrecursion limit 50',
        bg='#F5F3FF', fg=C['navy'], border='#7C3AED', fs=9, sfs=7)

    # ── FastAPI ───────────────────────────────────────────────────────────
    box(ax, 10.0, 1.25, 3.2, 1.6,
        'FastAPI', '/chat  ·  /calculate\n/calculate/quick/stream\nSSE streaming',
        bg='#F5F3FF', fg=C['navy'], border='#7C3AED', fs=9, sfs=7)

    # ── Frontend ──────────────────────────────────────────────────────────
    box(ax, 13.5, 1.25, 2.0, 1.6,
        'Frontend', 'Chat UI\nDebug drawer\nSSE client',
        bg=C['navy'], fg='#FFFFFF', border=C['navy'], fs=8.5, sfs=7)

    # Arrows → Agent/API
    arrow(ax, 2.1, 3.55, 2.85, 2.85, color='#7C3AED')
    arrow(ax, 5.5, 3.55, 7.5,  2.85, color='#7C3AED')
    arrow(ax, 9.0, 3.55, 12.2, 2.85, color='#7C3AED')
    arrow(ax, 13.2,2.05, 13.5, 2.05, color=C['navy'])

    # Title
    ax.text(8.0, 10.85, 'Development Architecture',
            ha='center', va='center', fontsize=13,
            fontweight='bold', color=C['navy'])
    ax.text(8.0, 10.6, 'Full pipeline from source PDF through ingestion, knowledge store, and inference layers',
            ha='center', va='center', fontsize=7.5, color=C['gray'])

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#FAFBFC')
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Architecture diagram — Production ────────────────────────────────────────
def make_prod_diagram(width_px=1600, height_px=900):
    fig, ax = plt.subplots(figsize=(width_px/100, height_px/100), dpi=100)
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FAFBFC')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    C = {
        'navy':   '#0F2744',
        'accent': '#2563EB',
        'teal':   '#0E7490',
        'green':  '#059669',
        'purple': '#7C3AED',
        'slate':  '#334155',
        'gray':   '#64748B',
        'light':  '#EFF6FF',
        'teal_l': '#ECFEFF',
        'green_l':'#ECFDF5',
        'white':  '#FFFFFF',
    }

    def box(ax, x, y, w, h, label, sublabel='', bg='#EFF6FF', fg='#0F2744',
            border='#2563EB', bw=2.0, fs=9.5, sfs=7.5):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.07",
                              facecolor=bg, edgecolor=border, linewidth=bw)
        ax.add_patch(rect)
        cx, cy = x + w/2, y + h/2
        dy = 0.18 if sublabel else 0
        ax.text(cx, cy + dy, label,
                ha='center', va='center', fontsize=fs,
                fontweight='bold', color=fg)
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel,
                    ha='center', va='center', fontsize=sfs,
                    fontweight='normal', color=C['gray'])

    def arrow(ax, x1, y1, x2, y2, label='', color='#2563EB', lw=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=lw, shrinkA=4, shrinkB=4))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.1, my, label, fontsize=6.5, color=color,
                    ha='left', va='center')

    def deploy_box(ax, x, y, w, h, label, bg, border, lc):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.12",
                              facecolor=bg, edgecolor=border,
                              linewidth=2.5, linestyle='--')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.28, label,
                ha='center', va='top', fontsize=9,
                fontweight='bold', color=lc)

    # ── Railway deployment boundary ────────────────────────────────────────
    deploy_box(ax, 0.3, 1.0, 11.0, 7.6,
               'Railway  —  Docker Container (python:3.13-slim)',
               '#F0F7FF', '#2563EB', '#1E40AF')

    # ── Vercel boundary ────────────────────────────────────────────────────
    deploy_box(ax, 11.8, 1.0, 3.9, 7.6,
               'Vercel  —  Global Edge CDN',
               '#F5F3FF', '#7C3AED', '#4C1D95')

    # ── Knowledge store inside Railway ────────────────────────────────────
    box(ax, 0.7, 6.0, 3.2, 1.6,
        'JSON Rate Store', '39 files committed to repo\nCopied into Docker image',
        bg=C['green_l'], fg=C['navy'], border=C['green'], fs=9, sfs=7)

    box(ax, 4.2, 6.0, 3.2, 1.6,
        'ChromaDB', 'Persistent prose store\nPre-built at ingestion time',
        bg=C['teal_l'], fg=C['navy'], border=C['teal'], fs=9, sfs=7)

    # ── MCP Servers ────────────────────────────────────────────────────────
    box(ax, 0.7, 4.1, 2.2, 1.5,
        'Rules Engine', 'Charge applicability\nGT thresholds',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=8.5, sfs=7)

    box(ax, 3.2, 4.1, 2.2, 1.5,
        'Calculator', '8 deterministic\nfunctions',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=8.5, sfs=7)

    box(ax, 5.7, 4.1, 2.2, 1.5,
        'Tariff RAG', 'JSON + ChromaDB\nsearch interface',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=8.5, sfs=7)

    box(ax, 8.2, 4.1, 2.6, 1.5,
        'Vessel Server', 'Profile validation\nnormalisation',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=8.5, sfs=7)

    # ── Chat Agent ────────────────────────────────────────────────────────
    box(ax, 1.0, 2.3, 5.0, 1.4,
        'Chat Agent  (LangChain)', 'Gemini 2.5 Flash Lite  ·  T=0  ·  Session memory (in-process dict)',
        bg='#F5F3FF', fg=C['navy'], border=C['purple'], fs=9, sfs=7)

    # ── Gemini external ────────────────────────────────────────────────────
    box(ax, 6.5, 2.3, 3.0, 1.4,
        'Gemini API', 'google.generativeai\nExternal (API key via env)',
        bg='#FFFBEB', fg=C['navy'], border='#D97706', fs=9, sfs=7)

    # ── FastAPI ────────────────────────────────────────────────────────────
    box(ax, 2.5, 1.3, 5.5, 0.85,
        'FastAPI + SSE  ·  /chat  ·  /health  ·  /calculate/quick/stream',
        bg=C['navy'], fg='#FFFFFF', border=C['navy'], fs=8.5)

    # ── Vercel static ─────────────────────────────────────────────────────
    box(ax, 12.0, 5.4, 3.3, 1.6,
        'Static Assets', 'index.html\napp.js  ·  style.css',
        bg=C['light'], fg=C['navy'], border=C['accent'], fs=9, sfs=7.5)

    # ── User browser ──────────────────────────────────────────────────────
    box(ax, 12.0, 2.3, 3.3, 2.7,
        'User Browser', 'Chat interface\nReal-time debug drawer\nSSE event stream',
        bg=C['navy'], fg='#FFFFFF', border=C['navy'], fs=9, sfs=7.5)

    # Arrows
    arrow(ax, 2.3, 6.8, 1.9, 5.6, color=C['green'])
    arrow(ax, 5.8, 6.8, 5.8, 5.6, color=C['teal'])
    arrow(ax, 1.9, 4.1, 3.7, 3.7, color=C['accent'])
    arrow(ax, 4.4, 4.1, 3.7, 3.7, color=C['accent'])
    arrow(ax, 6.8, 4.1, 3.7, 3.7, color=C['accent'])
    arrow(ax, 9.5, 4.1, 6.0, 3.7, color=C['accent'])
    arrow(ax, 3.7, 2.3, 4.3, 2.15, color=C['purple'])
    arrow(ax, 6.0, 2.3, 7.0, 2.15, color='#D97706', label='HTTPS + API key')
    arrow(ax, 8.0, 1.75, 11.8, 3.65, color=C['navy'], lw=2.5)
    ax.text(10.2, 2.8, 'HTTPS / SSE', fontsize=7, color=C['navy'],
            ha='center', va='center', rotation=28)
    arrow(ax, 13.65, 5.4, 13.65, 5.0, color=C['purple'])

    # Env var note
    ax.text(7.8, 1.25, 'GEMINI_API_KEY injected via Railway env vars  ·  HTTPS termination by Railway',
            ha='center', va='center', fontsize=7, color='#64748B', style='italic')

    # Title
    ax.text(8.0, 8.65, 'Production Architecture',
            ha='center', va='center', fontsize=13,
            fontweight='bold', color=C['navy'])
    ax.text(8.0, 8.38, 'Ingestion layer excluded  ·  Railway backend  ·  Vercel static frontend',
            ha='center', va='center', fontsize=7.5, color='#64748B')

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#FAFBFC')
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Page template with header/footer ─────────────────────────────────────────
def make_pdf(out_path: Path):
    ST = build_styles()
    W_inner = W - 36*mm

    # Full-bleed frame for cover page
    cover_frame = Frame(0, 0, W, H, leftPadding=0, bottomPadding=0,
                        rightPadding=0, topPadding=0, id='cover')

    # Normal content frame
    content_frame = Frame(18*mm, 20*mm, W_inner, H - 42*mm, id='content')

    def on_page(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        # Top rule
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(18*mm, H - 13*mm, W - 18*mm, H - 13*mm)
        # Header text
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(MID_GRAY)
        canvas.drawString(18*mm, H - 11*mm, 'Port Tariff AI  —  Architecture and Technical Approach')
        canvas.drawRightString(W - 18*mm, H - 11*mm, 'Hamza Shabbir  ·  Principal AI Solution Architect')
        # Footer rule
        canvas.line(18*mm, 14*mm, W - 18*mm, 14*mm)
        canvas.drawString(18*mm, 10*mm,
                          'Transnet National Ports Authority  ·  Durban 2024/25 Tariff Schedule')
        canvas.drawRightString(W - 18*mm, 10*mm, f'Page {doc.page}')
        canvas.restoreState()

    # ── Content ────────────────────────────────────────────────────────────
    story = []

    # ── Cover (full-bleed frame, then switch to Content template) ──────────
    story.append(NextPageTemplate('Content'))
    story.append(CoverPage(W, H))
    story.append(PageBreak())

    # ── 1. About the Author ───────────────────────────────────────────────
    story.append(SectionBar('About the Author', W_inner, ''))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(
        'Hamza Shabbir is a Principal AI Solution Architect with over eight years of experience '
        'leading and personally delivering enterprise AI systems across the UAE, Saudi Arabia, and Qatar. '
        'He has architected and built more than twenty production AI platforms for clients including '
        'Dubai Islamic Bank, RTA Dubai, Dammam Airports, the UAE Ministry of Defence, and the Federal Authority '
        'for Government Human Resources, with deep specialisation in Agentic AI, RAG pipelines, LLMOps, '
        'and AI governance aligned with UAE Central Bank, DESC, and GCC regulatory standards. '
        'He leads the full engagement lifecycle from executive solution design through to hands-on production delivery.',
        ST['body']))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph(
        'This document was prepared as part of the Generative AI Solutions Developer assessment. '
        'It describes the architectural decisions, component design, and deployment strategy '
        'for the Port Tariff AI system built against the Transnet National Ports Authority '
        '2024/25 tariff schedule.',
        ST['body']))
    story.append(Spacer(1, 6*mm))

    # ── 2. Executive Summary ───────────────────────────────────────────────
    story.append(SectionBar('Executive Summary', W_inner, '1.'))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        'Port Tariff AI is a conversational calculation system that allows shipping agents and vessel operators '
        'to obtain a complete breakdown of applicable port dues for any vessel calling at Durban, '
        'using the official TNPA 2024/25 tariff schedule as its authoritative source. '
        'The system accepts plain-language queries, identifies the relevant charge categories, '
        'retrieves the correct rate tables, performs deterministic arithmetic for each charge, '
        'and returns a fully itemised result with the calculation formula shown at every line.',
        ST['body']))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        'The central architectural decision is the separation of data extraction from runtime inference. '
        'Extracting structured rate data from the source PDF is a one-time, compute-intensive process '
        'that runs on a developer machine. Its output travels to production as committed files. '
        'The production system carries no extraction dependencies and runs on standard cloud infrastructure '
        'at low cost.',
        ST['body']))
    story.append(Spacer(1, 6*mm))

    # ── 3. Problem Statement ───────────────────────────────────────────────
    story.append(SectionBar('Problem Statement', W_inner, '2.'))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        'The TNPA tariff book is a dense regulatory document. Calculating port dues for a single vessel '
        'involves identifying which of eight charge categories apply based on vessel size, type, and '
        'operational intent; looking up the correct rate band from tables that use inconsistent formatting '
        'conventions; applying operational modifiers such as pilotage movement count and harbour-master-assigned '
        'tug count; and summing the results into a final figure. A complete calculation requires ten or more '
        'interdependent lookup and arithmetic steps, with the governing rules distributed across both '
        'tabular rate schedules and narrative prose sections.',
        ST['body']))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        'The goal was to make this process accessible through a plain-language conversation while '
        'maintaining complete transparency into how each figure was derived, so that a reviewer could '
        'follow the reasoning and verify every number against the source document.',
        ST['body']))
    story.append(Spacer(1, 6*mm))

    # ── 4. Development Architecture ───────────────────────────────────────
    story.append(SectionBar('Development Architecture', W_inner, '3.'))
    story.append(Spacer(1, 4*mm))

    # Diagram A
    dev_buf = make_dev_diagram()
    dev_img = Image(dev_buf, width=W_inner, height=W_inner * 1100/1600)
    story.append(dev_img)
    story.append(Paragraph('Figure 1  —  Development Architecture: Full pipeline from source PDF through ingestion, knowledge store, and inference layers.', ST['caption']))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph(
        'The development environment contains layers that do not travel to production. '
        'The ingestion pipeline, Docling, PyMuPDF, Pillow, and MinerU are present only on the developer machine. '
        'Their combined output, a set of structured JSON rate files and a ChromaDB collection, '
        'is committed to the repository and forms the foundation of the runtime system.',
        ST['body']))
    story.append(Spacer(1, 5*mm))

    # 3.1
    story.append(Paragraph('3.1  Ingestion Pipeline', ST['section_h2']))
    story.append(Paragraph(
        'The source document is the TNPA 2024/25 tariff book in PDF format. Extracting structured rate '
        'data from it is non-trivial. The tariff tables span multiple pages, contain merged cells, '
        'use space-separated thousands, and mix portrait and landscape orientations. '
        'The pipeline resolves this in three stages.',
        ST['body']))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        '<b>Stage 1 — Docling primary extraction.</b>  Docling processes the full PDF using its '
        'DocLayNet layout model and TableFormer table structure model. It parses the document directly '
        'without converting pages to images. It produces two outputs: prose chunks containing the qualitative '
        'rules, exemptions, and conditions found in the non-table text; and a set of table page entries, '
        'each containing the page number and a markdown rendering of what Docling recovered from the table '
        'structure on that page.',
        ST['body']))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        '<b>Stage 2 — Smart routing per page.</b>  Each table page is assessed for extraction quality '
        'by inspecting Docling\'s markdown. The assessment checks whether markdown table delimiters are '
        'present, whether the page contains a sufficient number of tariff-scale numeric values, and '
        'whether the ratio of empty cells indicates merged-header confusion. Pages that pass go to Gemini '
        'in a single batched text call, where the Docling markdown is the input and the model restructures '
        'it into the target JSON schema. Pages that fail, typically three to eight of the twenty-three '
        'table pages in the Durban tariff book, are routed to Gemini Vision. PyMuPDF renders those pages '
        'to high-resolution images, and Gemini processes each image alongside its Docling context text '
        'using a double-pass verification to confirm consistency. Any page that Vision still cannot resolve '
        'confidently is flagged for MinerU as a final fallback.',
        ST['body']))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        '<b>Stage 3 — Progressive save.</b>  Results are written per page as they complete. '
        'Rate rows go to the structured JSON store. Prose chunks and table descriptions go to ChromaDB. '
        'The complete output for Durban is 39 JSON files covering all charge types.',
        ST['body']))
    story.append(Spacer(1, 5*mm))

    # 3.2
    story.append(Paragraph('3.2  Structured Knowledge Store', ST['section_h2']))
    story.append(Paragraph(
        'Each of the 39 JSON files corresponds to one charge type and contains the rate rows exactly '
        'as extracted from the tariff book, including tonnage band bounds, per-port values, rate units, '
        'and incremental-row flags. Lookups are deterministic key-value reads with no model involvement '
        'at query time. This store is the authoritative source for all arithmetic in the system.',
        ST['body']))
    story.append(Spacer(1, 5*mm))

    # 3.3 — ChromaDB
    story.append(Paragraph('3.3  ChromaDB: Semantic Prose Store', ST['section_h2']))
    story.append(Paragraph(
        'ChromaDB holds two categories of document, both stored in a single persistent collection '
        'named <b>port_tariff_rules</b> with cosine similarity distance (HNSW index).',
        ST['body']))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        '<b>Prose chunks</b> are sections of text extracted by Docling from the non-table areas of the '
        'tariff document. These cover exemption conditions, special vessel category rules, general '
        'surcharge conditions, and qualitative guidance that cannot be expressed as numeric rate rows. '
        'Each chunk carries metadata recording the port, page number, section heading, and source.',
        ST['body']))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        '<b>Table descriptions</b> are natural-language descriptions of what each charge table covers, '
        'produced during extraction. They allow the system to answer scope and purpose questions about '
        'a charge without parsing the numeric rate data.',
        ST['body']))
    story.append(Spacer(1, 3*mm))

    # ChromaDB what is retrieved
    story.append(Paragraph(
        'ChromaDB is queried through the <b>search_rules</b> tool in the Tariff RAG MCP server. '
        'It is used specifically for the following query types:',
        ST['body']))
    story.append(Spacer(1, 2*mm))

    chroma_data = [
        ['Query Type', 'Example', 'What Is Returned'],
        ['Exemption lookup', 'Is a vessel in distress exempt from port dues?', 'Prose passage + metadata'],
        ['Special category', 'Do naval vessels pay pilotage?', 'Applicable conditions'],
        ['Condition check', 'What surcharges apply to tankers at Durban?', 'Relevant rule sections'],
        ['Charge scope', 'What does VTS cover and how is it applied?', 'Table description + prose'],
    ]
    ct = Table(chroma_data, colWidths=[3.5*cm, 6.5*cm, 5.5*cm])
    ct.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), NAVY),
        ('TEXTCOLOR',   (0,0), (-1,0), WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR',   (0,1), (-1,-1), SLATE),
        ('BACKGROUND',  (0,1), (-1,-1), WHITE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, ACCENT_LIGHT]),
        ('GRID',        (0,0), (-1,-1), 0.4, BORDER),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(ct)
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        'ChromaDB is not used for numeric lookups. Any calculation requiring an exact rate value goes '
        'directly to the JSON store. The embedding function uses Gemini text-embedding-004 with an '
        'automatic fallback to the local all-MiniLM-L6-v2 model built into ChromaDB if the Gemini '
        'API is unavailable.',
        ST['body']))
    story.append(Spacer(1, 5*mm))

    # 3.4 MCP Servers
    story.append(Paragraph('3.4  MCP Servers', ST['section_h2']))
    story.append(Paragraph(
        'The business logic is organised into four Model Context Protocol servers, each a focused '
        'and independently testable Python module built with FastMCP.',
        ST['body']))
    story.append(Spacer(1, 3*mm))

    mcp_data = [
        ['Server', 'Tools Exposed', 'Responsibility'],
        ['Rules Engine',
         'determine_applicable_charges\ncheck_exemptions\nget_vessel_charge_plan',
         'GT-threshold charge applicability rules, exemption logic, pilotage and tug-count defaults'],
        ['Calculator',
         'calculate_light_dues\ncalculate_vts\ncalculate_pilotage\ncalculate_tug_assistance\ncalculate_port_dues\ncalculate_cargo_dues\ncalculate_berth_dues\ncalculate_running_of_lines',
         'Eight pure Python arithmetic functions. No LLM calls. Returns ZAR amount plus step-by-step formula string.'],
        ['Tariff RAG',
         'get_tariff_table\nsearch_rules\nlist_available_charges',
         'Wraps JSON store and ChromaDB. Provides exact rate rows and semantic prose search.'],
        ['Vessel Server',
         'register_vessel\nget_vessel\nclassify_vessel_for_tariff',
         'Profile validation, type normalisation, GT categorisation, cargo classification.'],
    ]
    mt = Table(mcp_data, colWidths=[2.8*cm, 5.2*cm, 7.5*cm])
    mt.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), STEEL),
        ('TEXTCOLOR',   (0,0), (-1,0), WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 7.5),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR',   (0,1), (-1,-1), SLATE),
        ('BACKGROUND',  (0,1), (-1,-1), WHITE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, ACCENT_LIGHT]),
        ('GRID',        (0,0), (-1,-1), 0.4, BORDER),
        ('VALIGN',      (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',  (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(mt)
    story.append(Spacer(1, 5*mm))

    # 3.5 Agent Layer
    story.append(Paragraph('3.5  Agent Layer', ST['section_h2']))
    story.append(Paragraph(
        'Two agent implementations exist in the codebase, each serving a distinct interface.',
        ST['body']))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        '<b>Chat Agent</b> (chat_agent.py) is a LangChain tool-calling agent with two tools exposed to '
        'the model: determine_applicable_charges and calculate_all_dues. The latter internally '
        'orchestrates the rules engine and all eight calculator functions, keeping the model\'s '
        'tool-calling surface minimal and reducing the risk of incorrect sequencing. '
        'This agent maintains conversation history per session, stored in a server-side dictionary '
        'keyed by session UUID. Follow-up questions resolve correctly against the vessel parameters '
        'established earlier in the conversation without the user re-stating them. '
        'It powers the interactive frontend through the /chat SSE endpoint.',
        ST['body']))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        '<b>LangGraph ReAct Agent</b> (tariff_agent.py) is a full StateGraph implementation with '
        'seventeen tools spanning all four MCP servers. It follows an explicit workflow: charge plan '
        'retrieval, per-charge table lookup, individual charge calculation, and aggregation. '
        'This agent is exposed through the /calculate endpoint for programmatic use where a complete '
        'vessel profile is provided upfront. Both agents use Gemini 2.5 Flash Lite at temperature zero.',
        ST['body']))
    story.append(Spacer(1, 5*mm))

    # 3.6 API & Frontend
    story.append(Paragraph('3.6  API and Frontend', ST['section_h2']))
    story.append(Paragraph(
        'FastAPI exposes three endpoint categories. The /chat endpoint runs the conversational agent '
        'and streams typed Server-Sent Events for each step: LLM call, tool call, tool result, and '
        'final response. The /calculate/quick/stream endpoint runs the deterministic pipeline without '
        'the conversational layer and streams the same event types. The /calculate endpoint runs the '
        'LangGraph agent synchronously and returns JSON. '
        'The frontend is three static files. The chat interface streams events from the backend over SSE '
        'and routes each event to either the debug drawer (full execution trace, expandable) or the message '
        'area (final agent response with structured calculation card). Session continuity is maintained '
        'via a UUID in localStorage.',
        ST['body']))
    story.append(PageBreak())

    # ── 5. Production Architecture ────────────────────────────────────────
    story.append(SectionBar('Production Architecture', W_inner, '4.'))
    story.append(Spacer(1, 4*mm))

    prod_buf = make_prod_diagram()
    prod_img = Image(prod_buf, width=W_inner, height=W_inner * 900/1600)
    story.append(prod_img)
    story.append(Paragraph(
        'Figure 2  —  Production Architecture: Ingestion layer excluded. Railway hosts the backend in Docker. '
        'Vercel serves the static frontend from the global edge network.',
        ST['caption']))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph('4.1  What Is Excluded in Production', ST['section_h2']))
    story.append(Paragraph(
        'Docling, PyMuPDF, Pillow, and MinerU are absent from the production environment entirely. '
        'The extracted JSON rate files are committed to the repository and are part of the Docker image. '
        'ChromaDB is initialised from the same committed data directory. There is no extraction '
        'dependency in production, and the Docker image installs only the runtime dependency set.',
        ST['body']))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph('4.2  Backend: Railway', ST['section_h2']))
    story.append(Paragraph(
        'The FastAPI application runs in a Docker container on Railway, built from python:3.13-slim. '
        'The image installs LangChain, LangGraph, FastMCP, FastAPI, ChromaDB, pydantic, uvicorn, '
        'and python-dotenv. The 39 tariff JSON files are copied into the image at build time, '
        'making the container entirely self-contained. The Gemini API key is injected via '
        'Railway\'s environment variable system at deploy time. Railway provides automatic HTTPS '
        'termination, health check monitoring against /health, and a restart policy with three retries.',
        ST['body']))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph('4.3  Frontend: Vercel', ST['section_h2']))
    story.append(Paragraph(
        'The frontend is deployed to Vercel as a pure static site serving three files from the global '
        'edge network. A rewrite rule in vercel.json maps the /static/:file path pattern used by the '
        'FastAPI static mount to the file root of the deployment. A .vercelignore file excludes all '
        'Python code, requirements files, and Docker configuration, so Vercel sees only the three '
        'frontend files and produces no Lambda functions. The Railway URL is set in the HTML as a '
        'single constant, making local and production deployments structurally identical.',
        ST['body']))
    story.append(Spacer(1, 6*mm))

    # ── 6. Key Design Decisions ───────────────────────────────────────────
    story.append(SectionBar('Key Design Decisions', W_inner, '5.'))
    story.append(Spacer(1, 4*mm))

    decisions = [
        ('Extraction separated from runtime',
         'Running the ingestion pipeline at deploy time would require heavy dependencies and a '
         'GPU-accessible machine. Extraction runs once locally, and only its structured output ships. '
         'This reduces the production image size significantly and eliminates infrastructure complexity.'),
        ('Deterministic calculators, not LLM arithmetic',
         'The model manages conversation flow and tool sequencing. All arithmetic is performed by pure '
         'Python functions that are unit-testable without any API call. This separates the part of AI '
         'systems that can be non-deterministic from the part that must be exact for financial calculations.'),
        ('Two-layer knowledge store',
         'Exact rate lookups use keyed JSON reads. Condition and exemption queries use ChromaDB semantic '
         'search. Combining them in a single vector store would degrade numeric precision and add latency '
         'to the common calculation path.'),
        ('Streaming transparency via SSE',
         'Every step of the agent\'s reasoning streams as a typed event. Users and reviewers see which '
         'tools were called, what data they returned, and how the total was assembled. This is particularly '
         'important in a regulatory context where auditability of the result matters as much as the result itself.'),
        ('Minimal tool surface for the chat agent',
         'Exposing two tools instead of seventeen to the conversational model reduces incorrect tool '
         'sequencing risk and keeps prompt token consumption low. Aggregation logic lives inside '
         'calculate_all_dues rather than being orchestrated turn-by-turn by the model.'),
        ('Split deployment',
         'The frontend and backend are deployed independently, allowing static assets to be served '
         'from a CDN without any Python runtime cost, and enabling the two tiers to be updated '
         'and scaled independently.'),
    ]

    for title, body in decisions:
        row_data = [[Paragraph(f'<b>{title}</b>', ST['body_bold']),
                     Paragraph(body, ST['body'])]]
        t = Table(row_data, colWidths=[4.5*cm, 11.0*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), ACCENT_LIGHT),
            ('BACKGROUND', (1,0), (1,0), WHITE),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('GRID', (0,0), (-1,-1), 0.4, BORDER),
        ]))
        story.append(t)
        story.append(Spacer(1, 1.5*mm))

    story.append(Spacer(1, 6*mm))

    # ── 7. Charge Types ───────────────────────────────────────────────────
    story.append(SectionBar('Charge Types Covered', W_inner, '6.'))
    story.append(Spacer(1, 4*mm))

    charges_data = [
        ['Charge Type', 'Calculation Basis', 'Conditions'],
        ['Light Dues',        'Per 100 GT',
         'Applies to all vessels at all SA ports'],
        ['Vessel Traffic Services', 'Per GT — flat per port call',
         'Applies to all vessels'],
        ['Port Dues',         'Per 100 GT',
         'Marine services levy; applies to all vessels'],
        ['Cargo Dues',        'Per metric tonne by cargo category',
         'Bulk, break-bulk, containerised, liquid bulk, ro-ro rates differ'],
        ['Berth Dues',        'Per 100 GT per 24-hour period',
         'Pro-rated for partial periods; applies only when berthing'],
        ['Running of Lines',  'Flat rate per service',
         'Two services standard per port call; applies only when berthing'],
        ['Pilotage',          'Banded by GT, per movement',
         'Compulsory above 500 GT; two movements standard (inbound and outbound)'],
        ['Tug Assistance',    'Banded by GT, per tug per movement',
         'Compulsory above 3,000 GT; tug count set by Harbour Master'],
    ]
    ct2 = Table(charges_data, colWidths=[3.8*cm, 4.8*cm, 7.0*cm])
    ct2.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), NAVY),
        ('TEXTCOLOR',    (0,0), (-1,0), WHITE),
        ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,-1), 8),
        ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR',    (0,1), (-1,-1), SLATE),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, ACCENT_LIGHT]),
        ('GRID',         (0,0), (-1,-1), 0.4, BORDER),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING',  (0,0), (-1,-1), 6),
    ]))
    story.append(ct2)
    story.append(Spacer(1, 6*mm))

    # ── 8. Technology Stack ───────────────────────────────────────────────
    story.append(SectionBar('Technology Stack', W_inner, '7.'))
    story.append(Spacer(1, 4*mm))

    stack_data = [
        ['Component', 'Technology', 'Environment'],
        ['Conversational AI',    'Google Gemini 2.5 Flash Lite',        'Production'],
        ['Full agent',           'LangGraph StateGraph ReAct',          'Production'],
        ['Agent framework',      'LangChain with tool calling',         'Production'],
        ['Tool protocol',        'FastMCP (Model Context Protocol)',     'Production'],
        ['PDF structure parser', 'Docling (DocLayNet + TableFormer)',    'Development only'],
        ['Vision extraction',    'Gemini Vision Flash — complex pages', 'Development only'],
        ['PDF image rendering',  'PyMuPDF (fitz)',                      'Development only'],
        ['Extraction fallback',  'MinerU',                              'Development only'],
        ['Semantic store',       'ChromaDB (cosine, HNSW)',             'Both'],
        ['Embeddings',           'Gemini text-embedding-004',           'Both'],
        ['API',                  'FastAPI with Server-Sent Events',     'Production'],
        ['Backend hosting',      'Railway (Docker, python:3.13-slim)',  'Production'],
        ['Frontend hosting',     'Vercel (static, global CDN)',         'Production'],
        ['Runtime',              'Python 3.13',                         'Both'],
    ]
    st2 = Table(stack_data, colWidths=[4.5*cm, 6.5*cm, 4.5*cm])
    st2.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), STEEL),
        ('TEXTCOLOR',    (0,0), (-1,0), WHITE),
        ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,-1), 8),
        ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR',    (0,1), (-1,-1), SLATE),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, ACCENT_LIGHT]),
        ('GRID',         (0,0), (-1,-1), 0.4, BORDER),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING',  (0,0), (-1,-1), 6),
    ]))
    story.append(st2)
    story.append(Spacer(1, 3*mm))

    # Build with two templates: full-bleed cover + normal content
    cover_tpl   = PageTemplate(id='Cover',   frames=[cover_frame])
    content_tpl = PageTemplate(id='Content', frames=[content_frame], onPage=on_page)

    doc = BaseDocTemplate(
        str(out_path), pagesize=A4,
        pageTemplates=[cover_tpl, content_tpl],
        title='Port Tariff AI — Architecture and Technical Approach',
        author='Hamza Shabbir',
    )
    doc.build(story)
    print(f"PDF written to: {out_path}")


if __name__ == '__main__':
    make_pdf(OUT_PATH)
