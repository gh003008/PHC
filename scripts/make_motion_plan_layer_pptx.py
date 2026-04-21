"""Generate Motion Plan Layer methodology/progress pptx (Pretendard font).

Structure (conceptual → practical):
 1 Title
 2 Agenda
 3 배경 및 필요성 (motivation)
 4 전체 프레임워크 (3 components + hierarchical human model)
 5 Motion Plan Layer — 설계 목표 & C→G→R 구조
 6 G의 학습: 왜 PHC인가
 7 우리의 변형 — PHC → VIC-extended MPL (G module)
 8 진행 계획 — Experiment Ladder
 9 Phase 0 — AMASS pipeline verify 결과
 10 Round 1 — Velocity Command 결과
 11 Round 2 — Personalization + Incident
 12 6개 실험 통합 요약
 13 남은 과제 + 한 문장 정리
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree


FONT = "Pretendard"
OUT = Path(r"C:\Main_Firmware\PHC\02_research_dev\260420_motion_plan_layer_method_progress.pptx")

# Colors
NAVY = RGBColor(0x14, 0x3C, 0x82)
DARK = RGBColor(0x1E, 0x1E, 0x1E)
GREY = RGBColor(0x66, 0x66, 0x66)
ACCENT = RGBColor(0x2C, 0x6A, 0xB8)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
OK = RGBColor(0x28, 0x8E, 0x4F)
WARN = RGBColor(0xB6, 0x6B, 0x0C)
BAD = RGBColor(0xB0, 0x2E, 0x2E)
SOFT = RGBColor(0xE7, 0xEE, 0xF8)


# ---------- text helpers ----------
def set_run(run, size=18, bold=False, color=DARK, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        for el in rPr.findall(qn(tag)):
            rPr.remove(el)
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FONT)


def add_textbox(slide, left, top, width, height, text, size=18, bold=False,
                color=DARK, align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    set_run(r, size=size, bold=bold, color=color, italic=italic)
    return tb


def add_bullets(slide, left, top, width, height, items, size=16, color=DARK,
                line_spacing=1.3, bullet_char="•"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        if isinstance(item, tuple):
            text, is_sub = item
            indent = "    " if is_sub else ""
            ch = "–" if is_sub else bullet_char
        else:
            text, ch, indent = item, bullet_char, ""
        r = p.add_run()
        r.text = f"{indent}{ch}  {text}"
        set_run(r, size=size, color=color)
    return tb


def add_rect(slide, left, top, width, height, fill=LIGHT, line=None, text=None,
             text_size=14, text_bold=False, text_color=DARK,
             align=PP_ALIGN.CENTER, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    shp = slide.shapes.add_shape(shape, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    if text is not None:
        tf = shp.text_frame
        tf.margin_left = Emu(80000)
        tf.margin_right = Emu(80000)
        tf.margin_top = Emu(40000)
        tf.margin_bottom = Emu(40000)
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = text
        set_run(r, size=text_size, bold=text_bold, color=text_color)
    return shp


def set_shape_text(shape, lines, default_size=13, default_color=DARK,
                   default_align=PP_ALIGN.LEFT, line_spacing=1.25,
                   margins=(100000, 100000, 60000, 60000)):
    """lines: list of dict {text, size, bold, color, align}"""
    tf = shape.text_frame
    tf.margin_left = Emu(margins[0]); tf.margin_right = Emu(margins[1])
    tf.margin_top = Emu(margins[2]); tf.margin_bottom = Emu(margins[3])
    tf.word_wrap = True
    for i, entry in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = entry.get("align", default_align)
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = entry["text"]
        set_run(r,
                size=entry.get("size", default_size),
                bold=entry.get("bold", False),
                color=entry.get("color", default_color))


def add_header(slide, slide_number, total, title, subtitle=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                 Inches(13.333), Inches(0.08))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    add_textbox(slide, Inches(0.5), Inches(0.22), Inches(11.8), Inches(0.55),
                title, size=26, bold=True, color=NAVY)
    if subtitle:
        add_textbox(slide, Inches(0.5), Inches(0.78), Inches(11.8), Inches(0.35),
                    subtitle, size=13, color=GREY)
    add_textbox(slide, Inches(12.2), Inches(7.0), Inches(1.0), Inches(0.3),
                f"{slide_number} / {total}", size=11, color=GREY, align=PP_ALIGN.RIGHT)
    add_textbox(slide, Inches(0.5), Inches(7.0), Inches(11.0), Inches(0.3),
                "Motion Plan Layer — Method & Progress  |  2026-04-20",
                size=10, color=GREY)


def add_table(slide, left, top, width, height, header, rows,
              header_fill=NAVY, header_color=RGBColor(0xFF, 0xFF, 0xFF),
              body_size=12, header_size=13, col_widths=None,
              col_aligns=None):
    rows_n = 1 + len(rows)
    cols_n = len(header)
    tbl = slide.shapes.add_table(rows_n, cols_n, left, top, width, height).table
    if col_widths is not None:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(width * (w / total)))
    for i, h in enumerate(header):
        cell = tbl.cell(0, i)
        cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
        tf = cell.text_frame
        tf.margin_left = Emu(60000); tf.margin_right = Emu(60000)
        tf.margin_top = Emu(30000); tf.margin_bottom = Emu(30000)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; p.text = ""
        r = p.add_run(); r.text = h
        set_run(r, size=header_size, bold=True, color=header_color)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if ri % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            tf = cell.text_frame
            tf.margin_left = Emu(60000); tf.margin_right = Emu(60000)
            tf.margin_top = Emu(25000); tf.margin_bottom = Emu(25000)
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (col_aligns[ci] if col_aligns else PP_ALIGN.CENTER)
            p.text = ""
            r = p.add_run(); r.text = str(val)
            set_run(r, size=body_size, color=DARK)
    return tbl


# ==================================================================
# MAIN
# ==================================================================
def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    total = 13

    # ==============================================================
    # Slide 1 — Title
    # ==============================================================
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                            Inches(13.333), Inches(7.5))
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(0xFA, 0xFB, 0xFD); bg.line.fill.background()
    side = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                              Inches(0.35), Inches(7.5))
    side.fill.solid(); side.fill.fore_color.rgb = NAVY; side.line.fill.background()

    add_textbox(s, Inches(1.0), Inches(1.55), Inches(11.0), Inches(0.5),
                "Human–Robot Integrated Simulation Framework",
                size=18, color=ACCENT, bold=True)
    add_textbox(s, Inches(1.0), Inches(2.15), Inches(11.5), Inches(1.5),
                "Motion Plan Layer\n — 개념 · 방법론 · 진행 현황",
                size=42, bold=True, color=NAVY)
    add_textbox(s, Inches(1.0), Inches(4.5), Inches(11.0), Inches(0.5),
                "웨어러블 로봇 학습을 위한 개인화된 Human Agent 상위 제어 계층의 설계와 구현",
                size=18, color=DARK)
    add_rect(s, Inches(1.0), Inches(5.7), Inches(4.5), Inches(0.7),
             fill=NAVY, text="2026-04-21  |  Youn Jimin",
             text_size=15, text_bold=True, text_color=RGBColor(0xFF, 0xFF, 0xFF))
    add_rect(s, Inches(5.7), Inches(5.7), Inches(6.5), Inches(0.7),
             fill=LIGHT,
             text="Focus: Velocity Command 설계 진단 → Retiming 방법론 → 두 트랙 진행 중",
             text_size=13, text_color=DARK)

    # ==============================================================
    # Slide 2 — Agenda
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 2, total, "Agenda", "개념 → 방법론 → 진행 → 결과 순서로 전개")
    items = [
        "① 배경 및 필요성 — 왜 통합 human-robot 시뮬레이션이 필요한가",
        "② 전체 프레임워크 — 3 구성요소와 계층적 Human Model",
        "③ Motion Plan Layer — 설계 목표와 C → G → R 구조",
        "④ G의 학습 방법론 — 왜 PHC가 적합한가",
        "⑤ 우리의 변형 — PHC → VIC-extended MPL (4 axes)",
        "⑥ 진단 — v_motion vs v_cmd의 reward 충돌",
        "⑦ 중간 해결 — Reference Retiming (단일 clip 위에서의 토대 검증)",
        "⑧ 현재 진행 — Two parallel tracks (Baseline A / Baseline B)",
        "⑨ 결과 — motivating evidence (Phase 0 / Round 1)",
        "⑩ 궤도 수정 — 왜 multi-clip이 유일한 답인가",
        "⑪ 남은 과제와 한 문장 정리",
    ]
    add_bullets(s, Inches(1.0), Inches(1.7), Inches(11.0), Inches(5.0),
                items, size=19, line_spacing=1.45)

    # ==============================================================
    # Slide 3 — 배경 및 필요성
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 3, total, "배경 및 필요성",
               "웨어러블 로봇의 제어 학습은 사용자와의 폐루프 상호작용 안에서 이루어져야 한다")

    # Top: why integrated sim
    add_rect(s, Inches(0.6), Inches(1.5), Inches(12.2), Inches(1.1),
             fill=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "통합 시뮬레이션이 필요한 이유",
         "size": 15, "bold": True, "color": RGBColor(0xFF, 0xFF, 0xFF)},
        {"text": ("웨어러블 로봇 제어 네트워크는 다양한 신체·보행 특성의 사용자와 "
                  "폐루프 상호작용 안에서 학습되어야 한다. 따라서 인간의 passive·반사·자발 특성을 "
                  "포함한 human agent, 로봇 동역학, 그리고 둘 사이의 물리적 상호작용을 "
                  "같은 루프 안에서 다룰 수 있는 통합 시뮬레이터가 필요하다."),
         "size": 13, "color": RGBColor(0xFF, 0xFF, 0xFF)},
    ], line_spacing=1.3, margins=(200000, 200000, 80000, 80000))

    # Two camps of existing approaches
    add_rect(s, Inches(0.6), Inches(2.8), Inches(5.95), Inches(0.45),
             fill=BAD, text="기존 접근 ① Muscle-action (MyoSuite, KINESIS)",
             text_size=14, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    add_bullets(s, Inches(0.75), Inches(3.35), Inches(5.8), Inches(1.7), [
        "Muscle activation을 직접 action으로 두는 end-to-end 구조",
        "개인화 = 전체 정책 재학습 → 비실용적",
        "정책 내부에 joint-level 좌표계(로봇이 개입할 지점)가 드러나지 않음",
    ], size=13, line_spacing=1.3)

    add_rect(s, Inches(6.78), Inches(2.8), Inches(6.05), Inches(0.45),
             fill=BAD, text="기존 접근 ② Humanoid-like tracker (고정 PD)",
             text_size=14, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    add_bullets(s, Inches(6.93), Inches(3.35), Inches(5.9), Inches(1.7), [
        "관절 궤적만을 action으로 두는 구조",
        "co-contraction·강성 조절 등 인간 제어 핵심 자유도 표현 불가",
        "같은 궤적에서의 외란 대응 양상 차이 담을 수 없음",
    ], size=13, line_spacing=1.3)

    # Conclusion strip
    add_rect(s, Inches(0.6), Inches(5.25), Inches(12.2), Inches(1.6),
             fill=SOFT, line=ACCENT)
    set_shape_text(s.shapes[-1], [
        {"text": "→ 해결되지 않는 것: 개인마다 다른 근력·수동 강성·반사 지연·보행 장애와 같은 구조적 변동성",
         "size": 14, "bold": True, "color": NAVY},
        {"text": ("특히 보행 장애가 있는 사용자는 정상 보행 분포에 대한 일반화만으로는 대응할 수 없다. "
                  "각 사용자 고유의 동작 스타일과 외란 대응 양식을 재현해야만 "
                  "웨어러블 로봇의 보조 전략을 개인화된 형태로 학습할 수 있다."),
         "size": 13, "color": DARK},
        {"text": "⇒ Human agent를 적절히 계층화한 새로운 통합 시뮬레이션 프레임워크가 필요.",
         "size": 13, "bold": True, "color": ACCENT},
    ], line_spacing=1.35, margins=(200000, 200000, 80000, 80000))

    # ==============================================================
    # Slide 4 — 전체 프레임워크
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 4, total, "전체 프레임워크 — 3 구성요소와 계층적 Human Model",
               "통합 시뮬레이션의 뼈대")

    # Three component boxes (top)
    comps = [
        ("Robot Model",
         "웨어러블 로봇의 조인트 동역학",
         ACCENT),
        ("Human–Robot Interaction",
         "착용 접촉을 통한 양방향 힘·운동 전달",
         NAVY),
        ("Human Model (본 보고 범위)",
         "Motion Plan Layer  +  Muscle Activation Layer",
         OK),
    ]
    for i, (title, desc, color) in enumerate(comps):
        x = Inches(0.6 + i * 4.15)
        add_rect(s, x, Inches(1.5), Inches(4.0), Inches(1.15), fill=color)
        set_shape_text(s.shapes[-1], [
            {"text": title, "size": 15, "bold": True,
             "color": RGBColor(0xFF, 0xFF, 0xFF), "align": PP_ALIGN.CENTER},
            {"text": desc, "size": 12,
             "color": RGBColor(0xFF, 0xFF, 0xFF), "align": PP_ALIGN.CENTER},
        ], line_spacing=1.3, margins=(100000, 100000, 80000, 60000))

    # Human Model detail — two-layer structure
    add_rect(s, Inches(0.6), Inches(2.9), Inches(12.2), Inches(0.45),
             fill=OK, text="Human Model — 두 계층의 역할 분담",
             text_size=14, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)

    # MPL (highlighted) and MAL box
    add_rect(s, Inches(0.6), Inches(3.45), Inches(6.0), Inches(1.55),
             fill=NAVY, line=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "상위  ·  Motion Plan Layer  ★ 본 보고 범위",
         "size": 14, "bold": True, "color": RGBColor(0xFF, 0xFF, 0xFF)},
        {"text": ("의도·coordination·강성 수준의 제어.  "
                  "reference 관절 궤적 q_ref + 저차원 impedance κ 생성."),
         "size": 12, "color": RGBColor(0xFF, 0xFF, 0xFF)},
        {"text": "→ “어떤 스타일로, 얼마나 단단하게 움직일 것인가”",
         "size": 12, "italic": True, "color": SOFT},
    ], line_spacing=1.35, margins=(180000, 180000, 80000, 60000))

    add_rect(s, Inches(6.8), Inches(3.45), Inches(6.0), Inches(1.55),
             fill=LIGHT, line=GREY)
    set_shape_text(s.shapes[-1], [
        {"text": "하위  ·  Muscle Activation Layer", "size": 14, "bold": True, "color": DARK},
        {"text": ("피험자별 근력·수동 강성·반사 지연 등 근육·반사 동역학을 반영. "
                  "상위 reference를 근 활성화 → 실제 토크로 실현."),
         "size": 12, "color": DARK},
        {"text": "→ “subject-specific 실행기 (별도 라인)”",
         "size": 12, "italic": True, "color": GREY},
    ], line_spacing=1.35, margins=(180000, 180000, 80000, 60000))

    # Why layered (4 reasons — compact)
    add_rect(s, Inches(0.6), Inches(5.2), Inches(12.2), Inches(0.4),
             fill=LIGHT, line=ACCENT,
             text="왜 계층적으로 분리하나?",
             text_size=13, text_bold=True, text_color=ACCENT, align=PP_ALIGN.LEFT)
    add_bullets(s, Inches(0.75), Inches(5.65), Inches(12.0), Inches(1.35), [
        "① 개인화의 구조적 분해 — passive 생리 특성은 identification, coordination 차이는 학습",
        "② 궤적 너머의 강성·순응성 표현력 — κ가 있기 때문에 co-contraction·강성 조절 표현 가능",
        "③ 로봇과 공통 좌표계 위에서의 해석 — 실패 모드를 상위/하위로 분리 진단",
        "④ 인간 운동제어 계층과의 정합 — 의도 → coordination → 근활성 흐름을 그대로 반영",
    ], size=12, line_spacing=1.3)

    # ==============================================================
    # Slide 5 — MPL goals + C→G→R structure
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 5, total, "Motion Plan Layer — 설계 목표와 C → G → R 구조",
               "세 가지 요구를 동시에 만족하는 상위 제어 계층")

    # Left: 3 design goals
    add_rect(s, Inches(0.6), Inches(1.5), Inches(5.9), Inches(0.45),
             fill=ACCENT, text="설계 목표 — 동시에 만족해야 할 3 요구",
             text_size=14, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    add_bullets(s, Inches(0.75), Inches(2.05), Inches(5.8), Inches(3.3), [
        "① 임의 CoM 입력에 대한 동작 생성",
        "     · 주어진 u^user를 feasibility로 projection 후 tracking",
        "② 한 사람의 다양한 데이터 기반 개인화",
        "     · gait style·습관·반응 양식이 평균화되지 않음",
        "③ 위기상황에서의 능동적 반응",
        "     · residual correction을 넘어 step placement / timing reset / 강성 재구성",
    ], size=13, line_spacing=1.35)

    # Right: C → G → R structure
    add_rect(s, Inches(6.75), Inches(1.5), Inches(6.05), Inches(0.45),
             fill=NAVY, text="구조  ·  C → G → R (직렬 연결)",
             text_size=14, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    # C / G / R cards
    cgr = [
        ("C", "CoM Intention Projection",
         "LIP·XCoM 기반 QP projection. 학습 없이 매 tick 실행.  "
         "u^user를 feasible set으로 정규화해 ũ 출력.",
         GREY),
        ("G", "Nominal Gait Generator  ★ 현재 연구 집중",
         "Subject-specific 시퀀스 모델.  ũ · state history · retrieval된 reference window · "
         "style latent z_gait → (q_ref, κ_prior).",
         NAVY),
        ("R", "Structured Residual Correction",
         "α_t(불안정도)·φ_t(gait phase) 조건부 additive 보정. "
         "Fast/strategic 채널별로 학습.",
         ACCENT),
    ]
    y0 = Inches(2.05)
    card_h = Inches(1.05)
    gap = Inches(0.1)
    for i, (tag, title, desc, color) in enumerate(cgr):
        y = Emu(y0 + (card_h + gap) * i)
        # tag
        add_rect(s, Inches(6.75), y, Inches(0.95), card_h, fill=color)
        set_shape_text(s.shapes[-1], [
            {"text": tag, "size": 30, "bold": True,
             "color": RGBColor(0xFF, 0xFF, 0xFF), "align": PP_ALIGN.CENTER},
        ], margins=(20000, 20000, 60000, 20000))
        # body
        add_rect(s, Inches(7.8), y, Inches(5.0), card_h, fill=LIGHT, line=color)
        set_shape_text(s.shapes[-1], [
            {"text": title, "size": 13, "bold": True, "color": color},
            {"text": desc, "size": 11, "color": DARK},
        ], line_spacing=1.3, margins=(120000, 120000, 60000, 40000))

    # Bottom — composition equation
    add_rect(s, Inches(0.6), Inches(5.55), Inches(12.2), Inches(1.3),
             fill=SOFT, line=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "합성 규칙 — additive (게이팅 없음)",
         "size": 13, "bold": True, "color": NAVY, "align": PP_ALIGN.LEFT},
        {"text": "u_t  =  u_t^G  +  Δu_t^R        (R은 항상 켜져 있되 내부적으로 α_t·φ_t에 조건부 반응)",
         "size": 14, "bold": True, "color": DARK, "align": PP_ALIGN.CENTER},
        {"text": ("β(α_t)·R 형태의 외부 게이팅을 쓰지 않는 이유 — α_t noise로 인한 chattering 방지, "
                  "학습 분포 내에서 시간 스케일이 자연스럽게 분리되도록."),
         "size": 11, "italic": True, "color": GREY, "align": PP_ALIGN.LEFT},
    ], line_spacing=1.35, margins=(200000, 200000, 80000, 80000))

    # ==============================================================
    # Slide 6 — Why PHC for G (PHC 원본이 이미 제공하는 것)
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 6, total, "왜 PHC인가 — 원본이 이미 제공하는 것",
               "G를 새로 짜지 않아도 되는 이유 · “기본 제공 ↔ G 요구사항” 대응")

    # Intro strip
    add_rect(s, Inches(0.6), Inches(1.5), Inches(12.2), Inches(0.75),
             fill=SOFT, line=ACCENT)
    set_shape_text(s.shapes[-1], [
        {"text": ("PHC는 대규모 MoCap을 단일 goal-conditioned policy로 모사하는 humanoid motion "
                  "tracker. G가 요구하는 기능 다수가 원본에 이미 들어 있음 — 우리는 “그 위에” 얹기만 하면 된다."),
         "size": 13, "color": DARK, "align": PP_ALIGN.LEFT},
    ], margins=(200000, 200000, 60000, 60000))

    # Table: PHC 기본 기능 ↔ G 요구 대응
    header = ["PHC 원본이 기본 제공", "어떤 형태로", "G 요구 대응"]
    rows = [
        ["Goal-conditioned motion tracking",
         "매 step마다 motion reference의 미래 K 프레임(q·pos·vel)을 task obs로",
         "ũ에 해당하는 “무엇을 따라갈지” 신호"],
        ["AMP style discriminator",
         "Demo clip vs rollout 분포를 판별 → style reward 자동 부여",
         "Subject style 유지 기전 (D_S의 뼈대)"],
        ["per-env shape variation",
         "has_shape_variation=True → 각 env에 다른 SMPL betas robot 생성",
         "다양한 체형 노출 (personalization 재료)"],
        ["Shape observation",
         "has_shape_obs=True → policy obs에 betas 10 dims 추가",
         "Subject-style latent의 시뮬 현실판"],
        ["Action-space 확장 용이",
         "action dim을 yaml로 제어, 토크 계산 파이프라인이 dim 독립",
         "Impedance(VIC) latent κ_prior 붙이기 쉬움"],
        ["PPO + imitation + survival reward",
         "rl-games 기반, reward spec yaml 구성",
         "향후 R 단계의 recovery RL과 정합"],
    ]
    add_table(s, Inches(0.6), Inches(2.4), Inches(12.2), Inches(3.8),
              header, rows, header_size=12, body_size=11,
              col_widths=[3.4, 5.0, 3.8],
              col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

    # Conclusion strip
    add_rect(s, Inches(0.6), Inches(6.3), Inches(12.2), Inches(0.6),
             fill=NAVY,
             text=("⇒ PHC를 G의 골격으로 채택.  다음 페이지 = “그 위에 우리가 새로 얹은 것 vs 기본 flag로 켠 것”."),
             text_size=13, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)

    # ==============================================================
    # Slide 7 — 우리의 변형 (구현 종류별 분리)
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 7, total, "우리의 변형 — PHC → VIC-extended MPL",
               "“새로 코드로 구현한 축”과 “PHC 원본 기능을 flag로 켠 축”을 구분")

    # Section A: 새로 구현
    add_rect(s, Inches(0.6), Inches(1.5), Inches(12.2), Inches(0.45),
             fill=NAVY, text="A · 새로 코드로 구현한 축  (원본 PHC에는 없음)",
             text_size=13, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)

    # A1 — VIC
    add_rect(s, Inches(0.6), Inches(2.0), Inches(5.95), Inches(2.0),
             fill=LIGHT, line=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "A-1  ·  Impedance latent (VIC)", "size": 13, "bold": True, "color": NAVY},
        {"text": "왜 — MPL의 κ_prior 요구. 동일 궤적에서도 co-contraction·강성 조절 표현.",
         "size": 11, "color": DARK},
        {"text": "어떻게 — humanoid_im_vic.py · 4-group CCF · Stage 1/2 커리큘럼",
         "size": 11, "color": DARK},
        {"text": "토크식 — τ = kp·2^ccf·(q_ref − q) − kd·2^ccf·q̇   (ccf는 policy action dim)",
         "size": 11, "color": ACCENT},
    ], line_spacing=1.35, margins=(150000, 150000, 80000, 80000))

    # A2 — Velocity command (evolution)
    add_rect(s, Inches(6.78), Inches(2.0), Inches(6.05), Inches(2.0),
             fill=LIGHT, line=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "A-2  ·  Velocity Command  —  설계가 진화 중", "size": 13, "bold": True, "color": NAVY},
        {"text": ("왜 — PHC 원본 goal은 “motion frame 따라가라”뿐. 같은 걸음을 다양한 속도로 내려면 "
                  "CoM 속도 축이 필요."),
         "size": 11, "color": DARK},
        {"text": ("v1 (Round 1) — humanoid_im_vic_cmd.py · obs +3 · reward blend "
                  "(1−w)·base + w·cmd_track"),
         "size": 11, "color": DARK},
        {"text": ("v2 (Round 3, 진행 중) — humanoid_im_vic_cmd_retime.py · motion 자체를 retime · "
                  "cmd를 teacher 안으로"),
         "size": 11, "bold": True, "color": ACCENT},
    ], line_spacing=1.35, margins=(150000, 150000, 80000, 80000))

    # Section B: flag로 켠 축 / deferred
    add_rect(s, Inches(0.6), Inches(4.2), Inches(12.2), Inches(0.45),
             fill=OK, text="B · PHC 원본 기능을 yaml flag로 켠 축  (코드 변경 없음)",
             text_size=13, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)

    # B1 — personalization (shelved)
    add_rect(s, Inches(0.6), Inches(4.7), Inches(5.95), Inches(2.15),
             fill=LIGHT, line=GREY)
    set_shape_text(s.shapes[-1], [
        {"text": "B-1  ·  Personalization (shape variation + shape obs)  —  현재 보류",
         "size": 13, "bold": True, "color": GREY},
        {"text": ("개념 — robot.has_shape_variation + has_shape_obs 두 줄로 per-env 다른 SMPL betas · "
                  "obs에 betas 10 dims."),
         "size": 11, "color": DARK},
        {"text": ("Round 2(V2_A/B) 42% 지점에서 디스크 중단 → 드롭 결정:  "
                  "“단일-clip 위에 shape variation은 동기 부족”."),
         "size": 11, "color": DARK},
        {"text": "향후 — retime plateau 후 multi-clip 확장 시점에 재도입 검토.",
         "size": 11, "italic": True, "color": GREY},
    ], line_spacing=1.32, margins=(150000, 150000, 80000, 80000))

    # B2 — AMP discriminator
    add_rect(s, Inches(6.78), Inches(4.7), Inches(6.05), Inches(2.15),
             fill=LIGHT, line=OK)
    set_shape_text(s.shapes[-1], [
        {"text": "B-2  ·  AMP Style Discriminator",
         "size": 13, "bold": True, "color": OK},
        {"text": ("개념 — rl-games 기본 AMP path. 단일 subject demo라 population disc가 사실상 D_S 역할. "
                  "has_shape_obs_disc=False."),
         "size": 11, "color": DARK},
        {"text": ("Retime에 맞춰 AMP demo velocity도 per-demo scale(s')로 사후 스케일 → "
                  "positive 분포를 retimed rollout과 일치."),
         "size": 11, "color": DARK},
        {"text": "향후 — multi-subject library 확보 시 subject-filtered D_S로 자연스럽게 전환.",
         "size": 11, "italic": True, "color": ACCENT},
    ], line_spacing=1.32, margins=(150000, 150000, 80000, 80000))

    # ==============================================================
    # Slide 8 — 진단: v_motion vs v_cmd의 reward 충돌
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 8, total, "진단 — v_motion vs v_cmd의 reward 충돌",
               "먼저 용어부터: 두 속도와 두 reward를 구분해 정의")

    # Definitions box — 4 rows
    add_rect(s, Inches(0.6), Inches(1.45), Inches(12.2), Inches(0.4),
             fill=NAVY, text="정의  ·  두 속도와 두 reward",
             text_size=13, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)

    def_table = [
        ["기호",                 "무엇",                       "어떻게 정해지나"],
        ["v_motion",
         "Reference motion(AMASS forward walking clip)의 natural pelvis forward 속도",
         "Clip 내용 자체로 고정. 단일 clip이면 한 값."],
        ["v_cmd",
         "Task obs에 주입되는 명령 속도 (현재 구현: [v_cmd_x, v_cmd_y, ω_cmd] 3 dims)",
         "매 env reset마다  v_cmd_x ~ U(0.8, 1.3) m/s  로 독립 샘플  (v_motion과 무관)"],
        ["base imitation rwd",
         "시뮬 상태와 reference motion 상태의 일치도 (pose + body velocities 포함)",
         "간접적으로  v_motion  을 추종하게 유도"],
        ["cmd tracking rwd",
         "exp(−2·‖v_pelvis − v_cmd‖² − |ω_pelvis − ω_cmd|²)",
         "직접적으로  v_cmd  를 추종하게 유도"],
    ]
    add_table(s, Inches(0.6), Inches(1.9), Inches(12.2), Inches(2.0),
              def_table[0], def_table[1:],
              header_size=11, body_size=11,
              col_widths=[1.8, 5.4, 5.0],
              col_aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

    # Conflict + Evidence (side by side)
    add_rect(s, Inches(0.6), Inches(4.05), Inches(5.95), Inches(2.4),
             fill=LIGHT, line=BAD)
    set_shape_text(s.shapes[-1], [
        {"text": "두 reward가 가리키는 방향",
         "size": 13, "bold": True, "color": BAD},
        {"text": "•  base imitation   →   match  v_motion  (고정)",
         "size": 12, "color": DARK},
        {"text": "•  cmd tracking     →   match  v_cmd  (랜덤)",
         "size": 12, "color": DARK},
        {"text": ""},
        {"text": ("v_cmd ≠ v_motion 인 에피소드에서 정책은 두 reward의 가중합을 타협. "
                  "cmd_tracking_w 만 바꿔선 근본 충돌은 유지."),
         "size": 11, "italic": True, "color": DARK},
        {"text": ("주: reward 외에 AMP discriminator도 demo motion의 “속도 분포”를 보고 판별 "
                  "→ 충돌 채널이 하나 더 있음."),
         "size": 11, "italic": True, "color": GREY},
    ], line_spacing=1.3, margins=(180000, 180000, 60000, 60000))

    add_rect(s, Inches(6.78), Inches(4.05), Inches(6.05), Inches(2.4),
             fill=SOFT, line=ACCENT)
    set_shape_text(s.shapes[-1], [
        {"text": "증거 — Round 1 실측", "size": 13, "bold": True, "color": ACCENT},
        {"text": "•  CMD_A (w=0.3) : 실패율 35%,  av_rwd 527", "size": 12, "color": DARK},
        {"text": "•  CMD_B (w=0.5) : 실패율 14%,  av_rwd 440", "size": 12, "color": DARK},
        {"text": "Best-ep 역산 (cmd_reward≈1 가정) → implied base", "size": 12, "color": DARK},
        {"text": "•  CMD_A best 688.7  ⇒  983  (> VIC4 951)", "size": 12, "color": DARK},
        {"text": "•  CMD_B best 507.2  ⇒  1013  (> VIC4 951)", "size": 12, "color": DARK},
        {"text": ("⇒ 성공 시 품질은 baseline 초과.  비용은 “실패 에피소드 비율”로 나타남."),
         "size": 11, "italic": True, "color": ACCENT},
    ], line_spacing=1.3, margins=(180000, 180000, 60000, 60000))

    # Conclusion
    add_rect(s, Inches(0.6), Inches(6.6), Inches(12.2), Inches(0.35),
             fill=BAD,
             text=("결론 —  단일 clip에서  v_cmd ≠ v_motion 이 되는 순간마다 "
                   "base imitation ↔ cmd tracking 이 구조적으로 충돌.  "
                   "학습량이 아니라 reward 구조의 문제."),
             text_size=12, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)

    # ==============================================================
    # Slide 9 — 해결: Reference Retiming 방법론
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 9, total, "해결 — Reference Retiming 방법론",
               "Command를 “별도 reward 항”이 아니라 “teacher motion의 재생 속도”로 옮김")

    # Core idea box
    add_rect(s, Inches(0.6), Inches(1.5), Inches(12.2), Inches(1.1),
             fill=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "핵심 아이디어",
         "size": 14, "bold": True, "color": RGBColor(0xFF, 0xFF, 0xFF)},
        {"text": ("매 env reset마다 per-env speed scale  s ~ U(0.9, 1.1)  샘플. "
                  "Reference motion을 rate s로 재생 → retimed teacher의 pelvis 속도가 이미 s·v_nat. "
                  "Imitation reward를 그대로 쓰면 teacher 자체가 명령 신호를 담게 된다."),
         "size": 12, "color": RGBColor(0xFF, 0xFF, 0xFF)},
    ], line_spacing=1.35, margins=(200000, 200000, 80000, 80000))

    # Math strip
    add_rect(s, Inches(0.6), Inches(2.75), Inches(12.2), Inches(0.75),
             fill=LIGHT, line=ACCENT)
    set_shape_text(s.shapes[-1], [
        {"text": "수식",
         "size": 12, "bold": True, "color": ACCENT, "align": PP_ALIGN.LEFT},
        {"text": "retimed  =  start  +  s · (t − start)       "
                 "v_ref_retimed  =  s · v_ref(retimed)          "
                 "(위치는 retimed time에서, 속도는 chain rule로 s배)",
         "size": 12, "bold": True, "color": DARK, "align": PP_ALIGN.LEFT},
    ], line_spacing=1.25, margins=(200000, 200000, 60000, 60000))

    # Three implementation pillars
    pillars = [
        ("① Motion retiming",
         "_get_state_from_motionlib_cache를 wrapping.\n"
         "motion_times → start + s·(t−start), 그리고 "
         "root_vel / root_ang_vel / dof_vel / body_vel / body_ang_vel 모두 s배.",
         NAVY),
        ("② AMP demo 분포 일치",
         "AMP discriminator가 절대 속도 feature를 봄 → demo velocity도 per-demo s' 샘플로 s'배. "
         "Positive 분포가 retimed rollout과 일치하도록.",
         ACCENT),
        ("③ Command은 obs에만 보존",
         "v_cmd_x = s · v_nat 을 기존 3-dim obs 슬롯에 기록 → 네트워크 구조는 그대로. "
         "cmd_tracking_w = 0.0 (명시적 cmd reward 없음).",
         OK),
    ]
    y0 = Inches(3.65)
    card_h = Inches(1.1)
    card_gap = Inches(0.12)
    for i, (title, body, color) in enumerate(pillars):
        y = Emu(y0 + (card_h + card_gap) * i)
        add_rect(s, Inches(0.6), y, Inches(12.2), card_h, fill=LIGHT, line=color)
        set_shape_text(s.shapes[-1], [
            {"text": title, "size": 13, "bold": True, "color": color},
            {"text": body, "size": 12, "color": DARK},
        ], line_spacing=1.3, margins=(200000, 200000, 60000, 60000))

    # meaning strip
    add_rect(s, Inches(0.6), Inches(7.05), Inches(12.2), Inches(0.35),
             fill=SOFT, line=ACCENT,
             text="의미 — imitation-vs-cmd 구조적 충돌 제거.  “teacher가 곧 명령”이 되어 base reward가 자기모순 없음.",
             text_size=12, text_bold=True, text_color=ACCENT, align=PP_ALIGN.LEFT)

    # ==============================================================
    # Slide 10 — 현재 진행: Two parallel tracks + 평가 프로토콜
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 10, total, "현재 진행 — Two Parallel Tracks (2026-04-21)",
               "같은 질문을 두 각도에서 동시에 공격")

    # Two tracks side by side
    add_rect(s, Inches(0.6), Inches(1.5), Inches(5.95), Inches(0.5),
             fill=ACCENT, text="Baseline A  —  Ceiling check  (idx0 GPU)",
             text_size=14, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    add_rect(s, Inches(0.6), Inches(2.05), Inches(5.95), Inches(3.0),
             fill=LIGHT, line=ACCENT)
    set_shape_text(s.shapes[-1], [
        {"text": "질문", "size": 11, "bold": True, "color": ACCENT},
        {"text": "현재 설계가 덜 학습된 건 아닌지? 천장이 더 높지 않을까?",
         "size": 12, "color": DARK},
        {"text": "구성", "size": 11, "bold": True, "color": ACCENT},
        {"text": ("CMD_B checkpoint (20k) 이어서 → 30k epochs.  "
                  "코드 변경 없음. save_frequency 2500."),
         "size": 12, "color": DARK},
        {"text": "근거", "size": 11, "bold": True, "color": ACCENT},
        {"text": "Ep 20k에서 eps_len 185 → 202로 여전히 성장 중이었음. 미수렴 의심.",
         "size": 12, "color": DARK},
        {"text": "Sbatch", "size": 11, "bold": True, "color": ACCENT},
        {"text": "260421_AMASS_CMD_B_EXTEND/train_cmd_B_extend_gpu0.sh",
         "size": 11, "color": DARK},
    ], line_spacing=1.3, margins=(150000, 150000, 80000, 60000))

    add_rect(s, Inches(6.78), Inches(1.5), Inches(6.05), Inches(0.5),
             fill=NAVY, text="Baseline B  —  Structural fix (Retime R1)  (idx1 GPU)",
             text_size=14, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    add_rect(s, Inches(6.78), Inches(2.05), Inches(6.05), Inches(3.0),
             fill=LIGHT, line=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "질문", "size": 11, "bold": True, "color": NAVY},
        {"text": "Reward 충돌을 제거하면 VIC4 baseline 수준(100% success)을 회복할까?",
         "size": 12, "color": DARK},
        {"text": "구성", "size": 11, "bold": True, "color": NAVY},
        {"text": ("HumanoidImVICCmdRetime · s ∈ U(0.9, 1.1) · cmd_tracking_w = 0.0 · "
                  "fresh 20k epochs."),
         "size": 12, "color": DARK},
        {"text": "근거", "size": 11, "bold": True, "color": NAVY},
        {"text": "Teacher 자체가 retimed → base imitation만으로 명령 신호가 전달됨.",
         "size": 12, "color": DARK},
        {"text": "Sbatch", "size": 11, "bold": True, "color": NAVY},
        {"text": "260421_AMASS_RETIME/train_retime_R1_gpu1.sh",
         "size": 11, "color": DARK},
    ], line_spacing=1.3, margins=(150000, 150000, 80000, 60000))

    # Evaluation protocol
    add_rect(s, Inches(0.6), Inches(5.25), Inches(12.2), Inches(0.45),
             fill=OK, text="공통 평가 프로토콜  (두 트랙 종료 후 동일 배터리)",
             text_size=13, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    add_bullets(s, Inches(0.75), Inches(5.8), Inches(12.1), Inches(1.3), [
        "Test-greedy at fixed commands: v ∈ {0.90, 0.95, 1.00, 1.05, 1.10} · v_nat — success / av_steps / mean pelvis v / v_err / imitation rwd",
        "Best-episode 역산으로 bin별 implied base imitation 계산",
        "실패 모드 분포: slow / near-natural / fast 중 어디서 실패가 몰리는지",
        "→ 정리: 02_research_dev/260422_retime_vs_baseline_analysis.md (두 job 모두 종료 후)",
    ], size=12, line_spacing=1.3)

    # ==============================================================
    # Slide 11 — 결과 (motivating evidence)
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 11, total, "결과 — 지금까지의 motivating evidence",
               "Phase 0 · Round 1 은 “왜 Retime이 필요한가”를 드러낸 증거로 해석")

    header = ["단계", "실험", "설정", "Test-greedy rwd / steps", "Success", "역할"]
    rows = [
        ["Phase 0", "VIC4",
         "no cmd, 4-group",
         "951.19 / 299", "100%",
         "baseline: 파이프라인 건강"],
        ["Phase 0", "VIC8",
         "no cmd, 8-group",
         "940.71 / 299", "100%",
         "상체 2 dims는 노이즈 — 4-group mainline 확정"],
        ["Round 1", "CMD_A",
         "w=0.3, v_cmd U(0.8,1.3)",
         "526.95 / 236.4", "65.5%",
         "imitation 치우친 설정"],
        ["Round 1", "CMD_B ★",
         "w=0.5, v_cmd U(0.8,1.3)",
         "440.39 / 278.7", "86.1%",
         "cmd 치우친 설정 → Baseline A로 연장 중"],
        ["Round 2",
         "V2_A / V2_B",
         "+ shape variation",
         "(중단 / segfault)", "—",
         "드롭: 단일-clip 위 shape 동기 부족"],
    ]
    add_table(s, Inches(0.5), Inches(1.6), Inches(12.4), Inches(3.5),
              header, rows, body_size=11, header_size=12,
              col_widths=[1.1, 1.8, 2.5, 2.6, 1.1, 3.3],
              col_aligns=[PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.LEFT,
                          PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.LEFT])

    # Reinterpretation box
    add_rect(s, Inches(0.5), Inches(5.3), Inches(12.4), Inches(1.55),
             fill=SOFT, line=ACCENT)
    set_shape_text(s.shapes[-1], [
        {"text": "해석 (Retime 관점에서 다시 읽음)",
         "size": 13, "bold": True, "color": ACCENT},
        {"text": ("①  Phase 0 — VIC4 100% 성공, 951점. “단일 motion + 단일 속도”라 reward 충돌 자체가 없음."),
         "size": 12, "color": DARK},
        {"text": ("②  Round 1 — v_cmd가 motion의 natural speed와 어긋나는 비율만큼 실패가 누적. "
                  "Best-ep 시 imitation이 VIC4 초과인 것은 “충돌만 없으면 품질은 이미 있다”는 증거."),
         "size": 12, "color": DARK},
        {"text": ("③  Round 2 — 드롭. 기초 설계 충돌을 먼저 없애고 multi-clip 단계에서 재검토."),
         "size": 12, "color": DARK},
    ], line_spacing=1.3, margins=(200000, 200000, 60000, 60000))

    # ==============================================================
    # Slide 12 — 궤도 수정: Multi-clip Retrieval + Retime Hybrid
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 12, total, "궤도 수정 — Multi-clip Retrieval + Retime Hybrid",
               "“임의 v_cmd → 적절한 학습 동작”은 단일 clip에선 원리적으로 불가능")

    # Top — core realization
    add_rect(s, Inches(0.6), Inches(1.45), Inches(12.2), Inches(0.95),
             fill=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "핵심 인식",
         "size": 14, "bold": True, "color": RGBColor(0xFF, 0xFF, 0xFF)},
        {"text": ("Retime은 단일 clip ±10% 범위에서만 의미 있음.  “임의 속도 커맨드에 적절한 동작”의 진짜 답은 "
                  "multi-clip motion library + command-indexed retrieval + per-clip retime 의 3단 hybrid."),
         "size": 12, "color": RGBColor(0xFF, 0xFF, 0xFF)},
    ], line_spacing=1.35, margins=(200000, 200000, 60000, 60000))

    # Structure pseudocode
    add_rect(s, Inches(0.6), Inches(2.5), Inches(6.95), Inches(3.0),
             fill=LIGHT, line=ACCENT)
    set_shape_text(s.shapes[-1], [
        {"text": "구조  —  Episode reset마다",
         "size": 13, "bold": True, "color": ACCENT},
        {"text": "", "size": 6},
        {"text": "① v_cmd 샘플  ( 훈련 분포 내 )",
         "size": 12, "color": DARK},
        {"text": "② clip*  =  argmin  | v_motion(clip) − v_cmd |     # retrieval",
         "size": 12, "color": DARK},
        {"text": "③ s  =  v_cmd / v_motion(clip*)                        # retime scale",
         "size": 12, "color": DARK},
        {"text": "④ teacher  =  retime(clip*, scale = s)",
         "size": 12, "color": DARK},
        {"text": "", "size": 6},
        {"text": "훈련 reward — base imitation만  ( cmd_tracking_w = 0 )",
         "size": 12, "bold": True, "color": ACCENT},
        {"text": "→  retimed teacher가 이미 v_cmd 를 담음 → 구조적 충돌 없음",
         "size": 11, "italic": True, "color": DARK},
    ], line_spacing=1.3, margins=(180000, 180000, 80000, 60000))

    # Coverage diagram (6 clips example)
    add_rect(s, Inches(7.75), Inches(2.5), Inches(5.05), Inches(3.0),
             fill=SOFT, line=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": "Continuous coverage — 유한 clip으로",
         "size": 13, "bold": True, "color": NAVY},
        {"text": "", "size": 6},
        {"text": "예: 6 clip 등비수열  (ratio 1.20)",
         "size": 12, "color": DARK},
        {"text": "     0.60 — 0.72 — 0.86 — 1.03 — 1.24 — 1.48  m/s",
         "size": 11, "color": DARK},
        {"text": "", "size": 6},
        {"text": "각 clip ± 10% retime bandwidth →",
         "size": 12, "color": DARK},
        {"text": "전체 [ 0.54, 1.63 ] m/s 연속 커버",
         "size": 12, "bold": True, "color": NAVY},
        {"text": "", "size": 6},
        {"text": "Clip 간격은 Baseline B 결과로 결정 —",
         "size": 11, "italic": True, "color": DARK},
        {"text": "  success≥95% at ±10% → 간격 ≤ 20%",
         "size": 11, "italic": True, "color": DARK},
        {"text": "  부분 성공 → 간격 좁힘 (더 많은 clip)",
         "size": 11, "italic": True, "color": DARK},
    ], line_spacing=1.3, margins=(180000, 180000, 80000, 60000))

    # Evaluation protocol strip
    add_rect(s, Inches(0.6), Inches(5.6), Inches(12.2), Inches(0.45),
             fill=OK, text="Acceptance Criterion  —  “훈련 범위 내 임의 v_cmd 추종 + 안 넘어짐”",
             text_size=13, text_bold=True,
             text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    add_bullets(s, Inches(0.75), Inches(6.15), Inches(12.1), Inches(1.0), [
        "Fixed v_cmd bins (5~10개) × N=50~100 에피소드 — 각 bin당 success≥95%, v_err_rms ≤ 0.10 m/s",
        "Clip 경계 근처 v_cmd 집중 샘플 → retime bandwidth edge에서의 robustness 확인",
        "Ramp / step test → 에피소드 중 v_cmd 변화 대응력 (장기적으로 time-varying command의 전제)",
    ], size=12, line_spacing=1.3)

    # ==============================================================
    # Slide 13 — Remaining + one-sentence
    # ==============================================================
    s = prs.slides.add_slide(blank)
    add_header(s, 13, total, "남은 과제 & 한 문장 정리",
               "Baseline A/B 완료 → multi-clip 파이프라인 → 첫 multi-clip 학습까지")

    boxes = [
        ("W0  —  Baseline A/B 완료 + recipe 확정", OK,
         ["두 트랙 완료 후 fixed-v 배터리로 평가",
          "Retime bandwidth 결과 → clip 간격 파라미터 결정",
          "260422_retime_vs_baseline_analysis.md 정리"]),
        ("W0~W1 병렬  —  데이터 파이프라인", ACCENT,
         ["AMASS walking subset 필터링 (label or rule 기반)",
          "Per-clip annotation: v_mean, cadence, stride, duration",
          "amass_walking_subset_v0.pkl + index.json 산출"]),
        ("W1~W2  —  Multi-clip 첫 학습", NAVY,
         ["HumanoidImVICCmdMulti 구현 + motion_lib retrieval 경로",
          "AMP demo pool을 multi-clip × per-demo s'로 확장",
          "Narrow [0.7, 1.3] + Wider [0.5, 1.5] 두 variant 20k"]),
        ("Deferred / 장기", GREY,
         ["Personalization (B-1) — multi-clip 안정화 후 재도입",
          "Turning (ω_cmd) — turning clip 추가 이후",
          "Cadence 축 retrieval · rl-games resume epoch_num 보존"]),
    ]

    positions = [(0.6, 1.5), (6.93, 1.5), (0.6, 3.75), (6.93, 3.75)]
    for (title, color, items), (x, y) in zip(boxes, positions):
        add_rect(s, Inches(x), Inches(y), Inches(5.8), Inches(0.45),
                 fill=color, text=title,
                 text_size=14, text_bold=True,
                 text_color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
        add_bullets(s, Inches(x + 0.1), Inches(y + 0.55), Inches(5.7), Inches(1.7),
                    items, size=13, line_spacing=1.3)

    add_rect(s, Inches(0.6), Inches(6.1), Inches(12.2), Inches(0.8), fill=NAVY)
    set_shape_text(s.shapes[-1], [
        {"text": ("단일 clip + 랜덤 v_cmd는 정의 불능.  Multi-clip library + command-indexed retrieval + "
                  "per-clip retime hybrid로 구조적으로 해결한다 — 현재 Baseline A/B는 이 hybrid의 "
                  "bandwidth·reward-구조 recipe를 확정하는 진단 실험."),
         "size": 13, "bold": True,
         "color": RGBColor(0xFF, 0xFF, 0xFF), "align": PP_ALIGN.LEFT},
    ], line_spacing=1.25, margins=(220000, 220000, 80000, 80000))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
