#!/usr/bin/env python3
"""Generate MPL methodology .docx with formatted text and equations."""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import re

doc = Document()

# --- Style setup ---
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(11)

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.color.rgb = RGBColor(0, 0, 0)

def add_eq(doc, text, label=None):
    """Add an equation paragraph (centered, italic)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(11)
    run.font.name = 'Cambria Math'
    if label:
        run2 = p.add_run(f'    ({label})')
        run2.font.size = Pt(10)
    return p

def add_table(doc, headers, rows):
    """Add a simple table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Shading Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            table.rows[ri + 1].cells[ci].text = str(val)
    return table

def add_code_block(doc, text):
    """Add a monospace code block."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Cm(1)
    return p

# ============================================================
# TITLE
# ============================================================
title = doc.add_heading('Motion Plan Layer (MPL): Refined Methodology for Personalized Humanoid Control with Perturbation Response', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('Version 1.0 — 2026-04-13').bold = True
doc.add_paragraph('Scope: Lower-body (12 DoF), treadmill/level walking, perturbation recovery\nTarget hardware: Exoskeleton-assisted locomotion')

# ============================================================
# 1. Problem Statement
# ============================================================
doc.add_heading('1. Problem Statement', level=1)
doc.add_paragraph(
    'The goal of MPL is to produce a humanoid control policy that satisfies three requirements:'
)
doc.add_paragraph('Generates stable locomotion from arbitrary CoM velocity commands without falling', style='List Number')
doc.add_paragraph('Preserves individual movement style — trained on one person\'s data, the policy reproduces that person\'s cadence, stride, impedance patterns, and idiosyncratic habits', style='List Number')
doc.add_paragraph('Handles perturbations as a core capability — responses to external/robot forces must be learned from actual measured human reactions, not discovered purely through RL exploration', style='List Number')

doc.add_paragraph(
    'The prior v09 architecture proposed a two-module split — Nominal Generator G and Residual/Impedance Policy R. '
    'This document refines that architecture, with particular attention to how reflexive responses are captured, represented, and generalized.'
)

# ============================================================
# 2. Architecture
# ============================================================
doc.add_heading('2. Architecture Overview: Unified Latent-Conditioned Policy', level=1)

doc.add_heading('2.1 Motivation: Why Revise the G/R Split', level=2)
doc.add_paragraph(
    'The original G/R split assumes G produces nominal trajectories and R adds corrections. '
    'This is problematic for perturbation responses because:'
)
doc.add_paragraph('Human perturbation reactions are not residuals of nominal walking. A compensatory step involves a qualitatively different motor plan — altered foot placement, trunk counter-rotation — not a delta on top of normal gait.', style='List Bullet')
doc.add_paragraph('The G/R boundary creates an information bottleneck. R only sees G\'s output, not the full motion library context needed for recovery.', style='List Bullet')
doc.add_paragraph('Two separate networks that must coordinate adds complexity without clear benefit when the motion library already contains both walking and perturbation-response data.', style='List Bullet')

doc.add_heading('2.2 Proposed Architecture: Single Policy π with Mode-Conditioned Motion Prior', level=2)
doc.add_paragraph(
    'Instead of a hard G/R split, we propose a single policy network π conditioned on: '
    'current state s_t, stability margin α_t, motion context m_t (from library), and CoM command c_t.'
)
doc.add_paragraph('The policy outputs target joint positions and Compliance Control Factors (CCF):')
add_eq(doc, '(q_ref,t , ξ_t) = π_θ(s_t, α_t, m_t, c_t)', '1')

doc.add_paragraph('Torque computation follows the Variable Impedance Control (VIC) law:')
add_eq(doc, 'τ_j = K_p,j · 2^ξ_j · (q_ref,j − q_j) − K_d,j · 2^ξ_j · q̇_j', '2')
doc.add_paragraph(
    'where K_p,j, K_d,j are nominal PD gains and ξ_j ∈ [−1, 1] is the learned CCF giving '
    'an impedance modulation range of [0.5×, 2.0×] nominal stiffness.'
)

doc.add_heading('2.3 Network Architecture', level=2)
add_code_block(doc,
    '┌─────────────────────────────────────────────┐\n'
    '│            Observation Encoder               │\n'
    '│  s_t: proprioception (q, dq, body state)     │\n'
    '│  α_t: stability margin [0,1]                 │\n'
    '│  c_t: CoM velocity command (vx, vy, ωyaw)    │\n'
    '│  φ_t: gait phase (sin θ, cos θ)              │\n'
    '│  → MLP(512) → LayerNorm → SiLU               │\n'
    '└──────────────────┬──────────────────────────┘\n'
    '                   │\n'
    '                   ▼\n'
    '┌─────────────────────────────────────────────┐\n'
    '│           Motion Context Fusion              │\n'
    '│  m_t: retrieved motion window [t : t+K]      │\n'
    '│  α-weighted attention:                       │\n'
    '│    low α → nominal walking exemplars         │\n'
    '│    high α → perturbation-response exemplars  │\n'
    '│  → Cross-attention → MLP(512) → SiLU         │\n'
    '└──────────────────┬──────────────────────────┘\n'
    '                   │\n'
    '                   ▼\n'
    '┌─────────────────────────────────────────────┐\n'
    '│           GRU Temporal Module                 │\n'
    '│  hidden_dim = 256                            │\n'
    '└──────────────────┬──────────────────────────┘\n'
    '            ┌──────┴──────┐\n'
    '            ▼              ▼\n'
    '   ┌──────────────┐ ┌──────────────┐\n'
    '   │  q_ref head  │ │   CCF head   │\n'
    '   │  MLP(256,12) │ │  MLP(256,6)  │\n'
    '   └──────────────┘ └──────────────┘'
)

# ============================================================
# 3. Stability Margin α
# ============================================================
doc.add_heading('3. Stability Margin α: The Mode-Switching Signal', level=1)

doc.add_heading('3.1 Definition', level=2)
doc.add_paragraph(
    'α is a continuous scalar in [0, 1] that replaces discrete mode switching with a smooth transition signal:'
)
add_eq(doc, 'α_t = clamp(w₁·d_CoP + w₂·d_XCoM + w₃·‖L̇_t‖, 0, 1)', '3')

doc.add_paragraph('CoP margin d_CoP: normalized distance from Center of Pressure to support polygon edge.')
add_eq(doc, 'd_CoP = 1 − dist(p_CoP, ∂S) / dist(p_centroid, ∂S)', '4')

doc.add_paragraph('Extrapolated CoM (XCoM) based on the inverted pendulum model:')
add_eq(doc, 'p_XCoM = p_CoM + ṗ_CoM / ω₀,   ω₀ = √(g / h_CoM)', '5')
add_eq(doc, 'd_XCoM = max(0, signed_dist(p_XCoM, S) / ℓ_foot)', '6')

doc.add_paragraph('Angular momentum rate ‖L̇_t‖: rapid changes precede falls.')
doc.add_paragraph('Default weights: w₁ = 0.3, w₂ = 0.5, w₃ = 0.2. XCoM dominates as it is most predictive of whether a compensatory step is needed (Hof et al., 2005).')

doc.add_heading('3.2 How α Affects the Policy', level=2)
doc.add_paragraph('α modulates reward blending between tracking and recovery:')
add_eq(doc, 'r_t = (1 − α_t)·r_imit + α_t·r_balance + r_style + r_power', '7')

doc.add_heading('3.3 Human Balance Strategy Correspondence', level=2)
add_table(doc,
    ['α Range', 'Strategy', 'Policy Behavior'],
    [
        ['0.0–0.3', 'Ankle strategy', 'Small CCF adjustments, tight reference tracking'],
        ['0.3–0.6', 'Hip strategy', 'Larger hip/trunk corrections, increased stiffness'],
        ['0.6–1.0', 'Stepping strategy', 'Foot placement override, maximum CCF range'],
    ]
)
doc.add_paragraph()

# ============================================================
# 4. Motion Library
# ============================================================
doc.add_heading('4. Motion Library: Integrating Walking and Perturbation Responses', level=1)

doc.add_heading('4.1 The Core Insight', level=2)
doc.add_paragraph(
    'Human perturbation responses must be captured as motion data and included in the motion library, '
    'alongside normal walking. The policy cannot learn person-specific recovery strategies from RL exploration alone because: '
    '(1) human recovery involves learned motor synergies developed over a lifetime, '
    '(2) RL discovers physically valid but not human-like recovery motions, '
    '(3) for exoskeleton control, the robot must predict the human\'s actual response pattern.'
)

doc.add_heading('4.2 Required Data Collection Protocol', level=2)
doc.add_paragraph('A. Nominal walking data (existing):').bold = True
doc.add_paragraph('Treadmill walking at multiple speeds, level and incline, stop-and-go. MoCap + force plates.', style='List Bullet')

p = doc.add_paragraph()
run = p.add_run('B. Perturbation response data (new requirement):')
run.bold = True
doc.add_paragraph('Treadmill belt perturbations: sudden speed changes (trip-like and slip-like)', style='List Bullet')
doc.add_paragraph('Lateral perturbations: mediolateral platform translations or waist-pull cables', style='List Bullet')
doc.add_paragraph('Sagittal perturbations: anterior-posterior pushes at pelvis', style='List Bullet')
doc.add_paragraph('3+ intensity levels × 5+ repetitions per condition', style='List Bullet')

p = doc.add_paragraph()
run = p.add_run('C. Instrumentation:')
run.bold = True
doc.add_paragraph('Full-body optical MoCap (120+ Hz) → SMPL joint angles', style='List Bullet')
doc.add_paragraph('Ground reaction forces (force plates / instrumented treadmill)', style='List Bullet')
doc.add_paragraph('EMG on lower-limb muscles (optional but valuable for impedance estimation)', style='List Bullet')

doc.add_heading('4.3 Motion Library Structure', level=2)
doc.add_paragraph('The library L is indexed by command (CoM velocity) and motion type:')
add_eq(doc, 'L = {(clip_i, c_i, type_i, α_i^peak, subject_id)}_{i=1}^{N}', '8')
doc.add_paragraph(
    'where type_i ∈ {walk, perturb_trip, perturb_slip, perturb_lateral, perturb_push} '
    'and α_i^peak is the peak stability margin observed during the clip.'
)

doc.add_heading('4.4 Context Retrieval Mechanism', level=2)
add_eq(doc, 'm_t = Σ_{i ∈ top-K} w_i · clip_i[t : t+W]', '9')
add_eq(doc, 'w_i ∝ exp(−‖c_t − c_i‖² / σ_c²) · exp(−(α_t − α_i^peak)² / σ_α²)', '10')
doc.add_paragraph(
    'When α is low, nominal walking clips dominate. When α rises, perturbation-response clips '
    'with matching α^peak are retrieved. K=3, W=10 frames, σ_c=0.3 m/s, σ_α=0.2.'
)

# ============================================================
# 5. Training
# ============================================================
doc.add_heading('5. Training: Perturbation Curriculum', level=1)

doc.add_heading('5.1 Stage 1: Nominal Walking Imitation (epochs 0–5,000)', level=2)
doc.add_paragraph('Only walking clips are active. The imitation reward:')
add_eq(doc, 'r_imit = w_p·exp(−k_p·‖Δp‖²) + w_r·exp(−k_r·‖Δθ‖²) + w_v·exp(−k_v·‖Δṗ‖²) + w_ω·exp(−k_ω·‖Δω‖²) + w_f·exp(−k_f·‖Δp_foot‖²)', '11')

add_table(doc,
    ['Component', 'Weight w', 'Sharpness k'],
    [
        ['Body position', '0.4', '200'],
        ['Body rotation', '0.3', '10'],
        ['Linear velocity', '0.2', '1.0'],
        ['Angular velocity', '0.1', '0.1'],
        ['Foot position', '0.2', '40'],
    ]
)
doc.add_paragraph()

doc.add_heading('5.2 Stage 2: Perturbation Introduction (epochs 5,000–15,000)', level=2)
doc.add_paragraph('Perturbation clips activated. Random external forces applied:')
add_eq(doc, 'F_ext(t) ~ U(−F_max(e), F_max(e))', '12')
add_eq(doc, 'F_max(e) = F_min + (e − e_start)/(e_end − e_start) · (F_max,final − F_min)', '13')
doc.add_paragraph('Default: F_min=20N, F_max,final=200N at pelvis, 0.1–0.5s duration, 0.5–5s intervals.')
doc.add_paragraph('The reward transitions to α-weighted blending:')
add_eq(doc, 'r_t = (1 − α_t)·r_imit + α_t·r_balance + λ_style·r_amp + r_power', '14')

doc.add_paragraph('Balance reward components:')
add_eq(doc, 'r_balance = w_h·exp(−k_h·(h_CoM − h_target)²) + w_u·exp(−k_u·‖θ_trunk‖²) + w_c·exp(−k_c·‖v_CoM − v_cmd‖²)', '15')

add_table(doc,
    ['Balance Component', 'Weight', 'Description'],
    [
        ['CoM height', '0.3', 'Penalizes crouching/falling'],
        ['Trunk uprightness', '0.4', 'Penalizes excessive lean'],
        ['CoM velocity recovery', '0.3', 'Reward for returning to commanded velocity'],
    ]
)
doc.add_paragraph()

doc.add_heading('5.3 Stage 3: Full Perturbation with Style Enforcement (epochs 15,000–30,000)', level=2)
doc.add_paragraph('AMP discriminator weight increases to enforce subject-specific recovery patterns:')
add_eq(doc, 'r_amp = max(0, 1 − 0.25·(D_φ(s_t, s_{t+1}) − 1)²)', '16')
doc.add_paragraph(
    'Because training data includes both walking and perturbation responses, '
    'the discriminator learns a unified distribution that considers both as valid human motion — '
    'analogous to ASE (Peng et al. 2022) learning over diverse skills.'
)

# ============================================================
# 6. Personalization
# ============================================================
doc.add_heading('6. Personalization', level=1)

doc.add_heading('6.1 Subject-Specific Motion Library', level=2)
doc.add_paragraph('For personalization, the entire pipeline uses data from a single subject S:')
add_eq(doc, 'L_S = {clips from subject S only}', '17')
doc.add_paragraph(
    'The AMP discriminator D_S is also trained exclusively on S\'s data, '
    'ensuring style reward enforces S\'s specific movement patterns including perturbation responses.'
)

doc.add_heading('6.2 What Makes Perturbation Responses Person-Specific', level=2)
doc.add_paragraph('Biomechanics literature documents significant inter-individual variability:')
doc.add_paragraph('Response latency: 50–150ms stretch reflexes, 80–200ms triggered responses (varies with age, pathology)', style='List Bullet')
doc.add_paragraph('Strategy preference: some use hip strategy where others switch to stepping (Horak & Nashner, 1986)', style='List Bullet')
doc.add_paragraph('Compensatory step characteristics vary ~30% across healthy adults (Maki & McIlroy, 1997)', style='List Bullet')
doc.add_paragraph('Co-contraction (impedance) patterns differ between individuals', style='List Bullet')

doc.add_heading('6.3 Population Pre-training + Subject Fine-tuning', level=2)
add_eq(doc, 'θ_S = fine-tune(θ_pop, L_S, D_S)', '18')
doc.add_paragraph(
    'Pre-train on a population dataset (multiple subjects) for general physics understanding, '
    'then fine-tune on subject S\'s data with reduced learning rate.'
)

doc.add_heading('6.4 Physiological Parameter Injection', level=2)
add_table(doc,
    ['Parameter', 'Source', 'Effect'],
    [
        ['Body segment masses', 'DXA / anthropometry', 'Affects dynamics'],
        ['Joint ROM limits', 'Goniometry', 'Constrains action space'],
        ['Nominal PD gains', 'Isometric strength + EMG', 'Scales torque capacity'],
        ['Reflex delay', 'EMG onset latency', 'Adds control latency in sim'],
    ]
)
doc.add_paragraph()

# ============================================================
# 7. Robot/External Forces
# ============================================================
doc.add_heading('7. Handling Robot/External Forces', level=1)

doc.add_heading('7.1 Why This Is Critical', level=2)
doc.add_paragraph(
    'From the exoskeleton perspective, the human\'s response to robot-generated forces is the most '
    'important behavior to predict. An exoskeleton that cannot anticipate the human\'s reaction to '
    'its own torques will fight against the human, destabilize them, or fail to assist during critical moments.'
)

doc.add_heading('7.2 Robot Force as Observation', level=2)
doc.add_paragraph('The policy explicitly observes applied external forces:')
add_eq(doc, 's_t = (proprioception, α_t, φ_t, F_ext,t)', '19')
doc.add_paragraph('This makes force-response a first-class input, not a disturbance to be rejected.')

doc.add_heading('7.3 Training with Structured Force Profiles', level=2)
doc.add_paragraph('A. Joint-level torque perturbations mimicking exoskeleton interactions:', style='List Bullet')
add_eq(doc, 'τ_ext,j(t) = A_j · sin(2πf_j·t + φ_j),  A_j ~ U(0, τ_max)', '20')
doc.add_paragraph('B. Phased assistance/resistance: assist during swing, resist during stance', style='List Bullet')
doc.add_paragraph('C. Sudden onset/offset: simulating exoskeleton engaging/disengaging', style='List Bullet')

doc.add_heading('7.4 Impedance Response', level=2)
doc.add_paragraph(
    'The CCF output ξ provides impedance adaptation: '
    'stabilizing force → reduce ξ (accept assistance); '
    'destabilizing force → increase ξ (stiffen to resist); '
    'force disappears → transiently increase ξ, then relax. '
    'These patterns are learned from measured human responses.'
)

# ============================================================
# 8. Mathematical Summary
# ============================================================
doc.add_heading('8. Mathematical Summary', level=1)

doc.add_heading('8.1 State Space', level=2)
add_eq(doc, 's_t = (q_t, q̇_t, p_t^root, R_t^root, v_t^root, ω_t^root, α_t, sinφ_t, cosφ_t, F_ext,t) ∈ ℝ^(d_s)', '21')

doc.add_heading('8.2 Action Space', level=2)
add_eq(doc, 'a_t = (q_ref,t ∈ ℝ¹², ξ_t ∈ ℝ⁶) ∈ ℝ¹⁸', '22')

doc.add_heading('8.3 Reward Function', level=2)
add_eq(doc, 'r_t = (1−α_t)·r_imit + α_t·r_balance + λ_style·r_amp + r_power + r_ccf_bio', '23')

doc.add_heading('8.4 Policy Optimization (PPO + AMP)', level=2)
add_eq(doc, 'L_π = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t)]', '24')
add_eq(doc, 'L_D = −E_L[log D_φ(s,s\')] − E_π[log(1 − D_φ(s,s\'))] + λ_gp·GP', '25')

# ============================================================
# 9. Roadmap
# ============================================================
doc.add_heading('9. Implementation Roadmap', level=1)

add_table(doc,
    ['Phase', 'Description', 'Data Required', 'Deliverable'],
    [
        ['Phase 0 (current)', 'Single-policy VIC with walking', 'S001 treadmill walking', 'Feasibility: stable walking with CCF'],
        ['Phase 1', 'Add α computation, perturbation curriculum', 'Same + simulated pushes', 'Recovery from sim perturbations'],
        ['Phase 2', 'Collect perturbation response MoCap', 'NEW: S001 perturbed walking', 'Library with walking + perturbation'],
        ['Phase 3', 'Full MPL training', 'Walking + perturbation library', 'Personalized recovery policy'],
        ['Phase 4', 'Exoskeleton co-simulation', 'Phase 3 policy + exo model', 'Exo controller predicting human response'],
    ]
)
doc.add_paragraph()

doc.add_paragraph(
    'Critical Next Step: Phase 2 data collection. Without measured human perturbation response data, '
    'Phases 2–4 cannot proceed. Minimum viable dataset: 1 subject, 3 perturbation types, '
    '3 intensity levels, 5 repetitions each = ~45 perturbation trials.'
)

# ============================================================
# 10. Comparison
# ============================================================
doc.add_heading('10. Comparison with Related Work', level=1)

add_table(doc,
    ['Approach', 'Strengths', 'Limitation for Our Goals'],
    [
        ['PHC (Luo 2023)', 'Perpetual control, fail-state recovery', 'Recovery is RL-discovered, not personalized'],
        ['AMP (Peng 2021)', 'Natural motion from unstructured data', 'No explicit perturbation handling'],
        ['ASE (Peng 2022)', 'Skill latent space, reusable behaviors', 'Discrete skills; no continuous α blending'],
        ['CALM (Tessler 2023)', 'Conditional motion with latent space', 'No impedance control, no perturbation data'],
        ['MPL (Ours)', 'α-blended retrieval, perturbation data, VIC, personalized', 'Requires subject-specific perturbation MoCap'],
    ]
)
doc.add_paragraph()

# ============================================================
# References
# ============================================================
doc.add_heading('References', level=1)
refs = [
    'Peng, X.B., et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Animation." ACM TOG, 2021.',
    'Peng, X.B., et al. "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters." ACM TOG, 2022.',
    'Tessler, C., et al. "CALM: Conditional Adversarial Latent Models for Directable Virtual Characters." ACM TOG, 2023.',
    'Luo, Z., et al. "Perpetual Humanoid Control for Real-time Simulated Avatars." ICCV, 2023.',
    'Peng, X.B., et al. "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills." ACM TOG, 2018.',
    'Horak, F.B., Nashner, L.M. "Central programming of postural movements." J. Neurophysiol., 1986.',
    'Maki, B.E., McIlroy, W.E. "The role of limb movements in maintaining upright stance." Phys. Ther., 1997.',
    'Hof, A.L., et al. "The condition for dynamic stability." J. Biomech., 2005.',
    'Winter, D.A. "Biomechanics and Motor Control of Human Movement." 4th ed., Wiley, 2009.',
    'Peterka, R.J. "Sensorimotor integration in human postural control." J. Neurophysiol., 2002.',
]
for i, ref in enumerate(refs, 1):
    doc.add_paragraph(f'[{i}] {ref}')

# Save
outpath = '/home/exolab/Documents/GitHub/PHC/01_research_docs/260413_MPL_methodology_refined.docx'
doc.save(outpath)
print(f'Saved: {outpath}')
