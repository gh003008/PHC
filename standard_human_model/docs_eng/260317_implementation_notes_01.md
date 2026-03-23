# Implementation Notes 01 — R Matrix Structure

Created: 2026-03-17
Translated from Korean original: `260317_구현주의사항01.md`

---

## Q: Is a 20×69 R(q) matrix the right structure for the SMPL skeleton model? Is it okay to split each axis into XYZ?

### R(q) matrix structure: Is 20×69 correct?

Yes, it is structurally correct. The force direction (line of action) of each muscle is a vector in 3D space, and when decomposed onto each joint's rotation axes, each axis yields a different moment arm. For example, the hamstrings have a large moment arm with respect to the knee's flexion/extension axis (X), but nearly zero for the abduction/adduction axis (Y) or internal/external rotation axis (Z).

Therefore, the matrix is 20 × 69, but it is a very **sparse** matrix with most elements being zero. Each muscle group has meaningful moment arms on only about 2–4 joint-axis combinations.

### Is XYZ axis decomposition acceptable?

Yes. In fact, this is the anatomically correct approach. OpenSim internally performs the same operation — expressing each muscle's line of action as a 3D vector relative to the joint center and computing the moment arm independently for each rotation axis.

For example, the gluteus maximus has moment arms of approximately -0.05m for hip extension (X-axis), -0.02m for abduction (Y-axis), and 0.03m for external rotation (Z-axis). This is a single muscle acting on multiple axes simultaneously, and XYZ decomposition naturally captures this.

### Important considerations:

**1. SMPL axis matching:** SMPL uses XYZ Euler angles or axis-angles in each joint's local frame. You must verify which anatomical axis (flexion/extension, abduction/adduction, internal/external rotation) corresponds to X, Y, Z for each joint. Incorrect matching assigns moment arms to the wrong axes.

**2. Zero columns:** Many of the 69 DOFs will have entirely zero columns in R. For example, the knee's abduction/adduction and rotation axes are anatomically near-pure hinge joints, so most muscles have negligible moment arms on these axes. This is not a problem — it correctly represents that muscles do not generate torque in those directions.

**3. Sparse computation:** Out of 20 × 69 = 1,380 elements, only about 70–100 are non-zero. Dense matrix multiplication wastes most operations on 0 × 0. PyTorch sparse tensors or index-tracking could be more efficient. However, at 20 × 69, dense operations are fast enough on GPU that this may not matter in practice — profiling is recommended.
