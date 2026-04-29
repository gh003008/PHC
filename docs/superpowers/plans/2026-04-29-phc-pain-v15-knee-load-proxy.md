# PHC-Pain v1.5 Knee Load Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v1 torque-centric knee pain mechanism with an OA-style knee joint loading proxy through v1.5, without launching training.

**Architecture:** Keep `pain_body_state` as the reward-facing variable, but change what drives it. v1.4 adds a contact/compression proxy from foot GRF and knee geometry; v1.5 adds KAM/KFM-style moment-arm proxies so medial knee OA pain is driven by estimated tibiofemoral compartment loading rather than actuator torque.

**Tech Stack:** Python, PyTorch tensors, IsaacGym PHC task tensors, unittest/pytest-compatible tests, YAML config.

---

## File Structure

- Modify: `phc/env/util/pain_baseline.py`
  - Add pure-torch helpers for knee joint load proxies:
    - `compute_knee_contact_load_proxy`
    - `compute_knee_moment_load_proxy`
    - `combine_knee_oa_load_proxy`
  - Keep `compute_knee_torque_load_proxy` for backwards compatibility and ablation.

- Modify: `phc/env/tasks/humanoid_im_pain.py`
  - Import the new helpers.
  - Cache v1.4/v1.5 config values.
  - Estimate right/left knee joint loading from body positions and foot contact forces.
  - Log `pain_v1_*_compression`, `pain_v1_*_kam`, `pain_v1_*_kfm`, `pain_v1_*_contact_load`, and legacy torque metrics.

- Modify: `phc/data/cfg/env/env_im_pain_v1.yaml`
  - Add `knee_mechanism.proxy_mode: "oa_contact_v15"`.
  - Add references and weights for compression/KAM/KFM/loading-rate.
  - Keep legacy torque weights present but disabled by default in v1.5.

- Modify: `tests/test_pain_knee_proxy.py`
  - Add deterministic unit tests for the new pure-torch helpers.

- Optional modify: `docs/superpowers/phc_pain_v1_evaluation_protocol.md`
  - Update the main metric from raw torque reduction to OA joint-load proxy reduction.

---

## Task 1: Add Pure-Torch OA Knee Load Helpers

**Files:**
- Modify: `phc/env/util/pain_baseline.py`
- Test: `tests/test_pain_knee_proxy.py`

- [ ] **Step 1: Write failing tests for contact compression and KAM/KFM proxies**

Append these tests to `tests/test_pain_knee_proxy.py` and update the import line.

```python
from phc.env.util.pain_baseline import (
    combine_knee_oa_load_proxy,
    compute_knee_contact_load_proxy,
    compute_knee_moment_load_proxy,
    compute_knee_torque_load_proxy,
)
```

```python
    def test_knee_contact_proxy_uses_stance_foot_load_not_actuator_torque(self):
        foot_force = torch.tensor([[0.0, 0.0, 500.0], [30.0, 40.0, 0.0]])
        knee_pos = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
        foot_pos = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        knee_flex = torch.tensor([0.4, 0.4])

        load, components = compute_knee_contact_load_proxy(
            knee_pos=knee_pos,
            foot_pos=foot_pos,
            foot_force=foot_force,
            knee_flex=knee_flex,
            body_weight_ref=500.0,
            flex_compression_gain=0.5,
            loading_rate=None,
            loading_rate_ref=1000.0,
            w_compression=1.0,
            w_loaded_flex=0.0,
            w_loading_rate=0.0,
        )

        self.assertTrue(torch.allclose(components["compression"], torch.tensor([1.0, 0.1])))
        self.assertTrue(torch.allclose(load, torch.tensor([1.0, 0.1])))

    def test_knee_moment_proxy_separates_kam_and_kfm_geometry(self):
        knee_pos = torch.tensor([[0.0, 0.0, 0.5]])
        foot_pos = torch.tensor([[0.1, -0.2, 0.0]])
        foot_force = torch.tensor([[0.0, 0.0, 500.0]])

        load, components = compute_knee_moment_load_proxy(
            knee_pos=knee_pos,
            foot_pos=foot_pos,
            foot_force=foot_force,
            kam_ref=50.0,
            kfm_ref=50.0,
            w_kam=1.0,
            w_kfm=0.5,
        )

        self.assertTrue(torch.allclose(components["kam"], torch.tensor([2.0])))
        self.assertTrue(torch.allclose(components["kfm"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(load, torch.tensor([2.5])))

    def test_oa_load_combiner_keeps_components_visible(self):
        contact = torch.tensor([0.5])
        moment = torch.tensor([0.25])
        torque = torch.tensor([0.1])

        load, components = combine_knee_oa_load_proxy(
            contact_load=contact,
            moment_load=moment,
            torque_load=torque,
            w_contact=0.7,
            w_moment=0.3,
            w_torque=0.0,
        )

        self.assertTrue(torch.allclose(load, torch.tensor([0.425])))
        self.assertTrue(torch.allclose(components["contact_load"], contact))
        self.assertTrue(torch.allclose(components["moment_load"], moment))
        self.assertTrue(torch.allclose(components["torque_load"], torque))
```

- [ ] **Step 2: Run tests and verify they fail for missing functions**

Run:

```bash
conda run --no-capture-output -n phc python -m pytest tests/test_pain_knee_proxy.py -q
```

Expected: FAIL with an import error for `compute_knee_contact_load_proxy`, `compute_knee_moment_load_proxy`, or `combine_knee_oa_load_proxy`.

- [ ] **Step 3: Implement the pure-torch helpers**

Add this code after `compute_knee_torque_load_proxy` in `phc/env/util/pain_baseline.py`.

```python
def compute_knee_contact_load_proxy(
    knee_pos,
    foot_pos,
    foot_force,
    knee_flex,
    body_weight_ref=700.0,
    flex_compression_gain=0.25,
    loading_rate=None,
    loading_rate_ref=1000.0,
    w_compression=0.70,
    w_loaded_flex=0.20,
    w_loading_rate=0.10,
    eps=1e-6,
):
    """Approximate tibiofemoral compression from stance foot loading.

    This is an OA-style load proxy, not an actuator torque proxy. It uses the
    force transmitted through the stance foot plus a loaded-flexion term because
    knee flexion under compressive load increases tibiofemoral demand.
    """
    force_mag = torch.norm(foot_force, dim=-1)
    vertical_force = torch.relu(foot_force[..., 2])
    compression = torch.maximum(force_mag, vertical_force) / max(float(body_weight_ref), eps)
    loaded_flex = compression * torch.relu(knee_flex) * float(flex_compression_gain)

    if loading_rate is None:
        rate_proxy = torch.zeros_like(compression)
    else:
        rate_proxy = torch.relu(loading_rate) / max(float(loading_rate_ref), eps)

    load_proxy = (
        float(w_compression) * compression
        + float(w_loaded_flex) * loaded_flex
        + float(w_loading_rate) * rate_proxy
    )
    return load_proxy, {
        "compression": compression,
        "loaded_flex": loaded_flex,
        "loading_rate": rate_proxy,
    }


def compute_knee_moment_load_proxy(
    knee_pos,
    foot_pos,
    foot_force,
    kam_ref=50.0,
    kfm_ref=50.0,
    w_kam=0.70,
    w_kfm=0.30,
    eps=1e-6,
):
    """Approximate KAM/KFM load from GRF line of action around the knee.

    Coordinate convention follows IsaacGym PHC tensors where z is vertical.
    The frontal-plane adduction surrogate uses the mediolateral moment arm
    around the knee; the sagittal flexion surrogate uses the fore-aft moment arm.
    These are load proxies, not full inverse-dynamics joint moments.
    """
    lever = foot_pos - knee_pos
    force_z = torch.relu(foot_force[..., 2])
    kam = torch.abs(lever[..., 1] * force_z) / max(float(kam_ref), eps)
    kfm = torch.abs(lever[..., 0] * force_z) / max(float(kfm_ref), eps)
    load_proxy = float(w_kam) * kam + float(w_kfm) * kfm
    return load_proxy, {
        "kam": kam,
        "kfm": kfm,
    }


def combine_knee_oa_load_proxy(
    contact_load,
    moment_load,
    torque_load,
    w_contact=0.60,
    w_moment=0.40,
    w_torque=0.0,
):
    """Combine OA knee loading channels while preserving ablation visibility."""
    load_proxy = (
        float(w_contact) * contact_load
        + float(w_moment) * moment_load
        + float(w_torque) * torque_load
    )
    return load_proxy, {
        "contact_load": contact_load,
        "moment_load": moment_load,
        "torque_load": torque_load,
    }
```

- [ ] **Step 4: Run unit tests and verify they pass**

Run:

```bash
conda run --no-capture-output -n phc python -m pytest tests/test_pain_knee_proxy.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit pure helper changes**

```bash
git add phc/env/util/pain_baseline.py tests/test_pain_knee_proxy.py
git commit -m "feat: add OA knee load proxy helpers"
```

---

## Task 2: Wire v1.4 Contact Load Proxy Into HumanoidImPainV1

**Files:**
- Modify: `phc/env/tasks/humanoid_im_pain.py`
- Test: `tests/test_pain_knee_proxy.py`

- [ ] **Step 1: Import the new helpers**

Modify the import block in `phc/env/tasks/humanoid_im_pain.py`.

```python
from phc.env.util.pain_baseline import (
    apply_pain_action_guard,
    broadcast_body_pain_to_dof,
    combine_internal_pain,
    combine_knee_oa_load_proxy,
    compute_contact_pain,
    compute_joint_limit_pain,
    compute_knee_contact_load_proxy,
    compute_knee_moment_load_proxy,
    compute_knee_torque_load_proxy,
    compute_power_pain,
    compute_torque_pain,
    update_pain_state,
)
```

- [ ] **Step 2: Cache v1.4/v1.5 config values**

Add these attributes after the existing `self._knee_memory_alpha` assignment in `HumanoidImPainV1.__init__`.

```python
        self._knee_proxy_mode = str(knee_cfg.get("proxy_mode", "torque_v13"))
        self._knee_body_weight_ref = float(knee_cfg.get("body_weight_ref", 700.0))
        self._knee_flex_compression_gain = float(knee_cfg.get("flex_compression_gain", 0.25))
        self._knee_loading_rate_ref = float(knee_cfg.get("loading_rate_ref", 1000.0))
        self._knee_kam_ref = float(knee_cfg.get("kam_ref", 50.0))
        self._knee_kfm_ref = float(knee_cfg.get("kfm_ref", 50.0))
        self._knee_w_contact_load = float(knee_cfg.get("w_contact_load", 0.60))
        self._knee_w_moment_load = float(knee_cfg.get("w_moment_load", 0.40))
        self._knee_w_legacy_torque_load = float(knee_cfg.get("w_legacy_torque_load", 0.0))
        self._knee_w_compression = float(knee_cfg.get("w_compression", 0.70))
        self._knee_w_loaded_flex = float(knee_cfg.get("w_loaded_flex", 0.20))
        self._knee_w_loading_rate = float(knee_cfg.get("w_loading_rate", 0.10))
        self._knee_w_kam = float(knee_cfg.get("w_kam", 0.70))
        self._knee_w_kfm = float(knee_cfg.get("w_kfm", 0.30))
```

- [ ] **Step 3: Add body index lookup for knee and foot load tensors**

Add this method to `HumanoidImPainV1` near `_build_knee_dof_idx`.

```python
    def _build_knee_load_body_idx(self):
        return {
            "left": {
                "knee": self._body_names.index("L_Knee"),
                "ankle": self._body_names.index("L_Ankle"),
                "toe": self._body_names.index("L_Toe"),
            },
            "right": {
                "knee": self._body_names.index("R_Knee"),
                "ankle": self._body_names.index("R_Ankle"),
                "toe": self._body_names.index("R_Toe"),
            },
        }
```

Call it in `__init__` after `self._knee_dof_idx = self._build_knee_dof_idx()`.

```python
        self._knee_load_body_idx = self._build_knee_load_body_idx()
        self._prev_knee_compression = {
            "left": torch.zeros((self.num_envs,), device=self.device),
            "right": torch.zeros((self.num_envs,), device=self.device),
        }
```

- [ ] **Step 4: Add contact load computation**

Add this method to `HumanoidImPainV1`.

```python
    def _compute_knee_contact_proxy(self, side):
        ids = self._knee_load_body_idx[side]
        knee_pos = self._rigid_body_pos[:, ids["knee"], :]
        ankle_pos = self._rigid_body_pos[:, ids["ankle"], :]
        toe_pos = self._rigid_body_pos[:, ids["toe"], :]
        foot_pos = 0.5 * (ankle_pos + toe_pos)
        foot_force = (
            self.contact_forces[:, ids["ankle"], :]
            + self.contact_forces[:, ids["toe"], :]
        )
        knee_flex = self._dof_pos[:, self._knee_dof_idx[side]]

        force_mag = torch.norm(foot_force, dim=-1)
        compression = torch.maximum(force_mag, torch.relu(foot_force[..., 2])) / max(
            self._knee_body_weight_ref, 1.0e-6
        )
        loading_rate = (compression - self._prev_knee_compression[side]) / max(self.dt, 1.0e-6)
        self._prev_knee_compression[side][:] = compression.detach()

        return compute_knee_contact_load_proxy(
            knee_pos=knee_pos,
            foot_pos=foot_pos,
            foot_force=foot_force,
            knee_flex=knee_flex,
            body_weight_ref=self._knee_body_weight_ref,
            flex_compression_gain=self._knee_flex_compression_gain,
            loading_rate=loading_rate,
            loading_rate_ref=self._knee_loading_rate_ref,
            w_compression=self._knee_w_compression,
            w_loaded_flex=self._knee_w_loaded_flex,
            w_loading_rate=self._knee_w_loading_rate,
        )
```

- [ ] **Step 5: Reset previous compression buffers**

In `_reset_envs`, after zeroing `pain_body_drive`, add:

```python
            for side in ("left", "right"):
                self._prev_knee_compression[side][env_ids] = 0
```

- [ ] **Step 6: Run syntax check**

```bash
conda run --no-capture-output -n phc python -m py_compile phc/env/tasks/humanoid_im_pain.py
```

Expected: no output and exit code 0.

- [ ] **Step 7: Commit v1.4 wiring**

```bash
git add phc/env/tasks/humanoid_im_pain.py
git commit -m "feat: wire knee contact load pain proxy"
```

---

## Task 3: Add v1.5 KAM/KFM Moment-Arm Proxy And Mode Switch

**Files:**
- Modify: `phc/env/tasks/humanoid_im_pain.py`

- [ ] **Step 1: Add KAM/KFM proxy computation**

Add this method beside `_compute_knee_contact_proxy`.

```python
    def _compute_knee_moment_proxy(self, side):
        ids = self._knee_load_body_idx[side]
        knee_pos = self._rigid_body_pos[:, ids["knee"], :]
        ankle_pos = self._rigid_body_pos[:, ids["ankle"], :]
        toe_pos = self._rigid_body_pos[:, ids["toe"], :]
        foot_pos = 0.5 * (ankle_pos + toe_pos)
        foot_force = (
            self.contact_forces[:, ids["ankle"], :]
            + self.contact_forces[:, ids["toe"], :]
        )
        return compute_knee_moment_load_proxy(
            knee_pos=knee_pos,
            foot_pos=foot_pos,
            foot_force=foot_force,
            kam_ref=self._knee_kam_ref,
            kfm_ref=self._knee_kfm_ref,
            w_kam=self._knee_w_kam,
            w_kfm=self._knee_w_kfm,
        )
```

- [ ] **Step 2: Replace `_compute_knee_proxy` internals with mode-aware load selection**

Modify `_compute_knee_proxy` so it still computes the legacy torque proxy but returns OA contact/moment load when configured.

```python
        torque_load, torque_components = compute_knee_torque_load_proxy(
            tau=tau,
            dq=dq,
            rom_proxy=rom_proxy,
            asset_tau_limit=self.torque_limits[idx],
            torque_ref=self._knee_torque_ref,
            power_ref=self._p_power_ref,
            w_torque=self._knee_w_torque,
            w_flex=self._knee_w_flex,
            w_rom=self._knee_w_rom,
            w_work=self._knee_w_work,
        )

        if self._knee_proxy_mode == "torque_v13":
            return torque_load, torque_components

        contact_load, contact_components = self._compute_knee_contact_proxy(side)
        if self._knee_proxy_mode == "oa_contact_v14":
            components = dict(torque_components)
            components.update(contact_components)
            components["contact_load"] = contact_load
            return contact_load, components

        if self._knee_proxy_mode == "oa_contact_v15":
            moment_load, moment_components = self._compute_knee_moment_proxy(side)
            load, combined_components = combine_knee_oa_load_proxy(
                contact_load=contact_load,
                moment_load=moment_load,
                torque_load=torque_load,
                w_contact=self._knee_w_contact_load,
                w_moment=self._knee_w_moment_load,
                w_torque=self._knee_w_legacy_torque_load,
            )
            components = dict(torque_components)
            components.update(contact_components)
            components.update(moment_components)
            components.update(combined_components)
            return load, components

        raise ValueError(
            "env.pain.knee_mechanism.proxy_mode must be one of "
            "{'torque_v13', 'oa_contact_v14', 'oa_contact_v15'}; "
            f"got {self._knee_proxy_mode!r}"
        )
```

- [ ] **Step 3: Ensure extras names are stable**

No separate code is needed if Step 2 updates `components`, because the existing loop logs all component items:

```python
            for name, value in components.items():
                self.extras[f"{prefix}_{name}"] = float(value.mean().item())
```

After Step 2, expected new extras include:

```text
pain_v1_right_knee_compression
pain_v1_right_knee_loaded_flex
pain_v1_right_knee_loading_rate
pain_v1_right_knee_kam
pain_v1_right_knee_kfm
pain_v1_right_knee_contact_load
pain_v1_right_knee_moment_load
pain_v1_right_knee_torque_load
```

- [ ] **Step 4: Run syntax check**

```bash
conda run --no-capture-output -n phc python -m py_compile phc/env/tasks/humanoid_im_pain.py
```

Expected: no output and exit code 0.

- [ ] **Step 5: Commit v1.5 mode switch**

```bash
git add phc/env/tasks/humanoid_im_pain.py
git commit -m "feat: add OA contact and moment knee pain modes"
```

---

## Task 4: Update v1 Environment Config For OA Load Pain

**Files:**
- Modify: `phc/data/cfg/env/env_im_pain_v1.yaml`

- [ ] **Step 1: Replace the knee mechanism block**

Update `pain.knee_mechanism` to:

```yaml
  knee_mechanism:
    proxy_mode: "oa_contact_v15"  # torque_v13 | oa_contact_v14 | oa_contact_v15
    left_sensitivity: 0.0
    right_sensitivity: 1.0
    left_threshold: 0.30
    right_threshold: 0.30

    # Legacy actuator-torque ablation channel. Kept visible, disabled in v1.5 load.
    torque_ref: 100.0
    w_torque: 0.80
    w_flex: 0.10
    w_rom: 0.05
    w_work: 0.05

    # v1.4 contact/compression proxy.
    body_weight_ref: 700.0
    flex_compression_gain: 0.25
    loading_rate_ref: 1000.0
    w_compression: 0.70
    w_loaded_flex: 0.20
    w_loading_rate: 0.10

    # v1.5 medial OA moment-arm proxy.
    kam_ref: 50.0
    kfm_ref: 50.0
    w_kam: 0.70
    w_kfm: 0.30

    # Final OA load mixture.
    w_contact_load: 0.60
    w_moment_load: 0.40
    w_legacy_torque_load: 0.00
    memory_alpha: 0.05
```

- [ ] **Step 2: Add a short comment above `lambda_p`**

Use this exact comment:

```yaml
  # Reward penalizes pain state; pain state is driven by OA-style knee load proxy.
  lambda_p: 0.05
```

- [ ] **Step 3: Run YAML parse smoke check**

```bash
conda run --no-capture-output -n phc python - <<'PY'
import yaml
with open('phc/data/cfg/env/env_im_pain_v1.yaml') as f:
    cfg = yaml.safe_load(f)
k = cfg['pain']['knee_mechanism']
assert k['proxy_mode'] == 'oa_contact_v15'
assert k['w_legacy_torque_load'] == 0.0
print('env_im_pain_v1.yaml OK')
PY
```

Expected:

```text
env_im_pain_v1.yaml OK
```

- [ ] **Step 4: Commit config**

```bash
git add phc/data/cfg/env/env_im_pain_v1.yaml
git commit -m "config: use OA knee load pain proxy"
```

---

## Task 5: Add Local Probe Verification For New Metrics

**Files:**
- Modify only if needed: `phc/learning/im_amp_players.py`
- Generated output: `analysis/plots/phc_pain_oa_load_probe/`

- [ ] **Step 1: Confirm probe JSON hook captures the new metrics**

Open `phc/learning/im_amp_players.py` and verify the `PHC_PAIN_PROBE_JSON` filter captures keys starting with `pain_v1_`. If it does, make no edit.

Expected relevant logic:

```python
key.startswith("pain_v1_")
```

- [ ] **Step 2: Run a short headless play probe with pretrained checkpoint**

Use the already-local pretrained checkpoint. Do not train.

```bash
mkdir -p analysis/plots/phc_pain_oa_load_probe
PHC_PAIN_PROBE_JSON=analysis/plots/phc_pain_oa_load_probe/pretrained_oa_probe.json \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn_lowvram \
  env=env_im_pain_v1 \
  epoch=-1 \
  test=True \
  headless=True \
  env.num_envs=32 \
  env.motion_file=sample_data/amass_isaac_walking_forward_subset23.pkl \
  checkpoint=output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth
```

Expected: command exits 0 and prints a `PHC_PAIN_PROBE_JSON` line.

- [ ] **Step 3: Verify OA metrics are present**

```bash
conda run --no-capture-output -n phc python - <<'PY'
import json
from pathlib import Path
p = Path('analysis/plots/phc_pain_oa_load_probe/pretrained_oa_probe.json')
d = json.loads(p.read_text())
required = [
    'pain_v1_right_knee_compression',
    'pain_v1_right_knee_loaded_flex',
    'pain_v1_right_knee_kam',
    'pain_v1_right_knee_kfm',
    'pain_v1_right_knee_contact_load',
    'pain_v1_right_knee_moment_load',
    'pain_v1_right_knee_torque_load',
    'pain_v1_right_knee_load',
]
missing = [k for k in required if k not in d]
assert not missing, missing
for k in required:
    print(k, d[k]['mean'])
PY
```

Expected: all required metric names print with finite numeric means.

- [ ] **Step 4: Commit verification hook changes if any were required**

If `phc/learning/im_amp_players.py` was edited:

```bash
git add phc/learning/im_amp_players.py
git commit -m "chore: capture OA knee load probe metrics"
```

If no edit was required, skip this commit.

---

## Task 6: Update Evaluation Protocol And Final Validation

**Files:**
- Modify: `docs/superpowers/phc_pain_v1_evaluation_protocol.md`

- [ ] **Step 1: Add v1.5 metric hierarchy**

Append this section:

```markdown
## v1.5 OA Knee Load Proxy Metrics

Primary success metrics:
- `pain_v1_right_knee_load`: final reward-facing OA load proxy.
- `pain_v1_right_knee_state`: leaky pain state driven by OA load.
- `pain_v1_right_knee_contact_load`: compression and loaded-flexion contact component.
- `pain_v1_right_knee_moment_load`: KAM/KFM moment-arm component.
- `pain_v1_right_knee_kam`: medial-compartment loading surrogate.

Secondary diagnostics:
- `pain_v1_right_knee_kfm`: sagittal flexion loading surrogate.
- `pain_v1_right_knee_compression`: stance foot load normalized by body-weight reference.
- `pain_v1_right_knee_loaded_flex`: knee flexion under compressive load.
- `pain_v1_right_knee_tau_abs`, `pain_v1_right_knee_tau_rms`, `pain_v1_right_knee_tau_peak`: legacy actuator-torque diagnostics only.

Interpretation rule:
Do not claim OA pain reduction from reduced actuator torque alone. v1.5 claims must be based on reduced OA load proxy, especially contact load, KAM, and pain state.
```

- [ ] **Step 2: Run all relevant checks**

```bash
conda run --no-capture-output -n phc python -m pytest tests/test_pain_knee_proxy.py -q
conda run --no-capture-output -n phc python -m py_compile phc/env/util/pain_baseline.py phc/env/tasks/humanoid_im_pain.py phc/learning/im_amp_players.py
git diff --check
```

Expected:
- pytest passes
- py_compile exits 0
- `git diff --check` exits 0

- [ ] **Step 3: Commit docs**

```bash
git add docs/superpowers/phc_pain_v1_evaluation_protocol.md
git commit -m "docs: define OA knee load proxy evaluation"
```

---

## Self-Review

Spec coverage:
- v1.4 contact/compression proxy is covered by Tasks 1-2.
- v1.5 KAM/KFM moment-arm proxy is covered by Task 3.
- Config defaults are covered by Task 4.
- No training launch is included; Task 5 is headless play/probe only.
- Evaluation interpretation is covered by Task 6.

Placeholder scan:
- No `TBD`, `TODO`, or undefined future functions remain in the plan.
- Each new helper named in later tasks is defined in Task 1.

Risk:
- The KAM/KFM proxy uses PHC world axes and simple GRF moment arms, not full inverse dynamics. This is acceptable for v1.5 but should be reported as an estimated load proxy, not true medial contact force.

Execution order:
- Implement Tasks 1-4 first.
- Run Task 5 before any server training to confirm metrics exist.
- Only after v1.5 probe metrics look sensible should a future training job be submitted.
