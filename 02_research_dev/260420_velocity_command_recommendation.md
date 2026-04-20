# Recommendation on Velocity Command Design for Single-Clip Imitation

## Bottom line

I agree with the core diagnosis: in the current setup, the velocity command and the imitation target are structurally fighting each other.

For your current stage, I would **not** keep treating `v_cmd` as an independent objective that the policy must satisfy on top of a fixed-speed single walking clip. Instead, I would move the command **upward** into the **reference-generation layer**.

In practice, that means:

1. Measure the natural forward speed of the single walking clip, `v_nat`.
2. Replace the current “free absolute velocity command” idea with a **speed multiplier / phase-rate multiplier** around that natural gait.
3. **Retime the reference motion** using that multiplier, so the teacher itself becomes command-consistent.
4. Reduce the explicit velocity-tracking reward to a **small residual term** or remove it in the first ablation.
5. Keep the command range **narrow at first**.
6. Do **not** add personalization, body-shape variation, turning, or time-varying commands yet.

My short version is this:

> **For a single-clip walking teacher, the right immediate abstraction is not “general velocity command conditioning.” It is “command-conditioned reference retiming around one nominal gait.”**

---

## Why the current design is colliding

Right now the policy is being asked to do two different things at once:

- imitate one walking clip with one natural speed,
- while also tracking a randomly sampled commanded speed that is not derived from that clip.

That is not just a tuning problem. It is a **problem formulation** problem.

The important observation from your results is that the successful episodes are still high quality. The major cost of the command design is not “bad imitation all the time,” but **extra failure episodes**. That means the policy can still imitate well, but the reward landscape is inconsistent enough that training becomes less robust.

This is exactly the kind of issue that is worth removing early, because you already have plenty of learning difficulty from the PHC / simulation / control side. You do not want to add another contradictory signal before the basic simulation implementation is clean and repeatable.

---

## My recommendation: make the command generate the teacher

### 1. Reinterpret the current command

At this stage, I would rename the concept internally from:

- `velocity command`

into something closer to:

- `speed_scale`
- `phase_rate_scale`
- `nominal_speed_multiplier`

That is more honest to what a single-clip setup can actually support.

With one forward walking clip, you do **not** yet have a true velocity-conditioned walking family. What you really have is one gait that can be played a bit faster or slower.

### 2. Measure the clip’s actual natural speed first

Before changing the reward, compute:

```python
v_nat = (root_x_end - root_x_start) / clip_duration
```

Preferably use pelvis/root forward displacement projected onto the clip heading, and compute both:

- whole-clip mean forward speed,
- cycle-wise or mid-clip mean forward speed if you want robustness against start/end transient bias.

Do this first, because your current `[0.8, 1.3]` range is blind. If `v_nat` is already near `1.15` or `1.2`, then the slow side is much more aggressive than it looks.

### 3. Sample a scale, not a free velocity

Instead of sampling:

```python
v_cmd ~ U(0.8, 1.3)
```

sample:

```python
s ~ U(s_lo, s_hi)
v_cmd = s * v_nat
```

I would start with something conservative, for example:

```python
s in [0.9, 1.1]
```

and only widen after you confirm stable success and reasonable tracking.

If that works well, expand to maybe `[0.85, 1.15]`.

I would **not** jump straight back to the old absolute range unless it matches the measured `v_nat` well.

### 4. Retime the reference motion itself

This is the key change.

Update the reference phase with the scale:

```python
phi_{t+1} = phi_t + s * dt / T_clip
```

or equivalently scale the playback time index by `s`.

Then all teacher quantities should come from the **retimed reference**:

- pose,
- joint angles,
- joint velocities,
- root/pelvis velocity,
- possibly any other velocity-sensitive teacher signals.

So the policy is no longer asked to:

- imitate one fixed-speed teacher,
- while separately following another speed target.

Instead it is asked to:

- imitate **the command-conditioned teacher**.

That removes the main structural contradiction.

---

## Reward design I would use

### Recommended training reward

I would use the retimed reference as the main target, and make the explicit command reward weak.

#### Option R1: first ablation I would run

```python
rew = imitation_rew_against_retimed_ref
```

No explicit command reward.

This is the cleanest test of the idea.

If the retimed teacher is implemented correctly and the command is in the observation, the policy should still learn different speeds because the reward optimum itself is speed-dependent.

#### Option R2: safer practical version

```python
rew = imitation_rew_against_retimed_ref + lambda_cmd * cmd_track_rew
```

but with a **small** `lambda_cmd`, not a co-equal blended objective.

For example, I would start with the command term as a light regularizer rather than a major driver.

### What I would remove or reduce

If command-conditioned training is on, I would **remove or heavily downweight absolute pelvis/root velocity tracking inside the base imitation reward**, unless that term is already using the retimed reference and you have confirmed it is not double-counting.

The reason is simple:

- if the base imitation already compares against the retimed reference velocity,
- and the explicit command reward also punishes velocity error,

then velocity is being optimized twice.

That is not fatal, but it often over-focuses learning on root-speed matching.

### Important AMP check

One code-level check matters a lot:

- If your discriminator / AMP features include absolute body or root velocities, you should verify whether they also become command-consistent after retiming.

If AMP is still judging against fixed-speed motion statistics while the imitation reward is retimed, then some of the conflict may survive inside AMP even after you fix the task reward.

So I would verify one of these is true:

1. the discriminator already does not care much about absolute root velocity, or
2. the positive/reference motion windows supplied to AMP are also retimed, or
3. the most speed-sensitive absolute velocity features are removed from AMP for this experiment.

I would not change AMP blindly, but I would explicitly verify this point.

---

## Concrete experiment plan

### Stage 0 — no-code-baseline extension

Run your current best command-conditioned model to 30k as a baseline continuation.

Purpose:

- measure how much of the problem is just undertraining,
- establish a fair comparison against the retimed version.

But treat this as a **ceiling check**, not as the final solution.

### Stage 1 — quantify the current problem properly

Before redesigning too much, evaluate the current checkpoint at fixed commands:

- `v_cmd = 0.95 * v_nat`
- `v_cmd = 1.00 * v_nat`
- `v_cmd = 1.05 * v_nat`
- later `0.90`, `1.10` if stable

Record at least:

- success rate,
- episode length,
- mean pelvis forward velocity,
- mean absolute velocity error,
- failure rate as a function of command,
- foot slip / contact inconsistency if you can,
- imitation reward vs command bin.

This will tell you whether the failure distribution is symmetric or concentrated on the slow side / fast side.

### Stage 2 — retimed reference, narrow range

Implement command-conditioned retiming with:

- fixed SMPL body,
- single forward clip,
- constant command per episode,
- `s in [0.9, 1.1]`.

Use either:

- pure retimed imitation reward, or
- retimed imitation + very small cmd reward.

### Stage 3 — direct comparison

Compare at least these three conditions:

#### Baseline A
Current CMD_B resumed to 30k.

#### Baseline B
Retimed reference + no explicit cmd reward.

#### Baseline C
Retimed reference + small cmd reward.

That comparison will tell you whether the explicit cmd reward is still useful once the teacher is command-consistent.

---

## What I would *not* do now

I would explicitly avoid the following for this round:

### 1. No personalization / body-shape variation

That has already shown itself to add difficulty without helping this question.

### 2. No turning yet

Yaw command is outside the capability of the current single forward clip setup.

### 3. No time-varying commands yet

Do not add ramps, step changes, or joystick-like continuously changing commands until constant-per-episode behavior is stable.

### 4. No wide command range yet

Do not treat `[0.8, 1.3]` as automatically valid until `v_nat` is measured and the retimed setup is stable.

### 5. No multi-clip library yet

That is the right long-term solution for broader and more natural speed control, but it is too much machinery for the immediate question.

---

## When to move to a multi-clip library

If you want a **true velocity-conditioned walking skill** rather than a narrow nominal-gait speed scaling, then eventually you will outgrow the single-clip solution.

I would move to a slow/normal/fast walking library when one of these becomes true:

1. you need more than about `±15–20%` around `v_nat`,
2. cadence-only scaling starts looking unnatural,
3. step-length adaptation becomes important,
4. you want turning or richer locomotion styles.

At that point the right design is:

- choose the clip (or clip subset) closest to the commanded speed,
- then optionally do a small additional retiming around that clip.

That is much more physically natural than trying to stretch one clip too far.

---

## Why this recommendation fits your broader direction

This recommendation matches the direction you have already been converging to:

- keep the setup narrow,
- remove unnecessary complexity,
- make the simulation implementation stable first,
- produce a repeatable, measurable result before broadening the problem.

In other words, this is the **lowest-complexity change that directly attacks the root cause**.

It also gives you a clean paper/story later:

- **bad formulation:** fixed-speed teacher + independent speed reward,
- **better formulation:** command-conditioned teacher via retiming,
- **next step:** speed-conditioned multi-clip library.

That progression is coherent.

---

## My final recommendation

If I had to choose one concrete next move, it would be this:

> **Implement command-conditioned reference retiming around the single clip’s natural speed, shrink the range to a narrow speed multiplier, and reduce the explicit command reward to a minor regularizer or remove it in the first ablation.**

And in parallel:

> **Resume the current best checkpoint to 30k and run fixed-command evaluation so you know how much of the current result is undertraining versus structural conflict.**

That is the path I would take before touching personalization, turning, or multi-clip data.

