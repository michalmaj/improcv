# Design: deterministic PnP-based object-to-camera pose estimation

Status: **speculative — no development line has been opened for this feature.** This document
freezes a candidate API contract to determine whether one *can* be frozen durably; it does not by
itself approve building it. See §0 for why this file is deliberately not named after any version.
**Corrected post-merge, pre-implementation** — a static/runtime typing mismatch, a premature public
`CameraMatrix` alias, an underspecified type/dtype validation contract, an inaccurate "excludes no
real capability" claim, an unfrozen "tight tolerance" camera-matrix comparison, and an unstated
dataclass-construction semantic were all found and fixed; see §1-§13, §16, §19-§21, §23, and §26
for exactly what changed. No decision this document had already gotten right was reopened without
new evidence (§7 of the correcting task, restated in each corrected section as it applies).

## 0. Filename and status convention

Every existing design document in `docs/design/` is named after the exact version it shipped
under (e.g. `0.5.0a4-plot-confusion-matrix.md`), because in every prior case the version bump for
that alpha had already happened before or alongside the design PR — design docs in this repository
have never preceded a version decision. This document is the first exception: the preceding
boundary audit concluded **NO IMMEDIATE 0.6**, and the task that produced this document explicitly
requires not assuming a `0.6` line is approved merely because a design can be frozen. Naming this
file `0.6-pnp-pose-estimation.md` (as loosely suggested before this constraint was worked out) or
`0.6.0a1-pnp-pose-estimation.md` would misrepresent that a version decision has been made when it
has not. This document is therefore named **`pnp-pose-estimation.md`** — no version prefix at all
— the least committal choice consistent with the directory's own convention of naming files after
what has actually been decided. If a `0.6.0a1` line is later opened with this feature as its first
slice, the file should be renamed to match that convention at that time, not before.

## 1. Base state at design time

Base SHA `3212a1fc0233a4d8affdc42c5a1bac254027d9e4` (`origin/main` == local `main`), version
`0.5.0b1` (latest published release), working tree clean, `improcv.__all__` == 198,
`improcv.visualization.__all__` == 3. Zero existing public camera-calibration/3D-pose API — the
only superficial keyword matches (`calibrate_camera_response_debevec`/`_robertson` in `hdr.py`,
`find_homography`/`HomographyResult` in `features.py`) are HDR response-curve calibration and 2D
planar homography respectively, unrelated domains.

## 2. Two research-spike claims corrected by direct experiment

The preceding research spike's blanket "`SOLVEPNP_ITERATIVE` needs `N >= 4`" and "skew is
legitimately supported, don't reject it" claims were both re-tested directly and **both
corrected**.

### 2a. `SOLVEPNP_ITERATIVE` minimum point count depends on planarity — confirmed, not assumed

Tested with genuinely well-conditioned (non-collinear, well-spread) synthetic point sets — an
earlier same-day attempt using naive `linspace`-generated "non-planar" points was itself
accidentally collinear and produced meaningless garbage-vs-garbage comparisons; that mistake was
caught and the experiment redone with real box-corner-style 3D points before drawing any
conclusion. Result, identical in OpenCV 4.9.0 and 5.0.0:

| configuration | N=4 | N=5 | N=6 |
|---|---|---|---|
| planar (z=0) | correct | correct | correct |
| non-planar (box corners) | correct* | **`cv2.error`** | correct |

(*N=4 "non-planar" here was accidentally the box's planar bottom face — see below; the real
finding is the N=5 row.)

The decisive evidence is the **N=5 non-planar** case, which raises the identical error in both
versions:

```
OpenCV(4.9.0) .../calib3d/src/calibration.cpp:1173: error: (-2:Unspecified error) in function
'void cvFindExtrinsicCameraParams2(...)'
> DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences.
  (expected: 'count >= 6'), where 'count' is 5, must be greater than or equal to '6' is 6

OpenCV(5.0.0) .../geometry/src/calibration_base.cpp:1341: error: (-2:Unspecified error) in
function 'void cv::findExtrinsicCameraParams2(...)' — identical message text.
```

This is exactly the documented OpenCV behavior the task asked to verify: **planar object points
work from N=4 (homography-decomposition initialization); genuinely non-planar object points
require N>=6 (DLT initialization)**, confirmed with an explicit, unambiguous, cross-version-
identical assertion, not inferred. §10 decides what improcv does with this fact.

### 2b. Standard pinhole `K[0,1]` skew is silently ignored by `solvePnP`/`projectPoints` — confirmed

Directly tested, both versions, identical result:

- `cv2.projectPoints` with `K[0,1]=50` produces **byte-identical** output to `K[0,1]=0`
  (`max |diff| = 0.0` px).
- The manual projection formula that *includes* the skew term (`x = fx·X/Z + skew·Y/Z + cx`)
  disagrees with `cv2`'s actual output by up to 17 px; the formula that *omits* the skew term
  matches `cv2`'s output to floating-point noise (~1e-13).
- `solvePnP` recovers the **identical** correct pose whether given the true skewed `K` or a
  deliberately wrong no-skew `K` for the same (skew-ignorant) image points — because the skew term
  never entered the computation on either side.
- `K[1,0]` (the other conventionally-always-zero entry) was also confirmed ignored (`max diff =
  0.0`).

**This overturns the prior research spike's "skew is legitimately supported, don't reject it"
guidance for standard pinhole `solvePnP`/`projectPoints` specifically.** Silently accepting a
nonzero skew would let a caller believe their skew term is doing something when OpenCV's own
implementation discards it — exactly the "matrix entries OpenCV silently ignores while implying
they are meaningful" hazard the task warned against. §11 freezes `K[0,1] == K[1,0] == 0` as a hard
requirement.

## 3. Product-value gate

What this design adds over `ok, rvec, tvec = cv2.solvePnP(...)`:

1. One canonical point-array shape each for object/image points (`(N,3)`/`(N,2)`), never `cv2`'s
   `(N,1,·)`/`(1,N,·)` polymorphism.
2. Deterministic float64-normalized numeric contract, regardless of float32/float64 input.
3. An explicit, named, documented transform direction (§4) — raw `cv2.solvePnP`'s positional
   `(rvec, tvec)` tuple communicates no direction at all; a caller must already know the PnP
   convention.
4. A structured, immediately-usable result (a 3x3 rotation matrix and a `(3,)` translation vector)
   instead of a raw Rodrigues vector and a `(3,1)` column vector that both need conversion before
   use.
5. Pre-validated, specific `TypeError`/`ValueError`s (wrong type, wrong dtype, bad shape, point
   count, camera-matrix structure, distortion length — §23) in place of `cv2`'s internal C++
   assertion text (e.g. `"'6' is 6"`).
6. **Actively rejects** a silently-discarded skew term (§2b) rather than accepting it and
   discarding the caller's input without a trace — a real safety improvement raw `cv2` does not
   offer.
7. A cross-version-stable contract verified directly against both supported OpenCV lines, for
   exactly the function/flag/shape combination this design freezes — not assumed from
   documentation.

This is not a thin wrapper — it does not merely validate an ndarray and forward to one `cv2`
function with the same parameters and the same raw return shape (§24 restates this at the end
against the final signature).

## 4. Exact conceptual operation and terminology

Given `object_points` `X_o` (points expressed in some reference frame — the "object" or "model"
frame; this may represent a physical object, a calibration target, or literally the world, but the
function only ever knows it as "the frame `object_points` are expressed in"), `image_points` `x`,
camera intrinsics `K`, and optional distortion coefficients `d`, the function computes:

```
X_c = R_oc @ X_o + t_oc
```

verified directly (§2 of the research spike, re-confirmed unchanged here): the recovered
`(R, t)` map points **from the object frame into the camera frame**, residual 4.44e-16 against
ground truth; the reverse convention (`X_c = Rᵀ·(X_o − t)`) was explicitly tested and is off by
~4.28 — decisively ruled out.

**Frozen terminology: "object-to-camera"**, not "world-to-camera". `object_points` is the actual
parameter name and is correct regardless of what the caller's reference frame conceptually
represents; "world" would only be accurate in the special case where the caller's object points
happen to be literal world coordinates. "Object-to-camera" is unconditionally correct; "world-to-
camera" is not.

The inverse transform is documented for the caller's understanding only — **no inversion API is
part of this slice**:

```
X_o = R_oc.T @ (X_c − t_oc)
```

## 5. Function naming

| Option | Verdict |
|---|---|
| `solve_pnp(...)` | Technically precise, `cv2`-familiar, but names nothing about direction; "pnp" is opaque jargon outside CV practitioners. |
| `estimate_pose(...)` | Rejected — ambiguous which pose, relative to what. |
| `estimate_object_pose(...)` | Better, but "pose of the object" alone doesn't rule out a world-frame misreading. |
| `estimate_camera_pose(...)` | **Actively misleading** per the task's own explicit warning — the computed transform tells you where the *object* is relative to the camera, not where the *camera* is relative to the world (that would be the matrix inverse, §4). Rejected outright. |
| **`estimate_object_to_camera_pose(...)`** | Fully explicit about direction at every call site; matches this project's established preference for verbose-but-precise names (`expand_perspective_canvas`, `discover_image_mask_pairs`) over terse jargon. |

**Frozen: `estimate_object_to_camera_pose`.**

## 6. Result-type design

| Option | Verdict |
|---|---|
| Generic `Pose(rotation, translation)` | **Rejected** per the task's own strong caution — a bare "Pose" name invites a future different-direction pose (e.g. a hypothetical future camera-to-world result) to either reuse the same ambiguous name or fork into a confusingly-parallel type; the entire point of this exercise is to never let "pose" mean two things silently. |
| `PoseEstimationResult` | Rejected — wraps the ambiguity in a "Result" suffix without resolving it; still doesn't encode direction. |
| `PnPResult` | Rejected — names the type after the *method* (PnP) rather than the mathematical object it holds, inconsistent with this project's own naming precedent (`ConfusionMatrixResult`, `HomographyResult` are named after what they *are*, not how they were computed). |
| **`ObjectToCameraPose`** | Encodes both the semantic content (a pose/rigid transform) and the disambiguating direction in the type name itself, directly reusing §4's frozen terminology and matching the `HomographyResult`-style "named after the mathematical object" precedent. |
| tuple / no type | Rejected — the thin-wrapper anti-pattern this whole design exists to avoid. |

**Frozen: `ObjectToCameraPose`**, exactly two fields:

```python
rotation: npt.NDArray[np.float64]     # shape (3, 3)
translation: npt.NDArray[np.float64]  # shape (3,)
```

No `rvec` field (no independent justification for exposing it — internal-only, §17). No
`reprojection_error` field — `solvePnP` does not compute one; adding it would silently force this
function to do a second, distinct piece of work (a `projectPoints` call plus residual computation)
that is a legitimately separate concern, better left to a future, explicit, opt-in helper if
evidence ever justifies it. No `inverse`/`compose`/`homogeneous_matrix` — no independent
justification exists yet, matching the task's own strong default of "no" here.

`ObjectToCameraPose` is deliberately narrow enough (2 fields, one fixed direction) that a *future*
function computing the same semantic quantity by a different method could legitimately return the
same type — a positive, non-ambiguous form of reuse, unlike the rejected generic `Pose`.

## 7. Array ownership / equality contract

Matches the actual, verified precedent in `src/improcv/evaluation.py` (e.g.
`matrix.flags.writeable = False` on `ConfusionMatrixResult`'s array, confirmed directly by
grep — this is a real runtime-enforced pattern in this codebase, not merely a documentation
promise) and `RocCurve`'s `@dataclass(frozen=True, slots=True, eq=False)` + manual `__eq__` +
`__hash__ = None` pattern (necessary because default dataclass/tuple equality raises "truth value
of an array is ambiguous" on array fields — confirmed this is *why* `RocCurve`/`ConfusionMatrixResult`
use a manual `__eq__` rather than the plain-`NamedTuple` pattern `HomographyResult` uses, which
only works because nothing has directly exercised `HomographyResult.__eq__` against non-trivial
array content).

`ObjectToCameraPose`:
- `@dataclass(frozen=True, slots=True, eq=False)`
- `rotation`/`translation` are new, independent arrays — always copied out of `cv2`'s raw output,
  never a view into anything the caller passed in or that `cv2` returned
- both arrays: `dtype=np.float64`, `flags.writeable = False` set explicitly
- manual `__eq__` comparing both fields via `np.array_equal`
- `__hash__ = None` (unhashable, matching every other array-bearing result type in this project)

**Constructor-semantics clarification (explicit, to avoid a category of bug the original text
invited by omission):** the copied/float64/read-only/fixed-shape guarantees above describe values
*returned by* `estimate_object_to_camera_pose` — they are not enforced by `ObjectToCameraPose`
itself. `ObjectToCameraPose` has no `__post_init__` and performs no validation of its own; like
this project's other array-bearing result types, it is **a result container, not a
self-validating domain object** — manual construction (e.g. `ObjectToCameraPose(rotation=np.zeros((2,2)),
translation=np.array([1]))`) is never rejected, including with the wrong shape, a mutable array, or
a non-float64 dtype. This follows `MulticlassRocCurve`'s own documented precedent exactly (`src/
improcv/evaluation.py`: *"This is a result container, not a self-validating domain object --
manual construction ... is never rejected ... Only [the producing function] itself guarantees the
full contract ... for the value it returns"*) rather than inventing a new validating-dataclass
philosophy this project does not otherwise use. The docstring must state this precisely so it
never implies every manually-constructed `ObjectToCameraPose` is automatically copied, read-only,
or correctly shaped.

## 8. Object-point contract

- Must be `np.ndarray` — **not** a bare `Sequence` (unlike some evaluation functions that accept
  `Sequence[int]`, a 3D point set is naturally already array-shaped data, and requiring `ndarray`
  keeps the shape contract simple and unambiguous for a first slice). A non-`ndarray` input (a
  list, tuple, string, or any other object) raises **`TypeError`** — checked first, before any
  `.dtype`/`.ndim`/`.shape` access, so a plain Python object can never accidentally surface an
  `AttributeError` instead of a clear improcv error (§23 of this correction freezes the exact
  order across all four parameters).
- dtype policy: **accept `np.floating` (float32 or float64) only, normalize internally to
  float64** before calling `cv2` — matches this project's general pattern of accepting a
  reasonably wide numeric input and normalizing (e.g. `TransformMatrix = npt.NDArray[np.floating[Any]]`),
  rather than gatekeeping to float64-only. An integer-dtype (or any other non-floating-dtype)
  array raises **`TypeError`**, matching this project's existing precedent (`src/improcv/evaluation.py`'s
  `_normalize_zero_division`/label-validation helpers: `not isinstance(value, np.ndarray)` →
  `TypeError`; `not np.issubdtype(value.dtype, np.integer)` → `TypeError`) — integers are silently
  *not* accepted, never silently upcast.
- Public shape: **exactly `(N, 3)`** — never `cv2`'s `(N,1,3)`/`(1,N,3)` forms. A wrong `ndim` or
  wrong trailing shape raises **`ValueError`** (checked only after the type/dtype checks above have
  already passed, per this project's established `TypeError`-for-type/dtype,
  `ValueError`-for-shape/value convention, confirmed directly in `evaluation.py`'s validators).
- Must be finite (reject NaN/Inf) — closes a real gap: `projectPoints` silently propagates NaN,
  and neither version catches Inf anywhere in the object/camera-matrix path (§2 of the prior
  research spike). Raises `ValueError`.
- Internal C-contiguity is improcv's problem, not the caller's — normalize via
  `np.ascontiguousarray` internally rather than requiring the caller to think about it.
- Minimum N: **6**, universally — see §10. Raises `ValueError`.

## 9. Image-point contract

- Same type/dtype/shape/finiteness policy and exception types as §8: not-`ndarray` → `TypeError`;
  non-floating dtype → `TypeError`; wrong `ndim`/trailing shape (≠ `(N,2)`) → `ValueError`;
  non-finite → `ValueError`.
- `N` must equal `object_points`'s `N` exactly — mismatch is a `ValueError`, not left to `cv2`'s
  own (already-confirmed-clear-enough) internal assertion.
- Never expose `cv2`'s `(N,1,2)`/`(1,N,2)` forms publicly.

## 10. Planarity / degeneracy policy — the central open question, now resolved

Three policies were compared, using §2a's decisive new evidence:

- **Policy A** (detect planarity, N>=4 planar / N>=6 non-planar): requires improcv to itself
  compute point-cloud rank (e.g. smallest singular value of the centered point cloud) *before*
  calling `cv2`, so it can apply the correct method-specific minimum and raise a clear
  pre-validation error instead of `cv2`'s DLT assertion. Rejected for this first slice: picking a
  scale-invariant tolerance for "how close to zero counts as planar" is a genuine, non-trivial
  numerical design problem in its own right (the task's own instruction: "do not implement a
  fragile fixed absolute-rank tolerance without scale analysis") — solving it properly is a
  distinct piece of work this slice should not be blocked on.
- **Policy B** (require N>=6 universally, regardless of planar/non-planar; no planarity detection
  at all): §2a confirmed planar N=6 succeeds correctly in both OpenCV versions, so a universal
  N>=6 minimum does not break anything for callers who already have 6+ points. This entirely
  sidesteps the scale-dependent-tolerance problem, requires zero geometry-classification code, and
  cannot disagree with `cv2`'s own internal (not-fully-reverse-engineered) planarity test, because
  improcv never attempts to replicate it.
- **Policy C** (support only non-planar): rejected — planar target scenarios (fiducial markers,
  calibration checkerboards used purely for pose, etc.) are one of the most common real-world PnP
  use cases; excluding them gives *less* value than raw `cv2`.

**Frozen: Policy B — with an honest, explicitly-stated limitation.** `object_points`/
`image_points` must have `N >= 6`; the function does not distinguish planar from non-planar
internally at all. **This is a real, acknowledged capability restriction, not merely a
convenience simplification**: it excludes otherwise-valid planar PnP calls with exactly 4 or 5
correspondences — including the common case of a single planar four-corner fiducial/marker/square
target, which `cv2.solvePnP` itself handles correctly today (§2a, planar N=4). Any prior or later
wording in this document claiming this restriction "excludes no real capability" is **incorrect
and superseded here** — it does exclude a real, common four-point planar workflow. The restriction
is accepted anyway because it buys one uniform, deterministic public contract without any
planarity classification or scale-dependent tolerance code; it is strictly backward-compatible to
relax later (a future slice could add planarity detection and lower the minimum for planar inputs
without breaking any existing caller); and **callers who need four- or five-point planar PnP today
must call `cv2.solvePnP` directly** — this design does not yet serve them.

Re-running the product-value gate (§3) with this limitation stated honestly: the value-adds listed
there (canonical shapes, dtype normalization, explicit direction, structured result, safer
validation, cross-version stability, skew rejection) all still hold independently of the N>=6
restriction — none of them depended on the false "no capability excluded" claim. The gate still
passes; it simply passes for a *smaller* population of valid inputs than raw `cv2.solvePnP`
itself serves, which is an honest, disclosed trade-off rather than a hidden one.

Rank-1/collinear/identical-point degeneracy: the research spike found `cv2.solvePnP` does **not**
raise on identical or collinear point sets — it returns `ok=True` with silently wrong output.
Pre-validating point-cloud rank has the same scale-dependent-tolerance problem as planarity
detection above, so **no pre-solve rank/collinearity check is added**. An automatic post-solve
reprojection-residual rejection threshold was considered and **rejected** — per the task's own
instruction, "a solver succeeding is not the same thing as a solver being accurate," and a hidden
quality threshold risks rejecting valid noisy real-world solutions unpredictably. Instead, the
docstring will state plainly that a successful call does not itself certify well-conditioned input
geometry — the same honesty `HomographyResult`'s docstring already applies to degenerate
correspondences.

## 11. Camera-matrix contract

| Property | OpenCV's own behavior (both versions) | improcv policy |
|---|---|---|
| Not an `np.ndarray` | n/a | **`TypeError`** — checked first, before any attribute access |
| Non-floating dtype (e.g. integer) | n/a | **`TypeError`** — same convention as §8/§9, never silently upcast |
| Shape ≠ (3,3) | Rejected (`cv2.error`) | **Require exactly (3,3)** — `ValueError` |
| dtype (floating) | float32/float64 both accepted | Accept both, normalize to float64 |
| `K[0,1]`, `K[1,0]` ≠ 0 | **Silently ignored** (§2b) | **Require exactly `== 0.0`** — this is the corrected decision from §2b |
| `K[2,0]`, `K[2,1]` ≠ 0 | Not a supported pinhole parameterization | **Require exactly `== 0.0`** |
| `K[2,2]` ≠ 1 | Silently accepted, used as an unexplained scale | **Require exactly `== 1.0`** — no known legitimate alternative parameterization |
| fx, fy ≤ 0 | Silently accepted, produces sign-flipped/mirrored results | **Require > 0** — same "silently-wrong-but-accepted" hazard class as skew; a design choice, not independently re-tested this round, but justified by the same logic already validated for skew/Inf |
| NaN | Caught by `solvePnP`'s own assertion, **not** caught by `projectPoints` | **Require finite** — closes the `projectPoints` gap |
| Inf | **Neither version catches this** | **Require finite** — a real gap in `cv2` itself |
| Principal point (cx, cy) | Unrestricted | No restriction beyond finiteness — legitimately can be anywhere |

**Exact equality, no `rtol`/`atol`, frozen.** The five structural positions (`K[0,1]`, `K[1,0]`,
`K[2,0]`, `K[2,1]`, `K[2,2]`) are compared with plain Python/NumPy `==` against `0.0`/`1.0` *after*
dtype normalization to float64, not a tolerance-based comparison. This is safe because: a caller
constructing a proper pinhole matrix by hand places literal `0.0`/`1.0` values, which are exactly
representable in both float32 and float64 with no rounding error, and upcasting an exact float32
`0.0`/`1.0` to float64 introduces no error either — there is no legitimate path by which a
correctly-constructed camera matrix would have a merely-close-to-zero or merely-close-to-one value
in these positions. `cv2.calibrateCamera`'s own output `K` is assembled the same way internally
(the standard OpenCV calibration model fixes these positions structurally rather than fitting
them), so exact equality does not reject legitimate calibration-produced matrices either. A
tolerance-based comparison was considered and rejected as unnecessary complexity with no
supporting evidence that it is needed — per the task's own "no real evidence this rejects
legitimate matrices" test.

Ownership: `camera_matrix` is read, never mutated; any dtype normalization happens on an internal
copy, matching this project's "never hide expensive array copies, always document" rule.

## 12. No public `CameraMatrix` type — corrected decision

The originally-merged version of this document froze a public `CameraMatrix = npt.NDArray[np.float64]`
alias, reasoning it extended the existing `TransformMatrix` precedent in `src/improcv/types.py`.
**On correction, that decision is reversed: `CameraMatrix` is removed from this first slice.**

The `TransformMatrix` analogy does not actually hold up: `TransformMatrix` earns its place as a
named alias because it is used across roughly six different functions throughout
`src/improcv/augmentation.py` and `src/improcv/transforms.py` (`warp_affine`, `warp_perspective`,
`sample_affine`, `sample_perspective`, `expand_affine_canvas`, `expand_perspective_canvas`) — its
value comes from *consistent naming across many call sites*, not from labeling any single
parameter. `CameraMatrix` would have exactly **one** call site in the entire codebase today. A
type alias used in exactly one place adds no marginal signature-readability value the parameter
name `camera_matrix` was not already providing on its own — Python type aliases are not nominal
types; `CameraMatrix` and a bare `npt.NDArray[np.floating[Any]]` are literally interchangeable to
every type checker and at runtime, so the alias's only possible value is human-readability, and
one call site does not justify introducing a public symbol for that.

Separately, the originally-merged alias was **float64-only** (`npt.NDArray[np.float64]`) while
§11's own runtime contract accepts float32 *or* float64 and normalizes internally — meaning the
alias actively misrepresented what the function actually accepted (§1 of this correction). Fixing
the alias to the honest `npt.NDArray[np.floating[Any]]` would have made it *less* informative
still, since at that point it says nothing more than "a floating-point ndarray," the same thing
every other unaliased parameter in this design already says directly.

None of this rules out introducing `CameraMatrix` — or a genuinely validated `CameraIntrinsics`-
style object — later, once a second function in this codebase actually needs to accept the same
kind of camera matrix, giving the alias real multi-site value the way `TransformMatrix` has.
Adding a type alias to `src/improcv/types.py` (the project's existing shared location for such
aliases — not `geometry.py`, matching where `TransformMatrix`/`Mask`/`Image` already live) at that
point would be a purely additive, non-breaking change.

**Frozen: no `CameraMatrix` symbol in this slice.** `camera_matrix` is typed directly as
`npt.NDArray[np.floating[Any]]`, exactly matching what §11's validator actually accepts, and the
parameter name alone carries the semantic meaning.

## 13. Distortion-coefficient contract

Every tested shape (`None`, empty, `(4,)`, `(5,)`, `(8,)`, `(12,)`, `(14,)`, and column-vector
equivalents) was accepted by both `solvePnP` and `projectPoints` in both OpenCV versions, with no
rejections anywhere (prior research spike). Given this permissiveness and simplicity, **no public
`DistortionCoefficients` type is introduced** — the contract is thin enough to document directly.

**Frozen**: keyword parameter named `distortion` (not `dist_coeffs` — matches this project's
consistent preference for full English words over `cv2`-mirroring abbreviations, e.g.
`sample_weight` not `s_weight`), typed **`npt.NDArray[np.floating[Any]] | None = None`** — corrected
from the originally-merged `npt.NDArray[np.float64] | None`, which contradicted this very section's
own stated policy of accepting float32 as well (§1 of this correction).

- `None` means zero distortion (matches `cv2`'s own semantics directly) and skips all of the
  following checks entirely.
- When not `None`: not an `np.ndarray` → `TypeError`; non-floating dtype (e.g. integer) →
  `TypeError` — same convention as §8/§9/§11, checked before any shape/value inspection.
- An empty array (`shape (0,)`) is treated **identically to `None`** — a harmless, informationally
  equivalent representation some callers may naturally produce; rejecting it as an invalid length
  would be needlessly unforgiving.
- Otherwise: must be exactly **1-D** (wrong `ndim` → `ValueError`), length in `{4, 5, 8, 12, 14}`
  (`ValueError`), finite (`ValueError`). `cv2`'s column-vector forms (`(4,1)` etc.) are **not**
  accepted publicly — reshaped internally only if ever needed, never exposed as a caller-facing
  shape choice.
- Accepts float32 or float64, normalized to float64 internally.

## 14. `solvePnP` method scope

Given the N>=6-universal policy (§10) and the fact that `SOLVEPNP_P3P`/`AP3P`/`IPPE`/
`IPPE_SQUARE` each have their *own*, different, incompatible point-count/configuration
preconditions (exactly 4 points, or a planar square) — exposing method selection now would force
either per-method precondition-validation work (real, deferred design work) or let callers violate
an unvalidated method precondition and hit a raw `cv2` error, undermining this design's own §3
safety story for any non-default method.

**Frozen: no public `method` parameter. `SOLVEPNP_ITERATIVE` is used unconditionally in this
slice.** Documented explicitly as an additive extension point — a future `method:` keyword-only
parameter with a validated, narrow `Literal`/enum (never a raw `cv2` integer constant) could be
added later without breaking any existing caller, once each method's own precondition validation
is separately designed.

## 15. `useExtrinsicGuess` / initial-pose guess

**Out of scope, confirmed.** Exposing an initial-pose guess would force new decisions about how
that guess is represented (the same rotation/translation representation questions this whole
design exists to resolve, but for an *input* rather than the output) and interacts with
method-specific minimum-point relaxations not designed here. No evidence currently justifies it.

## 16. Solver failure contract

- Every structural/numeric precondition from §8-§13 is validated **before** calling `cv2`, raising
  either `TypeError` (wrong Python/array type, wrong dtype) or `ValueError` (wrong shape, wrong
  value/count/structure) with a clear message — matching this project's existing, directly-verified
  convention (`src/improcv/evaluation.py`'s validators: `not isinstance(x, np.ndarray)` and wrong
  dtype both raise `TypeError`; wrong `ndim`/shape/value raises `ValueError`). §23 freezes the
  exact order across all four parameters.
- If `cv2.solvePnP` still raises an unanticipated `cv2.error` after all pre-validation passes, it
  is **not** caught or reinterpreted — it propagates unchanged. improcv should not blanket-catch
  an internal `cv2` error it cannot meaningfully translate (matches this project's "catch specific
  exceptions, not bare `Exception`" rule).
- If `cv2.solvePnP` returns `ok=False`, improcv raises `RuntimeError` — a genuine solver failure on
  otherwise-valid input, distinct from a `ValueError` (bad input) or `HomographyResult`'s different
  precedent of representing "cannot compute" as `homography=None`. That precedent is *not*
  followed here deliberately: across every degenerate-input case actually tested for `ITERATIVE`
  (identical points, collinear points, coplanar points), `cv2` never once returned `ok=False` — it
  always returned `True` with either a correct or a silently-garbage result. An actual `ok=False`
  return therefore appears to be a rare, genuine failure mode for this specific method rather than
  a common, expected outcome the way homography-estimation failure is — `RuntimeError` fits better
  than a `None`-shaped "expected outcome" result.
- No runtime rotation-matrix-orthonormality assertion is added — `cv2.Rodrigues` converting a
  vector produced by a successful `solvePnP` call is guaranteed orthonormal by construction; a
  redundant runtime check would validate an invariant already covered by the deterministic test
  oracle (§22), not by a live assertion.
- No automatic reprojection-residual rejection (§10, restated: rejected as a hidden, potentially
  unpredictable quality gate).

## 17. Rotation conversion

`cv2.Rodrigues(rvec)` is called internally immediately after a successful `solvePnP`. Output is
normalized to `dtype=np.float64` (its natural dtype here already, per direct testing), copied
(never a view into `cv2`'s buffer), `flags.writeable = False`. Shape is trivially `(3,3)` by
`Rodrigues`'s own contract — no separate orthonormality check (§16). **The raw Rodrigues vector is
never exposed publicly.**

## 18. Translation conversion

`cv2.solvePnP` returns `tvec` as `(3,1)` float64. Normalized to public shape **`(3,)`** via
`.ravel()` on a fresh copy, `flags.writeable = False`. This is documented explicitly as a
public-API ergonomics normalization (idiomatic 1-D NumPy vector), not a claim about `cv2`'s
internal representation — and since this specific slice never calls `stereoRectify` or any other
function shown to be shape-sensitive about translation vectors (§25's hazard list), no
cross-version compatibility concern arises from this choice.

## 19. Namespace / module placement

The project's **only** existing namespace split, `improcv.visualization`, exists specifically to
isolate an *optional dependency* (matplotlib) — `import improcv` never imports it. Every other
domain, including the 4,000+-line `evaluation.py`, contributes flat top-level names regardless of
size or thematic coherence. `cv2` is already a hard, required dependency for the whole project —
there is no analogous optional-dependency story available for a geometry namespace, and the task's
own instruction is explicit: *"No optional dependency exists here, so do not copy visualization's
laziness mechanically without reason."*

Introducing `improcv.geometry` now — for exactly one function — would set a *new* kind of
precedent (namespacing purely for conceptual grouping) this project has never used, based on no
evidence yet that this domain will grow large enough to need it. That is exactly the kind of
premature abstraction this project's own stated philosophy rejects ("prefer plain functions over
abstractions until several real use cases justify one" applies to namespace decisions, not only to
function/type decisions).

**Frozen: the existing top-level module pattern.** New module `src/improcv/geometry.py` holding
`estimate_object_to_camera_pose` and `ObjectToCameraPose` (no `CameraMatrix`, §12); both
re-exported from `improcv.__init__` exactly like every other domain module. `import improcv`
imports this
module eagerly, same as every module except `visualization`.

This explicitly revises the prior research spike's tentative lean toward examining a `geometry`
namespace — after applying the project's actual demonstrated precedent rather than abstract
"avoid top-level growth" reasoning, flat top-level placement is the non-premature, historically-
consistent choice for a *first* slice in a new domain. If a second or third geometry-domain slice
is ever designed, *that* would be the natural point to revisit a namespace decision — not decided
here.

## 20. Exact public API delta — corrected

| Symbol | Kind | New? |
|---|---|---|
| `estimate_object_to_camera_pose` | function | + |
| `ObjectToCameraPose` | result type | + |

**No `CameraMatrix` alias (§12, corrected).** `improcv.__all__`: **198 → 200** (corrected from the
originally-merged 201, which counted the now-removed `CameraMatrix`). `improcv.visualization.__all__`:
unchanged at **3**. New module: `src/improcv/geometry.py`. New public types: **+1**
(`ObjectToCameraPose` only).

## 21. Typing — candidate final signature (corrected)

```python
def estimate_object_to_camera_pose(
    object_points: npt.NDArray[np.floating[Any]],
    image_points: npt.NDArray[np.floating[Any]],
    camera_matrix: npt.NDArray[np.floating[Any]],
    *,
    distortion: npt.NDArray[np.floating[Any]] | None = None,
) -> ObjectToCameraPose:
```

Corrected from the originally-merged signature, which typed `camera_matrix: CameraMatrix` (an
alias defined as float64-only, §12) and `distortion: npt.NDArray[np.float64] | None` — both
narrower than what §11/§13 actually validate and accept (float32 *or* float64, normalized
internally). This signature says exactly what the runtime contract accepts: a valid float32 caller
is never statically rejected by an annotation that doesn't match the function's real behavior.

`object_points`/`image_points`/`camera_matrix` are positional (matching `cv2`'s own conventional
ordering, already familiar to the target audience, and all three are always-required geometric
inputs, not tuning knobs). `distortion` is keyword-only with a default, matching this project's
consistent house rule that any parameter with a default is keyword-only (`confusion_matrix(...,
*, labels=None, sample_weight=None)`, `load_image(path, *, mode="color")`). No shape information is
encoded in the type annotations themselves — Python's type system cannot honestly enforce `(N,3)`
vs `(N,2)`; that contract lives in the docstring and in runtime validation (§23).

## 22. Minimal test-oracle plan (design only — not implemented here)

- Fixed, hand-authored synthetic scenes (not `np.random`-generated per test run, avoiding flaky
  camera configurations) — reuse the well-conditioned box-corner-style 3D points and a `z=0`
  planar point set already validated in §2a.
- Known `R_true`/`t_true`; project via `cv2.projectPoints` to obtain ground-truth `image_points`;
  call `estimate_object_to_camera_pose`; assert `result.rotation` ≈ `R_true` and
  `result.translation` ≈ `t_true`.
- Reprojection round-trip tolerance, derived directly from the prior research spike's measured
  numbers, with margin: **float64 < 1e-6 px** (measured ~1.3e-13), **float32 < 1e-3 px** (measured
  ~3.2e-5 px).
- Coverage required: float64 input; float32 input; the planar 6-point scene *and* the non-planar
  8-point scene (both must succeed under the frozen N>=6-universal policy, §10); `distortion=None`;
  a legitimate nonzero distortion vector (confirming it is actually plumbed through to `cv2`, using
  the *same* distortion for both the ground-truth projection and the estimation call); both
  OpenCV 4.9 and 5.0 (the project's existing floor/current CI matrix already covers this, no new
  infrastructure needed).

## 23. Validation tests and deterministic error order — corrected

The originally-merged order started directly with shape checks, silently assuming every input was
already a floating-dtype `np.ndarray` — a list, string, or integer array would have surfaced as an
uncontrolled `AttributeError` from `.ndim`/`.dtype`, not a clear improcv error. Corrected order,
grouped **type checks for every parameter first** (so nothing later ever touches `.ndim`/`.dtype`/
`.shape` on a value that isn't already confirmed to be a floating-dtype ndarray), then static shape
checks, then cross-parameter/value checks, then the `cv2` call:

1. `object_points` is `np.ndarray` → else `TypeError`
2. `object_points` dtype is floating → else `TypeError`
3. `image_points` is `np.ndarray` → else `TypeError`
4. `image_points` dtype is floating → else `TypeError`
5. `camera_matrix` is `np.ndarray` → else `TypeError`
6. `camera_matrix` dtype is floating → else `TypeError`
7. `distortion`, if not `None`, is `np.ndarray` → else `TypeError`
8. `distortion`, if not `None`, dtype is floating → else `TypeError`
9. `object_points` shape is `(N, 3)` for some `N` → else `ValueError`
10. `image_points` shape is `(M, 2)` for some `M` → else `ValueError`
11. `camera_matrix` shape is exactly `(3, 3)` → else `ValueError`
12. `distortion`, if not `None`, is 1-D with length in `{0, 4, 5, 8, 12, 14}` (0 treated as
    `None`-equivalent, §13) → else `ValueError`
13. `N == M` (`object_points`/`image_points` count match) → else `ValueError`
14. `N >= 6` → else `ValueError`
15. `object_points` all finite → else `ValueError`
16. `image_points` all finite → else `ValueError`
17. `camera_matrix` all finite → else `ValueError`
18. `camera_matrix` passes the strict pinhole-structure check (§11: skew/off-diagonal zeros,
    `K[2,2]==1`, fx/fy positivity) → else `ValueError`
19. `distortion`, if not `None`, all finite → else `ValueError`
20. only after all of the above pass: call `cv2.solvePnP`; propagate any unanticipated `cv2.error`
    unchanged; raise `RuntimeError` if it returns `ok=False`.

Step 17 (camera-matrix finiteness) is deliberately checked immediately before step 18 (its
structural comparison) rather than being grouped with steps 15-16: comparing a NaN camera-matrix
entry against `0.0`/`1.0` via `==` would silently evaluate to `False` and misreport a finiteness
problem as a structural-mismatch problem, so finiteness must be confirmed first for that specific
array.

Tests must include multiple-simultaneously-invalid-input cases confirming this exact firing order,
matching this project's established "error ordering must be deterministic and documented" pattern
used throughout `evaluation.py`'s validators — including at least one case per corrected boundary:
a Python `list` for `object_points` (must raise `TypeError`, never `AttributeError`), an
integer-dtype `object_points` array (must raise `TypeError`, never be silently upcast), and a
`camera_matrix` containing NaN in a structural position (must raise `ValueError` for
non-finiteness, not a confusing structural-mismatch message).

## 24. Thin-wrapper gate — final check

The frozen design: never exposes raw `rvec`; never exposes `tvec`'s raw `(3,1)` shape; never
exposes raw `cv2` integer method flags; never accepts `cv2`'s `(N,1,·)`/`(1,N,·)` shape
polymorphism; never lets a basic structural failure surface as a raw `cv2.error` (e.g. `"'6' is
6"`) instead of a clear `ValueError`; and **actively rejects** a skew term `cv2` itself silently
discards. This clears the gate — it is not a thin wrapper around `cv2.solvePnP`.

## 25. Explicit out-of-scope list

`solvePnPRansac`, `solvePnPGeneric`, `solvePnPRefineLM`/`RefineVVS`, method selection beyond
`ITERATIVE`, extrinsic guesses, camera calibration (`calibrateCamera`), checkerboard/ChArUco
detection, undistortion, stereo, triangulation, essential/fundamental matrices, `recoverPose`,
hand-eye calibration, fisheye calibration, pose inversion/composition helpers, a generic
homogeneous-transform framework, SfM, SLAM, bundle adjustment.

**Recorded future compatibility hazards** (discovered during this and the preceding research
spike; not blockers to this slice, but must not be lost from project history for whenever any of
the above is designed):

- `cv2.stereoRectify`'s `T` parameter: accepts flat `(3,)` in OpenCV 4.9 but requires exact
  `(3,1)` in OpenCV 5.0 (raises a `gemm` assertion otherwise).
- `cv2.triangulatePoints` with `(N,2)`-shaped input: raises a clear error in OpenCV 4.9 but
  **silently returns wrong, truncated output** in OpenCV 5.0 — a genuine silent-wrong-answer
  hazard, not merely an inconvenience.
- OpenCV 5.0.0's Python bindings are missing `calibrateHandEye`/`calibrateRobotWorldHandEye`
  (confirmed via source: `CV_EXPORTS` at the `5.0.0` tag vs. `CV_EXPORTS_W` already restored on the
  `5.x` development branch — a binding regression already being fixed upstream, not an intentional
  removal); `SOLVEPNP_DLS`/`SOLVEPNP_UPNP` constants were also removed in 5.0 but were confirmed
  pure aliases of `EPNP`, so no real capability is lost.

## 26. Version recommendation — corrected verdict

**DESIGN INTERNALLY CONSISTENT — READY FOR VERSION-GATE REAUDIT.**

(This supersedes the originally-merged verdict wording, "DESIGN READY — DO NOT OPEN 0.6 YET" — the
substance is the same, but this correction pass exists precisely because the original document was
*not yet* internally consistent: a static/runtime typing mismatch (§1 of this correction), a
premature public `CameraMatrix` alias (§2), an underspecified type/dtype validation contract (§3),
a materially inaccurate "excludes no real capability" claim (§4), an unfrozen "tight tolerance" for
the camera-matrix structural check (§5), and an unstated dataclass-construction semantic (§6) have
all now been corrected. The verdict vocabulary is deliberately different from the original
document's own to make unmistakably clear that a corrected design being self-consistent is not the
same claim as a design being *approved to build* — that remains, as before, a separate decision.)

Every item this design needed to freeze is now resolved into a concrete, justified, and internally
consistent decision: transform direction, naming, result-type shape/fields/construction semantics,
ownership/equality, point-array and camera-matrix contracts (including two corrected empirical
findings and one corrected typing/alias decision), distortion contract, degeneracy/planarity policy
(now stated honestly, including its real capability limitation), method scope, failure contract,
rotation/translation normalization, namespace placement, exact public delta, a corrected candidate
final signature, a concrete test-oracle plan with real numeric tolerances, and a corrected
deterministic validation order covering both type and shape/value failures. The design passes its
own acceptance gate (§27) and the thin-wrapper gate (§24).

This still does **not** mean 0.6 should open now. The preceding boundary audit's verdict (**NO
IMMEDIATE 0.6**, absent new user/workflow evidence) remains a separate, timing/evidence question
this correction does not touch — a design being internally consistent is not itself the kind of
evidence that audit was looking for. Whether this design is *approved* is a distinct, still-open
"version-gate reaudit" decision for whoever chooses to make it next, not concluded here.

## 27. Design acceptance gate — self-check

| Requirement | Status |
|---|---|
| Transform direction explicit | ✓ §4 |
| ITERATIVE minimum-point behavior correct | ✓ §2a, directly re-verified |
| Standard pinhole K semantics correct | ✓ §2b, directly re-verified (skew finding corrected the prior spike) |
| Result type avoids ambiguous generic `Pose` | ✓ §6 |
| Canonical shapes/dtypes frozen | ✓ §8, §9 |
| Distortion contract frozen | ✓ §13 |
| Planarity/degeneracy policy frozen | ✓ §10 |
| Solver-failure policy frozen | ✓ §16 |
| Method selection frozen/out-of-scope | ✓ §14 |
| Namespace decision frozen | ✓ §19 |
| Exact public delta known | ✓ §20 (198→200, corrected — no `CameraMatrix`) |
| OpenCV 4/5 first-slice behavior demonstrated stable | ✓ §2, all instabilities found affect only out-of-scope functions (§25) |
| Deterministic test oracle exists | ✓ §22 |
| Thin-wrapper gate passes | ✓ §24 |
| No broad calibration framework leaks in | ✓ §25 |

All criteria met.
