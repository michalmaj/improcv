# Design: deterministic PnP-based object-to-camera pose estimation

Status: **speculative — no development line has been opened for this feature.** This document
freezes a candidate API contract to determine whether one *can* be frozen durably; it does not by
itself approve building it. See §0 for why this file is deliberately not named after any version.

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
5. Pre-validated, specific `ValueError`s (bad shape, dtype, point count, camera-matrix structure,
   distortion length) in place of `cv2`'s internal C++ assertion text (e.g. `"'6' is 6"`).
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

## 8. Object-point contract

- Public shape: **exactly `(N, 3)`** — never `cv2`'s `(N,1,3)`/`(1,N,3)` forms, rejected explicitly
  with a clear message.
- Must be `np.ndarray` (not a bare `Sequence` — unlike some evaluation functions that accept
  `Sequence[int]`, a 3D point set is naturally already array-shaped data, and requiring `ndarray`
  keeps the shape contract simple and unambiguous for a first slice).
- dtype policy: **accept `np.floating` (float32 or float64), normalize internally to float64**
  before calling `cv2` — matches this project's general pattern of accepting a reasonably wide
  numeric input and normalizing (e.g. `TransformMatrix = npt.NDArray[np.floating[Any]]`), rather
  than gatekeeping to float64-only.
- Must be finite (reject NaN/Inf) — closes a real gap: `projectPoints` silently propagates NaN,
  and neither version catches Inf anywhere in the object/camera-matrix path (§2 of the prior
  research spike).
- Internal C-contiguity is improcv's problem, not the caller's — normalize via
  `np.ascontiguousarray` internally rather than requiring the caller to think about it.
- Minimum N: **6**, universally — see §10.

## 9. Image-point contract

- Public shape: **exactly `(N, 2)`**, `N` identical to `object_points`'s `N` — mismatch is a
  `ValueError`, not left to `cv2`'s own (already-confirmed-clear-enough) internal assertion.
- Same dtype/finite/contiguity policy as §8.
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
  at all): §2a confirmed planar N=6 succeeds correctly in both OpenCV versions — so a universal
  N>=6 minimum does not exclude *any* real capability, it only asks planar users for 2 more points
  than OpenCV's own bare minimum. This entirely sidesteps the scale-dependent-tolerance problem,
  requires zero geometry-classification code, and cannot disagree with `cv2`'s own internal
  (not-fully-reverse-engineered) planarity test, because improcv never attempts to replicate it.
- **Policy C** (support only non-planar): rejected — planar target scenarios (fiducial markers,
  calibration checkerboards used purely for pose, etc.) are one of the most common real-world PnP
  use cases; excluding them gives *less* value than raw `cv2`.

**Frozen: Policy B.** `object_points`/`image_points` must have `N >= 6`; the function does not
distinguish planar from non-planar internally at all. This is documented as a deliberate
simplification, not an oversight, and is a strictly backward-compatible thing to relax later (a
future slice could add planarity detection and lower the minimum for planar inputs without
breaking any existing caller).

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
| Shape ≠ (3,3) | Rejected (`cv2.error`) | **Require exactly (3,3)** |
| dtype | float32/float64 both accepted | Accept both, normalize to float64 |
| `K[0,1]`, `K[1,0]` ≠ 0 | **Silently ignored** (§2b) | **Require exactly 0** — this is the corrected decision from §2b |
| `K[2,0]`, `K[2,1]` ≠ 0 | Not a supported pinhole parameterization | **Require exactly 0** |
| `K[2,2]` ≠ 1 | Silently accepted, used as an unexplained scale | **Require exactly 1** (tight tolerance) — no known legitimate alternative parameterization |
| fx, fy ≤ 0 | Silently accepted, produces sign-flipped/mirrored results | **Require > 0** — same "silently-wrong-but-accepted" hazard class as skew; a design choice, not independently re-tested this round, but justified by the same logic already validated for skew/Inf |
| NaN | Caught by `solvePnP`'s own assertion, **not** caught by `projectPoints` | **Require finite** — closes the `projectPoints` gap |
| Inf | **Neither version catches this** | **Require finite** — a real gap in `cv2` itself |
| Principal point (cx, cy) | Unrestricted | No restriction beyond finiteness — legitimately can be anywhere |

Ownership: `camera_matrix` is read, never mutated; any dtype normalization happens on an internal
copy, matching this project's "never hide expensive array copies, always document" rule.

## 12. Public `CameraMatrix` type

Directly extends the existing `TransformMatrix = npt.NDArray[np.floating[Any]]` precedent in
`src/improcv/types.py` — a plain, coarse type alias documented with its shape/value contract in a
docstring comment, not a validated dataclass. A type alias cannot itself enforce shape or the
structural-zero/`K[2,2]==1`/positivity invariants from §11 — those remain the responsibility of a
private validator called at the top of the function, exactly like every other validated-but-alias-
typed parameter in this project.

**Frozen: `CameraMatrix = npt.NDArray[np.float64]`, public**, documented: *"A `(3,3)` standard
pinhole camera intrinsic matrix `[[fx,0,cx],[0,fy,cy],[0,0,1]]` — `fx`,`fy` > 0, all other
positions exactly as shown."* A validated dataclass wrapper (`CameraIntrinsics`-style) is rejected
as premature for a single first-slice parameter, matching `TransformMatrix`'s own precedent of
staying a plain alias.

## 13. Distortion-coefficient contract

Every tested shape (`None`, empty, `(4,)`, `(5,)`, `(8,)`, `(12,)`, `(14,)`, and column-vector
equivalents) was accepted by both `solvePnP` and `projectPoints` in both OpenCV versions, with no
rejections anywhere (prior research spike). Given this permissiveness and simplicity, **no public
`DistortionCoefficients` type is introduced** — the contract is thin enough to document directly.

**Frozen**: keyword parameter named `distortion` (not `dist_coeffs` — matches this project's
consistent preference for full English words over `cv2`-mirroring abbreviations, e.g.
`sample_weight` not `s_weight`), typed `npt.NDArray[np.float64] | None = None`.

- `None` means zero distortion (matches `cv2`'s own semantics directly).
- An empty array (`shape (0,)`) is treated **identically to `None`** — a harmless, informationally
  equivalent representation some callers may naturally produce; rejecting it as an invalid length
  would be needlessly unforgiving.
- Otherwise: must be exactly **1-D**, length in `{4, 5, 8, 12, 14}`, finite. `cv2`'s column-vector
  forms (`(4,1)` etc.) are **not** accepted publicly — reshaped internally only if ever needed,
  never exposed as a caller-facing shape choice.
- Accepts float32 or float64, normalized to float64.

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

- Every structural/numeric precondition from §8-§13 is validated **before** calling `cv2`,
  raising a specific `ValueError` with a clear message (§23 freezes the exact order).
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
`estimate_object_to_camera_pose`, `ObjectToCameraPose`, and `CameraMatrix`; all three re-exported
from `improcv.__init__` exactly like every other domain module. `import improcv` imports this
module eagerly, same as every module except `visualization`.

This explicitly revises the prior research spike's tentative lean toward examining a `geometry`
namespace — after applying the project's actual demonstrated precedent rather than abstract
"avoid top-level growth" reasoning, flat top-level placement is the non-premature, historically-
consistent choice for a *first* slice in a new domain. If a second or third geometry-domain slice
is ever designed, *that* would be the natural point to revisit a namespace decision — not decided
here.

## 20. Exact public API delta

| Symbol | Kind | New? |
|---|---|---|
| `estimate_object_to_camera_pose` | function | + |
| `ObjectToCameraPose` | result type | + |
| `CameraMatrix` | type alias | + |

`improcv.__all__`: **198 → 201**. `improcv.visualization.__all__`: unchanged at **3**. New module:
`src/improcv/geometry.py`.

## 21. Typing — candidate final signature

```python
def estimate_object_to_camera_pose(
    object_points: npt.NDArray[np.floating[Any]],
    image_points: npt.NDArray[np.floating[Any]],
    camera_matrix: CameraMatrix,
    *,
    distortion: npt.NDArray[np.float64] | None = None,
) -> ObjectToCameraPose:
```

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

## 23. Validation tests and deterministic error order

Frozen order (structural/shape checks first, cheapest-and-most-fundamental first):

1. `object_points` wrong ndim/trailing shape (≠ `(N,3)`) → `ValueError`
2. `image_points` wrong ndim/trailing shape (≠ `(N,2)`) → `ValueError`
3. N mismatch between `object_points` and `image_points` → `ValueError`
4. `N < 6` → `ValueError`
5. non-finite `object_points` → `ValueError`
6. non-finite `image_points` → `ValueError`
7. `camera_matrix` shape ≠ `(3,3)` → `ValueError`
8. `camera_matrix` non-finite → `ValueError`
9. `camera_matrix` fails the strict pinhole-structure check (§11: skew/off-diagonal/`K[2,2]`/
   fx,fy-positivity) → `ValueError`
10. `distortion` invalid length (not in `{0,4,5,8,12,14}`, treating 0 as `None`-equivalent) →
    `ValueError`
11. `distortion` non-finite → `ValueError`
12. only after all the above pass: call `cv2.solvePnP`; propagate any unanticipated `cv2.error`
    unchanged; raise `RuntimeError` if it returns `ok=False`.

Tests must include multiple-simultaneously-invalid-input cases confirming this exact firing order,
matching this project's established "error ordering must be deterministic and documented" pattern
used throughout `evaluation.py`'s validators.

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

## 26. Version recommendation

**DESIGN READY — DO NOT OPEN 0.6 YET.**

Every item this design needed to freeze was resolved into a concrete, justified decision: transform
direction, naming, result-type shape and fields, ownership/equality, point-array and camera-matrix
contracts (including two corrected findings from direct experiment), distortion contract,
degeneracy/planarity policy, method scope, failure contract, rotation/translation normalization,
namespace placement, exact public delta, a candidate final signature, a concrete test-oracle plan
with real numeric tolerances, and a deterministic validation order. The design passes its own
acceptance gate (§27) and the thin-wrapper gate (§24).

This does **not** mean 0.6 should open now. The preceding boundary audit's verdict (**NO IMMEDIATE
0.6**, absent new user/workflow evidence) was a separate, timing/evidence question this design
exercise does not change — producing a sound design for one candidate slice is not itself the kind
of evidence that audit was looking for. Per this task's own explicit framing, "design soundness"
and "whether to open a new development line" are two independent gates, and only the first is
answered here.

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
| Exact public delta known | ✓ §20 (198→201) |
| OpenCV 4/5 first-slice behavior demonstrated stable | ✓ §2, all instabilities found affect only out-of-scope functions (§25) |
| Deterministic test oracle exists | ✓ §22 |
| Thin-wrapper gate passes | ✓ §24 |
| No broad calibration framework leaks in | ✓ §25 |

All criteria met.
