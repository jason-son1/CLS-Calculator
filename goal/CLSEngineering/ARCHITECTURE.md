# Real-Space CLS 기반 Topological Flat Band Engineering — 아키텍처 설계

본 문서는 다음 두 연구 노트

- `실공간 CLS 관점에서 본 Finite Fourier-Sum Eigenvector의 비자명 Chern Number 조건.pandoc.md` (이하 **Note A**)
- `CLS 기반 Topological Flat Band Engineering 알고리즘.pandoc.md` (이하 **Note B**)

에 정의된 5-Phase 알고리즘을 `cls_finder/engineer/` 패키지로 구현하기 위한
설계 초안이다. 기존 `cls_finder.classify.chern` / `cls_finder.band.bands` /
`cls_finder.viz.plot` 의 검증·시각화 인프라를 **그대로 재사용**하는 것을
핵심 설계 원칙으로 한다 (중복 구현 금지).

## 0. 표기 및 핵심 항등식 (구현 전 수학적 정리)

설계 변수: 각 sublattice α, 특이점 kᵢ 에 대해 CLS site j 의
`(A_j^(α), θ_j^(α), R_j^(α))`.

### 0.1 ζ^(α) 의 게이지적 위치

`f_α(k) = e^{iζ^(α)} g_α(k)`, `g_α(k) = Σ_j A_j e^{iθ_j} e^{ik·R_j}`.

- **공통 영점 조건** `f(k_i)=0 ⟺ g(k_i)=0` — ζ 와 무관.
- **Jacobian** `A_{α,μ}=e^{iζ_α}(∂g_α/∂k_μ)` — 열 비례(rank-1)는 sublattice별
  위상 재정의에 불변.
- **Winding** `Im⟨A_x,A_y⟩ = Σ_α Im(e^{-iζ_α}\overline{a_g^(α)}·e^{iζ_α}b_g^(α))
  = Σ_α Im(\overline{a_g^(α)} b_g^(α))` — **ζ 완전히 상쇄**.

⇒ **Phase 2/3은 ζ=0으로 풀고, ζ^(α)는 Phase 4 조립 시에만 곱연산으로
주입한다.** (위상수학적 불변량은 ζ에 무관, 최종 hopping의 상대 위상에만
영향.)

### 0.2 Pairing Rule → Condition 1 자동 충족 (Module 2)

앵커: `R_{j2,p}=(0,0)`(원점), `R_{j1,p}=d_p=(n_p,m_p)` (정수 격자 벡터 좌표),
`m=0`.

```
θ_{j1,p} + k_i·d_p = θ_{j2,p} + π   (pairing rule, A_{j1,p}=A_{j2,p}=A_p)
  ⟹  e^{iθ_{j2,p}} + e^{i(θ_{j1,p}+k_i·d_p)} = 0      (각 pair가 개별적으로 0)
  ⟹  Σ_j A_j e^{iΘ_j} = 0  (Condition 1, Note A Eq.11 / Note B Phase2)
```

이는 `θ_{j1,p}, θ_{j2,p}` 둘 중 하나만 자유 변수이고 나머지는 **대수적으로
결정됨**을 의미 — Phase 3에서 그 자유도를 사용한다.

### 0.3 Bond-vector moment → Chiral 방정식 (Module 3)

`Φ_p ≡ Θ_{j1,p} + π/2 = θ_{j1,p} + k_i·d_p + π/2` 로 두면 (Note B 식, Note A
§5-7 Eq.33-46):

```
a^(α) = Σ_p A_p e^{iΦ_p} d_{p,x}
b^(α) = Σ_p A_p e^{iΦ_p} d_{p,y}
```

목표 `w_i=±1`에 대한 **Chiral Symmetry Equation** `a^(α) = -i w_i b^(α)`는

```
D_p ≡ d_{p,x} + i w_i d_{p,y}
Σ_p A_p e^{iΦ_p} D_p = 0                      ← 단일 복소 방정식
```

로 환원된다. **2-pair 닫힌 해 (P=2, gauge Φ_2=0)**:

```
A_1 = |D_2|,  A_2 = |D_1|,  Φ_2 = 0,  Φ_1 = π + arg(D_2) - arg(D_1)
```

`d_1, d_2 ≠ 0` 이고 **선형독립**(cross(d_1,d_2)≠0)이면 항상 풀리며
`(a,b)≠(0,0)` (first-order 영점). `θ_{j1,p}=Φ_p-π/2-k_i·d_p`,
`θ_{j2,p}=Φ_p+π/2` 로 역산.

**검산 (closed form)**: `a+iw_ib = Σ_p A_p e^{iΦ_p}D_p = 0` ⟹ `a=-iw_ib`
항등적으로 성립. 또한 `\bar a b = i w_i |b|^2` ⟹
`Im⟨A_x,A_y⟩ = w_i Σ_α|b^(α)|^2` ⟹ `sgn Im⟨A_x,A_y⟩ = w_i` —
`chern.local_winding`의 부호 규약과 정확히 일치.

### 0.4 축퇴(Degeneracy) 함정과 Shell 라이브러리

모든 α에 대해 동일한 `(d_1,d_2)`를 쓰면 `c_{α,R} = κ_α · c^{ref}_R`
(스칼라 배수 관계)가 되어 `f(k)=g(k)·v_0` (v_0 상수 벡터) — `P(k)`가
**전역적으로 상수**가 되어 kᵢ에서의 local winding=w_i 임에도
**C=0** (다른 곳의 보상 영점). ⇒ **sublattice마다 다른 (d_1,d_2) shell**을
순환 배정하는 `DEFAULT_SHELLS` 라이브러리를 사용 (모듈 3).

### 0.5 N=1 자명성

N=1이면 `P(k)=f_1\bar f_1/|f_1|^2 = 1` (1×1 항등) — 항상 C=0.
⇒ Phase 1에서 **목표 C≠0 ⟹ N≥2 필수** 검증.

## 1. 패키지 구조

```
cls_finder/engineer/
  __init__.py      - 공개 API export
  spec.py          - Module 1: LatticeSpec, SublatticeSpec, SingularityTarget, DesignTarget
  pairing.py       - Module 2: CLSSite, CLSPair, SublatticeCLS, CLSDesign, make_pair
  chiral.py        - Module 3: D_p/Φ_p 닫힌해, DEFAULT_SHELLS, build_sublattice_cls
  assembly.py      - Module 4: build_x_k, evaluate_psi, verify_projector_continuity
  hamiltonian.py   - Module 5: default_M_k, NumericHk, inverse_fourier_transform
  pipeline.py      - design_flat_band(): 전체 오케스트레이션 + 피드백 루프 + 로그
```

## 2. 데이터 흐름

```
LatticeSpec + DesignTarget                                  (사용자 입력)
        │  Phase1: target.validate(), N>=2 체크
        ▼
for singularity k_i (각 attempt마다 shell offset 변경):
    for α in 0..N-1:
        shell = DEFAULT_SHELLS[(α+offset) % L]
        build_sublattice_cls(α, k_i, shell)  → SublatticeCLS (3 sites: 0,d1,d2)
    → CLSDesign(k_i, [SublatticeCLS]*N)
        │  Phase2+3 (closed form, 대수적으로 정확)
        ▼
build_x_k(designs)  → x_k = [LaurentPoly]*N      (Phase4: f_α(k)=e^{iζ_α}Σc_R e^{ikR})
        │
verify_projector_continuity(x_k, k_i) for each k_i
   via chern.jacobian_at_zero / local_winding / loop_winding
analytic_chern(x_k)  → C_actual = Σ w(전체 공통영점)
        │
        ├─ C_actual == target.C and all loop windings match & continuous?
        │       NO → [WARNING] 로그 + offset 변경 후 재시도 (max_retries)
        ▼ YES
default_M_k / user M(k)  → MatrixPoly
NumericHk(x_k, M, E0, vortex_vectors)            (Phase5: P(k) + 해석적 패치)
   .evaluate / .evaluate_batch  ← bands.py / chern.py / viz.plot 재사용
        │
inverse_fourier_transform(H_k, n_grid, R_cut) → t_{αβ}(R), 절단 진단
        │
numerical_chern(H_trunc) vs target.C
   NOT MATCH & gapped → R_cut 증가 재시도 (bounded)
        ▼
DesignResult: cls_designs, x_k, ψ(k), H_k(NumericHk), hoppings, verification log
```

## 3. Module 5: Hamiltonian 및 IFT 세부

### 3.1 P(k) 의 해석적 극한 패치

`f(k_i)=0` 이므로 `P(k)=f f^†/‖f‖^2` 는 `k=k_i`에서 0/0. Phase3에서 구한
`(A_x^(i),A_y^(i))=jacobian_at_zero(x_k,k_i)`에 대해 `v_i := A_y^(i)`
(또는 더 큰 쪽)를 사용해

```
P(k_i) = v_i v_i^† / ⟨v_i|v_i⟩          (Note A Eq.16, rank-1 보장)
```

`NumericHk.evaluate_batch`는 `‖f(k)‖^2 < tol` 인 격자점에서만 이 패치를
적용한다 (그 외 점은 연속이므로 해석식 그대로 사용).

### 3.2 H(k) = E0·P(k) + (I-P(k)) M(k) (I-P(k))

`M(k)`는 기본적으로 순환 NN 호핑 `M(k)=t·(S e^{ik·a1}+S^T e^{-ik·a1}
+ S e^{ik·a2}+S^T e^{-ik·a2})` (S=cyclic shift, N×N) — 에르미트, 인접
hopping만 포함(locality). 사용자가 임의의 `MatrixPoly`로 교체 가능.

### 3.3 IFT

`t_{αβ}(R)=(1/N_k)Σ_k H_{αβ}(k)e^{-ik·R}`, `k`는 `n_grid×n_grid` 격자
(`chern._frac_grid`). `|n|,|m|≤R_cut` 으로 절단, Parseval로 절단된
스펙트럴 weight 비율을 진단으로 보고. 절단된 `t_{αβ}(R)` 로
`MatrixPoly H_trunc`를 재구성해 `numerical_chern`/`band_isolation`으로
재검증한다 (Rhim-Yang 가드레일).

## 4. 피드백 루프 & 검증 로그

- Phase3/4: `loop_winding(k_i).winding == w_i` AND `.projector_continuous`
  AND 전체 `analytic_chern(x_k).C == target.C` 모두 충족할 때까지
  shell-offset을 바꿔 최대 `max_retries`회 재시도.
- Phase5: `band_isolation(H_trunc)`이 gapped인데
  `numerical_chern(H_trunc).C != target.C` 이면 `R_cut`을 늘려 재시도.
- 종료 시 다음을 명시적으로 출력:
  - `C = Σ w_i` (목표) vs 해석적/수치적 실제 C (continuous H, truncated H)
  - 각 특이점의 `loop_winding`, `rank_ratio`(연속성 점수), `order`
  - Rhim-Yang 모순 발생 시 WARNING + 재시도 로그
