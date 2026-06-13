본 문서는 Flat band 및 dispersive band system의 Global Topology를
수치적으로 계산하고 검증하기 위한 4가지 핵심 방법론의 구현
가이드라인입니다. 각 방법론의 물리적 원리, 수치 해석적 알고리즘, 그리고
실제 코딩 시 유의해야 할 Technical detail을 포함합니다.

## 1. Fukui-Hatsugai-Suzuki (FHS) Method

**목적:** 2D 시스템의 Chern Number 계산 (Time-Reversal Symmetry가 깨진
시스템)

**물리적 배경 및 장점:** 기존의 Berry curvature 연속 적분 방식은
파동함수의 Gauge를 BZ 전체에 걸쳐 부드럽게(smooth) 잡아야 한다는
치명적인 단점이 있습니다. 수치적 대각화(Numerical Diagonalization)는 각
$k$-point마다 무작위의 Phase를 뱉어내므로(Random gauge), 이를 바탕으로
단순 미분을 계산하면 Dirac string 위치에서 발산 오류가 발생합니다.

FHS 방법은 Lattice Gauge Theory의 아이디어를 빌려와 연속적인 미분 대신
인접한 격자점 사이의 이산화된 위상 차이(Link variable)를 계산합니다. 이
방식은 명시적으로 Gauge-invariant하며, 아주 거친(coarse) $k$-mesh를
사용하더라도 수치적 오차 없이 완벽한 정수(Exact integer)의 Chern
Number를 도출하는 현존 최고의 알고리즘입니다.

### 1.1. 수학적 원리 및 알고리즘

1.  $k$**-space 이산화:** BZ를 $N_x \times N_y$ 개의 격자점으로
    나눕니다. $k_l = (k_{x, i}, k_{y, j})$.

2.  **Wavefunction 계산:** 각 $k_l$에서 Hamiltonian $\mathcal{H}(k_l)$을
    대각화하여 Occupied band들의 eigenvector $|u_n(k_l)\rangle$를
    구합니다 ($n=1, \dots, N_{occ}$). 이때 얻어지는 파동함수들의 Gauge는
    완전히 무작위여도 상관없습니다.

3.  **Link Variable (**$U_\mu$**) 계산:** 인접한 $k$ 포인트 간의 평행
    이동(Parallel transport)을 나타내는 행렬을 구합니다. 다중
    밴드(Multi-band) 시스템의 경우, $N_{occ} \times N_{occ}$ 크기의
    Overlap matrix $M_\mu(k_l)$을 먼저 구성합니다:

    $[M_\mu(k_l)]_{mn} = \langle u_m(k_l) | u_n(k_l + \hat{\mu}) \rangle$
    (여기서 $\hat{\mu}$는 $k_x$ 또는 $k_y$ 방향의 인접 격자 벡터)

    이후 행렬식(Determinant)을 취해 다체 파동함수의 겹침(Overlap)을
    스칼라로 만들고, 그 크기(Magnitude)로 나누어 순수한 $U(1)$
    Phase(Link variable)만 추출합니다:

    $$U_\mu(k_l) = \frac{\det M_\mu(k_l)}{|\det M_\mu(k_l)|}$$

4.  **Lattice Field Strength (**$\tilde{F}_{12}$**) 계산:** 격자의 단위
    사각형(Plaquette)을 반시계 방향으로 한 바퀴 도는 Berry flux를
    계산합니다.

    $$\tilde{F}_{12}(k_l) = \ln \left[ U_x(k_l) U_y(k_l + \hat{x}) U_x^{-1}(k_l + \hat{y}) U_y^{-1}(k_l) \right]$$

    이때 매우 중요한 제약 조건이 들어갑니다. 로그 함수 $\ln(z)$의
    허수부를 계산할 때, 반드시 **Principal branch**를 선택하여
    위상(Phase)이 $-\pi < \frac{1}{i} \tilde{F}_{12} \le \pi$ 영역에
    놓이도록 제한해야 합니다. 이를 통해 각 Plaquette를 지나는 Flux가
    과도하게 커지는 것을 막아줍니다.

5.  **Chern Number (**$C$**) 합산:** BZ 내의 모든 Plaquette에 대해
    $\tilde{F}_{12}$를 합산합니다.

    $$C = \frac{1}{2\pi i} \sum_{k_l \in BZ} \tilde{F}_{12}(k_l)$$

    내부의 Link variable들은 인접한 Plaquette끼리 더해지면서 서로
    상쇄(Cancel out)되고, 전체 BZ의 주기성과 Plaquette 내부의 위상
    꼬임(Winding)만이 남아 정확한 정수 $C$를 도출합니다.

### 1.2. 구현 시 주의사항 (Technical Detail)

- **Periodic Boundary Condition (PBC) 강제:** 코드로 배열 인덱싱을 할
  때, $k$-grid의 끝점($N_x, N_y$)이 다시 시작점($0, 0$)의 파동함수를
  참조하도록 모듈로(Modulo) 연산을 하거나 배열을 덮어씌워야 합니다. BZ
  경계에서 Wavefunction의 주기성(Periodicity)이 깨지면 최종 결과가
  정수가 아닌 실수로 도출되는 치명적인 에러가 발생합니다.

- **Degeneracy 및 Band Crossing 처리:** 밴드 간 교차점(Crossing)이
  있으면 수치적 대각화 과정에서 단일 밴드의 Eigenvector들이 무작위로
  섞입니다. 하지만 Occupied band 전체를 하나로 묶어 $\det M_\mu$를
  계산하면, 내부 밴드 간의 섞임은 행렬식
  성질($\det(AB) = \det A \det B$)에 의해 완벽하게 상쇄되어
  Gauge-invariant한 결과를 유지합니다. 분석 대상이 얽혀있는 $N$개의
  밴드라면, 반드시 $N \times N$ Overlap matrix를 구성해야 합니다.

- **Mesh Size (**$N_x, N_y$**) 설정:** FHS 방법은 기본적으로 성긴
  격자에서도 정수값을 도출하지만, 특정 Plaquette 내의 실제 Berry flux
  $\frac{1}{i}\tilde{F}_{12}$ 가 $\pi$를 초과할 정도로 곡률이 뾰족한
  경우 Principal branch cut을 넘어가버려 물리적으로 잘못된 위상
  정수(Aliasing error)를 뱉을 수 있습니다. 따라서 Band gap이 극도로
  작아지거나 곡률이 국소적으로 발산하는 구역이 있다면, 해당 부분의 격자
  밀도(dense mesh)를 충분히 높여주어야 안정적인 계산이 보장됩니다.

## 2. Wilson Loop and Wannier Charge Center (WCC) Flow

**목적:** $Z_2$ Invariant (TRS 보존) 및 Chern Number 판별, 시각적
Topology 증명

**물리적 배경 및 장점:**

Wilson Loop은 파동함수를 Brillouin Zone의 한쪽 끝에서 반대쪽 끝으로 평행
이동(Parallel transport)시킬 때 발생하는 위상(Phase)의 누적을
측정합니다. 다중 밴드(Multi-band) 시스템에서 이 누적된 위상은 단순한
스칼라가 아닌 행렬 형태(Non-Abelian gauge field)로 나타나며, 이 행렬의
Eigenvalue 위상각(Phase angle)을 Brillouin Zone 크기($2\pi$)로 나눈 값이
바로 실공간에서의 Wannier function의 중심 위치, 즉 Wannier Charge Center
(WCC)를 의미합니다.

이 방법의 가장 큰 장점은 Band crossing이나 Gauge 변환에 대해 완벽하게
불변(Gauge-invariant)이라는 점입니다. Flat band가 완전히 깨져
분산(dispersion)이 생기더라도 Occupied band manifold만 잘 정의되어
있다면, WCC의 연속적인 궤적(Flow)을 추적함으로써 Chern Number(Thouless
Pumping의 횟수)나 $Z_2$ Invariant(Kramers pair의 파트너 교환 현상)를
직관적이고 시각적으로 완벽하게 증명해 낼 수 있습니다.

### 2.1. 수학적 원리 및 알고리즘

1.  **경로 설정:** 2D BZ에서 $k_y$를 상수로 고정하고, $k_x$를 $-\pi$
    에서 $\pi$ 까지 $N_x$ 개의 점으로 이산화합니다
    ($k_{x,0}, k_{x,1}, \dots, k_{x,N_x-1}$).

2.  **Overlap Matrix 생성:** 인접한 $k_x$ 점들 사이의 Occupied band들의
    Overlap 행렬 $M^{(i, i+1)}$를 계산합니다. 이 행렬은
    $N_{occ} \times N_{occ}$ 차원을 가집니다.

    $$M_{mn}^{(i, i+1)} = \langle u_m(k_{x,i}, k_y) | u_n(k_{x,i+1}, k_y) \rangle$$

3.  **Wilson Loop Matrix (**$\mathcal{W}$**) 구성:** $k_x$ 경로를 따라
    생성된 모든 Overlap matrix를 Path-ordered product로 순차적으로
    곱합니다.

    $$\mathcal{W}(k_y) = M^{(N_x-1, 0)} M^{(N_x-2, N_x-1)} \dots M^{(1, 2)} M^{(0, 1)}$$

    (마지막 $M^{(N_x-1, 0)}$는 BZ의 주기성을 반영하여 $k_x=\pi$ 와
    $k_x=-\pi$ 의 상태를 연결합니다.)

4.  **Eigenvalue 계산:** $N_{occ} \times N_{occ}$ 행렬인
    $\mathcal{W}(k_y)$ 의 Eigenvalue $\lambda_j$
    ($j=1, \dots, N_{occ}$)를 구합니다. 이상적인 연속 극한에서
    $\mathcal{W}$는 Unitary 행렬이므로 고유값은 복소평면의 단위원 위에
    존재하며, $\lambda_j = e^{i 2\pi \nu_j(k_y)}$ 로 표현할 수 있습니다.

5.  **WCC 추출:** 고유값의 위상각으로부터 WCC $\nu_j(k_y)$ 를
    추출합니다.

    $$\nu_j(k_y) = \frac{1}{2\pi} \text{Im}[\ln \lambda_j] \quad (\text{mod } 1)$$

    $\nu_j$ 는 주기성을 가지므로 보통 $-\frac{1}{2}$ 에서 $\frac{1}{2}$
    (또는 $0$ 에서 $1$) 사이의 값을 가지도록 Principal branch를
    취합니다.

6.  **Topology 판별 (WCC Flow):** $k_y$를 $-\pi$ 에서 $\pi$ 까지
    변화시키며 $\nu_j(k_y)$ 의 궤적(Flow)을 플롯(Plot)합니다.

    - **Chern Number 판별:** $k_y$ 가 전체 주기를 도는 동안 WCC가 실공간
      Unit cell(세로축 기준선)을 가로질러 감싸는 횟수(Winding number)의
      합산입니다. WCC Flow가 아래에서 위로 경계를 뚫고 넘어가면 $+1$,
      위에서 아래로 내려가면 $-1$씩 더해 최종 합을 구합니다.

    - $Z_2$ **Invariant 판별:** Time-Reversal Symmetry가 존재하는
      시스템에서는 $k_y=0, \pi$ (TRIM points)에서 WCC가 반드시 두 개씩
      겹쳐야 합니다(Kramers pair). $k_y$ 축 중간(예: $\nu = 0$ 선)에
      임의의 수평선(Reference line)을 그었을 때, WCC Flow 곡선과
      교차하는 총 횟수가 홀수(Odd)면 위상학적으로 Non-trivial ($Z_2=1$),
      짝수(Even)면 Trivial ($Z_2=0$) 임을 보장합니다.

### 2.2. 구현 시 주의사항 (Technical Detail)

- **Unitarity 강제 보정 (SVD 테크닉 필수):** 이산화된 유한 격자에서는
  파동함수 내적의 수치적 한계로 인해 $\mathcal{W}(k_y)$ 행렬이 완벽한
  Unitary 성질을 잃어버리고 Norm이 보존되지 않을 수 있습니다. 이 행렬을
  그대로 Eigenvalue solver에 넣으면 위상 $\nu_j$ 에 허수부가 생기거나
  발산해 버립니다. 이를 막기 위해 각 $k_x$ 스텝에서 생성된
  $M^{(i, i+1)}$ 행렬에 대해 SVD 분해 ($M = U \Sigma V^\dagger$)를
  수행한 뒤, 특이값 행렬 $\Sigma$ 를 Identity 행렬 $I$ 로 대체하여
  $M_{unitary} = U V^\dagger$ 로 강제 직교화(Orthogonalization) 하는
  전처리 과정을 반드시 적용해야 안정적인 계산이 가능합니다.

- **Continuity 및 Sorting 알고리즘:** $k_y$ 에서 $k_y + \Delta k_y$ 로
  넘어갈 때 점으로 찍힌 WCC들을 선으로 부드럽게 이어주기 위해서는,
  반환된 $\nu_j$ 값들의 순서(Index)를 올바르게 추적해야 합니다. 단순히
  $\nu_j$ 값의 크기순으로 정렬하면 곡선이 교차하는 지점(Crossing
  point)에서 라인이 잘못 이어지는 오류가 생깁니다. 이를 방지하기 위해
  $k_y$ 스텝에서의 $\mathcal{W}$ Eigenvector와 다음
  스텝($k_y + \Delta k_y$)의 Eigenvector 간의 내적(Inner product) 행렬을
  구하고, 내적 절댓값이 가장 큰 쌍끼리 연결해주는 Maximum Overlap
  Sorting 알고리즘을 구현하여 곡선의 연속성을 담보해야 합니다.

## 3. Entanglement Spectrum (ES)

**목적:** 실공간 경계(Boundary)를 직접 만들지 않고 Bulk wavefunction
만으로 Edge state의 위상학적 구조 확인

**물리적 배경 및 장점:**

물리적인 실공간 경계(Open Boundary Condition)를 만들면 필연적으로 Edge의
모양(Zigzag, Armchair 등), finite-size effect, 또는 결함(Defect)에 의한
영향을 크게 받게 됩니다. Li와 Haldane은 시스템을 가상의 공간 절반으로
나누었을 때(Bipartition), 한쪽의 자유도를 적분해버린(Trace-out) Reduced
density matrix의 스펙트럼(Entanglement Spectrum)이 실제 물리적인 Edge
state의 에너지 스펙트럼과 위상학적으로 완벽히 동일하다는 것을
제안했습니다(Li-Haldane Conjecture).

특히 Free fermion (Non-interacting) 시스템에서는 Peschel의 방법론에 따라
복잡한 다체(Many-body) 계산 없이, 오직 단일 입자 상관
행렬(Single-particle correlation matrix)만을 계산하여 이를 부분적으로
추출하는 것으로 정확한 ES를 얻을 수 있습니다. 이는 Bulk 시스템 고유의
순수한 위상학적 성질을 평가하는 매우 우아하고 강력한 수단입니다.

### 3.1. 수학적 원리 및 알고리즘

1.  **Geometry 설정 (Cylinder):** 2D 시스템에서 $y$ 방향은 주기적
    경계조건(Periodic Boundary Condition, $k_y$ 보존)을 유지하고, $x$
    방향은 1차원 실공간 격자(Real-space lattice, $N_x$ unit cells)로
    변환합니다. 이때 Hamiltonian은 각 $k_y$ 마다 독립적인 블록 구조를
    띄며, 하나의 블록은 $N_x \times N_x$ (또는 내부 orbital 개수를
    고려한 크기)의 Block-diagonal 행렬 $\mathcal{H}(k_y)$ 가 됩니다.

2.  **Correlation Matrix (**$C$**) 계산:** 특정 $k_y$ 에서
    $\mathcal{H}(k_y)$ 를 대각화하여 Occupied state $|u_n(k_y)\rangle$
    들을 구합니다. 이를 바탕으로 전체 실공간 사이트 인덱스 $i, j$ 에
    대한 Correlation matrix $C(k_y)$ 를 구성합니다.

    $$C_{ij}(k_y) = \langle c_i^\dagger c_j \rangle = \sum_{n \in \text{occ}} u_{n, i}^*(k_y) u_{n, j}(k_y)$$

    이 행렬은 Occupied states로의 사영(Projection) 연산자와 동일하므로,
    전체 시스템에 대한 $C(k_y)$ 의 고유값은 정확히 1(Occupied) 또는
    0(Unoccupied)만 나옵니다.

3.  **Sub-system 분리 (Trace-out):** 시스템을 $x=N_x/2$ 기준으로 반으로
    자릅니다. 왼쪽 절반(Region A, $1 \le i, j \le N_x/2$)의 자유도만
    남기고 오른쪽 절반(Region B)은 무시(Trace-out)합니다. 알고리즘
    상으로는 전체 $C(k_y)$ 행렬에서 Region A에 해당하는 좌상단 부분
    행렬(Sub-block matrix) $C_A(k_y)$ 만 단순히 추출하면 됩니다.

4.  **Entanglement Energy (**$\epsilon_m$**) 계산:** 축소된 행렬
    $C_A(k_y)$ 를 대각화하여 Eigenvalue $\xi_m(k_y)$ 를 얻습니다. 정보의
    일부가 잘려나갔으므로, 이 값들은 더 이상 0과 1이 아닌
    $0 \le \xi_m \le 1$ 사이의 연속적인 분포를 가집니다. 이 값들을
    Fermi-Dirac 분포 형태의 관계식
    $\xi_m = \frac{1}{e^{\epsilon_m} + 1}$ 을 이용하여 물리적인 에너지와
    유사한 'Entanglement Energy' $\epsilon_m$ 로 변환합니다.

    $$\epsilon_m(k_y) = \ln \left( \frac{1 - \xi_m(k_y)}{\xi_m(k_y)} \right)$$

5.  **Topology 판별 (Plotting):** 가로축을 $k_y$, 세로축을 $\epsilon_m$
    으로 놓고 물리적 밴드 구조처럼 플롯(Plot)합니다. 위상학적으로
    Non-trivial한 시스템이라면, $\epsilon_m = 0$ (즉 $\xi_m = 0.5$)
    부근에서 가상의 Edge mode가 서로 교차하는 Gapless state(예: Dirac
    cone 형태의 Crossing)가 선명하게 나타납니다.

### 3.2. 구현 시 주의사항 (Technical Detail)

- **Finite-size Effect 극복:** 비록 물리적 경계를 실제로 자른 것은
  아니지만, Region A의 물리적 크기($N_x/2$)가 시스템의 Correlation
  length (또는 Wannier orbital의 퍼짐 정도)보다 작으면 가상의
  경계선(Virtual boundary) 양 끝에서 파동함수가 서로 겹쳐서 위상학적
  교차점이 인위적으로 벌어지는(Gap-out) 현상이 발생합니다. 뚜렷하고
  완벽한 Crossing을 보려면 $N_x$ 를 적어도 $30 \sim 50$ 이상으로 충분히
  크게 설정해야 합니다.

- **Numerical Instability (Log 발산 억제):** $\xi_m$ 이 $0$ 또는 $1$ 에
  극도로 가까울 경우 (시스템의 Deeply trivial states), 로그 함수 내의
  항이 0으로 나누어지거나 0이 되어 파이썬 수치 계산에서 발산(NaN, Inf)
  에러가 발생합니다. 실제 코드 구현 시에는
  `numpy.clip(xi, 1e-12, 1-1e-12)` 와 같이 값을 제한해주는 안전장치가
  필수적으로 들어가야 합니다.

- **Unit Cell 온전성 (Integrity of cut):** $C$ 행렬을 Sub-block으로 자를
  때, 다중 궤도(Multi-orbital) 기반의 Tight-binding 모델인 경우 절대로
  Unit cell 단위 내부를 쪼개서는 안 됩니다. 반드시 전체 Unit cell이
  Region A에 온전히 포함되도록 행렬의 Index 경계를 설정해야 하며, 특정
  Unit cell의 궤도 절반만 포함시킬 경우 인위적인 결함(Defect)이 발생하여
  스펙트럼 전체가 크게 오염됩니다.

## 4. Symmetry Indicators (Fu-Kane Formula)

**목적:** Inversion Symmetry ($P$)와 Time-Reversal Symmetry ($T$)가
공존하는 시스템에서 초고속 $Z_2$ Invariant 계산

**물리적 배경 및 장점:**

일반적인 시스템에서 $Z_2$ Invariant를 구하려면 Brillouin Zone (BZ)
절반에 걸쳐 Wilson Loop을 적분하거나 파동함수의 Gauge를 추적해야 합니다.
하지만 시스템에 공간 반전 대칭(Inversion Symmetry, $P$)이 존재할 경우,
Fu와 Kane은 이러한 복잡한 전역적 적분 없이 오직 Time-Reversal Invariant
Momenta (TRIM)에서의 파동함수 Parity eigenvalue만 확인하면 $Z_2$
Invariant를 완벽하게 결정할 수 있음을 증명했습니다.

이 방법은 전체 BZ 공간의 $\mathcal{O}(N_x N_y)$ 개의 $k$-point를
계산하는 대신 2D 기준 단 4개의 $k$-point 연산만 수행하므로 수치적 계산
비용이 $\mathcal{O}(1)$ 로 극단적으로 줄어듭니다. DFT(Density Functional
Theory)와 같은 무거운 제일원리 계산이나 복잡한 Tight-binding 모델에서
가장 강력하고 빠른 Topology 표준 검증 수단입니다.

### 4.1. 수학적 원리 및 알고리즘

1.  **TRIM Points 식별:** 2D BZ 내에서 Time-Reversal Symmetry 하에 자기
    자신으로 매핑되는 불변 모멘텀(TRIM) 4곳을 지정합니다. (3D의 경우
    8곳입니다.)

    $$\Gamma_1=(0,0), \quad \Gamma_2=(\pi,0), \quad \Gamma_3=(0,\pi), \quad \Gamma_4=(\pi,\pi)$$

2.  **Parity Operator (**$P$**) 정의:** 모델의 Basis(예: Sublattice,
    Orbital, Spin)에 작용하는 Inversion symmetry operator 행렬을
    구축합니다. 이 연산자는 TRIM point에서 Hamiltonian과
    교환(Commute)해야 합니다. 즉, $[ \mathcal{H}(\Gamma_i), P ] = 0$ 이
    성립합니다.

3.  **Simultaneous Diagonalization:** 각 $\Gamma_i$ 점에서
    $\mathcal{H}(\Gamma_i)$ 를 대각화하여 Occupied state
    $|u_m(\Gamma_i)\rangle$ 들을 찾습니다. Hamiltonian과 Parity 연산자가
    교환되므로, 이 상태들은 $P$ 의 Eigenstate이기도 하며 Parity
    eigenvalue $\xi_m(\Gamma_i) = \pm 1$ 을 가집니다.

4.  **Kramers Pair 처리 (핵심):** Time-Reversal Symmetry ($T$)가 있는
    Spin-1/2 시스템에서는 모든 에너지 레벨이 Kramers degeneracy로 인해
    두 배로 겹쳐 있습니다. 이때 상태 $|u\rangle$ 와 그 파트너인
    $T|u\rangle$ 는 **반드시 동일한 Parity**를 가집니다. 따라서 파생되는
    전체 $N_{occ}$ 개의 밴드 중, **한 쌍의 Kramers pair 당 오직 하나의
    State만 선택**하여 파생된 Parity $\xi_m$ 을 곱해야 합니다. (전부 다
    곱하면 항상 1이 되어버립니다.)

5.  $Z_2$ **Invariant (**$\nu$**) 계산:**

    각 TRIM 점에서의 Parity 곱을
    $\delta_i = \prod_{m=1}^{N_{pair}} \xi_m(\Gamma_i)$ 라 할 때, 전체
    시스템의 $Z_2$ Invariant는 다음과 같이 주어집니다.

    $$(-1)^\nu = \prod_{i=1}^{4} \delta_i = \prod_{i=1}^{4} \prod_{m=1}^{N_{pair}} \xi_m(\Gamma_i)$$

    결과값이 $-1$ 이면 $\nu=1$ (Topological, Non-trivial) 이고, $+1$
    이면 $\nu=0$ (Trivial) 을 의미합니다.

### 4.2. 구현 시 주의사항 (Technical Detail)

- **수치적 겹침 문제와 Sub-diagonalization (가장 흔한 에러 요인):** TRIM
  point에서는 에너지 레벨이 정확히 일치(Degenerate)합니다. 파이썬의
  `numpy.linalg.eigh` 같은 일반적인 수치 대각화 라이브러리는 겹쳐있는
  상태들($|u_{m,1}\rangle$, $|u_{m,2}\rangle$)이 무작위로 섞인 임의의
  선형 결합(Linear combination)을 Eigenvector로 반환할 가능성이
  높습니다.

  이 섞인 상태에 곧바로 $P$ 연산자를 작용시키면 기댓값
  $\langle u | P | u \rangle$ 이 $+1$ 이나 $-1$ 이 아닌 $0.34$, $-0.8$
  같은 실수값을 뱉어내어 알고리즘이 붕괴됩니다.

  **해결책:** 겹쳐있는 각 에너지 Subspace (예: 2개의 eigenvector 배열
  $V = [v_1, v_2]$)를 따로 떼어내어, 그 안에서 Parity 행렬
  $P_{sub} = V^\dagger P V$ 를 구성해야 합니다. 이 $2 \times 2$ 행렬을
  다시 대각화(Sub-diagonalization)하면, 완벽하게 $+1$ 과 $-1$ 의
  Eigenvalue를 가지는 순수한 Parity Eigenstate 기저(Basis)를 분리해 낼
  수 있습니다.

- **Symmetry Breaking Perturbation의 한계:** 계산 효율성은 압도적이지만,
  시스템에 주입된 무질서(Disorder)나 특정 Perturbation $\delta H$ 가
  Inversion Symmetry $P$ 를 조금이라도 깨뜨린다면 Fu-Kane 공식을 더 이상
  신뢰할 수 없습니다. 따라서 계산 코드를 구성할 때 맨 처음
  $[ \mathcal{H}(\Gamma_i), P ] = 0$ 조건을 `np.allclose()` 등을 통해
  수치적으로 검사하는 Assertion 단계를 반드시 넣어두고, 대칭성이 깨진
  경우에는 2번의 Wilson Loop 모듈로 자동으로 Fallback 하도록 아키텍처를
  짜야 합니다.

## 💡 개발 아키텍처 제안 (Module Structure)

기존 시스템에 통합하기 위해 다음과 같은 패키지 구조를 제안합니다.

    cls_calculator/
    │
    ├── core/
    │   ├── hamiltonian.py     # k-dependent Hamiltonian 생성기
    │   └── perturbation.py    # \delta H (Symmetry-breaking term 등) 주입 모듈
    │
    ├── topology/              # <--- 신규 개발 모듈
    │   ├── fhs_chern.py       # FHS method (2D Grid, Determinant 연산)
    │   ├── wilson_loop.py     # WCC flow 계산 및 Sorting 알고리즘
    │   ├── entangle_spec.py   # Correlation matrix 및 ES 도출
    │   └── fu_kane.py         # TRIM point 추출 및 Parity 계산
    │
    └── visualization/
        ├── plot_wcc.py        # Wilson loop flow 렌더링 (Plotly)
        └── plot_es.py         # Entanglement spectrum 렌더링
