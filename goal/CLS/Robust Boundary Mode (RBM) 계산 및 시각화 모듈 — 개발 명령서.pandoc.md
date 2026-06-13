> **이론적 근거:** Jun-Won Rhim & Bohm-Jung Yang, *"Classification of
> flat bands according to the band-crossing singularity of Bloch wave
> functions"*
>
> **모듈의 목적:** Periodic Boundary Condition (Torus)에서 파괴적
> 간섭(Destructive interference)으로 인해 정확히 소멸($0$)되는 CLS의
> 합이, **Open Boundary Condition (OBC)** 하에서는 경계면에서 완벽히
> 상쇄되지 않고 **Robust Boundary Mode**로 발현되는 현상을 수치적으로
> 계산하고 실공간(Real space)에 시각화한다.

## 1. 이론적 배경 및 알고리즘 원리

Singular Flat Band의 가장 큰 특징은 completeness의 붕괴이며, 이는 Bulk
시스템(Torus)에서 다음의 항등식으로 나타난다.

$$\sum_{R} c_R |\chi_R\rangle = 0$$

여기서 $|\chi_R\rangle$는 단위 셀 $R$에 중심을 둔 CLS이다.

그러나 유한한 크기의 **Open Boundary System**에서는 경계 밖으로
뻗어나가야 할 CLS 사본들이 존재하지 않으므로, 시스템 내부의 CLS들만
합산할 경우 경계면 부근에서 간섭 상쇄가 깨지게 된다. 즉, 경계 모드
$|\psi_\partial\rangle$는 다음과 같이 구성된다.

$$|\psi_\partial\rangle = \sum_{R \in \text{System}} c_R |\chi_R\rangle \neq 0$$

이 합산의 결과로 시스템의 내부(Deep bulk) 진폭은 정확히 $0$이 되고,
가장자리(Edge)에만 CLS의 skin depth와 동일한 두께로 유한한 진폭이 남게
된다.

## 2. 모듈 입출력 명세 (I/O Specification)

### 2.1 입력 (Input)

- **`Lattice` 객체:** 차원(d), 기본 병진 벡터($a_l$), 단위 셀 내 오비탈
  좌표 ($Q$개).

- **`CLS_Amplitude` 텐서 (**$A_{0, R, q}$**):** `CLS_Finder`를 통해
  획득한 단일 최소 단위 CLS의 실공간 진폭 (계수 딕셔너리 형태).

- **`System_Size` (**$N_x, N_y$**):** Open boundary를 구성할 유한 격자의
  크기 (예: $20 \times 20$ unit cells).

- **`Singularity_Type`:** 대상 Flat Band가 Singular인지
  Nonsingular인지의 여부 (Singular일 경우에만 Robust Boundary Mode가
  보장됨).

### 2.2 출력 (Output)

- **`Psi_Edge` 벡터:** OBC 상에서 모든 시스템 내 CLS를 합산한 결과 벡터
  (실공간 각 사이트에서의 진폭).

- **Bulk Cancellation 검증 결과 (Boolean):** 시스템 중심부(Bulk)에서
  진폭 합이 정확히 $0$으로 수렴하는지에 대한 검증 값 (오차 허용 범위
  `1e-10`).

- **시각화 (Plots):**

  1.  유한 격자 위에서의 $|\psi_\partial\rangle$ 진폭 분포 맵 (2D/3D
      Scatter Plot).

  2.  Edge로부터의 거리에 따른 진폭 감쇠 그래프 (Skin depth 확인용).

## 3. 핵심 수치 연산 알고리즘 (Step-by-Step)

### Step 1: Open Boundary 격자 생성 (Grid Initialization)

사용자가 지정한 `System_Size` $N_x \times N_y$ 에 해당하는 유한 격자점
$R = (n_x a_1, n_y a_2)$ ($0 \le n_x < N_x$, $0 \le n_y < N_y$)를
생성한다.

각 단위 셀마다 $Q$개의 오비탈이 존재하므로, 전체 상태 공간의 차원은
$N_x \times N_y \times Q$가 된다. 이를 담을 전역 상태 벡터
`Global_Psi`를 영벡터($0$)로 초기화한다.

### Step 2: CLS 계수 추출 및 페이즈(Phase) 설정

입력받은 단일 `CLS_Amplitude` 텐서에서 각 오비탈 $q$ 및 상대 좌표
$\Delta R$에 대한 진폭 $A_{0, \Delta R, q}$를 읽어온다.

이때, 각 단위 셀로 번역(Translation)할 때 필요한 위상 계수 $c_R$을
설정한다.

- *Note:* 대부분의 Singular flat band (예: Kagome, Lieb)에서는 중심점
  $k_0 = (0,0)$ 특이성을 가지므로 $c_R = 1$이다. 특이점이
  $k_0 = (\pi, \pi)$ 등일 경우 $c_R = e^{-i k_0 \cdot R}$ 형태로
  페이즈를 조정해야 한다.

### Step 3: 병진 및 합산 (Translation & Accumulation)

    def calculate_boundary_mode(lattice, cls_amplitude, system_size, k_singularity):
        Nx, Ny = system_size
        global_psi = initialize_zero_vector(Nx, Ny, lattice.Q)
        
        # 시스템 내부의 모든 단위 셀 R에 대해 Loop
        for nx in range(Nx):
            for ny in range(Ny):
                R = (nx, ny)
                phase_factor = exp(-1j * dot(k_singularity, R))
                
                # 단일 CLS의 진폭을 현재 R을 중심으로 뿌려줌
                for dR, q, amp in cls_amplitude:
                    target_R = (nx + dR[0], ny + dR[1])
                    
                    # 타겟 위치가 Open Boundary 시스템 '내부'에 있을 때만 더함
                    if 0 <= target_R[0] < Nx and 0 <= target_R[1] < Ny:
                        global_psi[target_R][q] += phase_factor * amp
                        
        return global_psi

### Step 4: Bulk Cancellation 검증 (Validation)

합산이 완료된 `global_psi` 텐서에서, 경계로부터 CLS의 반경(support size)
이상 떨어진 내부(Deep bulk) 영역을 슬라이싱한다.

해당 영역 내의 모든 사이트 진폭의 절댓값 합이 수치적 허용
오차(`tol = 1e-10`) 이하인지 검증한다.

$$ \sum_{R \in \text{Deep Bulk}} |\psi_\partial(R)| \approx 0 $$

이 검증을 통과해야만 파괴적 간섭이 올바르게 일어났음을 확증할 수 있다.

## 4. 분석 및 시각화 (Visualization Pipeline)

### 4.1 2D Real-space Amplitude Plot

- **배경 격자:** $N_x \times N_y$ 단위 셀과 사이트 위치를 연한 회색 점과
  선으로 플롯팅한다.

- **진폭 오버레이:** `global_psi` 값이 $0$이 아닌 사이트 위에 마커를
  표시한다.

  - `Marker Size` $\propto |\psi_\partial(R)|$

  - `Marker Color`: 위상(부호)에 따라 Red/Blue 계열 (Complex의 경우
    Phase angle에 따른 colormap 적용).

- **기대 결과:** 시각화 결과물은 시스템의 가장자리(테두리)를 따라서만
  마커가 존재하고 중앙은 완전히 비어있는(Void) 직사각형(혹은 지정된
  형태) 띠 형태가 되어야 한다.

### 4.2 Skin Depth 분석 프로파일

경계면(예: $n_x = 0$ 면)으로부터 수직 방향($n_x$ 축)으로 거리가 멀어짐에
따라 진폭의 절댓값 $\max_{n_y, q} |\psi_\partial(n_x, n_y, q)|$ 이
어떻게 변하는지 1D 그래프로 그린다.

- Robust Boundary Mode의 특성상, 진폭은 Exponential Decay가 아닌, CLS의
  최대 반경 내에서만 유한한 값을 가지고 그 이후에는 정확히 Step-function
  형태로 $0$으로 떨어져야 한다.

## 5. 실행 시나리오 (Test Cases)

이 모듈의 강건성을 테스트하기 위해 에이전트는 다음 두 가지
대조군(Control group)을 반드시 실행해야 한다.

1.  **Kagome Lattice (Singular):**

    - 입력: Kagome 평탄 밴드의 최소 CLS (육각형 고리 형태).

    - 기대 결과: Bulk 진폭은 완벽히 $0$이 되며, 경계를 따라 닫힌 띠
      형태의 Robust Boundary Mode가 선명하게 형성됨을 확인해야 함.
      임의로 경계의 CLS 1개를 빼거나 더하더라도 해당 부분의 국소적
      모양만 바뀔 뿐, 띠 전체가 붕괴되지 않음(Robustness).

2.  **Bilayer Square Lattice (Nonsingular):**

    - 입력: 이층 사각 격자의 수직 Dimer 형태 CLS.

    - 기대 결과: 경계를 따라 모드를 억지로 만들 수는 있으나(Fragile
      boundary mode), 본 알고리즘처럼 단순히 $\sum |\chi_R\rangle$를
      수행할 경우 특별한 위상학적 보호가 없어 경계의 단일 결함(CLS 하나
      누락)만으로도 간섭 패턴이 완전히 끊어짐을 확인.
