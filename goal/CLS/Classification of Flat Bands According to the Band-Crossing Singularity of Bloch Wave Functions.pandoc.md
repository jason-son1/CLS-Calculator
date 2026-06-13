본 문서는 Jun-Won Rhim 및 Bohm-Jung Yang의 연구 논문을 기반으로,
`flat band` 시스템을 Bloch wave function의 momentum space 내
특이성(singularity)에 따라 분류하는 이론적 체계와 이를 응용한 수치 계산
알고리즘 및 격자 모델링 프레임워크를 설명합니다.

## 제1장: 서론 및 연구 배경 (Introduction & Background)

### 1.1 Flat Band와 Strong Correlation

`flat band`는 Brillouin zone 전체에서 에너지 분산(dispersion)이 완전히
사라진 극단적인 밴드 구조를 의미합니다. 이 상태에서는 전자의 운동
에너지(kinetic energy)가 완전히 quenching 되기 때문에, 미세한 전자 간
상호작용(Coulomb interaction)에 의해서도 시스템의 거동이 지배됩니다.
이는 다음과 같은 고차원 물리 현상의 이상적인 배경이 됩니다.

- Wigner crystallization

- Fractional Chern Insulator (FCI)

- High-temperature superconductivity

### 1.2 기존 실공간 관점의 한계와 새로운 Momentum Space 접근법

기존 연구들은 주로 실공간(real space)의 기하학적 frustrated lattice(예:
Kagome, Lieb) 구조가 유도하는 파괴적 간섭(destructive interference)에만
집중하여 `flat band`를 이해하려 했습니다. 그러나 이러한 방식은 격자
구조에 종속적이며, 차원이나 대칭성에 독립적인 보편적(universal)인 특성을
규명하기 어렵습니다.

본 논문은 최초로 Bloch wave function의 momentum space
특이성(singularity)을 기준으로 `flat band`를 두 가지 근본적인 부류로
분류하는 보편적 프레임워크를 제시합니다.

## 제2장: 분류학적 체계 (Classification Framework)

Bloch wave function의 위상학적 성질에 따라 `flat band`는 다음과 같이 두
가지 클래스로 엄격하게 분류됩니다.

  ----------------------- ------------------------------------- -----------------------
  **분류 기준             **Singular Flat Band**                **Nonsingular Flat
  (Criteria)**                                                  Band**

  **Bloch Wave Function   $k$-space 내에 제거 불가능한          $k$-space 전체에서
  특성**                  불연속점(`immovable discontinuity`)   불연속점 없이 완벽하게
                          존재                                  연속적임

  **Vector Bundle 정의    밴드 교차점(`band touching`)에서의    Well-defined vector
  여부**                  특이성으로 인해 정의 불가             bundle 형성 가능

  **Complete Set of       $N$개의 translated CLS만으로는 전체   $N$개의 translated
  CLSs**                  밴드를 span할 수 없음                 CLS만으로 전체 밴드를
                                                                완전히 span 가능

  **Band Touching 성질**  타 dispersive band와 반드시 교차하며, 완전히
                          교차 해제 시 분산 발생                격리(isolated)되거나,
                                                                교차하더라도 flatness
                                                                유지하며 gap 생성 가능

  **1D Space 존재 여부**  1차원 공간에서는 물리적으로 존재      모든 1차원
                          불가능                                `flat band`는
                                                                nonsingular 클래스에
                                                                속함
  ----------------------- ------------------------------------- -----------------------

### 2.1 Immovable Discontinuity와 Singularity

- **Movable Singularity (Chern Band):** 일반적인 Chern band의 경우에도
  wave function에 특이점이 존재할 수 있으나, 이는 국소적인 gauge
  choice를 통해 다른 $k$ 지점으로 전이(shift)시킬 수 있습니다. 따라서
  여러 개의 patch를 엮어 전 공간에서 analytic한 vector bundle을 정의할
  수 있습니다.

- **Immovable Singularity (Singular Flat Band):** `singular flat band`가
  가지는 불연속점은 어떠한 국소적 gauge choice로도 제거하거나 위치를
  이동시킬 수 없습니다. 이 특이점 $k_0$는 다른 dispersive 밴드와의
  `band touching`에 기인하며, $k_0$로 접근하는 경로에 따라 Bloch wave
  function의 극한값이 달라지는 다가성(multi-valuedness)을 보입니다.

## 제3장: CLS의 불완전성과 비국소적 상태 (Incompleteness of CLS)

### 3.1 Compact Localized State (CLS)의 수학적 기초

격자 내 유닛 셀의 개수를 $N$, 각 유닛 셀 내의 오비탈 개수를 $Q$라 할 때,
임의의 `flat band`를 기술하는 Bloch wave function은 다음과 같이
표현됩니다.

$$|\psi_{k}\rangle = \frac{1}{\sqrt{N}}\sum_{R}\sum_{q=1}^{Q} e^{i k \cdot R} v_{k,q} |R,q\rangle$$

여기서 $v_{k,q}$는 정규화된 eigenvector $v_k$의 $q$번째 성분입니다.
이들의 선형 결합을 통해 실공간의 한정된 영역에서만 진폭을 가지는 CLS인
$|\chi_R\rangle$을 다음과 같이 정의합니다.

$$|\chi_{R}\rangle = c_{\chi}\sum_{k\in\text{BZ}}\alpha_{k}e^{-i k\cdot R}|\psi_{k}\rangle = \sum_{R'}\sum_{q=1}^{Q} A_{R,R',q}|R',q\rangle$$

이때 실공간 상에서의 완전한 국소화(compact localization)가 이루어지려면
역푸리에 변환의 성질에 의해 각 성분 $\alpha_k v_{k,q}$가 반드시 **Finite
Sum of the Bloch Phases (FSBP)** 형태를 만족해야 합니다.

$$\alpha_k v_{k,q} = \sum_{m_1,\dots,m_d} f_{m_1,\dots,m_d}^{(q)} \exp\left(i \sum_{l=1}^{d} m_l k_l \cdot a_l\right)$$

### 3.2 완전성 조건식 (Completeness Condition)

$N$개의 격자점 주위로 번역 이동된(translated) $N$개의 CLS 세트
${|\chi_{R_l}\rangle}$가 전체 `flat band`를 완벽히 span할 수 있는지
여부는 다음 행렬식(determinant) $D$의 거동으로 결정됩니다.

$$D = \det\left[ \langle \psi_{k_j} | \chi_{R_l} \rangle \right] \propto \prod_{i=1}^{N} \alpha_{k_i}$$

- $\alpha_k \neq 0$ **(Nonsingular):** Brillouin zone의 모든 $k$
  영역에서 $\alpha_k$가 0이 아닌 값을 가질 수 있다면 $D \neq 0$이므로
  CLS 세트는 선형 독립이며 완전성(completeness)을 가집니다.

- $\alpha_{k_0} = 0$ **(Singular):** 임의의 특이점 $k_0$에서 어떠한
  형태의 $\alpha_k$도 반드시 0이 되어야 한다면 $D = 0$이 되며,
  translated CLS 세트 사이에 선형 종속성(linear dependency)이
  발생합니다.

### 3.3 Non-contractible States의 필요성

`singular flat band`에서는 결손된 기저 상태를 보충하기 위해 공간적으로
완전히 국소화되지 않은 확장 상태(extended states)가 도입되어야 합니다.

- **Non-contractible Loop State (NLS - 2D):** 2차원 토러스(torus)
  기하학에서 한 방향(예: $a_1$)으로는 완전히 국소화되어 있으나, 다른
  방향($a_2$)으로는 무한히 뻗어 나가는 고리 형태의 상태입니다. 실공간
  상에서 위상학적으로 deforming을 통해 CLS로 축소시킬 수 없습니다.

- **Non-contractible Planar State (NPS - 3D):** 3차원 공간에서 두
  방향으로 무한히 확장되고 한 방향으로만 국소화된 평면 형태의
  상태입니다.

## 제4장: 대역 접촉 성질 및 위상 기하학적 변형 (Band Touching & Deformation)

`flat band`와 이웃한 dispersive 밴드 사이의 `quadratic band touching`
부근에서의 유효 해밀토니안(effective Hamiltonian) 해석을 통해 물리적
차이점을 극명히 구분할 수 있습니다.

### 4.1 Nonsingular Band Touching

Nonsingular 교차의 경우, 유효 해밀토니안은 단 하나의 Pauli matrix
성분만으로 단순화될 수 있습니다.

$$\mathcal{H}_k = (t_1' k_x^2 + t_2' k_x k_y + t_3' k_y^2)(\sigma_z + \sigma_0)$$

이 경우, $k$-space 상에서 다음과 같은 성질이 나타납니다.

- 적절한 외부 섭동(perturbation) $\mathcal{H}' = m\sigma_z$를 가했을 때,
  **밴드의 완벽한 평탄성(flatness)을 그대로 보존하면서** 에너지를
  이동시켜 gap을 열 수 있습니다.

- Gap이 열린 후에도 Berry curvature와 Berry connection이 모든 $k$
  영역에서 0이 되므로, 위상적으로 trivial하며 `Chern number`는 항상
  0입니다.

### 4.2 Singular Band Touching

반면, `singular band touching` 모델은 최소 두 개 이상의 Pauli matrix
성분을 포함해야 기술이 가능합니다.

$$\mathcal{H}_k = (t_1 k_x^2 + t_2 k_x k_y + t_3 k_y^2)\sigma_z + (t_4 k_x k_y + t_5 k_y^2)\sigma_y + t_6 k_y^2 \sigma_x + (b_1 k_x^2 + b_2 k_x k_y + b_3 k_y^2)\sigma_0$$

- **Warping Phenomenon:** Flatness 조건($\det \mathcal{H}_k = 0$) 하에서
  밴드 갭을 열어주는 어떠한 질량 항(mass term)도 밴드의 평탄성을
  파괴합니다. 즉, 외부 섭동이 가해지면 평탄했던 밴드가 무조건 **분산성을
  가지는 dispersive band로 warping**됩니다.

- **Chern Number Generation:** 특정 mass term 섭동(예: $m\sigma_x$)을
  적용해 갭을 유도하면, 기하학적으로 휘어진 nearly flat band가 형성되며
  유효한 위상 성질을 획득합니다. 예를 들어 다음 해밀토니안 시스템에서:

  $$\mathcal{H}_k = \frac{k_x^2 - k_y^2}{2}\sigma_z + k_x k_y \sigma_y + \frac{k_x^2 + k_y^2}{2}\sigma_0 + m\sigma_x$$

  이 계의 gappedNearly flat band의 `Chern number` $\nu$는 다음과 같이
  부호 함수에 비례하는 위상 위상을 가집니다.

  $$\nu = -\text{sgn}(m)$$

## 제5장: Bulk-Boundary Correspondence (대역-경계 대응성)

`singular flat band`가 가지는 momentum space 특이성은 개방 경계(open
boundary condition) 시스템 하에서 고유한 경계 상태로 실공간 상에
구현됩니다.

           [Torus Geometry]                       [Open Boundary System]
      ==========================               ============================
      |  Non-contractible      |               |  Robust Boundary Mode    |
      |  Loop States (NLS)     |  =========>   |  (Confined within CLS    |
      |  (Extended along BZ)   |   (Cutting)   |   skin depth, same       |
      |                        |               |   energy as bulk flat)   |
      ==========================               ============================

### 5.1 Robust Boundary Mode의 이론적 유도

시스템의 토러스 경계를 자르는 과정은 translated CLS 간의 완전한 소멸
간섭 조건을 붕괴시킵니다. 토러스 상에서는 다음 관계식이 만족됩니다.

$$\sum_{R} c_R |\chi_R\rangle = 0$$

그러나 가장자리가 존재하는 planar geometry 시스템에서는 경계면 부근에서
이 합이 완벽히 상쇄되지 않고 남게 되며, 이는 다음과 같은 경계 고유
상태($|\psi_\partial\rangle$)를 유도합니다.

$$|\psi_\partial\rangle = \sum_{R \in \text{boundary}} c_R |\chi_R\rangle$$

- **고유한 물리적 특징:**

  1.  이 경계 상태의 에너지는 벌크 gap 내부(in-gap)에 위치하지 않고,
      **bulk flat band의 에너지와 정확히 동일한 레벨**에 위치합니다.

  2.  이 모드의 공간적 침투 깊이(skin depth)는 CLS 자체의 크기보다
      작거나 같습니다.

  3.  **Robustness:** `singular flat band`에서 생성된 경계 모드는 거시적
      개수의 CLS가 얽혀 형성되므로, 외부에서 유한한 개수의 CLS를
      국소적으로 추가하더라도 파괴적 간섭에 의해 끊어지지 않고
      유지됩니다.

### 5.2 Nonsingular Flat Band의 Fragile Boundary Mode

반면, Nonsingular 클래스에 속하는 Bilayer square lattice 등에서 생성된
가장자리 상태는 단순히 경계를 따라 CLS를 일렬로 적층한 상태에
불과합니다. 따라서 경계부에 위상이 반대인 단 몇 개의 CLS를 추가해주는
것만으로도 간섭을 통해 경계 모드를 쉽게 분리(disconnect) 및 소멸시킬 수
있는 **fragile**한 특성을 가집니다.

### 5.3 Frustration과의 독립성 증명

`singular flat band`와 그에 따른 NLS는 기하학적 좌절(frustration)이 없는
격자에서도 존재할 수 있습니다.

- **Lieb Lattice (Non-frustrated):** 기하학적으로 frustrated 격자가
  아님에도 불구하고 $k=(\pi, \pi)$에서 `singular band touching`을 가지며
  NLS가 필연적으로 나타납니다.

- **Modified Kagome Lattice (Frustrated):** 기하학적 frustrated 격자
  형태를 취하고 있음에도 불구하고 적절한 이웃 호핑 제어를 통해 완전히
  isolated된 `nonsingular flat band`를 디자인할 수 있으며, 이 경우 NLS는
  요구되지 않습니다.

## 제6장: 수치 계산 및 해밀토니안 자동 모델링 파이프라인

AI coding agent나 시뮬레이션 엔진이 시스템을 자율적으로 분석하고
설계하기 위한 구체적인 수식 알고리즘 가이드입니다.

### 6.1 임의의 해밀토니안에서 $\alpha_k$ 수치 추출 알고리즘

주어진 격자 모델 해밀토니안 $\mathcal{H}_k$로부터 특이성 판별자
$\alpha_k$를 추출하는 수치 연산 프로세스입니다.

    def extract_alpha_k(H_k, eigenvector_v_k, p_index=0):
        # H_k: Q x Q Symbolic or Numerical Matrix
        # v_k: 정규화된 Eigenvector
        # p_index: 파싱 기준으로 삼을 오비탈 인덱스
        
        # Step 1: H_k에서 p_index 행과 열을 제거하여 (Q-1) x (Q-1) 서브 매트릭스 H_sub 제작
        H_sub = remove_row_and_column(H_k, p_index, p_index)
        
        # Step 2: 서브 매트릭스의 determinant 계산 (FSBP 분자 성분)
        numerator = symbolic_determinant(H_sub)
        
        # Step 3: 기준 오비탈의 eigenvector 성분 분모 설정
        denominator = eigenvector_v_k[p_index]
        
        # Step 4: alpha_k 도출
        alpha_k = numerator / denominator
        return alpha_k

### 6.2 임의 차원 Target Flat Band 해밀토니안 역설계 알고리즘

사용자가 원하는 대칭성과 특이성(singular/nonsingular)을 주입하여 격자
Tight-binding 모델 해밀토니안을 빌드하는 역공학 수치 알고리즘입니다.

    [입력 데이터 정의]
    - 격자의 오비탈 수: Q
    - 목표 밴드 갭 조건 및 Singular/Nonsingular 여부 선택
    - 원하는 국소 상태의 특성을 반영한 unnormalized eigenvector 기저 설정:
      v_k^{(0)} = [w_{k,1}, w_{k,2}, ..., w_{k,Q}]^T (각 성분은 유연하게 정의된 FSBP 다항식)

    [프로세스 실행]
    1. 정규화 계수 역할을 할 밴드 특이성 인자 함수를 정의합니다.
       - Singular 선택 시: \alpha_k 가 특정 고정 대칭점 k*에서 0이 되도록 설정.
       - Nonsingular 선택 시: \alpha_k 가 BZ 전체에서 0이 되지 않는 상수 혹은 삼각함수 조합으로 설정.

    2. 초기 기저 벡터 u_k^{(q)} (1 <= q <= Q-1) 세트를 설정합니다. (주로 단위 직교 벡터군 사용)

    3. Gram-Schmidt orthonormalization 프로세스를 k-space 심볼릭 연산상에서 실행합니다.
       * v_k^{(0)}를 시작점으로 하여 타 기저들과 직교하는 정규직교 벡터 세트 v_k^{(q)}를 순차적 유도:
         v_k^{(q+1)} = u_k^{(q+1)} - \sum_{p=0}^{q} [ (v_k^{(p)})^\dagger \cdot u_k^{(q+1)} ] v_k^{(p)}
       * 연산 과정 중 분모 소거를 위해 공통 인자 \prod (\alpha_k^{(p)})^2 를 곱해주어 
         모든 기저 성분이 FSBP 수식을 완벽히 유지하도록 보정합니다.

    4. 분산 밴드들의 에너지 함수를 빌드합니다.
       E_k^{(q)} = F_k^{(q)} * (\alpha_k^{(q)})^2   (F_k^{(q)}는 임의의 주기적 FSBP 형 분산 함수)
       Flat band 자체의 에너지는 E_k^{(0)} = 0 으로 상쇄하여 배치합니다.

    5. 최종 역격자 해밀토니안 행렬을 조립합니다.
       \mathcal{H}_k|_{ij} = \sum_{q=1}^{Q-1} E_k^{(q)} v_{k,i}^{(q)} (v_{k,j}^{(q)})^*

    [출력 데이터]
    - 역푸리에 변환을 통해 실공간 호핑 파라미터(Hopping parameters) 세트를 도출하여 격자 모델 완성.

## 제7장: 결론 및 연구의 확장성 (Conclusions & Extensibility)

본 논문은 `flat band` 연구에 있어 실공간 구조 연구에만 편향되어 있던
시각을 완전히 뒤바꾸어, **Bloch wave function이 담고 있는 대역간 결합
특이성**의 근본적인 물리적 가치를 입증했습니다.

1.  `singular flat band`는 밴드 교차점이 풀릴 때 필연적으로 뒤틀리며
    **Nearly Flat Chern Band**로 변모하여, Fractional Quantum Hall
    Effect 등을 모사하기 위한 FCI 플랫폼 설계의 최적의 원천 기술이
    됩니다.

2.  토러스 상의 위상적 결손 상태가 개방 격자에서 **Robust Bulk-Boundary
    Correspondence**에 의한 고유 경계 모드로 발현됨으로써,
    광결정(photonic crystal) 및 탄성 메타물질(acoustic metamaterials)
    분야에서 손실 없는 웨이브가이드(waveguide) 설계 기법으로 확장될 수
    있습니다.

<!-- -->

    eof
