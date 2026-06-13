Phase 1\~4를 통해 목표로 하는 Chern number를 가지며 rank-1 조건이 부여된
unnormalized CLS vector $f(\mathbf{k})$를 도출했습니다.

기존 알고리즘의 식
$H(\mathbf{k}) = E_0 P(\mathbf{k}) + [I - P(\mathbf{k})] M(\mathbf{k}) [I - P(\mathbf{k})]$는
수학적으로는 완벽하지만,
$P(\mathbf{k}) = \frac{f(\mathbf{k})f^\dagger(\mathbf{k})}{|f(\mathbf{k})|^2}$
에 포함된 분모 $|f(\mathbf{k})|^2$ 로 인해 역푸리에 변환 시 **무한한
범위의 Hopping parameter(Infinite range hoppings)** 가 발생하여 실제
Lattice model로 구현하기 어렵습니다.

따라서 Flat band의 완전한 Flatness를 유지하면서, Symmetry를 존중하고,
$k_i$에서의 Band touching을 자연스럽게 유도하는 **Finite Hopping TBM
(Tight-Binding Model)** 설계 방법론으로 Phase 5를 구체화합니다.

## 1. Denominator Clearing: Polynomial Hamiltonian 구성

무한 Hopping을 피하기 위해, 정규화된 Projector $P(\mathbf{k})$ 대신
**Unnormalized Projector** $\mathcal{P}(\mathbf{k})$ 를 정의합니다.

$$\mathcal{P}(\mathbf{k}) = f(\mathbf{k})f^\dagger(\mathbf{k})$$

이 행렬의 Trace는
$\text{Tr}(\mathcal{P}(\mathbf{k})) = |f(\mathbf{k})|^2$ 입니다. 이를
이용하여 분모가 없는(즉, 오직 Finite Fourier sum으로만 이루어진) Core
Hamiltonian을 다음과 같이 정의합니다.

$$H_{core}(\mathbf{k}) = |f(\mathbf{k})|^2 I_{N\times N} - f(\mathbf{k})f^\dagger(\mathbf{k})$$

**물리적 의미:**

- $H_{core}(\mathbf{k})$ 에 $f(\mathbf{k})$ 를 곱하면
  $H_{core}(\mathbf{k})f(\mathbf{k}) = (|f|^2 - |f|^2)f(\mathbf{k}) = 0$
  이 됩니다. 즉, $f(\mathbf{k})$ 는 에너지 $E=0$ 에 완벽하게 고정된 Flat
  band의 Eigenvector가 됩니다.

- $|f(\mathbf{k})|^2$ 와 $f(\mathbf{k})f^\dagger(\mathbf{k})$ 모두
  유한한 위상(Phase)들의 합이므로, $H_{core}(\mathbf{k})$ 는 철저히
  **다항식 형태(Polynomial matrix)** 가 되며, 역푸리에 변환 시
  **Strictly finite range hopping** 만을 생성합니다.

## 2. Dispersive Band 제어 및 Hamiltonian 합성

$H_{core}$ 만으로는 분산 밴드(Dispersive band)들이 단순한 형태를
가지거나 축퇴(Degenerate)될 수 있습니다. Flat band를 건드리지 않으면서
Dispersive band를 조형하기 위해, Symmetry를 만족하는 임의의 Finite
Fourier Hermitian matrix $\tilde{M}(\mathbf{k})$ 를 도입합니다.

최종 Hamiltonian $H(\mathbf{k})$ 는 다음과 같이 합성됩니다.

$$H(\mathbf{k}) = E_0 I + \alpha H_{core}(\mathbf{k}) + H_{core}(\mathbf{k}) \tilde{M}(\mathbf{k}) H_{core}(\mathbf{k})$$

- $E_0$: Target Flat band의 기준 Energy level.

- $\alpha$: Dispersive band의 기본 Bandwidth를 조절하는 Scaling factor.

- $\tilde{M}(\mathbf{k})$: Band shape과 Symmetry를 튜닝하는 제어 행렬.

이 구조는 여전히 $H(\mathbf{k})f(\mathbf{k}) = E_0 f(\mathbf{k})$ 를
완벽하게 만족시키며, 모든 항이 곱셈과 덧셈으로만 이루어져 있어 **Finite
hopping TBM**이 완벽히 보장됩니다.

## 3. Singularity $k_i$ 에서의 Band Touching 보장

Topological Flat band가 비자명한 Chern number를 갖기 위해서는 반드시
Dispersive band와의 Band touching point (Singularity)를 통해 Topological
charge를 교환해야 합니다. 위에서 구성한 $H(\mathbf{k})$ 는 이 성질을
수식적으로 매우 아름답게 보장합니다.

Phase 2에서 강제한 **Common zero 조건**에 의해, Singularity point $k_i$
에서는 다음이 성립합니다.

$$f(k_i) = 0 \quad \implies \quad |f(k_i)|^2 = 0 \quad \text{and} \quad f(k_i)f^\dagger(k_i) = 0$$

따라서 $H_{core}(k_i) = 0$ 이 되며, 최종 Hamiltonian에 대입하면 다음과
같은 결과가 나옵니다.

$$H(k_i) = E_0 I + 0 + 0 = E_0 I_{N\times N}$$

**결론:** Singularity $k_i$ 에서는 Hamiltonian이 단위 행렬에 비례하게
됩니다. 이는 **해당** $k_i$ **포인트에서 Flat band와 모든 Dispersive
band들이 에너지** $E_0$ **에서 완벽하게 만난다(Band touching / Exact
degeneracy)** 는 것을 의미합니다. $\tilde{M}(\mathbf{k})$ 를 어떻게
조작하든 이 Touching은 절대로 깨지지 않으며, 여기서 Phase 3에서 설계한
Winding에 따라 Chern number가 발생합니다.

## 4. Symmetry Constraint를 고려한 $\tilde{M}(\mathbf{k})$ 설계

설계하는 Lattice (Kagome, Honeycomb, Lieb 등)의 Point group symmetry와
Time-reversal symmetry (필요시 파기) 등을 고려하여
$\tilde{M}(\mathbf{k})$ 를 선택해야 합니다.

1.  **Lattice 및 Orbital Symmetry 파악:** 격자의 공간군(Space group)
    생성자 $S$ 에 대해 $S H(\mathbf{k}) S^{-1} = H(R\mathbf{k})$ 를
    만족해야 합니다. 이때 각 Site에 위치한 **궤도(Orbital)의 대칭성**
    역시 행렬 $S$ 에 반영되어야 합니다.

2.  **Complex Hopping Parameter의 도입 (중요):** $\tilde{M}(\mathbf{k})$
    를 구성하는 Hopping parameter들은 단순한 실수가 아니라
    **복소수(Complex numbers)** 의 조합으로 나타날 수 있음을 적극적으로
    고려해야 합니다.

    - **Orbital 특성 반영:** 예를 들어 $p_x, p_y$ 궤도의 결합이나
      스핀-궤도 결합(Spin-Orbit Coupling)이 포함된 경우, 대칭성 연산은
      Hopping의 진폭(Amplitude)뿐만 아니라 **복소 위상(Phase,**
      $t = |t|e^{i\phi}$**)** 에도 엄격한 조건을 부여합니다.

    - **Chern Band와 TRS 파괴:** 비자명한 Chern band를 생성하기 위해서는
      시간반전대칭(TRS)이 깨져야 합니다(예: Haldane model의
      Next-nearest-neighbor imaginary hopping). 따라서 특정 Symmetry
      조건에 의해 Hopping parameter 앞에는 순허수($\pm i$)나 특정 각도의
      복소 계수들이 필연적으로 도입됩니다.

3.  $\tilde{M}(\mathbf{k})$ **의 Basis 전개:** $H_{core}(\mathbf{k})$ 가
    이미 원래 격자의 대칭성을 내포하고 있으므로, $\tilde{M}(\mathbf{k})$
    는 위에서 분석한 복소수 Hopping 조건들을 만족하는 가장 짧은 거리의
    Hopping (예: Nearest-neighbor 및 Next-nearest-neighbor matrix)
    기저들의 선형 결합으로 초기값을 설정합니다.

4.  **Band gap tuning:** $\tilde{M}(\mathbf{k})$ 의 복소
    파라미터(Complex hopping amplitudes)들을 조절하여, Singularity $k_i$
    를 제외한 나머지 BZ 영역에서 Flat band와 Dispersive band 사이에
    명확한 Band gap이 열리도록 최적화합니다.

## 요약: Phase 5 실무 워크플로우

1.  $f(\mathbf{k})$ 를 이용하여 다항식 행렬
    $H_{core}(\mathbf{k}) = |f(\mathbf{k})|^2 I - f(\mathbf{k})f^\dagger(\mathbf{k})$
    를 계산한다.

2.  격자 및 Orbital 대칭성을 분석하고, 이를 만족하는 복소수
    Hopping(Complex hopping) 기반의 짧은 거리 Hermitian 행렬
    $\tilde{M}(\mathbf{k})$ 를 정의한다.

3.  $H(\mathbf{k}) = E_0 I + \alpha H_{core}(\mathbf{k}) + H_{core}(\mathbf{k}) \tilde{M}(\mathbf{k}) H_{core}(\mathbf{k})$
    를 조립한다.

4.  $H(\mathbf{k})$ 를 행렬 성분별로 역푸리에 변환(Inverse Fourier
    Transform)하여, 실제 Lattice의 각 Site간 Hopping parameter
    $t_{\alpha\beta}(\Delta R)$ (복소수 값 포함)를 추출한다.

5.  $k_i$ 에서의 Band touching을 확인하고, 나머지 영역에서 Gap이
    확보되는지 Band structure를 그려 검증한다.
