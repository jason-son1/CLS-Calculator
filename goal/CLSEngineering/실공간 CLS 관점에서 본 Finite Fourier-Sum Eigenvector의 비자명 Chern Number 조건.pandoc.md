*Real-Space CLS (Compact Localized State) Reformulation of the
Conditions for a Finite Fourier-Sum Eigenvector to Carry a Nonzero Chern
Number*

------------------------------------------------------------------------

## 0. 목적과 전략

업로드된 노트 *"Conditions for a Finite Fourier-Sum Eigenvector to Carry
a Nonzero Chern Number"* (June 3, 2026)는 Bloch eigenvector

$$ f(k)=\begin{pmatrix} f_1(k)\ \vdots \ f_N(k)\end{pmatrix},\qquad f_\alpha(k)=\sum_{R\in S_\alpha} c_{\alpha R},e^{ik\cdot R} $$

가 정규화 후 비자명 Chern number를 가지기 위한 세 가지 조건을 **운동량
공간(Fourier 계수)** 언어로 제시한다:

1.  **Common zero** --- $f$ 가 어떤 $k_i$ 에서 모든 성분이 동시에 0.
2.  **Projector continuity** --- 그 점에서 derivative matrix
    $A_{\alpha\mu}$ 가 $\mathrm{rank}_{\mathbb C}=1$.
3.  **Nonzero winding** --- $\mathrm{Im}(m_x^* m_y)\neq 0$.

이 문서는 동일한 내용을 **실공간 CLS(Compact Localized State)를
중심으로** 재정식화한다. 그 이유는:

- Flat band의 eigenvector는 본질적으로 유한개 Bloch phase의 합(Finite
  Sum of Bloch Phase, **FSBP**)이고, 실공간에서는 격자 site에 국소화된
  delta-function 형태(CLS)가 된다.
- 진폭/위상/site 위치라는 **물리적·기하학적 변수**로 조건이 직접
  표현되므로 더 직관적이며, **flat-band engineering**에서 곧바로 설계
  변수로 쓸 수 있다.

**핵심 결론(미리):** 세 조건은 실공간에서 단 두 개로 압축된다.

- **(C1) 0차(common zero) 조건** = CLS 진폭들의 **소멸간섭(destructive
  interference)**.
- **(C2≡2+3) 1차 chiral 조건** $;a^{(\alpha)}=\pm i,b^{(\alpha)};$ =
  projector 연속성과 winding을 **동시에** 강제하여 국소 vortex
  $q_x\pm iq_y$, 즉 $C_{\text{loc}}=\pm1$ 을 만든다.

------------------------------------------------------------------------

## 1. CLS 기반 FSBP 분해

각 성분(=sublattice/orbital index) $\alpha=1,\dots,N$ 에 대해 Fourier
계수를 극형식(polar form)으로 분해한다:

$$ c_{\alpha,j}=A^{(\alpha)}_{j},e^{,i\left(\theta^{(\alpha)}_{j}+\zeta^{(\alpha)}\right)},\qquad A^{(\alpha)}_{j}>0 . $$

그러면 eigenvector 성분은

$$ f_\alpha(k)=\sum_{j} A^{(\alpha)}_{j}; e^{,i\left(\theta^{(\alpha)}_{j}+\zeta^{(\alpha)}\right)}; e^{,ik\cdot R^{(\alpha)}_{j}} . $$

각 항의 물리적 의미는 다음과 같다.

  ------------------------------------------------------------------------------------------------------------
  기호                                                         이름                    의미
  ------------------------------------------------------------ ----------------------- -----------------------
  $A^{(\alpha)}_{j}>0$                                         **CLS amplitude**       항상 양수인 진폭항.
                                                                                       부호/위상은 전부
                                                                                       phase로 분리.

  $\theta^{(\alpha)}_{j}$                                      **CLS phase**           각 site의 CLS가 갖는
                                                                                       고유 위상.

  $\phi^{(\alpha)}_{j}=k_i\cdot R^{(\alpha)}_{j}$              **site phase**          singularity point $k_i$
                                                                                       가 원점이 아닐 때
                                                                                       선형전개에서 추가로
                                                                                       붙는 위상.

  $\zeta^{(\alpha)}$                                           **sublattice shift**    unit cell 내에서 기준
                                                                                       site로부터의
                                                                                       orbital/sublattice
                                                                                       변위에 따른 위상(성분
                                                                                       $\alpha$ 공통).

  $R^{(\alpha)}_{j}=n^{(\alpha)}_{j}a_1+m^{(\alpha)}_{j}a_2$   **격자 위치**           $a_1,a_2$ lattice
                                                                                       vector,
                                                                                       $n,m\in\mathbb Z$.
  ------------------------------------------------------------------------------------------------------------

여기서 $k_i$ 는 **Bloch wave function의 singularity point**(common
zero가 발생하는 BZ 내 점)이다. CLS를 중심으로 보기 위해 변위
$\Delta R^{(\alpha)}_{j}$ 를 도입하며, 아래에서 보듯 0차 조건이 성립하면
**CLS 중심의 선택은 결과에 무관**하므로 $\Delta R$ 대신 절대좌표 $R$ 를
써도 된다.

총위상을 한 기호로 묶으면

$$ \boxed{;\Theta^{(\alpha)}_{j}\equiv \theta^{(\alpha)}_{j}+\phi^{(\alpha)}_{j}+\zeta^{(\alpha)};} $$

------------------------------------------------------------------------

## 2. Singularity 근방 국소 전개

$k=k_i+q$, $q=(q_x,q_y)$ 로 두고 전개한다:

$$ f_\alpha(k_i+q)=\sum_{j}A^{(\alpha)}_{j},e^{,i\Theta^{(\alpha)}_{j}},e^{,iq\cdot \Delta R^{(\alpha)}_{j}} =\sum_{j}A^{(\alpha)}_{j},e^{,i\Theta^{(\alpha)}_{j}} \Big[1+i,q\cdot \Delta R^{(\alpha)}_{j}+O(q^2)\Big]. $$

$i=e^{i\pi/2}$ 를 써서 1차항의 $i$ 를 위상으로 흡수하면

$$ f_\alpha(k_i+q)=\underbrace{\sum_{j}A^{(\alpha)}_{j}e^{,i\Theta^{(\alpha)}_{j}}}_{O(q^0)} ;+;\underbrace{\sum_{j}A^{(\alpha)}_{j},e^{,i\left(\Theta^{(\alpha)}_{j}+\frac{\pi}{2}\right)},(q\cdot\Delta R^{(\alpha)}_{j})}_{O(q^1)} ;+;O(q^2). $$

이 두 항이 각각 노트의 **조건 1**과 **조건 2·3**에 대응한다.

------------------------------------------------------------------------

## 3. 조건 1 --- Common Zero = 소멸간섭

### 3.1 조건

$$ \boxed{;\sum_{j}A^{(\alpha)}_{j},e^{,i\left(\Theta^{(\alpha)}_{j}+\frac{\pi}{2}\right)}=0 \quad\Longleftrightarrow\quad \sum_{j}A^{(\alpha)}_{j},e^{,i\Theta^{(\alpha)}_{j}}=0,\quad \forall\alpha;} $$

($+\tfrac{\pi}{2}$ 는 전체 위상이므로 0차 소멸 여부에는 영향이 없다.)
이것은 노트 Eq. (56)의 common-zero 조건과 정확히 같다:
$\sum_R c_{\alpha R}e^{ik_i\cdot R}=0$.

### 3.2 실공간적 의미

진폭 $A^{(\alpha)}_{j}$ 가 전부 **양의 실수**이므로, 합이 0이 되는 길은
오직 **위상에 의한 상쇄**뿐이다. 이것이 "FSBP가 BZ의 어떤 점에서
사라진다"는 사실의 실공간 번역이다. (Chern band는 전역적으로 0이 되지
않는 매끄러운 section을 가질 수 없으므로, 유한 Fourier vector가 비자명
번들을 표현하려면 반드시 어딘가에서 사라져야 한다.)

### 3.3 현실적 격자 실현: 소멸간섭 쌍(pair)

가장 자연스럽고 격자 모형에서 실현 가능한 방식은 **진폭이 같고 위상이
$\pi$ 만큼 반대인 site 쌍**을 만드는 것이다. 같은 성분 $\alpha$ 의 두
site $j_1,j_2$ 에 대해

$$ A^{(\alpha)}_{j_1}=A^{(\alpha)}_{j_2}\quad\text{(진폭 동일)},\qquad \theta^{(\alpha)}_{j_1}+\phi^{(\alpha)}_{j_1} =\theta^{(\alpha)}_{j_2}+\phi^{(\alpha)}_{j_2}+(2m+1)\pi . $$

($\zeta^{(\alpha)}$ 는 같은 성분에서 두 항에 공통이므로 쌍 조건에서
자동으로 상쇄된다.) 이때

$$ A^{(\alpha)}_{j_1}e^{,i\Theta^{(\alpha)}_{j_1}}+A^{(\alpha)}_{j_2}e^{,i\Theta^{(\alpha)}_{j_2}} =A^{(\alpha)}_{j_1}e^{,i\Theta^{(\alpha)}_{j_1}}\big(1+e^{,i(2m+1)\pi}\big)=0 . $$

이런 **소멸간섭 쌍**이 각 성분마다 존재하면 조건 1이 항상 만족된다.

------------------------------------------------------------------------

## 4. 1차항과 계수 $a,b$

$q=(q\cos\phi,;q\sin\phi)$ 로 두면
$q\cdot\Delta R^{(\alpha)}_{j}=q\big(\Delta R^{(\alpha)}_{j,x}\cos\phi+\Delta R^{(\alpha)}_{j,y}\sin\phi\big)$
이고, 1차항은

$$ f_\alpha(k_i+q);\sim; q,\Big[a^{(\alpha)}\cos\phi+b^{(\alpha)}\sin\phi\Big], $$

$$ \boxed{;a^{(\alpha)}=\sum_{j}A^{(\alpha)}_{j},e^{,i\left(\Theta^{(\alpha)}_{j}+\frac{\pi}{2}\right)}\Delta R^{(\alpha)}_{j,x},\qquad b^{(\alpha)}=\sum_{j}A^{(\alpha)}_{j},e^{,i\left(\Theta^{(\alpha)}_{j}+\frac{\pi}{2}\right)}\Delta R^{(\alpha)}_{j,y};} $$

이 $a^{(\alpha)},b^{(\alpha)}\in\mathbb C$ 는 노트의 derivative matrix
성분과 정확히 일치한다:

$$ a^{(\alpha)}\equiv A^{(i)}_{\alpha x}=i\sum_R R_x,c_{\alpha R}e^{ik_i\cdot R},\qquad b^{(\alpha)}\equiv A^{(i)}_{\alpha y}=i\sum_R R_y,c_{\alpha R}e^{ik_i\cdot R}. $$

> **원점 무관성(origin independence).** 조건
> 1($\sum_j A_je^{i\Theta_j}=0$)이 성립하면,
> $\Delta R_j\to \Delta R_j+R_c$ (CLS 중심을 임의로 평행이동) 해도 $a,b$
> 는 $R_c\cdot(i\cdot 0)=0$ 만큼 변하므로 **불변**이다. 따라서 절대좌표
> $R_j$ 를 그대로 써도 된다.

### 4.1 소멸간섭 쌍의 1차 기여: $R_1-R_2$ 모멘트

진폭이 같고 위상이 $\pi$ 반대인
쌍($e^{i\Theta_{j_2}}=-e^{i\Theta_{j_1}}$)의 1차 기여를 합치면

$$ A,e^{,i(\Theta_{j_1}+\frac{\pi}{2})}(q\cdot\Delta R_{j_1}) +A,e^{,i(\Theta_{j_2}+\frac{\pi}{2})}(q\cdot\Delta R_{j_2}) =A,e^{,i(\Theta_{j_1}+\frac{\pi}{2})};q\cdot\big(\underbrace{\Delta R_{j_1}-\Delta R_{j_2}}_{=,R_{j_1}-R_{j_2}}\big). $$

즉 **소멸간섭 쌍은 1차에서 두 site 사이의 결합 벡터(bond vector)
$R_1-R_2$ 를 모멘트로 갖는 하나의 유효항**처럼 행동한다. 이 bond
벡터들과 그들의 위상이 $a^{(\alpha)},b^{(\alpha)}$ 를 결정하며, 따라서
아래 chiral 조건의 만족 여부를 좌우한다 --- 이것이 flat-band
engineering의 실제 설계 손잡이다.

------------------------------------------------------------------------

## 5. 조건 2+3 (통합) --- Chiral 조건 $a=\pm i,b$

### 5.1 조건

projector가 연속이면서 동시에 phase가 winding 하려면, **모든 성분
$\alpha$ 에 대해 같은 부호로**

$$ \boxed{;a^{(\alpha)}=\pm,i,b^{(\alpha)}\qquad(\forall\alpha,\ \text{부호 공통});} $$

### 5.2 왜 이것이 조건 2와 3을 한 번에 담는가

$a=-ib$ 를 대입하면

$$ a\cos\phi+b\sin\phi=b(-i\cos\phi+\sin\phi)=-i,b,(\cos\phi+i\sin\phi)=-i,b,e^{+i\phi}, $$

$a=+ib$ 이면

$$ a\cos\phi+b\sin\phi=b(i\cos\phi+\sin\phi)=i,b,(\cos\phi-i\sin\phi)=i,b,e^{-i\phi}. $$

따라서 1차항은

$$ f_\alpha(k_i+q);\sim; q,e^{\pm i\phi},\big(\mp i,b^{(\alpha)}\big). $$

여기서 **결정적 포인트**: 위상 인자 $e^{\pm i\phi}$ 는 성분 $\alpha$ 에
**의존하지 않는 공통 스칼라**다. 그러므로 벡터 전체가

$$ f(k_i+q);\sim; \underbrace{q,e^{\pm i\phi}}_{\text{scalar }h(q)};\underbrace{v}_{v_\alpha=,\mp i,b^{(\alpha)}}, \qquad h(q);\propto;q_x\pm i q_y . $$

이는 노트의 rank-one factorization $f\simeq h(q),v$ 와 정확히 같으며, 한
번에 두 조건을 만족한다:

- **조건 2 (projector continuity / rank-1):** 1차 선행항이 "스칼라 ×
  고정벡터 $v$" 이므로 $[f]\to[v]$ 가 방향($\phi$)에 무관 → projector
  연속. 즉 $;\mathbf A_x=\pm i,\mathbf A_y;$ (벡터 등식) 이므로 자동으로
  $\mathbf A_x\parallel\mathbf A_y$, $\mathrm{rank}_{\mathbb C}(A)=1$.
- **조건 3 (nonzero winding):** 스칼라 $h\propto q_x\pm iq_y$ 가 원점을
  한 바퀴 감으므로 winding $=\pm1$.

### 5.3 부호와 Chern number

$$ a^{(\alpha)}=-i,b^{(\alpha)}\ \Rightarrow\ h\propto q_x+iq_y\ \Rightarrow\ C_{\text{loc}}=+1, $$
$$ a^{(\alpha)}=+i,b^{(\alpha)}\ \Rightarrow\ h\propto q_x-iq_y\ \Rightarrow\ C_{\text{loc}}=-1 $$

(전체 부호는 Berry connection·BZ 방향 규약에 따른다 --- 노트 caveat
(v)). 즉 chiral 조건은 국소적으로 $B\cdot(k_x\pm i k_y)$ 형태를
**강제**하여 $C=\pm1$ 상태를 만든다.

### 5.4 노트의 $\mathrm{Im}(m_x^*m_y)$ 와의 일치

$h=m_xq_x+m_yq_y$, $a/b=m_x/m_y$. $a=\mp i b\Rightarrow m_x=\pm i,m_y$
이고

$$ \mathrm{Im}(m_x^*m_y)=\mathrm{Im}\big((\pm i,m_y)^*m_y\big)=\mp|m_y|^2\neq 0, $$

즉 winding $w_i=\mathrm{sgn},\mathrm{Im}(m_x^*m_y)=\pm1$ 과 부호까지
일치한다.

> **포함 관계(정확히 짚기).** 일반적 조건 2+3 (rank-1
> $+\ \mathrm{Im}(m_x^*m_y)\neq0$)은 타원형(anisotropic) vortex도
> 허용한다. $a=\pm i b$ 는 그중 **isotropic vortex $q_x\pm iq_y$** 로
> 고정하는 **충분조건**이며, 위상이 정확히 $\pm\pi/2$ 상대차일 때
> 성립한다. 즉 가장 깨끗하고 **engineering-friendly한 표준형**이다.
> (필요충분 일반형이 필요하면 $\mathrm{Im}(m_x^*m_y)\neq0$ 로 완화하면
> 된다.)

------------------------------------------------------------------------

## 6. 두 정식화의 대조표

  ----------------------------------------------------------------------------------------------------------------------
  실공간 CLS (본 문서)                                             운동량 공간 노트 (June 3)
  ---------------------------------------------------------------- -----------------------------------------------------
  소멸간섭 $\sum_j A_j e^{i\Theta_j}=0$                            common zero, Eq. (56)

  $a^{(\alpha)}=\sum_j A_j e^{i(\Theta_j+\pi/2)}\Delta R_{j,x}$,   $A^{(i)}_{\alpha x},A^{(i)}_{\alpha y}$, Eq. (57)
  $b$ ($y$)                                                        

  $a=\pm i b$ 중 **평행성** 부분                                   rank-1, Eq. (58)/(59)

  $a=\pm i b$ 중 **$\pm i$ 상대위상** 부분                         $\mathrm{Im}(m_x^*m_y)\neq0$, Eq. (60)

  부호 $\pm$                                                       $w_i=\mathrm{sgn},\mathrm{Im}(m_x^*m_y)$, Eq. (61)

  여러 singularity 합산                                            $C=\sum_i w_i$, Eq. (62)

  소멸쌍의 $R_1-R_2$ 모멘트                                        $A_{\alpha\mu}=i\sum_R R_\mu c_{\alpha R}e^{ik_iR}$
                                                                   의 격자 해석
  ----------------------------------------------------------------------------------------------------------------------

총 Chern number는 모든 singularity의 국소 winding 합

$$ C=\sum_i w_i,\qquad w_i=\mathrm{sgn},\mathrm{Im}\big[(m_x^{(i)})^*m_y^{(i)}\big]=\pm1 . $$

여러 영점은 서로 상쇄될 수 있다(노트 caveat (iv)).

------------------------------------------------------------------------

## 7. Flat-band Hamiltonian (projector 구성)

위 조건을 만족하는 정규화 상태
$|\psi(k)\rangle=u(k)=f(k)/\lVert f(k)\rVert$ 로 projector를 만들면

$$ P(k)=|\psi(k)\rangle\langle\psi(k)|, $$

flat band Hamiltonian은

$$ \boxed{;H(k)=E_0,P(k)+\big[I-P(k)\big],M(k),\big[I-P(k)\big];} $$

- 첫 항: $P$ 의 image(=CLS flat band)를 에너지 $E_0$ 의 평탄밴드로 고정.
- 둘째 항: 임의의 Hermitian $M(k)$ 로 나머지 $N-1$ 개 분산밴드를
  자유롭게 배치(평탄밴드 에너지는 건드리지 않음).

평탄밴드의 위상은 section $f$ 의 vortex로부터 $C=\sum_i w_i$ 를
물려받는다.

------------------------------------------------------------------------

## 8. Flat band 특이성과의 연결 (Rhim--Yang 관점 보충)

- common zero $k_i$ 는 단순한 영점이 아니라, **평탄밴드가 다른 밴드와
  닿는 특이점(singular point)** 으로 보는 것이 자연스럽다. $f$ 가 그
  점에서 사라진다는 것은 CLS 기반 평탄밴드가 그 $k$ 에서 잘 정의된 단일
  방향을 잃는다는 뜻이며, 이는 Rhim--Yang 분류의 **singular flat band**
  와 정확히 맞닿는다.
- **주의:** 유한 거리(finite-range) CLS만으로 만든 **완전히
  고립된(gapped)** 평탄밴드는 일반적으로 $C=0$ 이다. 비자명 $C$ 는 위의
  common zero에서의 band touching(특이성)을 통해서만 가능하다. 따라서 본
  구성에서 $C\neq0$ 이 나오면 그것은 **닿아 있는(특이) 평탄밴드**이지,
  격리된 Chern 평탄밴드가 아니다. 전체 밴드의 Chern 합은 0이므로
  $C=\pm1$ 평탄밴드는 분산밴드들이 보상한다.
- 이 관점에서 chiral 조건 $a=\pm ib$ 는 **그 특이점에서의 vorticity(국소
  winding)** 를 직접 설계하는 식이 된다.

------------------------------------------------------------------------

## 9. 실무 체크리스트 (실공간 버전)

주어진 CLS 사양 ${A^{(\alpha)}_j,\theta^{(\alpha)}_j,R^{(\alpha)}_j}$,
$\zeta^{(\alpha)}$, 그리고 singularity $k_i$ 에 대해:

1.  **총위상 계산:**
    $\Theta^{(\alpha)}_j=\theta^{(\alpha)}_j+k_i\cdot R^{(\alpha)}_j+\zeta^{(\alpha)}$.
2.  **조건 1 확인:**
    $\big|\sum_j A^{(\alpha)}_j e^{i\Theta^{(\alpha)}_j}\big|=0\ (\forall\alpha)$.
    (실현은 진폭 동일·위상 $\pi$ 반대 쌍.)
3.  **모멘트 계산:**
    $a^{(\alpha)}=\sum_j A^{(\alpha)}_j e^{i(\Theta^{(\alpha)}_j+\pi/2)}R^{(\alpha)}_{j,x}$,
    $b^{(\alpha)}$ (with $R_{j,y}$).
4.  **조건 2+3 확인:** $a^{(\alpha)}=\pm i,b^{(\alpha)}$ 가 **모든
    $\alpha$ 에서 같은 부호**로 성립하는가.
    - 실패 시: rank-1이 깨지면 projector 불연속, $\pm i$ 가 아니면
      winding 0.
5.  **국소 winding:**
    $w_i=\mathrm{sgn},\mathrm{Im}\big[(m_x^{(i)})^*m_y^{(i)}\big]$
    ($a=-ib\Rightarrow+1$, $a=+ib\Rightarrow-1$).
6.  **합산:** $C=\sum_i w_i$.

------------------------------------------------------------------------

## 10. 한눈에 보는 요약

> **유한 Fourier vector(=CLS)가 비자명 Chern number를 가지려면:**
>
> 1.  **(소멸간섭)** 각 성분의 CLS 진폭들이 고립된 점 $k_i$ 에서 위상
>     상쇄로 사라지고 --- 보통 \_진폭 동일·위상 $\pi$ 반대 쌍_으로 실현,
> 2.  **(공통 스칼라 vortex)** 그 1차 선행 거동이 모든 성분에 **공통인
>     스칼라 인자** $h\propto q_x\pm iq_y$ 로 인수분해되어 projector가
>     연속이고($a=\pm ib$ 의 평행성), 그 스칼라가 **위상 winding
>     $\pm1$** 을 가지면($\pm i$ 상대위상) 된다.
>
> 실공간 손잡이는 **소멸쌍의 bond 벡터 $R_1-R_2$ 와 CLS 위상 $\theta$**
> 이며, 이들을 배치해 $a=\pm ib$ 를 맞추는 것이 곧 $C=\pm1$ 평탄밴드
> 설계다.

# Real-Space CLS 기반의 Nonzero Chern Number Flat Band Engineering 조건

본 문서는 Brillouin zone 상의 유한한 푸리에 합(Finite Fourier-Sum)으로
표현되는 Eigenvector가 Nonzero Chern number를 갖기 위한 수학적 조건들을,
실제 Lattice 공간 상의 **Compact Localized State (CLS)** 관점에서
재해석하고 구체적인 Flat band engineering의 설계 지침을 제시한다.

## 1. Mathematical Framework and Physical Notation

일반적으로 Flat band의 Eigenvector는 Real space domain에서 Delta
function 형태로 Lattice sites에 Localization되는 특성을 갖는다. 따라서
$\mathbf{k}$-space에서의 Bloch eigenvector $f(\mathbf{k})$는 Finite Sum
of Bloch Phase (FSBP)로 다음과 같이 표현할 수 있다.

$$f(\mathbf{k}) = \begin{pmatrix} f_{1}(\mathbf{k}) \\ \vdots \\ f_{N}(\mathbf{k}) \end{pmatrix}$$

Brillouin zone 내의 Singularity point $\mathbf{k}_{i}$ 근처에서, 특정
Sublattice(또는 Orbital) 성분 $f^{(i)}(\mathbf{k})$는 실제 Real-space
CLS를 중심으로 다음과 같이 직관적인 Parameter들로 서술할 수 있다.

$$f^{(i)}(\mathbf{k}) = \sum_{j} A_{j}^{(i)} e^{ i \left( \theta_{j}^{(i)} + \phi_{j}^{(i)} + \zeta^{(i)} \right)} e^{i \mathbf{k} \cdot \mathbf{R}}$$

각 Parameter의 물리적 의미는 다음과 같다:

- $A_{j}^{(i)} > 0$ : **CLS Amplitude**. (항상 양수의 진폭을 갖도록
  정의)

- $\theta_{j}^{(i)}$ : **CLS Phase**. (해당 Lattice site의 CLS가 갖는
  고유한 Phase)

- $\phi_{j}^{(i)} = \mathbf{k}_{i} \cdot \mathbf{R}_{j}^{(i)}$ : **Site
  phase**. (Singularity point $\mathbf{k}_i$에 의해 결정되는 Phase
  shift)

- $\zeta^{(i)}$ : **Site shift term**. (Unit cell 내에서 현재 바라보는
  Site가 Unit cell 중심으로부터 얼마나 떨어져 있는지를 나타내는 항)

추가로, Singularity point $\mathbf{k}_{i}$ 근처에서의 거동을 보기 위해
$\mathbf{k} = \mathbf{k}_i + \mathbf{q}$ 로 두고
$\mathbf{q} \rightarrow 0$ 극한을 취하면, 선형 전개(Linear expansion)에
의해
$e^{i \mathbf{q} \cdot \Delta\mathbf{R}} \approx 1 + i \mathbf{q} \cdot \Delta\mathbf{R}$
이 되며, 이때 발생하는 허수 $i$는 $e^{i\frac{\pi}{2}}$ 의 추가적인 Phase
shift로 작용한다.

이때 위치 벡터의 차이 $\Delta \mathbf{R}_{j}^{(i)}$는 Lattice vector\_
$\mathbf{a}_1, \mathbf{a}_2$*와 정수* $n, m$\_을 이용하여 다음과 같이
표현된다.
$$\Delta \mathbf{R}_{j}^{(i)} = n_{j}^{(i)}\mathbf{a}_{1} + m_{j}^{(i)}\mathbf{a}_{2}$$

## 2. Condition 1: Common Zero and Real-Space Destructive Interference

Chern band가 되기 위한 첫 번째 필수 조건은 Vector $f(\mathbf{k})$가 전체
BZ 내에서 적어도 하나의 **Common zero (Singularity)** 를 가져야 한다는
것이다 ($f(\mathbf{k}_i) = 0$). 이를 앞서 정의한 CLS 기반 수식에
적용하면 0차 항이 소멸해야 함을 의미한다.

**\[Condition 1의 수학적 표현\]**

$$\sum_{j} A_{j}^{(i)} e^{ i \left( \theta_{j}^{(i)} + \phi_{j}^{(i)} + \zeta^{(i)} \right)} = 0$$

### 2.1. 현실적인 Lattice Model 반영 조건 (Pairing Rule)

위 조건을 Lattice 상에서 가장 직관적이고 현실적으로 만족시키는 방법은
**상쇄 간섭 쌍(Destructive interference pairs)** 을 설계하는 것이다.

특정 두 Site (Site 1, Site 2)가 다음 조건을 만족한다고 가정하자:

1.  **Equal Amplitude:** $A_{1}^{(i)} = A_{2}^{(i)}$

2.  **Opposite Phase:**
    $\theta_{1}^{(i)} + \phi_{1}^{(i)} = \theta_{2}^{(i)} + \phi_{2}^{(i)} + (2m + 1)\pi$
    (단, $m$은 정수)

이러한 쌍(Pair)들로 CLS 구조가 이루어져 있다면, 합산 과정에서 각 쌍이
소거되므로 Condition 1은 항상 자연스럽게 만족된다.

## 3. Condition 2 & 3: Projector Continuity and Chiral Vortex Generation

Condition 1이 만족되어 0차 항이 사라졌으므로,
$f^{(i)}(\mathbf{k}_i + \mathbf{q})$의 거동은 선형 항(Linear term)에
의해 지배된다. $\mathbf{q} = (q\cos\phi, q\sin\phi)$로 두었을 때, 선형
전개된 함수는 다음과 같이 근사된다.

$$f^{(i)}(\mathbf{k}_i + \mathbf{q}) \sim \sum_{j} A_{j}^{(i)} e^{ i \left( \theta_{j}^{(i)} + \phi_{j}^{(i)} + \zeta^{(i)} + \frac{\pi}{2} \right)} (\mathbf{q} \cdot \Delta\mathbf{R}_{j}^{(i)})$$

이 선형 조합은 $q_x$와 $q_y$에 대한 1차 함수이므로, 복소계수
$a^{(i)}, b^{(i)}$를 이용하여 다음과 같이 쓸 수 있다.

$$f^{(i)}(\mathbf{q}) \sim a^{(i)} (q\cos\phi) + b^{(i)} (q\sin\phi) = q (a^{(i)}\cos\phi + b^{(i)}\sin\phi)$$

### 3.1. 통합된 Winding Condition: $a^{(i)} = \pm i b^{(i)}$

수학적 이론에 따르면, Projector $P(\mathbf{k})$가 연속적으로 정의되기
위한 Rank-one 조건(Condition 2)과, Nonzero Chern number를 생성하기 위한
Nonzero scalar winding 조건(Condition 3)이 필요하다.

이 두 가지 복잡한 수학적 조건은 Real-space 구조에서 다음의 직관적인 단일
조건으로 강제(Force)될 수 있다.

**\[Condition 2 & 3의 통합된 물리적 조건\]**

$$a^{(i)} = \pm i b^{(i)}$$

이 조건을 적용하면 수식은 다음과 같이 변환된다:

$$q (a^{(i)}\cos\phi \pm i a^{(i)}\sin\phi) = a^{(i)} q e^{\pm i\phi} = B^{(i)} e^{i\varphi^{(i)}}$$

즉, 결과적으로 이 항은 $B \times (q_x \pm i q_y)$ 또는
$B \times (k_x \pm i k_y)$ 형태의 완벽한 **Chiral Vortex**가 된다.

이 Vortex는 Singularity 주위를 한 바퀴 돌 때 Phase가 $\pm 2\pi$ 만큼
변화함을 보장하므로, **Local Chern number(Winding number)**
$w_i = \pm 1$ 인 상태를 완벽하게 만족시킨다. 여기서 앞서 설명한 상쇄
간섭 쌍의 좌표 차이인 $\Delta \mathbf{R}_{j}^{(i)}$가
$a^{(i)}, b^{(i)}$의 값을 결정하는 핵심 기하학적 요소가 된다.

## 4. Hamiltonian Construction

위의 조건들을 만족하도록 Real space상의 Phase $\theta_j$ 및 Amplitude
$A_j$를 설계하여 성공적으로 Target Bloch state
$|\psi(\mathbf{k})\rangle$ 를 구축했다면, 이 상태가 Flat band를
형성하도록 만드는 Real-space Hamiltonian $H(\mathbf{k})$는 Projector
formalism을 통해 곧바로 구성할 수 있다.

정규화된 Projector
$P(\mathbf{k}) = |\psi(\mathbf{k})\rangle \langle \psi(\mathbf{k})|$ 를
이용하여, 다음과 같이 역설계(Reverse engineering)를 수행한다.

$$H(\mathbf{k}) = E_0 P(\mathbf{k}) + \left[ I - P(\mathbf{k}) \right] M(\mathbf{k}) \left[ I - P(\mathbf{k}) \right]$$

- $E_0$ : Target Flat band의 Energy level

- $M(\mathbf{k})$ : Dispersive band들의 Energy와 형태를 조절하는 임의의
  Hermitian Matrix

이러한 Construction을 통해 생성된 Hamiltonian은 설계된
$|\psi(\mathbf{k})\rangle$ 를 고유 상태로 가지는 완벽한 Flat band를
포함하며, 앞서 증명한 조건들($f(\mathbf{k}_i) = 0$ 및
$a^{(i)} = \pm i b^{(i)}$)에 의해 이 Flat band는 반드시 Nonzero Chern
number를 갖게 된다.

## 요약 (Conclusion)

Topological Flat Band를 설계하기 위해 추상적인 푸리에 계수의 제약 조건을
푸는 대신, **Real-space의 CLS 진폭(**$A_j$**)과 위상(**$\theta_j$**)**
을 컨트롤하는 것으로 문제를 치환할 수 있다.

1.  **Destructive pair (**$\pi$ **phase shift)** 를 만들어
    Singularity(Common zero)를 보장하고,

2.  Singularity 주변에서의 기하학적 분포 $\Delta \mathbf{R}$ 가
    $a^{(i)} = \pm i b^{(i)}$ 의 Chiral symmetry를 가지도록 배치하면,

    자연스럽게 Chern number $\pm 1$을 가지는 Topological Flat Band를
    구현할 수 있다.
