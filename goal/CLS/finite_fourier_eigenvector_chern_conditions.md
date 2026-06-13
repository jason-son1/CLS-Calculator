| Conditions |     |     | for a    | Finite  |     | Fourier-Sum |        |            | Eigenvector |     |     | to Carry | a   |
| ---------- | --- | --- | -------- | ------- | --- | ----------- | ------ | ---------- | ----------- | --- | --- | -------- | --- |
|            |     |     |          | Nonzero |     |             | Chern  | Number     |             |     |     |          |     |
|            |     |     |          | Notes   |     | based       | on the | discussion |             |     |     |          |     |
|            |     |     |          |         |     | June        | 3,     | 2026       |             |     |     |          |     |
| 1 Problem  |     | and | notation |         |     |             |        |            |             |     |     |          |     |
Consider an unnormalized Bloch eigenvector with N internal components,
f (k)
1
.
|     |     |     | f(k) | =   | .   | ,   | k   | = (k | ,k ) | BZ  | 2,  |     | (1) |
| --- | --- | --- | ---- | --- | --- | --- | --- | ---- | ---- | --- | --- | --- | --- |
|     |     |     |      | 󰀳   | .   | 󰀴   |     | x    | y    |     | T   |     |     |
|     |     |     |      |     |     |     |     |      |      | ∈   | ≃   |     |     |
f (k)
N
|            |           |     |       | 󰁅      |        | 󰁆              |     |          |     |      |      |     |     |
| ---------- | --------- | --- | ----- | ------ | ------ | -------------- | --- | -------- | --- | ---- | ---- | --- | --- |
|            |           |     |       | 󰁃      |        | 󰁄              |     |          |     |      |      |     |     |
| where each | component |     | is a  | finite | sum    | of exponential |     | factors, |     |      |      |     |     |
|            |           | f   | (k) = | c      | eik R, |                | c   | C,       | R   | = (R | ,R ) | 2.  | (2) |
|            |           | α   |       |        | αR ·   |                | αR  |          |     |      | x y  | Z   |     |
|            |           |     |       |        |        |                |     | ∈        |     |      |      | ∈   |     |
R Sα
󰁛∈
| The normalized |     | state | is formally |     |     |     |     |     |     |     |     |     |     |
| -------------- | --- | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
N
f(k)
|     |     |     | u(k) | =   |     | ,   | f(k) | 2   | =   | f (k) | 2.  |     | (3) |
| --- | --- | --- | ---- | --- | --- | --- | ---- | --- | --- | ----- | --- | --- | --- |
α
|     |     |     |     |     | f(k) |     | 󰀂   | 󰀂   |     | |   | |   |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     | 󰀂    | 󰀂   |     |     | α=1 |     |     |     |     |
󰁛
| The associated |          | rank-one | projector |      | is  |      |      |       |         |       |     |     |     |
| -------------- | -------- | -------- | --------- | ---- | --- | ---- | ---- | ----- | ------- | ----- | --- | --- | --- |
|                |          |          |           |      |     |      |      | f(k)f |         | † (k) |     |     |     |
|                |          |          |           | P(k) | =   | u(k) | u(k) | =     |         | .     |     |     | (4) |
|                |          |          |           |      |     |      |      | f     | (k)f(k) |       |     |     |     |
|                |          |          |           |      |     | |    | 〉〈   | |     | †       |       |     |     |     |
| The central    | question |          | is:       |      |     |      |      |       |         |       |     |     |     |
Under what conditions on the Fourier coefficients c can such an unnormalized
αR
finiteFourier-sumeigenvectorproduceanonzeroChernnumberafternormalization?
| The | Chern | number | of a | smooth | rank-one |     | projector |     | is  |     |     |     |     |
| --- | ----- | ------ | ---- | ------ | -------- | --- | --------- | --- | --- | --- | --- | --- | --- |
1
(k)d2k,
|     |     |     | C = |     | F xy |     | F   | xy = | iTrP[∂ | kx  | P,∂ ky | P]. | (5) |
| --- | --- | --- | --- | --- | ---- | --- | --- | ---- | ------ | --- | ------ | --- | --- |
|     |     |     | 2π  |     |      |     |     |      | −      |     |        |     |     |
󰁝BZ
| Equivalently, |     | in a smooth | local |     | gauge, |     |     |     |     |       |     |     |     |
| ------------- | --- | ----------- | ----- | --- | ------ | --- | --- | --- | --- | ----- | --- | --- | --- |
|               |     |             | F     | =   | ∂ A    | ∂   | A , | A   | =   | i u ∂ | u . |     | (6) |
|               |     |             |       | xy  | kx y   | ky  | x   | µ   |     | kµ    |     |     |     |
|               |     |             |       |     |        | −   |     |     | −   | 〈 |   | 〉   |     |     |
1

| 2 First |     | basic | fact: | if there |     | is  | no common |     | zero, | then | C = 0 |     |
| ------- | --- | ----- | ----- | -------- | --- | --- | --------- | --- | ----- | ---- | ----- | --- |
Suppose
|     |     |     |      | 2   |     |       | 2   |     |           |     |     |     |
| --- | --- | --- | ---- | --- | --- | ----- | --- | --- | --------- | --- | --- | --- |
|     |     |     | f(k) | =   |     | f (k) | > 0 | for | all k BZ. |     |     | (7) |
|     |     |     | 󰀂    | 󰀂   |     | | α   | |   |     | ∈         |     |     |     |
α
󰁛
Thenu(k) = f(k)/ f(k) isagloballydefined,smooth,periodicBlocheigenvectoronthewhole
|     |     | 󰀂   | 󰀂   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Brillouin torus. Hence the Berry connection A is globally defined and
|          |     |          |           |     |     | F   | = dA. |     |     |     |     | (8) |
| -------- | --- | -------- | --------- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
| Since BZ |     | 2 has no | boundary, |     |     |     |       |     |     |     |     |     |
T
|     | ≃   |     |     |     |     | 1   |     |      |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
|     |     |     |     |     | C   | =   | dA  | = 0. |     |     |     | (9) |
2π
󰁝T2
Therefore:
Proposition 1 (Necessary condition for nonzero Chern number). A nonzero Chern number is
possible only if the finite Fourier vector has at least one common zero,
|          |            | k             | BZ  | such | that  | f (k | ) = | f (k ) =     | = f (k | ) = 0. |     | (10) |
| -------- | ---------- | ------------- | --- | ---- | ----- | ---- | --- | ------------ | ------ | ------ | --- | ---- |
|          |            |               | i   |      |       | 1    | i   | 2 i          | N      | i      |     |      |
|          |            | ∃             | ∈   |      |       |      |     |              | ···    |        |     |      |
| In terms | of Fourier | coefficients, |     |      |       |      |     |              |        |        |     |      |
|          |            |               |     | c    | eiki· | R =  | 0,  | α = 1,...,N. |        |        |     | (11) |
αR
∀
R
󰁛
This condition is very natural: a Chern band cannot admit a globally nonvanishing smooth
section. A finite Fourier vector can represent a global section, but if the line bundle has nonzero
| first Chern | number, | this       | section | must | vanish    |     | somewhere. |      |        |            |     |     |
| ----------- | ------- | ---------- | ------- | ---- | --------- | --- | ---------- | ---- | ------ | ---------- | --- | --- |
| 3 Second    |         | condition: |         | the  | projector |     |            | must | remain | continuous |     |     |
A common zero of f(k) is not enough. At a common zero k , the normalized eigenvector u(k) is
i
ill-defined, but the physical object is the projector P(k). Thus the relevant requirement is not
| that u(k) | be  | continuous, | but | that |      |     |       |     |     |     |     |      |
| --------- | --- | ----------- | --- | ---- | ---- | --- | ----- | --- | --- | --- | --- | ---- |
|           |     |             |     |      |      |     | f(k)f | (k) |     |     |     |      |
|           |     |             |     |      | P(k) | =   |       | †   |     |     |     | (12) |
f (k)f(k)
†
| can be | continuously | extended |     | to k | = k | .   |     |     |     |     |     |     |
| ------ | ------------ | -------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
i
| Equivalently, |     | the | projective | vector   |       |       |       |      |      |     |     |      |
| ------------- | --- | --- | ---------- | -------- | ----- | ----- | ----- | ---- | ---- | --- | --- | ---- |
|               |     |     |            |          |       |       |       |      | N 1  |     |     |      |
|               |     |     |            | [f 1 (k) | : f 2 | (k) : | : f N | (k)] | CP − |     |     | (13) |
|               |     |     |            |          |       |       | ···   | ∈    |      |     |     |      |
must have a direction-independent limit as k k . This is the precise meaning of projector
i
→
| continuity | at  | the common | zero. |     |     |     |     |     |     |     |     |     |
| ---------- | --- | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 Ratio  |     | criterion  |       |     |     |     |     |     |     |     |     |     |
A useful explicit criterion is the following. Suppose that, at the limiting projective vector, the
j-th component is nonzero. Then P(k) is continuous at k if and only if all ratios
i
f α (k)
|     |     |     |     |     |     | r (k) | =   |     |     |     |     | (14) |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | ---- |
α
f j (k)
| have finite, | direction-independent |     |     |     | limits | as k | k :   |     |     |     |     |      |
| ------------ | --------------------- | --- | --- | --- | ------ | ---- | ----- | --- | --- | --- | --- | ---- |
|              |                       |     |     |     |        |      | → i   |     |     |     |     |      |
|              |                       |     |     |     | f      | (k)  |       |     |     |     |     |      |
|              |                       |     |     |     | lim    | α    | = ℓ , | ℓ = | 1.  |     |     | (15) |
|              |                       |     |     |     |        |      | α     | j   |     |     |     |      |
|              |                       |     |     |     | k ki f | (k)  |       |     |     |     |     |      |
→ j
2

| Then the | limiting | projector |     | is  |     |     |     |     |     |     |     |     |
| -------- | -------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ℓ ℓ
|     |     |     |     |     |     | P(k | i ) = | | 〉〈 |. |     |     |     | (16) |
| --- | --- | --- | --- | --- | --- | --- | ----- | ------- | --- | --- | --- | ---- |
ℓ ℓ
〈 | 〉
| Equivalently, |     | in  | component |     | form, |     |        |     |     |     |     |      |
| ------------- | --- | --- | --------- | --- | ----- | --- | ------ | --- | --- | --- | --- | ---- |
|               |     |     |           |     |       |     | f (k)f | (k) |     |     |     |      |
|               |     |     |           |     |       |     | α      | β∗  |     |     |     |      |
|               |     |     |           |     |       | lim |        |     |     |     |     | (17) |
2
|            |     |           |     |      |     | k ki     |     | f γ (k) |      |     |     |     |
| ---------- | --- | --------- | --- | ---- | --- | -------- | --- | ------- | ---- | --- | --- | --- |
|            |     |           |     |      |     | →        | γ|  |         | |    |     |     |     |
| must exist | for | all α,β.  |     |      |     |          | 󰁓   |         |      |     |     |     |
| 4 Local    |     | expansion |     | near |     | a common |     |         | zero |     |     |     |
Let
|            |     |     |        |     | k = | k +q, |     | q = | (q ,q ), |     |     | (18) |
| ---------- | --- | --- | ------ | --- | --- | ----- | --- | --- | -------- | --- | --- | ---- |
|            |     |     |        |     |     | i     |     |     | x y      |     |     |      |
| and expand | Eq. | (2) | around | k : |     |       |     |     |          |     |     |      |
i
|     |     |     | f (k | +q) = |     | c eiki· | Reiq | R   |     |     |     | (19) |
| --- | --- | --- | ---- | ----- | --- | ------- | ---- | --- | --- | --- | --- | ---- |
|     |     |     | α i  |       |     | αR      | ·    |     |     |     |     |      |
R
󰁛
1
|     |     |     |     |     |     | eiki· | R    |     |       | R)2+ |     |      |
| --- | --- | --- | --- | --- | --- | ----- | ---- | --- | ----- | ---- | --- | ---- |
|     |     |     |     | =   |     | c     | 1+iq |     | R (q  |      | .   | (20) |
|     |     |     |     |     |     | αR    |      | ·   | − 2 · |      | ··· |      |
|     |     |     |     |     | R   |       | 󰀗    |     |       |      | 󰀘   |      |
󰁛
| At a common |     | zero, | the zeroth-order |     |     | term | vanishes: |     |     |     |     |      |
| ----------- | --- | ----- | ---------------- | --- | --- | ---- | --------- | --- | --- | --- | --- | ---- |
|             |     |       |                  |     |     |      | eiki·     | R   |     |     |     |      |
|             |     |       |                  |     |     |      | c         | =   | 0.  |     |     | (21) |
αR
R
󰁛
| The homogeneous |     | terms |     | are |     |     |     |     |     |     |     |     |
| --------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1
|     |     |     |     |     | F (n)(q) | =   | c   | eiki· | R(iq R)n, |     |     | (22) |
| --- | --- | --- | --- | --- | -------- | --- | --- | ----- | --------- | --- | --- | ---- |
|     |     |     |     |     | α        |     |     | αR    |           |     |     |      |
|     |     |     |     |     |          | n!  |     |       | ·         |     |     |      |
R
󰁛
so that
F(n)(q),
|     |     |     |     |     | f   | (k +q) | =   |     |     |     |     | (23) |
| --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | ---- |
|     |     |     |     |     | α   | i      |     |     | α   |     |     |      |
|     |     |     |     |     |     |        | n   | mi  |     |     |     |      |
󰁛≥
| where m | is the | lowest | order | for | which | the | vector |     |     |     |     |     |
| ------- | ------ | ------ | ----- | --- | ----- | --- | ------ | --- | --- | --- | --- | --- |
i
|                    |               |       |       | F(mi)(q) |        |     | (mi)      |     | (mi) | T   |     |      |
| ------------------ | ------------- | ----- | ----- | -------- | ------ | --- | --------- | --- | ---- | --- | --- | ---- |
|                    |               |       |       |          |        | = F | (q),...,F |     | (q)  |     |     | (24) |
|                    |               |       |       |          |        |     | 1         |     | N    |     |     |      |
|                    |               |       |       |          |        | 󰀃   |           |     |      | 󰀄   |     |      |
| is not identically |               | zero. |       |          |        |     |           |     |      |     |     |      |
| For                | a first-order |       | zero, | m =      | 1, and |     |           |     |      |     |     |      |
i
|     |     |     |     | f   | (k +q) | = A | q    | +A  | q +O(q2), |     |     | (25) |
| --- | --- | --- | --- | --- | ------ | --- | ---- | --- | --------- | --- | --- | ---- |
|     |     |     |     | α   | i      |     | αx x | αy  | y         |     |     |      |
with
|     |     |     |     |     |     |     | eiki· | R,  |          |     |     |      |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | -------- | --- | --- | ---- |
|     |     |     |     | A   | = i | R   | c     |     | µ = x,y. |     |     | (26) |
|     |     |     |     |     | αµ  | µ   | αR    |     |          |     |     |      |
R
󰁛
Let
|     |     |     | A   | = (A | ,...,A | )T, |     | A = | (A ,...,A |     | )T. | (27) |
| --- | --- | --- | --- | ---- | ------ | --- | --- | --- | --------- | --- | --- | ---- |
|     |     |     | x   |      | 1x     | Nx  |     | y   | 1y        | Ny  |     |      |
Then
|     |     |     |     |     | f(k +q) | =   | A q | +A q | +O(q2). |     |     | (28) |
| --- | --- | --- | --- | --- | ------- | --- | --- | ---- | ------- | --- | --- | ---- |
|     |     |     |     |     | i       |     | x x | y    | y       |     |     |      |
3

| 5 Projector |     | continuity |     |     | for | a first-order |     | zero |     |
| ----------- | --- | ---------- | --- | --- | --- | ------------- | --- | ---- | --- |
For a first-order zero, the leading direction of f(k +q) is determined by
i
|     |     |     |     |     |     | A x q x | +A y | q y . | (29) |
| --- | --- | --- | --- | --- | --- | ------- | ---- | ----- | ---- |
If A x and A y are not complex-linearly dependent, the direction of f depends on how one
| approaches | q = | 0. For example, |     |     |         |     |     |       |      |
| ---------- | --- | --------------- | --- | --- | ------- | --- | --- | ----- | ---- |
|            |     |                 |     | q   | = (ρ,0) |     | [f] | [A ], | (30) |
x
|     |     |     |     |     |     | ⇒   |     | →   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
whereas
|     |     |     |     | q   | = (0,ρ) |     | [f] | [A ]. | (31) |
| --- | --- | --- | --- | --- | ------- | --- | --- | ----- | ---- |
|     |     |     |     |     |         | ⇒   |     | → y   |      |
If [A ] = [A ], the projective limit does not exist, and the projector is discontinuous.
| x ∕       | y   |              |             |     |            |     |           |     |      |
| --------- | --- | ------------ | ----------- | --- | ---------- | --- | --------- | --- | ---- |
| Therefore |     | the explicit | first-order |     | continuity |     | condition | is  |      |
|           |     |              |             |     |            | A   | A .       |     | (32) |
|           |     |              |             |     |            | x   | y         |     |      |
󰀂
Equivalently,
|         |          |     |     | A A |     | A A  | =     | 0, α,β. | (33) |
| ------- | -------- | --- | --- | --- | --- | ---- | ----- | ------- | ---- |
|         |          |     |     | αx  | βy  | αy   | βx    |         |      |
|         |          |     |     |     | −   |      |       | ∀       |      |
| This is | the same | as  |     |     |     |      |       |         |      |
|         |          |     |     |     |     | rank | (A) = | 1,      | (34) |
C
| where A | is the | N 2 matrix |     | with | entries | A   | .   |     |     |
| ------- | ------ | ---------- | --- | ---- | ------- | --- | --- | --- | --- |
αµ
×
When Eq. (33) holds, there exist a nonzero vector v C N and two complex numbers
i
∈
| m ,m | C such | that |     |     |     |     |     |     |     |
| ---- | ------ | ---- | --- | --- | --- | --- | --- | --- | --- |
x y ∈
|     |     |     |     | A   | x = v | i m x , | A   | y = v i m y . | (35) |
| --- | --- | --- | --- | --- | ----- | ------- | --- | ------------- | ---- |
Hence
|            |        |     | f(k | +q) | =     | v (m | q +m | q )+O(q2). | (36) |
| ---------- | ------ | --- | --- | --- | ----- | ---- | ---- | ---------- | ---- |
|            |        |     |     | i   |       | i x  | x    | y y        |      |
| The scalar | factor |     |     |     |       |      |      |            |      |
|            |        |     |     |     | h (q) | = m  | q +m | q          | (37) |
|            |        |     |     |     | i     |      | x x  | y y        |      |
is the local object that carries the phase winding of the common zero.
| 6 Third   |         | condition: |     | nonzero    |     | scalar |     | winding |     |
| --------- | ------- | ---------- | --- | ---------- | --- | ------ | --- | ------- | --- |
| The local | winding | number     | of  | the common |     | zero   | is  |         |     |
1
|     |     |     |     |     | w = |     | dargh | (q), | (38) |
| --- | --- | --- | --- | --- | --- | --- | ----- | ---- | ---- |
|     |     |     |     |     | i   |     |       | i    |      |
2π
󰁌∂Di
| where ∂D | i is a        | small counterclockwise |     |     | loop  | around |      | k i . |      |
| -------- | ------------- | ---------------------- | --- | --- | ----- | ------ | ---- | ----- | ---- |
| For      | a first-order | zero,                  |     |     |       |        |      |       |      |
|          |               |                        |     |     | h (q) | = m    | q +m | q .   | (39) |
|          |               |                        |     |     | i     |        | x x  | y y   |      |
Set
|     |     |     |     | q x | = ρcosφ, |     | q y | = ρsinφ. | (40) |
| --- | --- | --- | --- | --- | -------- | --- | --- | -------- | ---- |
Then
|     |     |     |     | h (φ) | = ρ(m |     | cosφ+m | sinφ). | (41) |
| --- | --- | --- | --- | ----- | ----- | --- | ------ | ------ | ---- |
|     |     |     |     | i     |       | x   |        | y      |      |
Write
|     |     |     |     | m   | = a+ib, |     | m   | = c+id. | (42) |
| --- | --- | --- | --- | --- | ------- | --- | --- | ------- | ---- |
|     |     |     |     | x   |         |     |     | y       |      |
4

Then
|     |     |     |     |     | Reh |     | a c | q   |      |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
|     |     |     |     |     | i   | =   |     | x . | (43) |
|     |     |     |     |     | Imh |     | b d | q   |      |
|     |     |     |     | 󰀕   | i 󰀖 | 󰀕   | 󰀖󰀕  | y 󰀖 |      |
2 2
| The determinant |     | of this | real map | R   | R   | C         | is  |      |      |
| --------------- | --- | ------- | -------- | --- | --- | --------- | --- | ---- | ---- |
|                 |     |         |          |     | →   | ≃         |     |      |      |
|                 |     |         |          |     | ad  | bc = Im(m |     | m ). | (44) |
|                 |     |         |          |     | −   |           | ∗x  | y    |      |
Therefore:
|     |     |     |     |     | Im(m | m   | ) = | 0   | (45) |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | ---- |
|     |     |     |     |     |      | ∗x  | y ∕ |     |      |
means that the map (q ,q ) h (q) is locally invertible. A small circle in q-space is mapped
|     |     |     | x y | i   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
󰀁→
to an ellipse in the complex h -plane enclosing the origin. Hence the phase of h winds once.
i i
| For | a first-order | zero, |     |     |     |         |     |      |      |
| --- | ------------- | ----- | --- | --- | --- | ------- | --- | ---- | ---- |
|     |               |       |     |     | w = | sgnIm(m |     | m ). | (46) |
|     |               |       |     |     | i   |         | ∗x  | y    |      |
The sign convention depends on the orientation of the loop in momentum space and on the
convention used for the Berry connection. With the convention of Eq. (38), Eq. (46) gives the
| local winding    |     | of the scalar | factor. |      |     |     |     |     |     |
| ---------------- | --- | ------------- | ------- | ---- | --- | --- | --- | --- | --- |
| 7 Interpretation |     |               | of      | Im(m | m   | ) = | 0   |     |     |
∗x y
∕
Thequantitiesm andm arethescalarcoefficientsthatdescribehowthecommonzerochanges
|     |     | x   | y   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
under q and q displacements. In terms of the Fourier coefficients, the first derivatives are
|     | x   | y   |     |     |     |     |      |          |      |
| --- | --- | --- | --- | --- | --- | --- | ---- | -------- | ---- |
|     |     |     |     |     |     |     |      | eiki· R. |      |
|     |     |     |     | A   | =   | i   | R c  |          | (47) |
|     |     |     |     |     | αµ  |     | µ αR |          |      |
R
󰁛
| After the | rank-one | factorization |     |     |     |     |     |     |      |
| --------- | -------- | ------------- | --- | --- | --- | --- | --- | --- | ---- |
|           |          |               |     |     | A   | =   | v m | ,   | (48) |
|           |          |               |     |     |     | αµ  | i,α | µ   |      |
m and m are the scalar versions of the x- and y-direction Fourier moments of the coefficients.
| x    | y             |     |     |     |      |     |     |     |      |
| ---- | ------------- | --- | --- | --- | ---- | --- | --- | --- | ---- |
| Thus | the condition |     |     |     |      |     |     |     |      |
|      |               |     |     |     | Im(m | m   | ) = | 0   | (49) |
|      |               |     |     |     |      | ∗x  | y ∕ |     |      |
means that the x-direction and y-direction moments have a nontrivial relative complex phase.
If they have the same complex phase, then h (q) is essentially real-valued up to an overall phase
i
andcannotwindaroundtheorigin. Iftheirrelativephaseisneither0norπ, thenh (q)becomes
i
| a genuine | complex   | vortex. |     |     |      |     |     |     |      |
| --------- | --------- | ------- | --- | --- | ---- | --- | --- | --- | ---- |
| The       | canonical | example | is  |     |      |     |     |     |      |
|           |           |         |     |     | h(q) | = q | +iq | .   | (50) |
|           |           |         |     |     |      |     | x   | y   |      |
Then
|     |     |     |     |     | m x = | 1,  | m y | = i, | (51) |
| --- | --- | --- | --- | --- | ----- | --- | --- | ---- | ---- |
and
|     |     |     |     |     | Im(m | m   | ) = 1 | > 0. | (52) |
| --- | --- | --- | --- | --- | ---- | --- | ----- | ---- | ---- |
∗x y
| A loop         | around | q = 0 gives |     |                |      |     |       |       |      |
| -------------- | ------ | ----------- | --- | -------------- | ---- | --- | ----- | ----- | ---- |
|                |        |             |     | h(ρcosφ,ρsinφ) |      |     | =     | ρeiφ, | (53) |
| so the winding |        | is +1.      |     |                |      |     |       |       |      |
| By contrast,   |        | if          |     |                |      |     |       |       |      |
|                |        |             |     |                | h(q) | = q | x +2q | y ,   | (54) |
| then m         | = 1,   | m = 2, and  |     |                |      |     |       |       |      |
|                | x      | y           |     |                |      |     |       |       |      |
|                |        |             |     |                | Im(m | m   | ) =   | 0.    | (55) |
|                |        |             |     |                |      | ∗x  | y     |       |      |
The image of a small loop is a line segment on the real axis, not a loop enclosing the origin, so
| the winding | is  | zero. |     |     |     |     |     |     |     |
| ----------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
5

| 8 Putting | the three | conditions |     |     | together |     |     |     |
| --------- | --------- | ---------- | --- | --- | -------- | --- | --- | --- |
For isolated first-order common zeros, the conditions for a nonzero Chern number can be stated
as follows.
Proposition2(First-order-zerocriterion). Letf(k)beafiniteFourier-sumvectorasinEq.(2).
Supposethecommonzerosk i off areisolatedandfirst-order. Thenthefollowingdatadetermine
whether f/ f can yield a nonzero Chern number through continuous projectors:
󰀂 󰀂
| (i) Common | zero condition: |     |     |     |         |      |     |      |
| ---------- | --------------- | --- | --- | --- | ------- | ---- | --- | ---- |
|            |                 |     |     | c   | eiki· R | = 0, | α.  | (56) |
αR
∀
R
󰁛
| (ii) Projector | continuity | condition: |     | define |     |      |          |      |
| -------------- | ---------- | ---------- | --- | ------ | --- | ---- | -------- | ---- |
|                |            |            |     | A( i ) |     |      | eiki· R. |      |
|                |            |            |     | =      | i R | c    |          | (57) |
|                |            |            |     | α µ    |     | µ αR |          |      |
R
󰁛
Then
|     |     |     |       | (i) | (i)   |      |      |      |
| --- | --- | --- | ----- | --- | ----- | ---- | ---- | ---- |
|     |     |     | A(i)A |     | A(i)A | = 0, | α,β. | (58) |
|     |     |     | αx    | βy  | αy βx |      |      |      |
|     |     |     |       | −   |       |      | ∀    |      |
Equivalently,
|               |               |            | A(i) | m(i), |        | A(i) | m(i). |      |
| ------------- | ------------- | ---------- | ---- | ----- | ------ | ---- | ----- | ---- |
|               |               |            |      | = v i |        |      | = v i | (59) |
|               |               |            | x    |       | x      | y    | y     |      |
| (iii) Nonzero | local winding | condition: |      |       |        |      |       |      |
|               |               |            |      |       | (m(i)) | m(i) |       |      |
|               |               |            |      | Im    | ∗      |      | = 0.  | (60) |
|               |               |            |      |       | x      | y    | ∕     |      |
|               |               |            |      | 󰁫     |        | 󰁬    |       |      |
Then
|           |              |        |     |           | (m(i)) |         | m(i)     |      |
| --------- | ------------ | ------ | --- | --------- | ------ | ------- | -------- | ---- |
|           |              |        | w   | i = sgnIm |        |         | ∗ .      | (61) |
|           |              |        |     |           |        | x       | y        |      |
|           |              |        |     |           | 󰁫      |         | 󰁬        |      |
| The total | Chern number | is the | sum | of the    | local  | winding | numbers, |      |
|           |              |        |     | C         | = w    | ,       |          | (62) |
i
i
󰁛
up to the global sign convention for Berry curvature. Therefore, a finite nonzero Chern number
requires
|     |     |     |     |     | w i = | 0.  |     | (63) |
| --- | --- | --- | --- | --- | ----- | --- | --- | ---- |
∕
󰁛 i
| 9 Remarks | on higher-order |     |     | zeros |     |     |     |     |
| --------- | --------------- | --- | --- | ----- | --- | --- | --- | --- |
The first-order criterion above is the cleanest and most useful generic case. For higher-order
zeros, the same logic applies, but the first-derivative matrix A is not sufficient.
Let m be the smallest order for which the homogeneous vector F(mi)(q) is not identi-
i
cally zero. Projector continuity requires that the leading homogeneous vector have a direction-
independent projective limit. A sufficient and natural condition is the factorization
|     |     |     | F(mi)(q) |     | = h (mi) | (q)v | ,   | (64) |
| --- | --- | --- | -------- | --- | -------- | ---- | --- | ---- |
|     |     |     |          |     | i        |      | i   |      |
wherev N isfixedandh ( mi) (q)isascalarhomogeneouspolynomialinq ,q . Moregenerally,
| i C |     |     |      |     |      |     | x y |     |
| --- | --- | --- | ---- | --- | ---- | --- | --- | --- |
| ∈   |     | i   |      |     |      |     |     |     |
|     |     |     | (mi) |     | (mi) |     |     |     |
one must require that all ratios F α (q)/F (q) have direction-independent limits wherever
j
| the denominator | is used. |     |     |     |     |     |     |     |
| --------------- | -------- | --- | --- | --- | --- | --- | --- | --- |
6

| If Eq. | (64) holds, | then | the | local | contribution |     | is  |     |     |
| ------ | ----------- | ---- | --- | ----- | ------------ | --- | --- | --- | --- |
1
|     |     |     |     |     | w = |     | dargh | (q), | (65) |
| --- | --- | --- | --- | --- | --- | --- | ----- | ---- | ---- |
|     |     |     |     |     | i   | 2π  |       | i    |      |
󰁌∂Di
| which may | be an | integer | with | magnitude |     | larger | than | one. |     |
| --------- | ----- | ------- | ---- | --------- | --- | ------ | ---- | ---- | --- |
For example,
|             |         |         |     |     | h(q) | = (q | +iq | )n  | (66) |
| ----------- | ------- | ------- | --- | --- | ---- | ---- | --- | --- | ---- |
|             |         |         |     |     |      |      | x y |     |      |
| has local   | winding | n.      |     |     |      |      |     |     |      |
| 10 Physical |         | picture |     |     |      |      |     |     |      |
AfiniteFouriervectorf(k)isanalogoustoaglobalsectionofalinebundle. Ifitnevervanishes,
it trivializes the line bundle and the Chern number must be zero. Nonzero Chern number
becomes possible only when this section has zeros. At such zeros, the normalized vector is
singular, but the projector can still be smooth if the zero is a common scalar vortex shared by
all components.
| In the | first-order | case, | the | local | structure | is     |          |          |      |
| ------ | ----------- | ----- | --- | ----- | --------- | ------ | -------- | -------- | ---- |
|        |             |       |     | f(k   | i +q)     | v i (m | x q x +m | y q y ). | (67) |
≃
The internal vector v i determines the limiting projector, while the scalar factor
|         |             |     |           |     | h (q) | = m | q +m | q   | (68) |
| ------- | ----------- | --- | --------- | --- | ----- | --- | ---- | --- | ---- |
|         |             |     |           |     | i     | x   | x    | y y |      |
| carries | the vortex. | The | condition |     |       |     |      |     |      |
|         |             |     |           |     | Im(m  | m   | ) =  | 0   | (69) |
|         |             |     |           |     |       | ∗x  | y    |     |      |
∕
means that the q and q variations enter the scalar zero with a nontrivial relative complex
|        |               | x                 | y     |          |         |              |     |        |      |
| ------ | ------------- | ----------------- | ----- | -------- | ------- | ------------ | --- | ------ | ---- |
| phase. | This is the   | same              | local | geometry | as      | the familiar |     | vortex |      |
|        |               |                   |       |          |         | q x +iq      | y . |        | (70) |
| Thus   | the intuitive | coefficient-space |       |          | picture | is:          |     |        |      |
TheFouriercoefficientsmustbearrangedsothatthevectorf(k)vanishesatisolated
points, all components vanish through a common scalar factor so that the projector
remainscontinuous,andthiscommonscalarfactormusthavenonzerophasewinding
| around       | the             | zero.  |           |           |     |       |      |     |      |
| ------------ | --------------- | ------ | --------- | --------- | --- | ----- | ---- | --- | ---- |
| 11 Checklist |                 | for    | practical |           |     | use   |      |     |      |
| Given a      | finite Fourier  | vector |           | f(k):     |     |       |      |     |      |
| 1. Solve     | the common-zero |        |           | equations |     |       |      |     |      |
|              |                 |        |           |           |     | eiki· | R    |     |      |
|              |                 |        |           |           |     | c     | = 0, | α.  | (71) |
|              |                 |        |           |           |     | αR    |      | ∀   |      |
R
󰁛
| 2. At | each isolated |     | zero k | , compute |     |     |     |     |     |
| ----- | ------------- | --- | ------ | --------- | --- | --- | --- | --- | --- |
i
|     |     |     |     |     | A(  | i )   |       | eiki· R. |      |
| --- | --- | --- | --- | --- | --- | ----- | ----- | -------- | ---- |
|     |     |     |     |     | α   | µ = i | R µ c | αR       | (72) |
|     |     |     |     |     |     | 󰁛     | R     |          |      |
7

| 3. For a first-order | zero, check |       |       |          |      |
| -------------------- | ----------- | ----- | ----- | -------- | ---- |
|                      |             | (i)   |       | (i)      |      |
|                      |             | A(i)A | A(i)A | = 0 α,β. | (73) |
|                      |             | αx βy | αy    | βx       |      |
|                      |             |       | −     | ∀        |      |
If this fails, the rank-one projector is generally discontinuous at that zero.
| 4. If the rank-one | condition         | holds, factorize |         |                |      |
| ------------------ | ----------------- | ---------------- | ------- | -------------- | ---- |
|                    |                   | A(i) =           | v m(i), | A(i) = v m(i). | (74) |
|                    |                   | x                | i x     | y i y          |      |
| 5. Compute         | the local winding |                  |         |                |      |
|                    |                   |                  |         | (m(i)) m(i)    |      |
|                    |                   | w i =            | sgnIm   | ∗ .            | (75) |
x y
󰁫 󰁬
| 6. Sum all | local contributions: |     |     |     |      |
| ---------- | -------------------- | --- | --- | --- | ---- |
|            |                      |     | C = | w . | (76) |
i
󰁛 i
A nonzero sum gives a finite nonzero Chern number, subject to the Berry-curvature sign
convention.
| 12 Important | caveats |     |     |     |     |
| ------------ | ------- | --- | --- | --- | --- |
(i) The first-order matrix condition is valid only when the zero is genuinely first-order. If
the first derivatives vanish or the leading behavior is higher order, one must analyze the
| lowest | nonzero homogeneous | term. |     |     |     |
| ------ | ------------------- | ----- | --- | --- | --- |
(ii) A common zero alone is not sufficient. Without projector continuity, the object may fail
| to define | a smooth rank-one | band | projector | at the zero. |     |
| --------- | ----------------- | ---- | --------- | ------------ | --- |
(iii) ProjectorcontinuityaloneisnotsufficientfornonzeroChernnumber. Thecommonscalar
| zero must | also have nonzero | winding. |     |     |     |
| --------- | ----------------- | -------- | --- | --- | --- |
(iv) The total Chern number is the sum of all local windings. Multiple zeros may cancel one
another.
(v) The overall sign depends on the convention for the Berry connection, the orientation of
| the Brillouin | zone, and | the definition | of local | winding. |     |
| ------------- | --------- | -------------- | -------- | -------- | --- |
8
