**기본 예제 설정의 단일 실험 결과**

2026-09-23 17:54 EDT에 `robust_ip_estimation` conda 환경에서 실행했다.
`experiments.run_experiment`를 호출하여 세 estimator를 각각 한 번 실행했다.
아래 명령으로 같은 설정을 실행할 수 있다. Runtime은 재실행 시 달라질 수 있다.

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0.05 --nu 5 --s 2 --delta 0.05 \
    --dim 4 --scale 1 --seed 42
```

실험 설정은 다음과 같다. CLI에서 필수인 인자들은 현재 README의 예제 값을 사용했다.

| 설정 | 값 |
| --- | --- |
| Sample size `n` | 200 |
| Dimension `d` | 4 |
| Sparsity `s` | 2 |
| Contamination rate `epsilon` | 0.05 |
| Degrees of freedom `nu` | 5 |
| Failure probability `delta` | 0.05 (confidence 0.95) |
| t shape matrix | `scale * I_d = I_4` |
| Seed | 42 |
| 반복 횟수 | 1 |
| Contamination | `adversarial_sparse_contamination`, `strength=3`, 10개 행 변경 |
| 공통 block 수 `K` | 21 |
| Outer tolerance | 0.591607978310 |
| Algorithm 1 inner / separation tolerance | 0.147901994577 / 0.073950997289 |
| Brute-force net | Euclidean 1/4-covering net, 3,129개 방향 |

Clean data는 multivariate t distribution에서 생성했다. `loc ~ Uniform(1, 3)`을
한 번 추출하고, 무작위로 선택한 s개 좌표에 같은 값을 넣었다.
이번 실행의 `loc = 1.749080237694725`이고 true mean은
`mu = (0, 1.749080237694725, 0, 1.749080237694725)`이다.
오염 함수에는 true support를 전달하지 않았다.

Clean covariance와 알고리즘에 전달한 bound는
`Sigma = (nu / (nu - 2)) * I_4 = (5/3) * I_4`,
`lambda_upper = 2 * lambda_max(Sigma) = 10/3`이다.
`C=2`로 두고 `2 * max(s*log(d/s), epsilon*n, log(1/delta))` 이상의
최소 홀수를 K로 선택했다. 세 방법은 같은 오염 데이터와 seed에 따른 동일한
block 분할을 사용했다. `tol = sqrt(K * lambda_upper / n)`이다.

| Method | L2 error | Support recovery ratio | Runtime (seconds) |
| --- | ---: | ---: | ---: |
| Brute-force (covering net) | 0.200608212 | 1.000000 | 0.035337458 |
| Algorithm 1 | 0.136816711 | 1.000000 | 0.117660542 |
| Coordinate-wise MoM + hard thresholding | 0.136816711 | 1.000000 | 0.000126708 |

L2 error는 반환된 estimator에 대한 `||mu_hat - mu||_2`이고, support recovery는
true support 중 `abs(mu_hat_j) > 1e-8`인 좌표의 비율이다.
Runtime에는 각 estimator의 block 생성, net 생성 및 projection, 모델 구성과
solver 실행이 포함된다. 공통 데이터 생성·오염, 패키지 import, metric 계산은 제외했다.
실행 순서는 표의 순서와 같고, 별도 warm-up이나 반복 평균은 사용하지 않았다.

두 최적화 방법 모두 `converged=True`, outer iteration 1회로 종료했다.
Brute-force의 최종 gap은 0.494644059071, Algorithm 1은 0.553192134085로,
둘 다 기본 tolerance 0.591607978310 이하였다. 이 gap들은 각각 finite-net
objective와 continuous-direction objective에 대한 값이다.
Algorithm 1은 SOCP 2회, separation oracle 3회를 실행했다.

이번 실행에서 Algorithm 1은 초기 MoM 추정치를 그대로 반환했으므로,
coordinate-wise MoM + hard thresholding과 추정치 및 error가 동일하다.
공통 추정치는 `(0, 1.8849566340871564, 0, 1.7330672011458903)`이다.
이 seed에서는 두 방법의 L2 error가 brute-force보다 작았고, MoM의 runtime이 가장 짧았다.
위 수치는 seed 42에서 얻은 단일 실행 결과이다.

실행 환경: macOS 15.7.3 arm64, Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1,
gurobipy 13.0.3, CVXPY 1.9.3, Clarabel 0.11.1.

---

**Contamination rate를 0.2로 올린 단일 실험 결과**

2026-09-23 18:08 EDT에 같은 환경에서 실행했다. 앞선 실험에서 epsilon만
0.05에서 0.2로 변경했다. 나머지는 `n=200, d=4, s=2, nu=5, delta=0.05,
scale=1, seed=42`이고, 공격 강도는 `strength=3`이다. 별도로 tolerance를
줄이지 않고 기존 기본 규칙을 사용했다.

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0.2 --nu 5 --s 2 --delta 0.05 \
    --dim 4 --scale 1 --seed 42
```

Seed와 데이터 생성 설정이 같으므로 clean data와 true mean은 앞선 실험과
동일하다. `lambda_upper=10/3`과 net 크기 3,129도 동일하다.
Epsilon에 따라 오염 행 수, block 수, 기본 tolerance는 다음처럼 달라졌다.

| 설정 | epsilon=0.05 | epsilon=0.2 |
| --- | ---: | ---: |
| 오염 행 수 | 10 | 40 |
| Block 수 K | 21 | 81 |
| Block별 sample 수 | 9 또는 10 | 2 또는 3 |
| Outer tolerance | 0.591607978310 | 1.161895003862 |
| Algorithm 1 inner tolerance | 0.147901994577 | 0.290473750966 |
| Algorithm 1 separation tolerance | 0.073950997289 | 0.145236875483 |

아래는 epsilon=0.2에서 각 estimator를 한 번 실행한 결과다. Metric과 runtime의
정의 및 측정 범위는 앞선 실험과 같다.

| Method | L2 error | Support recovery ratio | Runtime (seconds) |
| --- | ---: | ---: | ---: |
| Brute-force (covering net) | 0.466492737 | 1.000000 | 0.038921708 |
| Algorithm 1 | 0.382506009 | 1.000000 | 5.821839542 |
| Coordinate-wise MoM + hard thresholding | 0.382506009 | 1.000000 | 0.000308958 |

두 최적화 방법 모두 `converged=True`, outer iteration 1회로 종료했다.
Brute-force의 최종 gap은 0.597043169522, Algorithm 1은 0.774038915264로,
둘 다 기본 tolerance 1.161895003862 이하였다.
Algorithm 1은 SOCP 4회, separation oracle 5회를 실행했다.

이번에도 Algorithm 1은 초기 MoM 추정치를 그대로 반환했다. 공통 추정치는
`(0, 1.5543260529418896, 0, 1.4198665468345826)`이다. Brute-force의 추정치는
`(0, 1.5863520026492144, 0, 1.3118903529866008)`이다.

MoM의 L2 error는 epsilon=0.05의 0.136816711에서 0.382506009로 증가했지만,
세 방법 모두 true support를 복구했다. 현재 기본 tolerance에서는 Algorithm 1이
MoM initialization을 개선하지 않았다. Epsilon 증가에 따라 종료 tolerance도
커졌으므로, 이 결과만으로 더 엄격한 최적화에서도 개선이 없는지는 판단할 수 없다.

---

**epsilon=0.2에서 e_tol을 기본값의 1/4로 줄인 결과**

2026-09-23 18:10 EDT에 같은 환경에서 각 방법을 한 번씩 실행했다.
`n=200, d=4, s=2, epsilon=0.2, nu=5, delta=0.05, scale=1, seed=42`와
`strength=3`을 유지했다. Clean data, 오염 데이터, true mean, K=81과 block
분할은 직전 실험과 동일하다. `lambda_upper=10/3`, net 크기는 3,129이다.

Outer tolerance를 `sqrt(K * lambda_upper / n) / 4`로 지정했다.
Algorithm 1의 inner/separation tolerance도 기존 비율에 따라 줄였다.
Coordinate-wise MoM에는 최적화 종료 조건이 없으므로 추정치는 영향을 받지 않는다.

| 설정 | 직전 실험 | 이번 실험 |
| --- | ---: | ---: |
| Outer e_tol | 1.161895003862 | 0.290473750966 |
| Algorithm 1 inner tolerance | 0.290473750966 | 0.072618437741 |
| Algorithm 1 separation tolerance | 0.145236875483 | 0.036309218871 |

`experiments.py`에 추가한 `--tol` 옵션으로 재실행할 수 있다. 옵션을 생략하면
기존 기본 tolerance를 사용한다.

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0.2 --nu 5 --s 2 --delta 0.05 \
    --dim 4 --scale 1 --seed 42 --tol 0.2904737509655563
```

| Method | L2 error | Support recovery ratio | Runtime (seconds) | Outer iterations |
| --- | ---: | ---: | ---: | ---: |
| Brute-force (covering net) | 0.466492737 | 1.000000 | 0.052463375 | 2 |
| Algorithm 1 | 0.519446091 | 1.000000 | 16.299130834 | 3 |
| Coordinate-wise MoM + hard thresholding | 0.382506009 | 1.000000 | 0.000321083 | — |

두 최적화 방법 모두 `converged=True`로 종료했다. Brute-force의 objective
upper bound는 0.597043169522, lower bound는 0.443033975081,
gap은 0.154009194442였다. Algorithm 1의 objective upper bound는
0.676986178957, lower bound는 0.507170383781, gap은 0.169815795176이었다.
두 gap 모두 새 tolerance 0.290473750966 이하다.
Algorithm 1은 SOCP 16회, separation oracle 17회를 실행했고, 생성한 (S,B) 제약은 14개였다.

Brute-force와 MoM의 반환 추정치는 직전 실험과 동일하다. Algorithm 1의 반환
추정치는 `(0, 1.6120494876919274, 0, 1.248034516200677)`로 바뀌었으나,
L2 error는 0.382506009에서 0.519446091로 증가했다. 이번에는 초기화에서
이동했지만 MoM보다 정확해지지는 않았다. 최적화 gap을 줄이는 것이 true mean에
대한 L2 error 감소를 보장하지 않음을 보여주는 단일 seed의 결과이다.
Runtime 측정 범위와 metric 정의는 앞선 실험과 동일하다.

---

**epsilon=0.2에서 e_tol을 기본값의 1/10로 줄인 결과**

2026-09-23 18:11 EDT에 같은 환경에서 각 방법을 한 번씩 실행했다.
`n=200, d=4, s=2, epsilon=0.2, nu=5, delta=0.05, scale=1, seed=42`,
`strength=3`을 유지하고, 원래 기본 tolerance의 1/10을 지정했다.
데이터, true mean, K=81, block 분할, `lambda_upper=10/3`과 net 크기 3,129는
앞선 epsilon=0.2 실험들과 동일하다.

| 설정 | 값 |
| --- | ---: |
| Outer e_tol = sqrt(K * lambda_upper / n) / 10 | 0.116189500386 |
| Algorithm 1 inner tolerance | 0.029047375097 |
| Algorithm 1 separation tolerance | 0.014523687548 |

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0.2 --nu 5 --s 2 --delta 0.05 \
    --dim 4 --scale 1 --seed 42 --tol 0.11618950038622251
```

| Method | L2 error | Support recovery ratio | Runtime (seconds) | Outer iterations |
| --- | ---: | ---: | ---: | ---: |
| Brute-force (covering net) | 0.466492737 | 1.000000 | 0.069373750 | 3 |
| Algorithm 1 | 0.484675902 | 1.000000 | 16.983899333 | 3 |
| Coordinate-wise MoM + hard thresholding | 0.382506009 | 1.000000 | 0.000330083 | — |

두 최적화 방법 모두 `converged=True`로 종료했다. Brute-force의 objective
upper/lower bound는 모두 0.597043169522이며, 보고된 gap은 0이었다.
이는 수치 허용오차 내에서 finite-net objective의 최적성을 인증한 것이다.
Algorithm 1의 objective upper bound는 0.654045811544, lower bound는
0.583658200651, gap은 0.070387610893으로 이번 tolerance 이하였다.
Algorithm 1은 SOCP 18회, separation oracle 19회를 실행했고, 생성한 (S,B) 제약은 16개였다.

Brute-force와 MoM의 반환 추정치는 앞선 실험들과 동일하다. Algorithm 1의 반환
추정치는 `(0, 1.6675969231406946, 0, 1.2713028847203791)`이다.
Algorithm 1의 L2 error는 기본값의 1/4 실험에서 0.519446091이었고,
이번에는 0.484675902로 줄었지만 MoM의 0.382506009보다 여전히 컸다.
Brute-force 역시 finite-net objective의 보고된 gap이 0이어도 MoM보다
L2 error가 컸다. Runtime 측정 범위와 metric 정의는 앞선 실험과 동일하다.

---

**d=6, epsilon=0.2, e_tol=기본값의 1/10 결과**

2026-09-23 18:14 EDT에 같은 환경에서 각 방법을 한 번씩 실행했다.
직전 실험에서 d만 4에서 6으로 변경하고 `n=200, s=2, epsilon=0.2, nu=5,
delta=0.05, scale=1, seed=42`, `strength=3`을 유지했다.
차원이 달라졌으므로 d=6의 clean data와 오염 데이터를 새로 생성했으며,
이 데이터를 세 방법에 공통으로 사용했다.

이번 true mean은
`mu = (0, 1.749080237694725, 0, 0, 0, 1.749080237694725)`이다.
Clean covariance는 `(5/3) * I_6`이고 `lambda_upper=10/3`이다.
오염 행 수는 40개이며, 오염 항이 block 수를 결정하여 K=81과 tolerance는
d=4 실험과 동일하다. 같은 seed의 block 분할을 세 방법에 사용했다.

| 설정 | 값 |
| --- | ---: |
| Outer e_tol = sqrt(K * lambda_upper / n) / 10 | 0.116189500386 |
| Algorithm 1 inner tolerance | 0.029047375097 |
| Algorithm 1 separation tolerance | 0.014523687548 |
| Brute-force 1/4-covering net 크기 | 29,613 |

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0.2 --nu 5 --s 2 --delta 0.05 \
    --dim 6 --scale 1 --seed 42 --tol 0.11618950038622251
```

| Method | L2 error | Support recovery ratio | Runtime (seconds) | Outer iterations |
| --- | ---: | ---: | ---: | ---: |
| Brute-force (covering net) | 0.548250814 | 1.000000 | 0.575751542 | 4 |
| Algorithm 1 | 0.580303281 | 1.000000 | 119.760470500 | 4 |
| Coordinate-wise MoM + hard thresholding | 0.548250814 | 1.000000 | 0.000429042 | — |

두 최적화 방법 모두 `converged=True`로 종료했다. Brute-force의 objective
upper/lower bound는 모두 0.732941497351, 보고된 gap은 0이다.
Algorithm 1의 objective upper bound는 0.769143877174, lower bound는
0.759339909869, gap은 0.009803967305로 이번 tolerance 이하였다.
Algorithm 1은 SOCP 22회, separation oracle 23회를 실행했고,
생성한 (S,B) 제약은 19개였다.

Brute-force는 초기 MoM 추정치를 그대로 반환했다. 공통 추정치는
`(0, 1.3065882748074238, 0, 0, 0, 1.425383128736632)`이고,
Algorithm 1의 추정치는 `(0, 1.2107058349850766, 0, 0, 0, 1.5325048450621082)`이다.
이 seed에서는 d=6에서도 Algorithm 1의 L2 error가 MoM보다 컸다.
Algorithm 1의 runtime은 d=4 실험의 약 17초에서 이번에는 약 120초로 증가했다.
Metric과 runtime 측정 범위는 앞선 실험과 동일하며, 이번 Algorithm 1 실행에는
진행 확인을 위한 oracle 완료 로그 출력이 포함되었다.

---

**d=6, epsilon=0.2에서 block multiplier C=1 결과**

2026-09-23 18:17 EDT에 같은 환경에서 각 방법을 한 번씩 실행했다.
직전 실험의 `n=200, d=6, s=2, epsilon=0.2, nu=5, delta=0.05, scale=1,
seed=42`, `strength=3`을 유지하고 block 수의 계수 C를 2에서 1로 변경했다.
같은 clean data, 40개 행이 변경된 같은 오염 데이터와 true mean을 사용했다.
`lambda_upper=10/3`과 net 크기 29,613은 동일하다.

K는 `C * max(s*log(d/s), epsilon*n, log(1/delta))` 이상의 최소 홀수로
선택했다. `e_tol = sqrt(K * lambda_upper / n) / 10` 규칙을 유지했으므로,
K가 줄어들면서 절대 tolerance도 아래처럼 달라졌다.

| 설정 | C=2 | C=1 |
| --- | ---: | ---: |
| Block 수 K | 81 | 41 |
| Block별 sample 수 | 2 또는 3 | 4 또는 5 |
| 오염점을 하나 이상 포함한 block 수 | 34 | 24 |
| Outer e_tol | 0.116189500386 | 0.082663978451 |
| Algorithm 1 inner tolerance | 0.029047375097 | 0.020665994613 |
| Algorithm 1 separation tolerance | 0.014523687548 | 0.010332997306 |

세 방법이 같은 C와 block 분할을 사용하도록 `C` 인자를 전달했다.
`experiments.py --C`로 계수를 지정할 수 있으며, 생략 시 기본값은 2이다.

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0.2 --nu 5 --s 2 --delta 0.05 \
    --dim 6 --scale 1 --seed 42 --C 1 --tol 0.08266397845091497
```

| Method | L2 error | Support recovery ratio | Runtime (seconds) | Outer iterations |
| --- | ---: | ---: | ---: | ---: |
| Brute-force (covering net) | 0.408800696 | 1.000000 | 0.550885208 | 4 |
| Algorithm 1 | 0.508235271 | 1.000000 | 8.179338667 | 4 |
| Coordinate-wise MoM + hard thresholding | 0.714112107 | 1.000000 | 0.000315042 | — |

두 최적화 방법 모두 `converged=True`로 종료했다. Brute-force의 objective
upper/lower bound는 모두 0.638635075702이고, 보고된 gap은 0이다.
Algorithm 1의 objective upper bound는 0.686872990819, lower bound는
0.681738275637, gap은 0.005134715182로 이번 tolerance 이하였다.
Algorithm 1은 SOCP 15회, separation oracle 16회를 실행했고,
생성한 (S,B) 제약은 12개였다.

각 반환 추정치는 다음과 같다.

| Method | mu_hat |
| --- | --- |
| Brute-force | `(0, 1.4506100143907676, 0, 0, 0, 1.4697354061884196)` |
| Algorithm 1 | `(0, 1.3577375658166768, 0, 0, 0, 1.4248056567391711)` |
| Coordinate-wise MoM + hard thresholding | `(0, 1.176129801527128, 0, 0, 0, 1.3228362948373036)` |

C=2에서 C=1로 변경하자 MoM의 error는 0.548250814에서 0.714112107로
커졌고, 이번에는 두 최적화 방법 모두 MoM보다 error가 작았다.
Algorithm 1의 runtime은 약 119.76초에서 8.18초로 줄었다.
이는 K와 그에 따른 tolerance가 함께 달라진 단일 seed의 결과이다.

이번 C=1 분할에서는 전체 41개 중 24개 block에 오염점이 들어가므로,
오염점이 전혀 없는 block이 과반이라는 조건은 성립하지 않는다.
위 block 수는 실제 교체된 행을 추적해 계산했다.
Metric과 runtime 측정 범위는 앞선 실험과 동일하며, Algorithm 1 실행에는
진행 확인을 위한 oracle 완료 로그 출력이 포함되었다.

---

**모든 최적화 solver를 Gurobi로 통일한 뒤의 검증 실행**

2026-09-23에 brute-force의 LP를 HiGHS에서 Gurobi로, Algorithm 1의
restricted SOCP를 Clarabel에서 Gurobi로 변경했다. Separation oracle과
outer cutting-plane MILP는 기존부터 Gurobi를 사용했다. 앞선 결과는 변경 전
solver의 기록이며, 아래는 변경 후 `experiments.py`를 한 번 실행한 결과다.

`n=200, d=6, s=2, epsilon=0.2, nu=5, delta=0.05, scale=1, seed=42,
C=1`, `strength=3`을 사용했다. K=41, `lambda_upper=10/3`,
`e_tol=sqrt(K*lambda_upper/n)/10=0.08266397845091497`로 직전 실험과
같은 데이터, block 분할과 종료 기준이다. 각 Gurobi 모델은 thread 1개를 사용했다.

```sh
conda run -n robust_ip_estimation python experiments.py \
    --n 200 --epsilon 0.2 --nu 5 --s 2 --delta 0.05 \
    --dim 6 --scale 1 --seed 42 --C 1 --tol 0.08266397845091497
```

| Method | L2 error | Support recovery ratio | Runtime (seconds) |
| --- | ---: | ---: | ---: |
| Brute-force (Gurobi LP) | 0.643197 | 1.000000 | 0.440143 |
| Algorithm 1 (Gurobi SOCP) | 0.513567 | 1.000000 | 6.122471 |
| Coordinate-wise MoM + hard thresholding | 0.714112 | 1.000000 | 0.000209 |

두 최적화 방법 모두 수렴했다. Brute-force의 반환 추정치가 solver 변경에 따라
달라져 별도로 목적값을 확인했다. 이전 HiGHS 해는
`(0, 1.4506100143907676, 0, 0, 0, 1.4697354061884196)`이고,
새 Gurobi 해는 `(0, 1.129382011801307, 0, 0, 0, 1.5768114070515733)`이다.
두 해의 net 목적값은 수치 허용오차 내에서 모두 0.638635075702이며,
Gurobi가 계산한 global lower bound 역시 0.638635075702였다.
따라서 같은 최적 목적값을 갖는 다른 추정치가 선택되어 L2 error가 달라진 경우다.

변경한 구현은 기존 HiGHS/Clarabel 전수 열거 기준의 테스트 8개를 통과했다.
이후 테스트용 최적화 코드도 Gurobi로 변환하고, 해석적으로 계산한
distance-to-box 문제와 비교하는 LP/SOCP dual 검증을 추가했다.
최종 테스트 10개가 모두 통과했다. 위 runtime은 개별 실행 측정값이다.
