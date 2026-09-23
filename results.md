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
