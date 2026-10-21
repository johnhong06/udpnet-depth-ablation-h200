# UDPNet depth prior ablation — 항공 영상 안개 제거

## 실험

UDPNet은 안개 이미지와 depth 지도를 함께 받아 안개를 걷어 내는 모델이다. 같은 UDPNet을 **depth 입력만 바꿔 세 번**
학습해서 두 가지를 확인한다.

1. depth가 항공 영상의 안개 제거에 실제로 도움이 되는가
2. DAv2(단안 depth 추정)가 항공 영상에서 틀리는 만큼 성능을 얼마나 잃는가

| 모델 | depth 입력 | 학습 장소 |
|---|---|---|
| dav2 | 안개 이미지에 Depth Anything V2 Large를 돌린 결과 (UDPNet 원래 방식) | 연구실 데스크탑 |
| gt | 실측 depth, dav2와 같은 형식 (1/거리, 0~255) | H200 |
| const | 전부 128 (depth 정보 없음) | H200 |

- **데이터**: AerialMetric 항공 영상에 실측 depth로 안개를 합성했다 (β = 1, 2.5, 6, 15 /km)
- **분할**: 학습 45,664장, val 1,564장, test 3,080장(770프레임), test_heldout 848장(학습에 없는 장소 3곳)
- **설정**: 세 모델 모두 같다. 300만 장(66 epoch), batch 6 × 누적 2, lr 1e-4, fp32
- **지표**: gain = UDPNet 출력 PSNR − 안개 입력 PSNR (dB). 세 모델의 입력이 같아서 gain 차이가 곧 PSNR 차이다

## test와 test_heldout

| | test | test_heldout |
|---|---|---|
| 구성 | AerialMetric 공식 Oblique test 전체 | test 중 학습에 없는 장소 3곳 (Artsci, caliterra-output, polytech) |
| 크기 | 770프레임 (3,080장) | 212프레임 (848장) |
| 장소 | 20곳, 이 중 16곳은 학습에도 있음 | 3곳 모두 학습에 없음 |
| 경사 시점 비율 (pitch < 70°) | 58% | 88% |
| 보는 것 | 아는 장소의 처음 보는 프레임 | 처음 보는 장소로의 일반화 |

test_heldout은 test의 일부다. 두 세트는 시점 구성도 다르다. 학습 데이터는 경사 시점이 25%라서 3배로 반복해 약 50%로
맞췄다. 그래서 test_heldout에서 성능이 떨어지면 새 장소 때문인지 경사 시점 때문인지 함께 따져 봐야 한다.

## 결과

| 모델 | test PSNR | test gain | test_heldout PSNR | test_heldout gain | val 최고 PSNR (epoch) |
|---|---|---|---|---|---|
| 안개 입력 | 17.12 | – | 15.18 | – | – |
| dav2 | | | | | |
| gt | 34.65 | +17.53 | 28.93 | +13.75 | 41.89 (56) |
| const | | | | | |

단위는 dB. SSIM은 gt 기준 test 0.738 → 0.984, test_heldout 0.690 → 0.972 (안개 입력 → 출력).

## 비교

| 비교 | 의미 | 예상 | test | test_heldout |
|---|---|---|---|---|
| dav2 − const | DAv2 depth가 주는 이득 | 양수, 셋 중 가장 작음 | | |
| gt − const | 정확한 depth가 주는 이득 | 양수, 가장 큼 | | |
| gt − dav2 | DAv2 오차로 잃는 성능 | 양수 | | |

예상 순서는 gt > dav2 > const다. 안개를 실측 depth로 만들었으니 gt는 실제보다 유리한 상한선으로 읽는다.

## 실행 (H200)

GPU 할당량 `7`로 요청한다. 명령은 `bash run_gt.sh check` → `smoke` → `full` 순서다. const는 `bash run_gt.sh full const`로
돌린다. 요청서 작성법은 [docs/README_ko.md](docs/README_ko.md), 영어 상세 설명은 [docs/details_en.md](docs/details_en.md)에 있다.

UDPNet 원본: Zuo et al., arXiv 2601.06909 (Apache-2.0). 수정 내역은 [NOTICE.md](NOTICE.md).
