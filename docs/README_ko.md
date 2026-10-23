# H200에서 gt·const·dav2 모델 돌리기

depth prior ablation은 같은 UDPNet을 depth 입력만 바꿔 세 번 학습한다. 이 저장소로 세 모델 모두 항공드론 H200에서
같은 코드, 같은 설정으로 돌린다.

| 모델 | depth 입력 | 데이터 |
|---|---|---|
| gt | 실측 depth (1/거리, 0~255 정규화) | 드라이브에 올린 조각 |
| const | 전부 128 | 같은 조각 (128 PNG는 스크립트가 만든다) |
| dav2 | 안개 이미지에 Depth Anything V2 Large를 돌린 결과 | 같은 조각 (DAv2 depth는 스크립트가 H200에서 만든다) |

신청, 이슈 작성, 컨테이너 경로 같은 서비스 사용법은 "Aerodrone H200 사용법" 페이지를 따른다.

## 1. 데이터 전달

관리자에게 드라이브 폴더 `h200_udpnet_gt`를 공유하고 `/app/data` 아래 아무 폴더에 받아 달라고 요청한다.

| 파일 | 개수 | 크기 |
|---|---|---|
| `udpnet_gt_data.tar.part_00` ~ `part_26` | 27 | 각 2 GiB, 합계 57 GB |
| `SHA256SUMS_parts` | 1 | 체크섬 |
| `MANIFEST.txt` | 1 | 파일 목록과 개수 |
| `README_ADMIN_DRIVE.txt` | 1 | 관리자용 안내 |

- 압축은 풀지 않아도 된다. 스크립트가 체크섬을 확인한 뒤 `/app/scratch`에 푼다.
- 업로드 중에도 `check`는 돌릴 수 있다. 몇 번 조각까지 올라왔는지 보여 준다. `smoke`·`full`은 조각이 다 오기 전에는 체크섬 검사에서 멈춘다.
- `/app/data`가 읽기 전용이어도 되고 드라이브 폴더째 받아 zip으로 묶여 와도 된다.
- `/app/data`와 `/app/output`은 공간을 같이 쓴다. 57 GB가 필요하다.

## 2. 요청서 작성

이슈마다 아래 표대로 채운다. 바뀌는 칸은 **코드 실행 명령어** 하나다.

| 항목 | 입력값 |
|---|---|
| 사용자 ID | `johnhong06` |
| GitHub 링크 | `https://github.com/johnhong06/udpnet-depth-ablation-h200.git` |
| 코드 실행 명령어 | 아래 순서표의 명령 한 줄 |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 | **`7`** (H200 한 장) |

GPU 할당량 `1`(16 GB 조각)은 쓰지 않는다. 이 학습은 batch 6에서 18~20 GiB를 쓰고 조각은 연산 유닛도 1/7이라
너무 느리다. 메모리에 맞추려고 batch를 줄이면 dav2와 조건이 달라져 비교가 공정하지 않다.

### 요청 순서

| 순서 | 코드 실행 명령어 | 걸리는 시간 | 끝나면 확인할 것 |
|---|---|---|---|
| 1 | `bash run_gt.sh check` | 2분 | `data upload: COMPLETE`(조각 27/27)인지, `/app/scratch` 여유 공간 57 GB 이상, GPU 메모리, CPU 코어 수 |
| 2 | `bash run_gt.sh smoke` | 15분 안팎 | 에러 없이 끝났는지, 마지막 줄 `[smoke] … ms/optimizer step -> … h` |
| 3 | `bash run_gt.sh full` | 35시간 안팎 | 30분마다 나오는 `[hb]` 줄, 마지막 결과 표 |
| 4 | `bash run_gt.sh smoke const` | 15분 안팎 | 2번과 같음 |
| 5 | `bash run_gt.sh full const` | 35시간 안팎 | 3번과 같음 |
| 6 | `bash run_gt.sh smoke dav2` | 1시간 안팎 | 2번과 같음 + `[dav2 check] OK` |
| 7 | `bash run_gt.sh full dav2` | 36시간 안팎 | 3번과 같음 + `[dav2 check] OK` |

3번, 5번, 7번은 할당량이 허락하면 동시에 요청해도 된다.

dav2는 학습 전에 안개 이미지 34,972장 각각에 Depth Anything V2 Large를 돌려 depth를 만든다(작업마다 30~40분, 모델은
Hugging Face에서 토큰 없이 받는다). 만든 depth는 로컬에서 만든 것의 통계(200장)와 대조하고, 다르면 학습하지 않고 멈춘다.

## 3. 결과 받기

작업이 끝나면 관리자에게 이슈 번호와 파일 이름을 적어 요청한다.

| 모델 | 받을 파일 |
|---|---|
| gt | `/app/output/udpnet_gt/results_am_v2_lr1e-4_gt.zip` |
| const | `/app/output/udpnet_const/results_am_v2_lr1e-4_const.zip` |
| dav2 | `/app/output/udpnet_dav2/results_am_v2_lr1e-4_dav2.zip` |

zip에는 결과 JSON(test, test_heldout), 학습·평가 로그, TensorBoard 기록, val 최고 checkpoint, `last.ckpt`가 들어 있다.

## 4. 스크립트가 알아서 하는 것

| 상황 | 동작 |
|---|---|
| 학습 로그가 30분 동안 늘지 않음 | 학습을 끝내고 `last.ckpt`에서 다시 시작 (최대 10번) |
| GPU 메모리 부족 | 재시도하지 않고 멈춤 |
| `torch.compile` 실패 | compile을 끄고 다시 시작 (계산은 같은 fp32, 속도만 느려짐) |
| 평가가 4시간 넘게 걸림 | 평가를 끝내고 오류로 표시 |
| torch가 2.6 미만 | torch 2.11 + torchvision 0.26 설치 |

**작업이 중간에 끊겨 이어서 할 때**: 결과 zip 안의 `last.ckpt`를 관리자에게 보내 `/app/data/<폴더>/`에 아래 이름으로
올려 달라고 한 뒤, 같은 `full` 명령을 다시 요청한다. 거기서부터 이어서 학습한다.

| 모델 | 파일 이름 |
|---|---|
| gt | `am_v2_lr1e-4_gt_last.ckpt` |
| const | `am_v2_lr1e-4_const_last.ckpt` |
| dav2 | `am_v2_lr1e-4_dav2_last.ckpt` |

화면 출력은 65,000자 제한 때문에 요약만 나온다. 전체 로그는 `/app/output/udpnet_<모델>/` 안의 파일에 있다.

## 5. 학습 설정

세 모델 모두 같다. 한 모델만 바꾸지 않는다.

| 항목 | 값 |
|---|---|
| 학습량 | 300만 장 (66 epoch, 251,130 optimizer step) |
| batch | 6 × gradient 누적 2 = 실질 12 (원본 UDPNet의 GPU 2장 × 6과 같음) |
| 학습률 | 1e-4 (논문은 4e-4지만 이 데이터에서 발산) |
| gradient clip | 0.01 |
| 정밀도 | fp32 (`torch.compile` 사용) |
| 패치 | 256 × 256 |
| checkpoint 선택 | val split, 512 중앙 crop, 3 epoch마다 |
| 평가 | test, test_heldout을 원본 해상도로 |

학습률을 낮춘 이유는 [details_en.md](details_en.md)의 "Stability"에 적어 두었다.

## 6. 시간 예상

연구실 RTX PRO 4500 Blackwell에서는 730 ms/step × 251,130 step으로 검증까지 합쳐 약 52시간이 걸린다.
H200 첫 `smoke`(worker 6)는 505 ms/step, 전체 약 35시간이었다. 정확한 값은 `smoke`의 마지막 줄로 확인한다.
