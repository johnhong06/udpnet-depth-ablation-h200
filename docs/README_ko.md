# GT depth 모델 H200 실행 안내

depth prior ablation 세 모델(dav2 / gt / const) 중 **gt**를 항공드론 H200에서 돌리기 위한 저장소입니다.
dav2와 const는 연구실 데스크탑에서 같은 코드·같은 설정으로 돌립니다.
H200 서비스 자체의 사용법(신청, 이슈, 경로 규칙)은 별도 정리된 "Aerodrone H200 사용법" 페이지를 따릅니다.

## 1. 데이터 전달 (관리자에게)

`udpnet_gt_data.tar.part_00` ~ `part_26` (27개, 각 2 GiB, 합계 57 GB) + `SHA256SUMS_parts`를
`/app/data/<아무 폴더>`에 올려 달라고 요청합니다. 드라이브 폴더째 받아 zip으로 묶여 와도 스크립트가 처리합니다.
`/app/data`가 읽기 전용이어도 됩니다 (실행 중 `/app/scratch`에 풀고, `/app/output`에는 풀지 않음).

## 2. 요청서

| 항목 | 값 |
|---|---|
| GitHub 링크 | 이 저장소 |
| 사용할 이미지 | `pytorch/pytorch:latest` (패키지는 스크립트가 시작할 때 설치) |
| GPU 할당량 | **`7`** (H200 한 장). `1`(16 GB 조각)은 batch 6에 메모리가 부족하고 속도도 1/7 |
| 코드 실행 명령어 | 아래 순서대로 한 줄씩 |

| 순서 | 명령 | 시간 | 확인할 것 |
|---|---|---|---|
| 1 | `bash run_gt.sh check` | 1분 | `/app/data`에서 조각을 찾았는지, `/app/scratch` 여유 공간(57 GB 이상), GPU 메모리 |
| 2 | `bash run_gt.sh smoke` | 15분 정도 | 에러 없이 끝나는지, 마지막 줄의 `ms/optimizer step`과 예상 시간 |
| 3 | `bash run_gt.sh full` | 예상 시간대로 | 30분마다 나오는 `[hb]` 줄(진행률, 남은 시간, 건너뛴 batch, val 최고값) |

**const 모델**도 같은 데이터로 돌릴 수 있습니다 (128 depth는 스크립트가 만듦, 추가 데이터 없음):
`bash run_gt.sh smoke const` → `bash run_gt.sh full const` → 결과 `/app/output/udpnet_const/results_am_v2_lr1e-4_const.zip`.
이어 하기용 checkpoint 이름은 `am_v2_lr1e-4_const_last.ckpt`.

끝나면 관리자에게 `/app/output/udpnet_gt/results_am_v2_lr1e-4_gt.zip`을 요청합니다
(결과 JSON, 로그, TensorBoard, val 최고 checkpoint와 last.ckpt가 들어 있음).

## 3. 스크립트가 알아서 하는 것

- **멈춤 감시**: 학습 로그가 30분 동안 한 줄도 안 늘면 죽이고 `last.ckpt`에서 다시 시작 (최대 10번). 메모리 부족이면 재시도하지 않고 멈춤
- **시간 제한**: 작업 시작 후 34시간 − 45분이 되면 학습을 멈추고, 그때까지의 val 최고 checkpoint로 평가 + 결과 zip까지 끝냄
  (예전에 35시간 만에 강제 종료된 적이 있어서). 이어서 하려면 zip 안의 `last.ckpt`를
  `/app/data/<폴더>/am_v2_lr1e-4_gt_last.ckpt`로 올려 달라고 한 뒤 `bash run_gt.sh full`을 다시 요청 → 거기서 이어 감
- **화면 출력**: 65,000자 제한 때문에 요약만. 전체 로그는 `/app/output/udpnet_gt/` 안의 파일
- `torch.compile`이 실패하면 끄고 다시 시도 (계산은 같은 fp32, 속도만 느려짐)

## 4. 설정 (세 모델 공통, 바꾸지 말 것)

lr 1e-4, batch 6 × accumulation 2 = 실질 12, gradient clip 0.01, fp32, 256² 패치, 300만 장(66 epoch, 251,130 step),
val split으로만 checkpoint 선택(512 중앙 crop), 평가는 원본 해상도. lr을 논문값 4e-4에서 1e-4로 낮춘 이유는
README의 "Stability" 참고 (4e-4는 이 데이터에서 발산).

## 5. 시간 참고

연구실 RTX PRO 4500 Blackwell에서 730 ms/step × 251,130 step ≈ 52시간(검증 포함). H200 한 장은 이보다 빠를 것으로
예상하지만 실측은 `smoke` 결과로 확인. 34시간 안에 안 끝나는 속도면 시간 제한에서 멈추고 한 번 더 이어서 돌리면 됨.
