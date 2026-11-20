# 개발 기록

## 0. 뭘 만들어야 했나

- 해커톤 미션은 불규칙하게 놓인 포장물을 PIPER 로봇 팔이 집어서 파티션 상자의 빈 칸에 넣는 거였다.
- 팀은 글로벌 카메라로 작업공간을 보고, 손목 카메라로 가까이서 다시 확인하는 구조를 잡았다. 나는 그중 비전 코드를 맡았다.
- 비전이 로봇에 줘야 하는 건 세 가지였다. 상자가 어디 있는지, 칸이 몇 × 몇이고 각 칸 경계가 어딘지, 칸마다 뭐가 들어 있는지.
- ROS2나 Gazebo 없이 Windows USB 카메라 + OpenCV로 먼저 MVP를 만들고, 카메라 부분은 나중에 ROS2 어댑터로 바꿀 수 있게 `camera/`로 분리해뒀다.

---

## 1. 처음 버전: 꼭짓점 찾고 등간격으로 나누기

- 처음엔 윤곽선을 찾아서 approxPolyDP로 사각형 꼭짓점 4개를 잡고, Homography로 펴서 Top View를 만들었다.
- 칸은 영상 전체의 가로 · 세로 프로파일을 평균내서 "같은 간격으로 나뉜다"고 보고 R × C를 맞췄다.
- 칸 상태는 칸마다 공통 바닥색 · 질감을 추정하고 다르면 OCCUPIED로 봤다. 아이디어 요약서 단계에선 빈 칸 기준 영상이랑 지금 영상을 비교하는 방식도 생각했다.

---

## 2. 실제 상자에서 다 틀어짐

- 실제 4×4 상자를 찍었더니 2×2로 나왔다. 가운데 칸막이 두 개만 또렷하고 나머지가 흐리니까, 등간격 맞추기가 제일 단순한 2×2를 골랐다. 그 뒤 Top View, 칸 상태가 다 2×2 기준으로 틀어졌다.
- 외곽 오른쪽 위 꼭짓점도 잘렸다. 실제로는 x = 760 px 넘는 곳인데 x = 674 px로 잡혔다. 모서리가 조금만 가려져도 approxPolyDP 꼭짓점이 안쪽으로 들어왔다.
- 윗변은 상자 테두리(y ≈ 312)가 아니라 바닥이랑 닿는 그림자 경계(y ≈ 336)로 끌려 내려갔다. 대비가 제일 센 선을 고르니까 그림자가 이겼다.
- 칸 바닥에 생긴 칸막이 그림자도 세로선처럼 보였다.
- 칸막이 윗면이랑 벽면이 같이 보이면 선 하나가 두 줄로 잡혀서 칸 수가 늘기도 했다.

---

## 3. 외곽 다시 잡기

- 꼭짓점을 직접 찾지 않고, 네 변을 각각 직선 피팅해서 서로 교차시켜 꼭짓점을 계산하게 바꿨다. 모서리가 가려져도 변은 대부분 보이니까 꼭짓점이 안 움직인다.
- 외곽 벽을 고를 때는 "종단 점수"를 넣었다. 칸막이는 상자 안쪽에만 있고 바깥에서 끝나니까, 안쪽엔 수직 칸막이가 많고 바깥엔 없는 선이 진짜 외곽이다.
- 오른쪽 위 꼭짓점은 x > 760 px로 돌아왔다.

---

## 4. 등간격 버리고 보이는 경계로만 세기

- 같은 간격이라는 가정 자체가 문제라서 `_fit_divisions` 같은 등간격 맞추기 코드를 아예 지웠다. 테스트에도 그 함수가 다시 생기면 실패하게 넣어뒀다.
- 경계 증거를 다시 만들었다 (`boundary_evidence`). "이 픽셀이 밝냐"가 아니라 "여기 한 방향으로 정렬된 경계가 있냐"를 보는 맵이다.
  - 영상을 Lab으로 바꾸고 밝기는 log(1+L)로 썼다. 그림자는 밝기를 곱해서 줄이니까 log를 하면 빼기가 돼서 기울기에서 대부분 빠진다. 색은 a*, b*를 따로 넣었다(가중치 0.7). 이웃 칸끼리 밝기는 같은데 색만 다른 경우도 잡으려고.
  - 채널마다 Gaussian(σ 1.0)으로 살짝 흐리고 `Scharr`로 gx, gy를 구했다.
  - 구조 텐서: 채널마다 gx², gx·gy, gy²를 더해서 Gaussian(σ 1.5)으로 동네 평균을 낸다. 2×2 행렬의 고유값 λ1 ≥ λ2로
    - 세기 = √λ1
    - 일관성 = (λ1 − λ2) / (λ1 + λ2). 주변 기울기가 한 방향으로 정렬돼 있으면 1, 사방으로 흩어져 있으면 0이다. 칸막이 경계는 1에 가깝고, 바닥 질감이나 잡음은 0에 가깝다.
    - 기울기를 제곱해서 쓰니까 밝→어둡이든 어둡→밝이든 같은 값이 나온다. 칸막이가 바닥보다 밝든 어둡든 상관없게 된 이유가 이거다.
  - 방향 분리: cos2θ = (Jxx − Jyy) / √((Jxx − Jyy)² + 4Jxy²). +면 기울기가 가로 방향이라 세로 경계, −면 가로 경계. 세로 맵 = 세기 × 일관성 × max(cos2θ, 0).
  - 칸막이 윗면은 얇은 띠라 에지 두 줄 사이에 묻힐 때가 있어서 Hessian 항을 더했다. Gaussian(σ 1.6) 후 `Sobel`로 2차 미분 dxx, dyy, dxy를 구하고 |λ1 − λ2| / 2를 띠 응답으로 썼다. 한 방향으로만 휘는 곳(선 모양)에서 커지고, 절댓값이라 밝은 띠 · 어두운 띠 둘 다 잡힌다. 방향은 위와 같은 식으로 나눠서 두 맵 중 큰 값을 썼다.
  - 정규화: 상자 영역 안 세기의 상위 10% 값으로 나눠서 조명이 전체적으로 밝거나 어두워도 같은 기준이 되게 했다. 임계값은 max(0.3 × 상위 0.5% 값, 4 × 잡음(중앙값), 0.08).
  - LSD 직선 검출은 보조로만 썼다. 화면 짧은 변의 3%보다 짧은 선은 버리고, 세로 · 가로로 2 : 1 이상 기운 선만 남긴다. 그리고 이미 임계값의 절반 이상 증거가 있는 픽셀만 임계값 살짝 위로 올려준다. LSD 혼자서는 경계를 만들지 못하게 했다.
- 경계 그래프 (`boundary_graph`)
  - 기울어지거나 휜 곡선 가설마다, 길이 대비 증거가 있는 비율로 점수를 매겼다. 중간에 끊겨도 된다.
  - 좁은 통로 안에서 DP로 실제 곡선을 따라가게 하고, 연속성 · 곡률 패널티를 줘서 칸 안쪽 그림자나 물건 테두리로 튀지 못하게 했다.
  - 칸막이 윗면이랑 벽면처럼 거의 평행한 두 선은 간격 비율로 하나로 합쳤다. 외곽 벽은 안쪽 벽면을 흡수한다.
  - T / X 교차점에서 가로 · 세로 선이 서로 받쳐주는지 점수를 줬다.
  - 선마다 프레임 누적 존재 확률(LineTracker)을 뒀다. 계속 보이면 올라가고 안 보이면 내려간다.
  - 최종적으로 교차점 노드 + 곡선 엣지 + 면(face)으로 된 평면 그래프를 만들고, 모든 면이 단순한 칸일 때만 행 · 열을 확정한다.
- 끊긴 구간은 강한 임계값으로 선을 먼저 채택하고, 이미 확인된 선을 따라서만 약한 임계값(히스테리시스)으로 이어붙였다.
- 증거가 아예 없는 구간은 선을 만들어내지 않기로 했다. 그 대신 두 칸을 합친 채로 두고 `TOPOLOGY UNCERTAIN`을 띄운다. 그럴듯하게 채우는 것보다 모른다고 하는 게 로봇한테는 낫다.
- 행 · 열이 바뀌는 건 3프레임 연속 같은 결과가 나와야 반영한다. 한 프레임 흔들림으로 칸 번호가 바뀌면 로봇이 엉뚱한 칸으로 간다.
- `dense_edges`: 그래프 곡선을 카메라 영상에 다시 투영해서 약 2 \~ 4 px 간격으로 실제 기울기에 맞춘 점을 만들었다. 교차점 계산용 연장선은 내부에서만 쓰고 화면이나 칸 polygon에는 안 넣었다.

---

## 5. 휘어진 칸막이

- 골판지 칸막이는 휘어 있어서, 네 꼭짓점 Homography 하나로는 가운데 칸이 안 맞았다. 단일 Homography는 평면일 때만 정확하다.
- 문서 펴기(dewarping) 연구에서 제어점 · 격자 정규화로 휜 종이를 펴는 아이디어를 가져왔다. 파티션은 격자 구조가 명확하니까 더 단순하게 할 수 있었다.
- 4점 Homography로 대충 맞춘 다음, 내부 교차점을 제어점으로 찾아서 칸마다 따로 local Homography를 적용했다 (`mesh_rectifier`). 화면의 `MESH` 값이 제어점 신뢰도다.
- 휘어진 격자 합성 장면에서 가운데 제어점 오차 15 px 이내, 신뢰도 0.5 이상이 나왔다.

참고한 것
- [Document Dewarping with Control Points](https://github.com/gwxie/Document-Dewarping-with-Control-Points)
- [Revisiting Document Image Dewarping by Grid Regularization](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.pdf)
- [UVDoc](https://github.com/tanguymagne/UVDoc), [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet)

---

## 6. 색으로 칸 상태 보는 걸 포기하고 높이로 바꿈

- 칸 상태를 색 · 밝기로 보니까 버블랩 반사, 조명 변화, 칸마다 다른 바닥색에 계속 흔들렸다. 바닥이랑 같은 색 물건은 EMPTY로 나올 수도 있었다.
- 생각해보면 로봇한테 필요한 건 "바닥이랑 다르다"가 아니라 "얼마나 높이 쌓였나"였다. 카메라 한 대 색 정보로는 높이를 알 수가 없다.
- 그래서 손목 카메라를 더해서 스테레오로 높이를 직접 재기로 했다 (`stereo_height`).
  1. 렌즈 왜곡 보정
  2. `stereoRectify`로 에피폴라 정렬
  3. 같은 에피폴라 선이랑 물리적으로 가능한 시차 범위 안에서만 SIFT 대응 (비율 검사 + 상호 검사)
  4. Fundamental 행렬 RANSAC이랑 양쪽 카메라 재투영 검사로 이상치 제거
  5. 칸별 3D 점으로 RANSAC 높이 합의
  6. 삼각측량
  7. 보정해둔 바닥 평면까지 부호 있는 거리
- 칸마다 자기 영역 안 3D 점만 써서 median, MAD, p10 / p90, 유효 점 수, 불확실도를 따로 계산했다. 다른 칸 결과를 기준으로 쓰지 않으니까 한 칸이 바뀌어도 다른 칸이 안 흔들린다.
- 색 · 밝기는 판정에 아예 안 쓴다. 점이 부족하면 `UNKNOWN`, 보정 파일이 없으면 `UNCALIBRATED`로 두고 mm 값을 안 낸다.
- USB 카메라 두 대는 동기화가 안 돼서, 로봇이 멈춰 있을 때만 판정하게 했다.
- 합성 장면에서 0 \~ 60 mm가 섞인 4×4 칸의 칸별 오차는 최대 0.2 mm 이하였다. 색만 바꾼 칸은 높이 차 2 mm 이내로 같게 나왔고, 같은 색 · 다른 높이 9칸은 다 구분됐다. 무늬 없는 칸은 UNKNOWN으로 나왔다.
- 상자 구조가 바뀌면(칸막이 위치가 달라지면) 이전 높이 이력을 버리고, 새 칸이 연속 프레임에서 확인된 다음 적용한다. 칸 경계가 0.35 px 정도 떨리는 건 같은 구조로 본다.

---

## 7. 테스트를 먼저 만들어둔 것

- 카메라 없이도 돌 수 있게 합성 장면 생성기(`vision/synthetic.py`)를 만들었다. 격자 크기, 칸 크기, 원근, 칸마다 다른 색 · 밝기, 그림자 · 조명 기울기, 끊긴 경계, 가림, 휘어짐, 벽면을 마음대로 넣을 수 있다.
- 고칠 때마다 다른 경우가 깨지는지 보려고 회귀 테스트를 43개까지 늘렸다 (`python -m unittest tests.test_pipeline`).
- 실제 카메라 프레임 2장(`tests/data/`)도 회귀 테스트에 넣었다.
  - 예전 화면 캡처(오버레이가 그려진 것): 4×4로 세는지, 오른쪽 위 꼭짓점이 안 잘리는지
  - 지금 카메라 원본: 4×4 + 확정(CONFIRMED) + 윗변이 테두리에 붙는지 + 바닥 그림자가 칸막이가 안 되는지
- 지금은 원본 프레임 테스트가 실패한다. 4×4 개수는 맞는데 확정까지 못 가고 UNCERTAIN에 머문다. 이게 다음에 볼 문제다.

---

## 8. 남은 것

- 실제 프레임에서 확정이 안 되는 경계 구간이 어딘지 따로 뽑아서 봐야 한다.
- 스테레오 높이는 합성 장면에서만 숫자로 확인했다. 높이를 아는 블록으로 실제 카메라 쌍 오차를 재야 한다.
- OpenCV 4.x에서는 합성 스테레오 장면의 모서리 칸 하나가 3D 점 부족으로 UNKNOWN이 되는 경우가 있어서, 대응점 수가 버전에 따라 달라지는 이유도 확인해야 한다.

---

## 실행

VS Code에서 폴더를 열고 터미널에서 실행한다.

```powershell
python main.py
```

`q` 또는 `ESC`로 종료한다. 카메라가 여러 대면 index를 확인하고 지정한다.

```powershell
python main.py --list-cameras
python main.py --camera-index 1 --backend dshow
```

- 카메라 없이 전체 파이프라인 확인: `python main.py --self-test`
- `--rows/--cols`를 주면 자동 판별을 끄고 그 크기로 고정(FIXED)한다.
- 테스트(카메라 불필요): `python -m unittest tests.test_pipeline`
- 카메라가 안 열리면 다른 카메라 앱을 닫고, Windows 설정 → 개인정보 및 보안 → 카메라 → 데스크톱 앱의 카메라 액세스를 허용한다.

### 설정 (`config/vision.json`)

- `camera.index`: `auto` 또는 정수, `camera.backend`: `auto` / `dshow` / `msmf` / `any`
- `grid.rows`, `grid.cols`: 기본 3 × 3 (고정 모드일 때)
- `grid.auto_detect`: `true`면 경계 그래프로 행 · 열 자동 판별, `grid.min_rows/max_rows/min_cols/max_cols`로 범위 지정
- `partition.min_confidence`: 실제 조명에서 외곽 후보 판정 임계값
- `occupancy.empty_threshold/occupied_threshold`: 단일 카메라 외관 참고값의 EMPTY / UNKNOWN / OCCUPIED 범위
- `output.write_json`: `true`면 `latest_scene.json` 갱신

### 렌즈 보정 (카메라마다 한 번)

```powershell
python calibrate_camera.py --make-board
python calibrate_camera.py
```

체커보드를 출력해서 여러 위치 · 각도로 보여주고 `SPACE`로 10장 이상 저장한 뒤 `q`. `config/camera_calibration.json`이 생기면 다음 실행부터 자동 적용되고 화면에 `LENS: CAL`이 뜬다. 없으면 원본을 쓰고 `LENS: RAW`.

### 스테레오 보정과 높이 측정

```powershell
python dual_camera_main.py --list-cameras
python calibrate_camera.py --camera-index 0 --output config/camera_calibration.json
python calibrate_camera.py --camera-index 1 --output config/wrist_camera_calibration.json
python calibrate_dual_view.py --stereo --global-index 0 --wrist-index 1
python dual_camera_main.py --global-index 0 --wrist-index 1
```

- `--stereo`: 그리퍼를 검사 자세에 고정하고 체커보드가 두 카메라에 같이 보이게 해서 서로 다른 10쌍 이상 저장한다. 거의 같은 자세는 자동으로 거부한다. 체커보드를 칸 바닥에 평평하게 놓고 `F`, 그다음 `C`. RMS랑 카메라 간격이 기준을 통과해야 `config/stereo_calibration.json`이 저장된다. 체커보드 인자는 `--inner-cols 9 --inner-rows 6 --square-mm 25`.
- 측정 화면에는 칸마다 ID, `±mm` 높이, confidence(`c`), 유효 3D 점 수(`n`), 불확실도가 뜬다. 결과는 `latest_dual_scene.json`에 저장된다.
- 기본은 실시간 경계 검출이라 상자를 바꿔도 재시작할 필요가 없다. 저장된 고정 격자가 꼭 필요할 때만 `--fixed-grid`를 쓴다.
- 카메라나 그리퍼 장착이 바뀌면 다시 보정해야 한다.

## 저장소 구성

```text
camera/        OpenCV 카메라 어댑터, 렌즈 보정, 두 카메라 동시 읽기
vision/        외곽 검출, 경계 증거 · 그래프, 격자 판별, mesh 보정, 스테레오 높이, 시각화, 합성 장면
models/        로봇 쪽에 넘기는 구조화 결과 (partition, cells, blocks, camera)
config/        설정 파일
tests/         합성 장면 · 실제 프레임 회귀 테스트
docs/images/   README 그림 (실제 프레임 검출 결과, 합성 장면 결과)
```
