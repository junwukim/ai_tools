# AI_Tool_v5 변경 사항 정리

이 문서는 현재 기준으로 `AI_Tool_v5.py` 및 함께 추가된 보조 파일들에 반영되어 있는 변경 사항을 정리한 문서다.

중간에 시도했다가 되돌린 UI 변경은 제외하고, 현재 코드에 실제로 남아 있는 기능만 기준으로 정리했다.

## 1. 전체 목적

`AI_Tool_v5.py`는 기존 `AI_Tool_v4.py` 기반 UI를 유지하면서 아래 기능들을 추가/개선한 버전이다.

- 좌측 기능 레이어를 체크박스로 선택적으로 표시
- `App Start` 영역을 고정 4개 하드코딩 방식에서 동적 입력 방식으로 변경
- `Touch` 좌표를 사용자가 직접 입력 가능하도록 변경
- `Auto Touch`를 토글 버튼으로 변경
- `Washer Test`용 체크박스 그룹화
- `.tflite` 모델 분석 기능 추가
- 레이어별 연산량 추정 및 CSV 저장 기능 추가
- timing 측정값 기반 회귀식 추정 기능 추가

## 2. UI 표시 제어 관련 변경

### 2.1 View 체크박스 기반 레이어 표시

좌측 레이아웃 상단에 `View` 영역이 추가되어, 사용자가 필요한 기능만 선택해서 표시할 수 있도록 변경했다.

지원 방식:

- 각 기능별 체크박스 선택 시 해당 section만 표시
- 여러 개 중복 선택 가능
- `All select` 버튼으로 전체 선택 가능
- 전체 선택 상태에서는 버튼 텍스트가 `All deselect`로 변경

### 2.2 Shell Port는 항상 고정 표시

`Shell Port`는 기본 제어 포트이므로 아래 정책으로 유지했다.

- 체크박스 대상에서 제외
- 항상 표시
- 다른 항목 선택 여부와 무관하게 고정 표시

### 2.3 Washer Test 체크박스

세탁기 테스트 관련 기능 중 `Relay Port`와 `Washer Control`은 별도 체크박스를 노출하지 않고, 하나의 `Washer Test` 체크박스로 묶었다.

동작:

- `Washer Test` 체크 시 `Relay Port` + `Washer Control` 표시
- 해제 시 둘 다 숨김

별도 체크박스를 유지하는 항목:

- `Display Control`
- `Touch`
- `App Start`
- `Watchdog`
- `Memory Check`
- `Windows CMD`
- `TFLite Parser`

## 3. App Start 관련 변경

기존의 고정된 `App#1 ~ App#4` 방식 대신, 사용자가 직접 package name을 입력하는 구조로 변경했다.

### 3.1 App Start 헤더

`App Start` 행에는 아래 버튼이 존재한다.

- `Add (max:4)`
- `Delete`
- `All Start`
- `All Stop`

### 3.2 App 행 구조

기본적으로 `App #1`은 항상 보이도록 구성했고, 최대 `App #4`까지 추가 가능하다.

각 App 행 구성:

- `App #n` label
- package name 입력 칸
- `Start` 버튼
- `Repeat` 버튼

### 3.3 package name 처리 방식

사용자는 suffix만 입력하면 되고, 내부에서는 자동으로 prefix를 붙여 사용한다.

예:

- 입력: `multi_ai_demo1`
- 내부 사용: `lupa.usr.multi_ai_demo1`

### 3.4 버튼 동작

`Start` 버튼:

- 초기 상태에서는 `Start`
- 실행 중에는 `Stop`으로 변경
- `Stop` 클릭 시 앱 종료 후 다시 `Start`로 복귀

`Repeat` 버튼:

- 클릭 시 start/stop 반복 모드 진입
- 활성화되면 연노란색 표시
- 버튼 텍스트가 `Stop (count)` 형식으로 변경
- count에는 현재 repeat 횟수 반영

### 3.5 Add / Delete 동작

- `Add` 클릭 시 `App #2`, `App #3`, `App #4` 순으로 추가
- 최대 4개까지만 허용
- `Delete` 클릭 시 마지막 App부터 제거
- `App #1`은 삭제되지 않도록 유지

## 4. Touch 관련 변경

기존에는 좌우 터치 좌표가 코드에 하드코딩되어 있었으나, 사용자가 직접 입력할 수 있도록 변경했다.

### 4.1 좌표 입력

입력 항목:

- `L(x,y)`
- `R(x,y)`

사용자는 좌측/우측 터치 좌표를 직접 수정할 수 있다.

### 4.2 버튼 구성

버튼:

- `Left`
- `Right`
- `Auto touch`

### 4.3 Auto touch 토글화

기존 `Auto` / `Auto Stop` 2개 버튼 구조를 하나의 토글 버튼으로 변경했다.

동작:

- 시작 전: `Auto touch`
- 동작 중: `Auto Stop`
- 동작 중에는 연노란색 표시

## 5. 레이아웃/정렬 관련 변경

UI 사용 중 자주 요청되었던 정렬 문제를 일부 조정했다.

대표적으로:

- 좌측 상단 `View` 영역을 고정 배치
- 좌측/우측 메인 레이아웃의 상단 정렬 유지
- `App #n` 행의 `Start` 버튼 위치를 `App Start` 헤더의 `All Start` 열과 맞추도록 보정

참고:

- 이 부분은 배치 계열 특성상 폰트/OS 스타일에 따라 1~2px 정도 차이가 보일 수 있다.

## 6. Watchdog / Memory / CMD / Shell 관련 유지 사항

기존 기능은 유지하면서, 새 체크박스 표시 제어 구조에 맞게 섹션 가시성만 제어하도록 연결했다.

유지된 주요 기능:

- Shell Port 연결/해제
- Relay Port 연결/해제
- Watchdog 로그 표시
- Memory Check 결과 표시
- Windows CMD 입력/출력
- Shell 로그 출력

## 7. TFLite Parser 기능 추가

새로운 `TFLite Parser` section과 버튼들이 추가되었다.

버튼:

- `Parse .tflite`
- `Batch Folder`
- `Fit Timing`

이 기능은 `.tflite` 모델을 직접 분석해서, MCU 환경에서 성능 추정에 필요한 연산량 지표를 얻기 위한 목적이다.

### 7.1 분석 목적

주 사용 목적:

- 모델 전체 연산량 파악
- 레이어별 연산량 파악
- 파라미터 수 확인
- 실제 측정 시간과 연산량을 연결하는 근사식 도출

즉, 아래와 같은 질문에 답하기 위한 기능이다.

- 이 모델은 총 몇 번의 곱셈/덧셈이 필요한가?
- FC / Conv / DepthwiseConv 중 어디서 연산이 많이 발생하는가?
- 파라미터 수와 연산량이 증가하면 시간은 얼마나 늘어나는가?

## 8. TFLite Parser 상세 기능

### 8.1 단일 모델 분석 (`Parse .tflite`)

단일 `.tflite` 파일을 선택하면 아래 정보를 출력한다.

- 모델 이름
- schema version
- subgraph 이름
- operator 개수
- graph input / output tensor 정보
- 각 레이어의 input / output tensor 정보
- 지원 레이어에 대한 연산량 통계

출력되는 주요 수치:

- `params`
- `mul`
- `add`
- `total`
- `mac`
- `output_elements`

### 8.2 지원 레이어

현재 연산량 추정 지원 레이어:

- `FULLY_CONNECTED`
- `CONV_2D`
- `DEPTHWISE_CONV_2D`

### 8.3 연산량 계산 기준

#### FULLY_CONNECTED

- `mul = output_elements * input_depth`
- `add = output_elements * (input_depth - 1)`
- bias가 있으면 `output_elements` 만큼 add 추가

#### CONV_2D

- `mul = output_elements * kernel_h * kernel_w * in_channels`
- `add = output_elements * (kernel_h * kernel_w * in_channels - 1)`
- bias가 있으면 `output_elements` 만큼 add 추가

#### DEPTHWISE_CONV_2D

- `mul = output_elements * kernel_h * kernel_w`
- `add = output_elements * (kernel_h * kernel_w - 1)`
- bias가 있으면 `output_elements` 만큼 add 추가

### 8.4 주의 사항

현재 계산은 tensor shape 기반의 산술 연산 추정치다.

포함하지 않는 항목:

- activation 연산 비용
- requantization 비용
- 메모리 이동 비용
- cache / bus / DMA 영향
- scheduler / runtime overhead
- framework 내부 관리 비용

즉, 이 값은 `순수 계산량`에 가까운 1차 근사치다.

또한:

- `FLOAT32`, `FLOAT16` tensor라면 float ops로 해석 가능
- quantized 모델이라면 arithmetic op count로 보는 것이 더 정확

## 9. CSV 저장 기능

### 9.1 단일 모델 분석 후 저장

단일 모델 분석 시 아래 파일이 자동 저장된다.

- `*_summary.csv`
- `*_layers.csv`

`summary.csv`에는 모델 단위 합계 정보가 들어가고,  
`layers.csv`에는 레이어별 상세 정보가 들어간다.

### 9.2 Batch Folder 분석

폴더를 선택하면 하위 폴더까지 포함하여 `.tflite` 파일을 수집하고 일괄 분석한다.

생성 파일:

- `tflite_batch_summary.csv`
- `tflite_batch_layers.csv`

이 파일들은 여러 모델을 한 번에 비교하기 위한 용도다.

### 9.3 timing 측정을 위한 CSV 컬럼

summary CSV에는 `measured_time_ms` 컬럼이 포함되어 있다.

사용 흐름:

1. `Parse .tflite` 또는 `Batch Folder` 실행
2. 생성된 summary CSV 열기
3. 각 모델의 실제 측정 시간(ms)을 `measured_time_ms`에 입력
4. `Fit Timing` 버튼으로 다시 CSV 선택

## 10. Timing 회귀식 추정 기능

`Fit Timing` 버튼은 summary CSV와 측정 시간을 바탕으로 계수를 추정한다.

기본적으로 아래 형태의 근사식을 얻기 위한 구조다.

```text
time_ms ~= c0
         + c1 * (fc_mac / 1e6)
         + c2 * (conv_mac / 1e6)
         + c3 * (depthwise_mac / 1e6)
         + c4 * (total_add / 1e6)
         + c5 * (total_output_elements / 1e6)
```

### 10.1 회귀 입력 feature

현재 사용되는 feature:

- `fc_mac`
- `conv_mac`
- `depthwise_mac`
- `total_add`
- `total_output_elements`

모든 값은 내부에서 `1e6` 단위로 정규화하여 사용한다.

### 10.2 회귀 결과

출력:

- 추정식
- `MAE`
- `RMSE`
- `R^2`
- coefficient CSV
- prediction CSV

생성 파일:

- `*_timing_fit_coefficients.csv`
- `*_timing_fit_predictions.csv`

### 10.3 구현 방식

- 외부 라이브러리 없이 내부 선형 방정식 풀이 기반
- 아주 작은 ridge 항을 더한 회귀 방식 사용

목적:

- MCU 실측 시간과 모델 구조 지표를 연결하는 경험식 확보
- 이후 보드/커널 최적화 방향 검토

## 11. 추가된 보조 파일

이번 작업 과정에서 아래 파일들이 추가되었다.

### 11.1 TFLite 샘플/생성기

- `generate_fc3_test_tflite.py`
- `fc3_test_model.tflite`
- `generate_fc3_runnable_tflite.py`
- `fc3_runnable_model.tflite`

용도:

- parser 테스트용 최소 FC 3개 모델
- weight/bias buffer까지 포함한 runnable 형태 샘플 모델

### 11.2 TFLM 연동 참고 파일

- `emit_c_array.py`
- `fc3_runnable_model_data.h`
- `tflm_fc3_harness.cc`
- `fc3_reference_inference.py`

용도:

- `.tflite`를 C array header로 변환
- TFLite Micro 쪽에 바로 넣어볼 수 있는 harness 제공
- Python 기준의 참조 출력값 확인

참고:

- 현재 메인 사용 방향은 `.py` 기반 분석이므로, 위 파일들은 부가 참고 자료 성격이 강하다.

## 12. 현재 정리 기준

현재 기준으로 가장 핵심적인 사용 흐름은 아래와 같다.

1. `AI_Tool_v5.py` 실행
2. 좌측 `View` 체크박스로 필요한 기능 표시
3. `Touch`, `App Start`, `Washer Test` 등 필요한 제어 기능 사용
4. `TFLite Parser`에서 모델 분석
5. CSV 저장
6. 실제 보드 측정 시간 입력
7. `Fit Timing`으로 시간 근사식 도출

## 13. 향후 확장 후보

추가로 확장 가능성이 있는 항목:

- `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `MUL`, `ADD` 등 연산량 추정 추가
- activation 비용 모델링
- quantized 모델에 대한 별도 cost model
- 메모리 access / bandwidth 기반 cost 추가
- CSV 대신 그래프 출력 기능 추가
- 여러 보드별 회귀식 profile 분리

