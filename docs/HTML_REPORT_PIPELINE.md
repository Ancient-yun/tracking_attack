# 완료 결과에서 HTML 보고서 만들기

대상은 `lossstudy_allframes_20261004_183750`의 clean 8조건과 공격 40조건이다. 분석 정의와 해석 기준은 [HTML_REPORT_GUIDE.md](HTML_REPORT_GUIDE.md)의 6.6절을 따른다. 이 문서는 구현된 CPU 후처리의 실행 방법이다. 합성 자료의 테스트 통과는 실제 실험이나 최종 보고서의 완료를 의미하지 않는다.

실제 데이터·vendor·가상환경·저장 결과는 `C:\code\st4rtrack_pgd`에 있고, 공개 코드는 `C:\code\tracking_attack`에 있다. GitHub의 중간 결과 snapshot을 최종 결과로 사용하지 않는다. 아래 명령은 실제 프로젝트에 같은 후처리 파일을 설치한 뒤 실행한다.

## 사용하는 코드

| 파일 | 역할 |
|---|---|
| `scripts/component_report_data.py` | 48조건 완료, 8클립 전체128프레임, 원래 분석·audit·scalar SHA를 검증하고 영향표·산점도·직접 paired 차분을 계산 |
| `scripts/component_projection.py` | 원본 NPZ·GT 카메라·저장 배열과 source SHA를 검증하고 모든 query의 tracking 및 고정 GT mask의 reconstruction 지표 재현 |
| `scripts/render_component_comparisons.py` | CPU RGB·tracking MP4와 frame0/64/127 스틸, reconstruction 오차 지도, PGD·normalization 진단, 동일 프레임 세 열 비교 생성 |
| `scripts/build_component_html_report.py` | 새로운 통계 그림 9개와 기존 그림 3개를 묶어 상대 경로의 한국어 HTML 및 수치 JSON 생성 |
| `templates/component_report.html` | 인터넷 없는 표·dataset/목적/clip 필터, 영상, 이미지 확대, 인쇄 레이아웃 |
| `scripts/verify_component_html_report.cjs` | 실제 브라우저에서 file://, 표 수치, 필터, 자산, 영상 재생, 모바일·인쇄 및 다른 폴더로 복사한 보고서 검증 |
| `scripts/run_component_report_pipeline.py` | 원래 GPU 컨테이너 종료 및 완료 증거 확인 후 순차 CPU 렌더링·HTML 생성·브라우저 QA·최종 폴더 게시 |

어떤 후처리도 모델을 불러오거나 PGD를 재실행하지 않는다. frozen 공격 소스, 설정, manifest, 원본 결과, 기존 분석 JSON은 수정하지 않는다. 영상 인코딩은 로컬 CPU ffmpeg/ffprobe를 사용하며, 없으면 고정 Docker image의 CPU ffmpeg/ffprobe를 사용한다. 이 인코딩 컨테이너에는 `--gpus`를 주지 않는다.

## 완료를 먼저 확인

```powershell
Set-Location C:\code\st4rtrack_pgd
$reportRun = 'runs\docker\lossstudy_allframes_20261004_183750'
$reportContainer = '97cd3b3105d3942caf71fd399885e8f90444d87b9d125d0c8da450aad9fee074'
.\.venv\Scripts\python.exe scripts\run_component_report_pipeline.py `
  --run-dir $reportRun --project-root . --container-id $reportContainer --check-only
```

이 검사는 원래 컨테이너가 exit 0·OOM 없음으로 종료됐고, `campaign_execution.json`과 `resume_execution.json`이 complete이며, independent audit과 강화된 분석기가 전체48조건·PO4/DR4·128프레임·PGD20·ε4/255의 완료를 확인했을 때만 통과한다. `result.json` 수가48이거나 GPU가 쉬는 것만으로 시작하지 않는다. 누락·실패·SHA 불일치는 이유를 출력한다. GPU 작업을 재시작하거나 정지하는 옵션은 없다.

## 전체 후처리 실행

```powershell
.\.venv\Scripts\python.exe scripts\run_component_report_pipeline.py `
  --run-dir $reportRun --project-root . --container-id $reportContainer `
  --node 'C:\Users\eodbs\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  --playwright-module 'C:\Users\eodbs\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\playwright' `
  --browser-executable 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
```

Node 및 Playwright 경로는 설치된 runtime에 따라 다르므로 실제 디렉터리를 확인하고 넘긴다. Playwright 모듈을 별도로 설치했다면 그 경로를 사용한다. 대용량 배열과 영상은 Git에 넣지 않는다.

파이프라인은 run 안의 새 staging 디렉터리에 생성하고, 다섯 공격 목적을 순차 처리한다. 40개 clean/attack 쌍마다 RGB·tracking 영상 두 개, 총80개를 만들며 각 영상의 decoded frame 수128을 ffprobe로 확인한다. Preview는10fps·12.8초이며 원본 촬영 FPS나 실험 실행 시간을 의미하지 않는다. 시각화에 표시한 query subset과 모든 query를 사용하는 정량 지표를 구분한다.

Builder는 완료 자료만 받아 `built_pending_browser_qa` 상태로 저장한다. 실제 브라우저 QA와 출력 hash 검증을 통과한 뒤에만 최종 `html_report` 폴더를 게시한다. 합성 테스트는 `synthetic_qa_passed`로 표시하며 실제 완료 결과로 게시하지 않는다. 후처리 상태는 별도의 `html_postprocessing.json`에 기록한다. 실패한 staging과 이유를 보존하고 이전 완료 보고서를 덮어쓰지 않는다.

## 결과와 검증 근거

최종 진입점은 `<RUN>\html_report\index.html`이다. `assets`에는 통계 PNG, 조건별 스틸·영상, timeline, 동일 프레임 clean/tracking_3d/reconstruction_3d 패널과 `visualization_manifest.json`이 있다. `data`에는 보고서 수치, 새 `task_coupling_analysis.json`, CSV와 존재하는 runtime 관측 근거가 있다. `qa`에는 브라우저 검증 JSON, screenshot과 인쇄 PDF가 있다.

`report_manifest.json`은 원래 분석·audit·scalar 입력, 원본 배열·source NPZ, 시각화 코드, guide, 보고서 자산의 SHA와 명령, CI seed/samples 및 QA 결과를 연결한다. 보고서 폴더 전체를 복사하면 인터넷 없이 열린다. 원본 NPZ·checkpoint·대용량 float32 배열은 보고서에 포함하지 않는다.

보고서의 핵심 순서는 영향표 → 같은 clip의 두 상대 APD 손상 산점도 → 두 기하 목적의 직접 paired 차분 → 내부 진단·동일 프레임 영상이다. 직접 차분의 CI는 같은 clip의 차이에서 새로 계산하며 기존 CI를 빼지 않는다. 개선된 음수 변화와 N/A, 과거 실패 기록도 보존한다. PO/DR 첫 manifest 클립을 고정 사례로 제공하고, tracking APD 공격 간 차이 절댓값이 최대인 사례는 결과 기반 탐색 사례라고 명시한다.

## CPU 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_component_report_data.py tests\test_component_visualization.py `
  tests\test_build_component_html_report.py tests\test_component_report_pipeline.py -q
```

테스트의 작은 합성 배열·probe stub은 완료 gate와 계산·출처 연결을 검사하기 위한 자료다. 별도의 Docker CPU 인코딩 smoke 및 실제 브라우저 검증이 영상·HTML 동작을 검사한다. 최종 실제 보고서는 원래 run의 완료 증거를 다시 검증해야 한다.

2026-10-05 준비 검증에서는 CPU 테스트110개가 통과했다. Windows의 symlink 생성 권한을 요구하는 검사1개는 skip했다. 고정 코드로 생성한 합성 보고서의 브라우저 QA에서는 상세40행·집계15행·직접 차분12행의 값, 80개 합성 영상의40조건 metadata와 두 영상의 실제 재생, 필터·프레임 선택·1440/390px 레이아웃·인쇄·다른 폴더로 옮긴 file:// 동작을 확인했다. 외부 요청과 JavaScript 오류는0이었다. 이 검증은 실제 캠페인 완료 증거가 아니다.
