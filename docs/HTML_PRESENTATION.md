# 실험 결과 발표용 HTML

`lossstudy_allframes_20261004_183750`의 검증된 결과를 12장, 16:9 발표 자료로 요약한다. 다섯 공격 로스의 정의, 두 과제의 성능 감소, 기하 공격 간 직접 비교와 대표 영상 여덟 개를 담는다. PO와 DR 각각에서 Tracking 기하 공격 및 Reconstruction 기하 공격의 Tracking 결과와 Reconstruction 결과를 나란히 보여준다.

5장의 막대는 **공격 후 APD 값(%)**을 표시한다. 원본 보고서의 `dataset == "all"` 집계에서 `tracking_apd_drop_pp_attacked`와 `reconstruction_apd_drop_pp_attacked`를 사용한다. Clean APD는 Tracking 69.69%, Reconstruction 63.25%로 함께 표시하며 두 막대는 공통 0~100% 척도다. APD는 높을수록 좋다. 7장의 두 기하 공격 직접 비교는 같은 클립의 손상 차이와 paired CI를 유지한다.

완성 파일은 `runs/docker/lossstudy_allframes_20261004_183750/presentation/st4rtrack_pgd_slides.html`이다. 약 53MB이며 이미지, 영상, 스타일, 스크립트, 발표에 사용한 수치를 HTML 안에 포함한다. **이 HTML 한 파일만 복사하거나 이름을 바꿔도 된다.** 별도의 assets 폴더나 인터넷 연결은 필요하지 않다. 입력 RGB의 절대 차분 영상은 포함하지 않는다.

| 슬라이드 | 대표 클립 | 공격 목적 | 함께 표시하는 결과 |
|---|---|---|---|
| 8 | PO `cab_e_3rd_13` | `tracking_3d` | Tracking + Reconstruction |
| 9 | PO `cab_e_3rd_13` | `reconstruction_3d` | Tracking + Reconstruction |
| 10 | DR `9c43b3-3_obj_source_left_3` | `tracking_3d` | Tracking + Reconstruction |
| 11 | DR `9c43b3-3_obj_source_left_3` | `reconstruction_3d` | Tracking + Reconstruction |

## 발표 조작

Edge 또는 Chrome에서 HTML을 연다. 실제 검증은 Edge에서 수행했다.

| 조작 | 기능 |
|---|---|
| `←` / `→`, 하단 화살표 | 이전 / 다음 슬라이드 |
| `Home` / `End` | 첫 장 / 마지막 장 |
| `F`, 전체 화면 버튼 | 전체 화면 전환 |
| `O`, 목차 버튼 | 슬라이드 선택 |
| `N`, 발표 메모 버튼 | 현재 슬라이드의 발표 메모 |
| `Esc`, 닫기 버튼 | 열린 목차 또는 메모 닫기 |
| 영상의 재생 버튼 | 해당 영상 재생 |
| 두 영상 함께 재생 / 일시 정지 | 같은 공격의 Tracking과 Reconstruction을 처음부터 함께 재생하거나 정지 |
| 같은 프레임 슬라이더 | 두 영상을 같은 source frame 0~127로 이동하고 정지 |

영상은 해당 장에 진입할 때 메모리에 준비하고 사용자가 재생한다. 다른 장으로 이동하거나 목차·발표 메모를 열면 재생을 멈춘다. 준비 도중 재생 취소나 같은 프레임 이동을 눌러도 이전 요청이 뒤늦게 재생되지 않도록 처리한다. 브라우저 인쇄에서는 영상 대신 포스터를 출력하며 12페이지가 된다.

## 사용하는 코드

| 파일 | 역할 |
|---|---|
| `scripts/render_component_task_results.py` | 검증된 저장 배열에서 두 공격 × 두 과제 × PO/DR의 128프레임 영상을 새로 생성하고 평가 수치 재현 |
| `scripts/build_component_presentation.py` | 완료 보고서와 새 시각화 자산의 SHA를 확인하고 영상 여덟 개를 CPU로 압축하여 단일 HTML 생성 |
| `templates/component_presentation.html` | 슬라이드 구성, 수식 표, 수치에 연결된 막대와 CI, 키보드 조작, 발표 메모, 내장 영상 로딩 |
| `scripts/verify_component_presentation.cjs` | 실제 브라우저에서 원본과 HTML만 복사한 단독 파일의 화면·조작·영상·외부 요청 검증 |

실제 데이터와 결과는 `C:\code\st4rtrack_pgd`에 있다. 공개 코드 저장소는 `C:\code\tracking_attack`이다. 생성 HTML과 대용량 미디어는 Git의 `runs/` 제외 규칙을 따른다.

## 다시 생성

완료된 원본 실험의 저장 배열과 `html_report`, 로컬 Docker 이미지 `sha256:c07e78bf94c671f41392c98e5558cb49296e4187fc7682c51f74467d76539398`가 필요하다. 입력 보고서 내부에 출력하지 않는다.

```powershell
Set-Location C:\code\st4rtrack_pgd
$slidesRun = 'runs\docker\lossstudy_allframes_20261004_183750'
.\.venv\Scripts\python.exe -X utf8 scripts\render_component_task_results.py `
  --run-dir $slidesRun --project-root . `
  --output-dir "$slidesRun\presentation\.build\task_visualization_v2"
.\.venv\Scripts\python.exe -X utf8 scripts\build_component_presentation.py `
  --report-dir "$slidesRun\html_report" `
  --visualization-dir "$slidesRun\presentation\.build\task_visualization_v2" `
  --output "$slidesRun\presentation\st4rtrack_pgd_slides.html"
```

영상 인코딩에는 CPU ffmpeg를 사용한다. GPU 옵션과 모델 추론은 사용하지 않는다. 발표에 내장하는 최종 영상은 H.264 CRF 21, yuv420p로 압축하며 해상도와 전체 128프레임을 보존한다. 여덟 영상 모두 ffprobe decoded frame count 128, 10fps, 12.8초를 확인한다. 10fps는 미리보기 속도이며 실험 소요 시간이 아니다. 원본 보고서 자산을 수정하지 않는다.

Reconstruction은 저장된 raw head2 pointmap에 각 조건의 기존 benchmark scale을 한 번 적용한다. 첫 카메라 좌표계에서 GT, clean, PGD 점군을 같은 시점과 축으로 그린다. 두 공격에서 같은 GT-valid raster 표시 subset과 clean RGB 색을 사용한다. 공통 GT 중심을 사용하며 각 예측별로 중심화하거나 confidence로 점을 제거하지 않는다. 큰 뷰는 GT 장면의 공통 상세 범위이며, 축 밖의 표시점 수와 전체 범위의 작은 inset을 함께 보여준다. 실제 모델 좌표를 clamp하지 않는다. 현재 프레임의 dense 픽셀은 물체 trajectory로 연결하지 않는다. 두 meter 오차지도는 같은 GT mask와 SymLog 색상척도(0.1m 이하 선형, 이상 로그)를 사용한다. 표시용 subset과 전체 valid pixel의 APD/EPE 계산을 구분한다.

Renderer는 기존 manifest가 있는 폴더를 덮어쓰지 않으므로 다시 만들 때는 새 출력 폴더를 지정하고 builder에도 같은 폴더를 넘긴다. 이전 중단된 렌더와 layout preview는 완료 영상과 구분하여 보존한다.

출력 옆의 `presentation_manifest.json`은 입력·생성 코드·출력 SHA와 영상 인코딩 정보를 기록한다. Builder의 `built` 상태는 브라우저 검증 통과를 의미하지 않는다. `.build/media`는 인코딩 캐시이며 이동 시 필요하지 않다.

## 실제 브라우저 검증

아래 Node, Playwright, 브라우저 경로는 설치 환경에 맞게 확인한다. 검증기는 네 인수를 모두 절대 경로로 받는다. 기존 검증 근거를 보존하려면 새로운 QA 디렉터리를 사용한다.

```powershell
& 'C:\Users\eodbs\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' `
  scripts\verify_component_presentation.cjs `
  --presentation 'C:\code\st4rtrack_pgd\runs\docker\lossstudy_allframes_20261004_183750\presentation\st4rtrack_pgd_slides.html' `
  --qa-dir 'C:\code\st4rtrack_pgd\runs\docker\lossstudy_allframes_20261004_183750\presentation\.build\qa\new_check' `
  --playwright-module 'C:\Users\eodbs\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\playwright' `
  --browser-executable 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
```

새 영상의 출처 및 frame count 근거는 `presentation/.build/task_visualization_v2/task_visualization_manifest.json`이다. APD 표시 수정본의 브라우저 검증 근거는 `presentation/.build/qa/revision03_apd_values/presentation_qa.json`이다. 5장의 원본 집계 연결, 표시값과 단위, 공통 막대 척도, 화면 배치와 단독 HTML 복사본을 검사한다.

이전의 전체 재생 검증 근거 `presentation/.build/qa/revision02_fixed/presentation_qa.json`도 보존한다. 원본과 한글·공백 이름의 단독 복사본 각각에서 12장 × 4뷰포트, 로스 수식과 막대·대표 수치 연결, 이미지 로드, 여덟 영상의 실제 시간 진행 및 마지막 프레임 12.7초와 종료 재생, 두 결과 함께 재생·동일 프레임 이동·정지, 장 이동 시 정지, 목차·메모·전체 화면을 확인한 기록이다. 현재 수정본과 이전 검증본의 데이터·영상·이미지 및 APD 막대 렌더링 이외의 실행 코드가 같음을 대조하여 재생 검증을 연결한다. 이전 검증기의 정확한 소스는 `presentation/.build/verifier_history/revision02_full_verify_component_presentation.cjs`에 보존한다.

검증기를 기본 인수로 실행하면 전체 브라우저 검사를 수행한다. 한 장의 APD 표시만 바꾼 경우에는 `--scope slide5`와 `--prior-presentation`, `--prior-qa`, `--prior-review`, `--prior-verifier`의 절대 경로를 함께 지정한다. 이 모드는 이전 검증 근거의 SHA와 변경 범위를 대조하고 현재 화면을 검사하며, 영상 재생을 새로 검사한 것으로 기록하지 않는다.

원본 48조건에서 APD를 다시 평균한 근거는 `presentation/.build/apd_values_source_audit_revision03.json`, 새 발표의 내용 대조는 `presentation/.build/content_review_revision03.json`, 화면 검토는 `presentation/.build/visual_review_revision03.json`에 남긴다. 이전 과학 감사와 빠른 재생 취소·실제 영상 디코딩 증거도 유지한다. 이 검증 파일과 생성 manifest는 HTML 재생에 필요하지 않다.

## 내용의 범위

선정된 WorldTrack 8클립(PO4/DR4), 클립당 전체 128프레임, clean 8조건과 공격 40조건을 요약한다. Tracking MSE, Tracking 기하, Reconstruction 기하, Confidence, Joint 목적을 각각 고정 모델의 입력 공격에서 최대화했다. Joint에는 Tracking MSE가 포함되지 않는다. Confidence 항의 최대화는 confidence 점수를 낮추는 방향이며 이 점수는 확률이 아니다.

APD 감소는 %p, 상대 감소율은 %, EPE 증가는 m로 구분한다. 두 기하 공격의 차이는 같은 클립의 직접 차분과 새 paired CI를 사용하며 Reconstruction EPE 차이의 전체 CI에는 0이 포함된다. 두 과제의 동반 손상을 관찰한 결과를 제시하고, 특정 연결의 독립 인과 효과나 전체 WorldTrack 100클립의 성능으로 일반화하지 않는다.

PO와 DR 첫 manifest 클립을 대표 사례로 고정했다. 영상에 표시한 query subset과 정량 지표에 사용하는 모든 평가 query를 구분한다. 발표용 HTML의 내장 데이터는 전체 목적별 집계, 직접 비교와 선택한 네 조건의 수치다. 전체 클립별 결과와 상세 신뢰구간은 원본 보고서에서 확인한다.
