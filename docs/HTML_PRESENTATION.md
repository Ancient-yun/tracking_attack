# 실험 결과 발표용 HTML

`lossstudy_allframes_20261004_183750`의 실제 저장 결과를 24장, 16:9 HTML 발표 자료로 만든다. 발표에서는 `tracking_3d`, `reconstruction_3d`, `confidence`, `joint_training` 네 공격 목적을 비교한다. Tracking MSE는 발표의 표·집계·차트·영상에서 제외한다. 원본 실험과 상세 보고서의 다섯 목적, 48조건은 보존한다.

8클립 전부에 대해 **클립마다 Tracking 한 장, Reconstruction 한 장**, 총 **16개 비교 페이지와 영상 64개**를 넣는다. 한 페이지는 같은 원본 클립의 같은 과제를 네 로스로 공격한 결과를 2×2로 보여준다. 네 영상 모두 저장된 전체 128프레임이다.

완성 파일은 `runs/docker/lossstudy_allframes_20261004_183750/presentation/st4rtrack_pgd_slides.html`이다. 이미지, 64개 영상, 스타일, 스크립트, 발표 수치를 모두 내장한다. **HTML 한 파일만 복사하거나 이름을 바꿔도 된다.** 인터넷과 별도 assets 폴더는 필요하지 않다.

## 슬라이드 구성

| 슬라이드 | 내용 |
|---|---|
| 1~4 | 발표 목적, 두 과제의 공통 구조, 네 로스, 조건·지표 |
| 5 | 공격 후 실제 APD 값(%), 네 목적 × 두 과제 |
| 6 | 같은 클립의 상대 APD 감소, 32개 점과 기하 목적 간 8개 연결선 |
| 7 | 두 기하 공격의 같은 클립 차분·paired CI |
| 8~23 | 아래 원 manifest 순서의 8클립 × Tracking/Reconstruction 비교 |
| 24 | 관찰 결과와 해석 범위 |

| 클립 | 실제 sequence | Tracking 페이지 | Reconstruction 페이지 |
|---|---|---|---|
| PO 1 | `cab_e_3rd_13` | 8 | 9 |
| DR 1 | `9c43b3-3_obj_source_left_3` | 10 | 11 |
| PO 2 | `dancingroom1_3rd_7` | 12 | 13 |
| DR 2 | `e09bca-3_obj_source_left_2` | 14 | 15 |
| PO 3 | `dancingroom1_3rd2_4` | 16 | 17 |
| DR 3 | `fec654-3_obj_source_left_4` | 18 | 19 |
| PO 4 | `seminar_g110_0315_ego2_3` | 20 | 21 |
| DR 4 | `423875-3_obj_source_left_0` | 22 | 23 |

배치는 좌상 Tracking 기하, 우상 Reconstruction 기하, 좌하 Confidence, 우하 Joint다. 영상 바깥에 로스 ID, 선택 저장 step, 해당 클립의 Clean→PGD APD/EPE 값과 범례를 표시한다. APD는 높을수록, EPE는 낮을수록 좋다.

## 설명 없는 실험 영상

영상에는 **실험 시각화만** 넣는다. 제목, 설명 문장, 숫자, query ID, 성능 그래프, 범례, 축 글자와 오차지도는 굽지 않는다. 입력 RGB 절대 차분도 제외한다.

Tracking 영상은 1024×288이다. 좌측 Clean, 우측 PGD의 저장 RGB(각512×288) 위에 같은 GT query의 점과 짧은 궤적을 그린다. 녹색은 GT, 청록색은 Clean 예측, 주황색은 PGD 예측이다. 표시 query는 GT만으로 고정 선택하며 네 로스에서 같다. 지표는 모든 평가 query로 계산한다. 화면 밖이나 카메라 뒤의 예측점은 RGB 영상에서 보이지 않지만 전체 지표와 내장 projection status에 보존된다.

Reconstruction 영상은 1600×600의 GT/Clean/PGD 점군이다. 저장 raw head2 pointmap에 기존 조건별 benchmark scale을 한 번 적용한다. 같은 첫 카메라 좌표계, GT 중심, 카메라 시점, 상세축과 clean RGB 색으로 비교한다. 표시 pixel은 같은 GT-valid stride8 raster다. 예측별 중심화, confidence filtering, 좌표 clamp를 하지 않는다.

주뷰는 전체128프레임 GT 장면의 공통 상세 범위다. 이 범위 밖의 점은 작은 **전체범위 inset**에서 보존한다. Inset도 clean과 네 공격의 모든 표시점을 함께 포함하며 로스마다 자동으로 바뀌지 않는다. 상세 범위·inset·표시 subset 설명은 HTML 본문과 발표 메모에 둔다. Dense pointmap은 현재 프레임 재구성이므로 물체 trajectory로 연결하지 않는다. 정량 APD/EPE는 stride8 선택 전에 전체 GT-valid pixel과 전체128프레임으로 재현한다.

## 발표 조작

Edge 또는 Chrome에서 연다. `←`/`→`는 이전·다음, `Home`/`End`는 첫·마지막 장, `F`는 전체 화면, `O`는 목차, `N`은 발표 메모, `Esc`는 목차·메모와 확대 화면을 닫는다. 영상 바깥의 확대 버튼으로 해당 실험 영상만 전체 화면에서 볼 수 있다.

**네 영상 함께 재생**은 현재 장의 네 로스 영상을 frame0부터 함께 재생한다. **일시 정지**는 네 영상을 멈춘다. **같은 프레임**은 네 영상을 source frame0~127로 함께 이동하고 정지한다. 장 이동, 목차·메모 열기, 준비 중 취소 시 이전 요청이 뒤늦게 재생되지 않도록 처리한다.

10fps·12.8초는 발표용 재생 속도이며 실험 소요 시간이 아니다. 인쇄에서는 실제 첫 프레임 포스터를 사용하고 24페이지가 된다.

## 사용하는 코드와 재생성

| 파일 | 역할 |
|---|---|
| `scripts/render_component_task_results.py` | all8 × four losses × two tasks의 128프레임을 저장 배열에서 CPU로 다시 그린다. 64MP4, 0/64/127 포스터, 전체 valid 지표와 원자료 SHA를 기록한다. |
| `scripts/build_component_presentation.py` | 원 보고서와 시각화 SHA를 확인한다. 4목적·32점 차트를 새로 그리고 64영상을 압축·내장한다. |
| `templates/component_presentation.html` | 24장 구성, APD 원값·CI, 16개 네 영상 비교 장과 발표 조작을 구현한다. |
| `scripts/verify_component_presentation.cjs` | 원본과 이름을 바꾼 단독 복사본의 내용·화면·64영상·16그룹 조작·외부 요청을 검증한다. |

실험은 `C:\code\st4rtrack_pgd`, 공개 소스는 `C:\code\tracking_attack`이다. 대용량 생성 HTML·영상은 Git의 `runs/` 제외 규칙을 따른다. 완료 원본 저장 배열과 `html_report`, 로컬 CPU ffmpeg 이미지 `sha256:c07e78bf94c671f41392c98e5558cb49296e4187fc7682c51f74467d76539398`가 필요하다. 원 보고서 안에 출력하지 않는다.

```powershell
Set-Location C:\code\st4rtrack_pgd
$slidesRun = 'runs\docker\lossstudy_allframes_20261004_183750'
.\.venv\Scripts\python.exe -X utf8 scripts\render_component_task_results.py `
  --run-dir $slidesRun --project-root . `
  --output-dir "$slidesRun\presentation\.build\task_visualization_fourloss_plain"
.\.venv\Scripts\python.exe -X utf8 scripts\build_component_presentation.py `
  --report-dir "$slidesRun\html_report" `
  --visualization-dir "$slidesRun\presentation\.build\task_visualization_fourloss_plain" `
  --output "$slidesRun\presentation\st4rtrack_pgd_slides.html"
```

기본값은 all8/four losses다. Layout 확인에는 `--preview-only --clip-limit 2`를 사용한다. 기존 manifest가 있는 폴더를 덮어쓰지 않으므로 재생성은 새 폴더를 지정하고 builder에도 같은 경로를 넘긴다.

최종 내장 영상은 CPU H.264 CRF21·yuv420p로 압축하며 해상도와 전체128프레임을 보존한다. 64개 모두 ffprobe 전체 decoded frame count128, 10fps, 12.8초를 확인한다. 새 PGD·모델 추론·GPU 사용은 없다. 원 실험 수치와 상세 보고서를 보존한다.

## 새 검증 근거

검증기는 `--presentation`, `--qa-dir`, `--report-dir`, `--visualization-dir`, `--playwright-module`, `--browser-executable`의 절대 경로를 받는다. 환경의 Node로 `scripts/verify_component_presentation.cjs`를 실행하고 새 QA 폴더를 사용한다. Playwright 모듈과 Edge 실행 파일의 설치 경로를 지정한다.

Revision4 renderer 근거는 `.build/task_visualization_fourloss_plain/task_visualization_manifest.json`, 브라우저 근거는 `.build/qa/revision04_fourloss_plain_fixed_bounded/presentation_qa.json`, 내용·원자료 감사는 `.build/content_review_revision04.json`, 실제 화면 검토는 `.build/visual_review_revision04.json`에 둔다. 이전 revision2·3은 보존하지만 새64영상 재생 증거로 대신 사용하지 않는다.

원본과 한글·공백 이름의 HTML 단독 복사본 각각에서24장×4뷰포트,64영상 실제 재생 진행·frame0/64/127·마지막12.7초와 종료12.8초,16그룹 동시 재생·정지·이동·빠른 취소를 확인한다. 전체128프레임의 실제 디코딩은 ffprobe로, 브라우저 주요 프레임 표시와 끝 재생은 브라우저로 확인하며 두 근거를 구분한다. 외부 파일·네트워크 요청과 JS 오류가 없어야 한다.

출력 옆 `presentation_manifest.json`은 코드·원자료·자산·HTML SHA를 기록한다. Builder의 `built`는 검증 통과와 다르다. 인코딩 캐시 `.build/media`와 별도 QA 파일은 HTML 재생에 필요하지 않다.

## 수치와 해석

발표는 선정 개발8클립(PO4/DR4), 전체128프레임, clean8+공격32=40조건이다. 내장 `source_scope`는 원 실험48조건, `scope`는 발표40조건을 명시한다. 32공격 조건과4집계 행은 원 기록 그대로다.

5장은 감소량 대신 **공격 후 APD 값(%)**이다. 원 `impact`의 `dataset == "all"` 행에서 `tracking_apd_drop_pp_attacked`, `reconstruction_apd_drop_pp_attacked`를 쓰고 공통0~100%로 표시한다. Clean69.69/63.25%다. 네 목적 중 Tracking 기하의 두 APD가 가장 낮다(14.00/3.93%).

6장의 상대 감소율(%)과7장의 같은 클립 손상 차이(%p 또는 m)는 별도 분석이다. 같은 클립의 목적들을 독립 표본으로 해석하지 않는다. 두 기하 공격의 Reconstruction EPE 차이 전체CI는0을 포함한다. 두 과제의 동반 손상과 목적별 균형 변화의 관찰이며 특정 연결의 독립 인과 효과나 전체 WorldTrack으로 일반화하지 않는다. Joint는 두 기하 항과 confidence 항의 합이다. Confidence 항의 최대화는 confidence 점수를 낮추는 방향이며 점수는 확률이 아니다.
