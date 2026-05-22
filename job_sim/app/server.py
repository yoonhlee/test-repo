"""
server.py — JOBSIM 채점 백엔드
Flask 경량 서버. Claude API 로 사용자 답변을 실제로 분석해 5축 점수를 반환.

실행:
    cd job_sim/app
    ANTHROPIC_API_KEY=sk-... python server.py

또는 .env 파일에 ANTHROPIC_API_KEY=sk-... 를 저장한 뒤 실행.
"""

import os
import json
import time
import logging
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import anthropic

# ── 로깅 설정 ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── .env 파일 수동 로드 (python-dotenv 없이) ─────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

app = Flask(__name__, static_folder=".")

# ── Claude 클라이언트 ────────────────────────────────────────────
def get_client():
    """요청마다 새 클라이언트 생성 (API 키 변경 반영)"""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None, "ANTHROPIC_API_KEY 가 설정되지 않았습니다."
    return anthropic.Anthropic(api_key=key), None


# ── 채점 프롬프트 ────────────────────────────────────────────────
SCORE_PROMPT = """\
당신은 직무 적성 평가 전문가입니다. 아래 직무 미션에 대한 응시자의 답변을 분석하여 5개 사고방식 축의 점수를 매겨주세요.

## 미션 정보
- 직무: {job_name}
- 미션 제목: {title}
- 시나리오: {scenario}
- 수행 과제: {task}

## 응시자 답변
{answer}

## 평가 기준 — 5개 사고방식 축 (각 0~100점)

**AX1 정보분석·논리**
높은 점수: 수치·데이터를 근거로 논리적 추론, 원인-결과 관계를 명확히 서술, 체계적 분석 틀 사용
낮은 점수: 막연한 주장, 데이터 언급 없음, 직관에만 의존

**AX2 관찰·탐색**
높은 점수: 현상의 세부 요소를 분리해 관찰, 이상 신호·패턴 발견, 원인을 단계적으로 추적
낮은 점수: 표면적 현상만 언급, 탐색 과정 없이 결론으로 도약

**AX3 전략·판단**
높은 점수: 복수 대안 검토 후 우선순위 제시, 판단 근거 명확, 단·장기 구분
낮은 점수: 단일 해결책만, 왜 그 결정인지 설명 없음

**AX4 리더십·조직**
높은 점수: 이해관계자 역할 구분, 팀 협업 방안, 자원 배분, 보고 라인 언급
낮은 점수: 개인 관점에서만 접근, 조직 맥락 무시

**AX5 대인서비스**
높은 점수: 상대방 감정·입장 공감, 관계 기반 해결, 명확한 커뮤니케이션 방법 제안
낮은 점수: 일방적 해결, 상대방 배려 없음

## 주의사항
- 답변이 짧거나 성의 없으면 모든 축을 낮게 주세요 (10~20점).
- 답변이 해당 미션과 무관하거나 매우 단순하면 40점 이하로 채점하세요.
- 특정 축이 이 미션과 관련이 낮더라도, 답변에서 해당 사고가 나타나면 점수를 주세요.
- 과도하게 후한 점수는 삼가세요. 평균 50~65점을 기준으로 조정하세요.

## 응답 형식
아래 JSON 만 출력하세요. 마크다운, 설명 텍스트 절대 금지.
{{
  "AX1": <정수 0-100>,
  "AX2": <정수 0-100>,
  "AX3": <정수 0-100>,
  "AX4": <정수 0-100>,
  "AX5": <정수 0-100>,
  "feedback": "<이 답변의 가장 두드러진 강점 1가지와 개선 포인트 1가지를 2문장 이내 한국어로>"
}}
"""


# ── 라우트 ───────────────────────────────────────────────────────

@app.route("/")
def index():
    """index.html 서빙"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    """서버 상태 + API 키 존재 여부 확인"""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return jsonify({
        "status": "ok",
        "api_key_set": bool(key),
        "api_key_hint": key[:8] + "..." if key else None,
    })


@app.route("/api/score", methods=["POST"])
def score():
    """
    요청 바디:
      {
        "answer": "사용자 답변 텍스트",
        "mission": { mission 객체 (title, scenario, task, job_name, axis_signals) }
      }
    응답:
      {
        "AX1": 70, "AX2": 55, "AX3": 80, "AX4": 30, "AX5": 40,
        "feedback": "...",
        "source": "llm"
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    answer  = (data.get("answer") or "").strip()
    mission = data.get("mission") or {}

    # ── 입력 검증 ─────────────────────────────────────────────────
    if not answer:
        return jsonify({"error": "answer 필드가 비어 있습니다."}), 400
    if not mission.get("title"):
        return jsonify({"error": "mission 정보가 없습니다."}), 400

    # ── 프롬프트 구성 ─────────────────────────────────────────────
    prompt = SCORE_PROMPT.format(
        job_name = mission.get("job_name", ""),
        title    = mission.get("title", ""),
        scenario = mission.get("scenario", ""),
        task     = mission.get("task", ""),
        answer   = answer[:2000],  # 토큰 제한 방어
    )

    # ── Claude API 호출 ───────────────────────────────────────────
    client, err = get_client()
    if err:
        log.warning("API 키 없음 → 폴백")
        return jsonify({"error": err, "fallback": True}), 503

    t0 = time.time()
    try:
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",   # 빠르고 저렴
            max_tokens = 400,
            temperature= 0.15,   # 일관된 채점을 위해 낮은 온도
            messages   = [{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        log.info(f"채점 완료 ({time.time()-t0:.2f}s) | 직업={mission.get('job_name')} | 미션={mission.get('title')}")
        log.debug(f"  원시 응답: {raw_text[:120]}")
    except anthropic.APIStatusError as e:
        log.error(f"Claude API 오류: {e.status_code} {e.message}")
        return jsonify({"error": f"API 오류: {e.message}", "fallback": True}), 502
    except Exception as e:
        log.error(f"예상치 못한 오류: {e}")
        return jsonify({"error": str(e), "fallback": True}), 500

    # ── JSON 파싱 ─────────────────────────────────────────────────
    # LLM 이 마크다운 코드블록으로 감쌀 경우 제거
    if "```" in raw_text:
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        log.error(f"JSON 파싱 실패: {raw_text[:200]}")
        return jsonify({"error": "응답 파싱 실패", "raw": raw_text, "fallback": True}), 500

    # ── 값 범위 보정 ──────────────────────────────────────────────
    axes = ["AX1", "AX2", "AX3", "AX4", "AX5"]
    for ax in axes:
        result[ax] = max(0, min(100, int(result.get(ax, 50))))

    result["source"] = "llm"
    return jsonify(result)


@app.route("/api/set_key", methods=["POST"])
def set_key():
    """
    런타임에 API 키를 설정 (브라우저 UI 에서 입력).
    보안 주의: 이 서버는 로컬 전용입니다.
    """
    data = request.get_json(force=True, silent=True) or {}
    key  = (data.get("key") or "").strip()
    if not key:
        return jsonify({"ok": False, "error": "키가 비어 있습니다."}), 400
    os.environ["ANTHROPIC_API_KEY"] = key
    log.info(f"API 키 설정됨: {key[:8]}...")
    return jsonify({"ok": True})


# ── 메인 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    key  = os.environ.get("ANTHROPIC_API_KEY", "")
    print("=" * 55)
    print("  JOBSIM 채점 서버")
    print(f"  http://localhost:{port}")
    print(f"  API 키: {'설정됨 (' + key[:8] + '...)' if key else '⚠ 미설정 (UI 에서 입력 가능)'}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)
