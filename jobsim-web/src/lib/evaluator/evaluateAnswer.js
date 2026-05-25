import OpenAI from "openai";
import { AXES } from "./schema.js";
import { buildEvaluationPrompt, PROMPT_VERSION, SYSTEM_PROMPT } from "./prompts.js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function emptyAxis(reason) {
  return {
    score: 0, confidence: 0, evidence: [],
    reason: reason ?? "답변에서 해당 축과 관련된 구체적인 행동이나 근거를 찾을 수 없습니다."
  };
}

function clamp(v, min, max) { return Math.min(Math.max(v, min), max); }
function normalizeText(v) { return v.replace(/\s+/g, " ").trim().toLowerCase(); }
function countNumbers(v) { return v.match(/\d+(?:[.,]\d+)?%?|\d+(?:[.,]\d+)?/g)?.length ?? 0; }
function uniqueTokenRatio(v) {
  const t = normalizeText(v).split(/\s+/).filter(Boolean);
  return t.length === 0 ? 0 : new Set(t).size / t.length;
}
function countMarkers(answer, markers) {
  const n = normalizeText(answer);
  return markers.filter(m => n.includes(normalizeText(m))).length;
}
function keywordHits(answer, keywords) {
  const n = normalizeText(answer);
  return (keywords ?? []).map(k => k.trim()).filter(k => k && n.includes(normalizeText(k)));
}
function quoteInAnswer(answer, quote) {
  return Boolean(quote) && normalizeText(answer).includes(normalizeText(quote));
}

function validateAndRecalculate(answer, evaluation) {
  const usedQuotes = new Set();
  for (const axis of AXES) {
    const r = evaluation.axes[axis];
    r.evidence = (r.evidence ?? []).filter(item => {
      const key = normalizeText(item.quote ?? "");
      if (item.primary_axis === axis && quoteInAnswer(answer, item.quote) && !usedQuotes.has(key)) {
        usedQuotes.add(key);
        return true;
      }
      return false;
    });
    if (r.evidence.length === 0) {
      const hasReason = r.reason && r.reason.length > 5;
      evaluation.axes[axis] = emptyAxis(hasReason ? `[검증 후 근거 불인정] ${r.reason}` : undefined);
      continue;
    }
    const primary = r.evidence.filter(e => e.primary_axis === axis);
    const scoreEv = primary.length > 0 ? primary : r.evidence;
    r.score = Math.max(...scoreEv.map(e => e.level));
    if (primary.length === 0) { r.score = Math.min(r.score, 2); }
    if (r.evidence.length < 2 && r.score >= 4) { r.score = 3; }
    r.confidence = clamp(r.confidence ?? 0, 0, 1);
    if (r.confidence < 0.55) r.score = Math.min(r.score, 2);
  }

  // cap if all axes high
  const high = AXES.filter(a => evaluation.axes[a].score >= 3)
    .sort((a, b) => {
      const ea = evaluation.axes[a], eb = evaluation.axes[b];
      return (eb.confidence * 10 + eb.evidence.length) - (ea.confidence * 10 + ea.evidence.length);
    });
  if (high.length >= 4) {
    high.slice(2).forEach(a => {
      evaluation.axes[a].score = Math.min(evaluation.axes[a].score, 2);
      evaluation.axes[a].confidence = Math.min(evaluation.axes[a].confidence, 0.54);
    });
    if (!evaluation.flags.includes("low_confidence")) evaluation.flags.push("low_confidence");
  }

  if (evaluation.flags.includes("off_topic")) {
    for (const axis of AXES) {
      if (evaluation.axes[axis].score > 1) evaluation.axes[axis].score = 1;
      evaluation.axes[axis].confidence = Math.min(evaluation.axes[axis].confidence, 0.4);
    }
  }
  return evaluation;
}

function heuristicFallback(mission, answer, cause) {
  const length = normalizeText(answer).length;
  const numberCount = countNumbers(answer);
  const uniqueness = uniqueTokenRatio(answer);
  const repPenalty = uniqueness < 0.45 ? 0.9 : uniqueness < 0.6 ? 0.35 : 0;
  const lenLevel = length >= 260 ? 3 : length >= 140 ? 2 : length >= 45 ? 1 : 0;
  const markersByAxis = {
    AX1: ["data", "log", "metric", "rate", "ratio", "compare", "분석", "비교", "수치", "%"],
    AX2: ["check", "관찰", "패턴", "조사", "탐색", "확인", "파악"],
    AX3: ["priority", "전략", "우선순위", "결정", "계획", "방향", "제안"],
    AX4: ["team", "팀", "조직", "역할", "부서", "관리"],
    AX5: ["고객", "사용자", "공감", "응대", "서비스", "경험"],
  };
  const axes = Object.fromEntries(AXES.map(axis => {
    const hits = keywordHits(answer, mission.rubric?.[axis] ?? []);
    const markerHits = countMarkers(answer, markersByAxis[axis]);
    const signal = mission.axis_signals?.[axis] ?? 0;
    const kwLevel = Math.min(hits.length, 5);
    const mrkLevel = Math.min(markerHits, 4);
    const numLevel = axis === "AX1" ? Math.min(numberCount, 4) : 0;
    const sigLevel = signal * 4;
    const evidenceBonus = kwLevel >= 2 && (mrkLevel >= 1 || numLevel >= 1) ? 0.55 : 0;
    const raw = kwLevel * 0.55 + mrkLevel * 0.35 + numLevel * 0.28 + sigLevel * 0.38 + lenLevel * 0.22 + evidenceBonus - repPenalty;
    let score = 0;
    if (raw >= 4.2 && length >= 120 && uniqueness >= 0.55) score = 4;
    else if (raw >= 3.0 && length >= 80 && uniqueness >= 0.5) score = 3;
    else if (raw >= 1.65 && length >= 35 && uniqueness >= 0.45) score = 2;
    else if (raw >= 0.45) score = 1;
    if (hits.length === 0 && markerHits === 0 && numLevel === 0) score = 0;
    const confidence = score === 0
      ? clamp(0.08 + signal * 0.1, 0.05, 0.2)
      : clamp(0.16 + score * 0.08 + Math.min(hits.length, 4) * 0.035 + Math.min(markerHits, 3) * 0.025 + signal * 0.14 + lenLevel * 0.025 - repPenalty * 0.08, 0.18, 0.68);
    return [axis, { score, confidence: Number(confidence.toFixed(2)), evidence: [], reason: `Heuristic fallback (${cause}). keyword_hits=${hits.length}, signal=${signal.toFixed(2)}` }];
  }));
  const flags = ["low_confidence"];
  if (length < 15) flags.push("too_short");
  if (uniqueness < 0.45) flags.push("ambiguous");
  return { mission_id: mission.mission_id, axes, flags, prompt_version: `${PROMPT_VERSION}-fallback` };
}

function toMissionScore(evaluation, mission) {
  return Object.fromEntries(AXES.map(axis => {
    const signal = mission.axis_signals?.[axis] ?? 0;
    return [axis, (evaluation.axes[axis].score / 4) * signal];
  }));
}

export async function evaluateAnswer({ mission, answer }) {
  if (!process.env.OPENAI_API_KEY) {
    const evaluation = heuristicFallback(mission, answer, "missing_api_key");
    return { evaluation, missionScore: toMissionScore(evaluation, mission) };
  }

  let evaluation;
  try {
    const response = await openai.chat.completions.create({
      model: process.env.OPENAI_EVAL_MODEL ?? "gpt-4.1-mini",
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: buildEvaluationPrompt({ mission, answer }) }
      ],
    });
    const raw = response.choices[0]?.message?.content;
    if (!raw) throw new Error("OpenAI returned empty response.");
    const parsed = JSON.parse(raw);
    parsed.prompt_version = parsed.prompt_version || PROMPT_VERSION;
    // ensure all axes exist
    for (const axis of AXES) {
      if (!parsed.axes?.[axis]) {
        parsed.axes = parsed.axes ?? {};
        parsed.axes[axis] = emptyAxis();
      }
    }
    if (!parsed.flags) parsed.flags = [];
    evaluation = validateAndRecalculate(answer, parsed);
  } catch (error) {
    const cause = error instanceof Error ? error.message : "unknown_llm_error";
    console.error("[evaluate] LLM error, using heuristic:", cause);
    evaluation = heuristicFallback(mission, answer, cause);
  }

  return { evaluation, missionScore: toMissionScore(evaluation, mission) };
}
