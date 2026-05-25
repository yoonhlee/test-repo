import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import { evaluateAnswer } from "./src/lib/evaluator/evaluateAnswer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const port = Number(process.env.PORT ?? 8080);

app.use(express.json({ limit: "2mb" }));
app.use(express.static(path.join(__dirname, "app")));
app.use("/missions", express.static(path.join(__dirname, "missions")));
app.use("/data", express.static(path.join(__dirname, "data")));

// ── 미션 인덱스 API
app.get("/api/missions", (req, res) => {
  try {
    const idx = JSON.parse(readFileSync(path.join(__dirname, "missions/index.json"), "utf8"));
    res.json(idx);
  } catch (e) {
    res.status(500).json({ error: "Failed to load mission index." });
  }
});

// ── 채점 API
app.post("/api/evaluate", async (req, res) => {
  try {
    const { mission, answer } = req.body ?? {};
    if (!mission || typeof answer !== "string") {
      return res.status(400).json({ error: "mission and answer are required." });
    }
    const result = await evaluateAnswer({ mission, answer });
    return res.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[/api/evaluate]", message);
    return res.status(500).json({ error: message });
  }
});

// ── job_weights 데이터 API (적합도 계산용)
app.get("/api/job-weights", (req, res) => {
  try {
    const data = JSON.parse(readFileSync(path.join(__dirname, "data/processed/job_weights.json"), "utf8"));
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: "Failed to load job weights." });
  }
});

app.listen(port, () => {
  console.log(`✅ JOBSIM server running at http://localhost:${port}`);
});
