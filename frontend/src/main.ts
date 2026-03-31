import { AudioCapture } from "./audio-capture";
import { WebSocketClient } from "./websocket-client";
import type { Protocol, ExamData, Vitals } from "./websocket-client";

type Stage = "idle" | "calibrating" | "calibrated" | "recording" | "processing" | "stopped" | "finalizing" | "done";

const EXAM_FIELDS: { key: keyof ExamData; label: string }[] = [
  { key: "complaints", label: "Жалобы" },
  { key: "complaints_details", label: "Детали жалоб" },
  { key: "anamnesis", label: "Анамнез заболевания" },
  { key: "life_anamnesis", label: "Анамнез жизни" },
  { key: "allergies", label: "Аллергии" },
  { key: "medications", label: "Лекарства" },
  { key: "diagnosis", label: "Диагноз" },
  { key: "treatment_plan", label: "План лечения" },
  { key: "patient_recommendations", label: "Рекомендации пациенту" },
];

const VITAL_FIELDS: { key: keyof Vitals; label: string; unit: string }[] = [
  { key: "height_cm", label: "Рост", unit: "см" },
  { key: "weight_kg", label: "Вес", unit: "кг" },
  { key: "bmi", label: "ИМТ", unit: "" },
  { key: "pulse", label: "Пульс", unit: "уд/мин" },
  { key: "spo2", label: "SpO₂", unit: "%" },
  { key: "systolic_bp", label: "АД", unit: "" },
];

class App {
  private audio = new AudioCapture();
  private ws = new WebSocketClient();
  private stage: Stage = "idle";
  private timerInterval: ReturnType<typeof setInterval> | null = null;
  private recordStart = 0;

  // Elements
  private btnCalibrate!: HTMLButtonElement;
  private btnStopCalibrate!: HTMLButtonElement;
  private btnRecord!: HTMLButtonElement;
  private btnStop!: HTMLButtonElement;
  private btnFinalize!: HTMLButtonElement;
  private statusEl!: HTMLElement;
  private statusText!: HTMLElement;
  private stageLabel!: HTMLElement;
  private timerEl!: HTMLElement;
  private transcript!: HTMLElement;
  private protocol!: HTMLElement;
  private processingEl!: HTMLElement;
  private calibrationHint!: HTMLElement;
  private patientInfoEl!: HTMLElement;

  init(): void {
    this.btnCalibrate = document.getElementById("btn-calibrate") as HTMLButtonElement;
    this.btnStopCalibrate = document.getElementById("btn-stop-calibrate") as HTMLButtonElement;
    this.btnRecord = document.getElementById("btn-record") as HTMLButtonElement;
    this.btnStop = document.getElementById("btn-stop") as HTMLButtonElement;
    this.btnFinalize = document.getElementById("btn-finalize") as HTMLButtonElement;
    this.statusEl = document.getElementById("connection-status")!;
    this.statusText = this.statusEl.querySelector(".status-text")!;
    this.stageLabel = document.getElementById("stage-label")!;
    this.timerEl = document.getElementById("timer")!;
    this.transcript = document.getElementById("transcript-container")!;
    this.protocol = document.getElementById("protocol-container")!;
    this.processingEl = document.getElementById("processing-indicator")!;
    this.calibrationHint = document.getElementById("calibration-hint")!;
    this.patientInfoEl = document.getElementById("patient-info")!;

    this.renderProtocol();
    this.setupWS();
    this.setupButtons();
    this.ws.connect();
    this.setStage("idle");
  }

  private setStage(s: Stage): void {
    this.stage = s;
    // Buttons
    this.btnCalibrate.disabled = s !== "idle";
    this.btnStopCalibrate.disabled = s !== "calibrating";
    this.btnRecord.disabled = s !== "calibrated";
    this.btnStop.disabled = s !== "recording";
    this.btnFinalize.disabled = s !== "stopped";

    this.btnCalibrate.classList.toggle("hidden", s !== "idle");
    this.btnStopCalibrate.classList.toggle("hidden", s !== "calibrating");
    this.btnRecord.classList.toggle("hidden", !["calibrated"].includes(s));
    this.btnStop.classList.toggle("hidden", s !== "recording");
    this.btnFinalize.classList.toggle("hidden", !["stopped", "done"].includes(s));

    this.calibrationHint.classList.toggle("visible", s === "calibrating");
    this.processingEl.classList.toggle("visible", s === "processing" || s === "finalizing");

    const labels: Record<string, string> = {
      idle: "Готов к работе",
      calibrating: "🎯 Калибровка...",
      calibrated: "✅ Калибровка завершена — нажмите «Запись»",
      recording: "🔴 Запись приёма",
      processing: "⏳ Обработка...",
      stopped: "⏸ Запись остановлена — проверьте поля",
      finalizing: "⏳ Формирование заключения...",
      done: "✅ Заключение готово",
    };
    this.stageLabel.textContent = labels[s] || s;

    // Make fields editable only after stop
    const editable = s === "stopped" || s === "done";
    this.protocol.querySelectorAll<HTMLTextAreaElement>(".field-input").forEach(el => {
      el.readOnly = !editable;
      el.classList.toggle("editable", editable);
    });
  }

  private setupWS(): void {
    this.ws.onConnectionChange = (connected, connecting) => {
      this.statusEl.className = `status-indicator ${connected ? "status-connected" : connecting ? "status-connecting" : "status-disconnected"}`;
      this.statusText.textContent = connected ? "Подключено" : connecting ? "Подключение..." : "Отключено";
      if (connected) this.btnCalibrate.disabled = false;
    };

    this.ws.onCalibrationComplete = (msg) => {
      this.setStage("calibrated");
      const pi = msg.patient_info;
      this.patientInfoEl.textContent = `${pi.full_name || "—"}, ${pi.age || "?"} лет, ${pi.gender || "?"}`;
      this.patientInfoEl.classList.add("visible");
    };

    this.ws.onTranscriptUpdate = (msg) => {
      for (const u of msg.utterances) {
        this.addTranscriptLine(u.speaker, u.text, u.start, u.end);
      }
      if (this.stage === "processing") this.setStage(this.stage === "processing" ? "recording" : this.stage);
    };

    this.ws.onProtocolUpdate = (msg) => {
      this.updateProtocolFields(msg.protocol);
    };

    this.ws.onStatus = (msg) => {
      if (msg.status === "processing" && this.stage === "recording") {
        this.processingEl.classList.add("visible");
      } else if (msg.status === "recording" || msg.status === "calibrating") {
        this.processingEl.classList.remove("visible");
      } else if (msg.status === "stopped") {
        this.setStage("stopped");
      } else if (msg.status === "done") {
        this.setStage("done");
      } else if (msg.status === "calibrated") {
        this.setStage("calibrated");
      } else if (msg.status === "finalizing") {
        this.setStage("finalizing");
      }
    };
  }

  private setupButtons(): void {
    this.btnCalibrate.addEventListener("click", () => this.startCalibration());
    this.btnStopCalibrate.addEventListener("click", () => this.stopCalibration());
    this.btnRecord.addEventListener("click", () => this.startRecording());
    this.btnStop.addEventListener("click", () => this.stopRecording());
    this.btnFinalize.addEventListener("click", () => this.finalize());
  }

  private async startCalibration(): Promise<void> {
    try {
      this.audio.onChunk = (chunk) => this.ws.sendAudio(chunk);
      await this.audio.start();
      this.ws.send({ type: "start_calibration", config: { num_speakers: 2 } });
      this.setStage("calibrating");
      this.transcript.innerHTML = "";
    } catch {
      alert("Не удалось получить доступ к микрофону.");
    }
  }

  private stopCalibration(): void {
    this.audio.stop();
    this.ws.send({ type: "stop_calibration" });
  }

  private async startRecording(): Promise<void> {
    try {
      this.audio.onChunk = (chunk) => this.ws.sendAudio(chunk);
      await this.audio.start();
      this.ws.send({ type: "start_recording" });
      this.setStage("recording");
      this.recordStart = Date.now();
      this.timerInterval = setInterval(() => this.updateTimer(), 1000);
      this.updateTimer();
    } catch {
      alert("Не удалось получить доступ к микрофону.");
    }
  }

  private stopRecording(): void {
    this.audio.stop();
    this.ws.send({ type: "stop_recording" });
    if (this.timerInterval) { clearInterval(this.timerInterval); this.timerInterval = null; }
  }

  private finalize(): void {
    this.ws.send({ type: "finalize" });
    this.setStage("finalizing");
  }

  private updateTimer(): void {
    const sec = Math.floor((Date.now() - this.recordStart) / 1000);
    const m = Math.floor(sec / 60).toString().padStart(2, "0");
    const s = (sec % 60).toString().padStart(2, "0");
    this.timerEl.textContent = `${m}:${s}`;
  }

  // ---- Transcript ----
  private addTranscriptLine(speaker: string, text: string, start: number, end: number): void {
    const el = document.createElement("div");
    el.className = `utterance ${speaker}`;
    const icon = speaker === "doctor" ? "👨‍⚕️" : "🤒";
    const label = speaker === "doctor" ? "Врач" : "Пациент";
    const t0 = this.fmtTime(start);
    const t1 = this.fmtTime(end);
    el.innerHTML = `
      <span class="utt-speaker ${speaker}">${icon} ${label}</span>
      <span class="utt-text">${this.esc(text)}</span>
      <span class="utt-time">${t0}–${t1}</span>
    `;
    this.transcript.appendChild(el);
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }

  // ---- Protocol ----
  private renderProtocol(): void {
    this.protocol.innerHTML = `
      <div class="proto-section">
        <div class="proto-title">Данные осмотра</div>
        ${EXAM_FIELDS.map(f => `
          <div class="proto-field">
            <label class="proto-label">${f.label}</label>
            <textarea class="field-input" id="field-${f.key}" data-field="${f.key}" rows="${f.key.includes("plan") || f.key.includes("recommendation") ? 4 : 2}" readonly></textarea>
          </div>
        `).join("")}
      </div>
      <div class="proto-section">
        <div class="proto-title">Витальные показатели</div>
        <div class="vitals-grid">
          ${VITAL_FIELDS.map(v => `
            <div class="vital-card" id="vital-${v.key}">
              <div class="vital-label">${v.label}</div>
              <div class="vital-value" id="vital-val-${v.key}">—</div>
              ${v.unit ? `<div class="vital-unit">${v.unit}</div>` : ""}
            </div>
          `).join("")}
        </div>
      </div>
    `;

    // Edit handler — send changes to server
    this.protocol.querySelectorAll<HTMLTextAreaElement>(".field-input").forEach(el => {
      el.addEventListener("change", () => {
        const field = el.dataset.field;
        if (field && (this.stage === "stopped" || this.stage === "done")) {
          this.ws.send({ type: "edit_field", field, value: el.value });
        }
      });
    });
  }

  private updateProtocolFields(proto: Protocol): void {
    const exam = proto.exam_data ?? {};
    for (const f of EXAM_FIELDS) {
      const el = document.getElementById(`field-${f.key}`) as HTMLTextAreaElement | null;
      if (!el) continue;
      const val = exam[f.key];
      if (val && typeof val === "string" && val.trim()) {
        if (el.value !== val) {
          el.value = val;
          el.classList.add("updated");
          setTimeout(() => el.classList.remove("updated"), 1500);
        }
      }
    }
    const vitals = proto.vitals ?? {};
    for (const v of VITAL_FIELDS) {
      const valEl = document.getElementById(`vital-val-${v.key}`);
      const card = document.getElementById(`vital-${v.key}`);
      if (!valEl || !card) continue;
      const val = vitals[v.key];
      if (val != null) {
        const txt = typeof val === "number" ? String(val) : val;
        if (valEl.textContent !== txt) {
          valEl.textContent = txt;
          valEl.classList.remove("empty");
          card.classList.add("updated");
          setTimeout(() => card.classList.remove("updated"), 1500);
        }
      }
    }
  }

  private fmtTime(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2,"0")}:${s.toString().padStart(2,"0")}`;
  }

  private esc(t: string): string {
    const d = document.createElement("div"); d.textContent = t; return d.innerHTML;
  }
}

new App().init();
