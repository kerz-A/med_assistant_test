import { AudioCapture } from "./audio-capture";
import { WebSocketClient } from "./websocket-client";
import type { Protocol, ExamData, Vitals, QualityCriteria, DialogueAnalytics } from "./websocket-client";

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

const QC_LABELS: { key: keyof QualityCriteria; label: string }[] = [
  { key: "greeting_and_contact", label: "Приветствие и установление контакта" },
  { key: "conversation_structure", label: "Структура разговора" },
  { key: "needs_identification", label: "Выявление потребностей пациента" },
  { key: "current_complaints_identification", label: "Выявление текущих жалоб" },
  { key: "disease_history", label: "Анамнез текущего заболевания" },
  { key: "general_medical_history", label: "Общий медицинский анамнез" },
  { key: "medication_history", label: "Лекарственный анамнез" },
  { key: "family_history", label: "Семейный анамнез" },
  { key: "prevention_and_risk_control", label: "Профилактика и контроль факторов риска" },
  { key: "treatment_planning", label: "Планирование лечения" },
  { key: "visit_closure", label: "Заключение визита" },
];

const DA_LABELS: { key: keyof DialogueAnalytics; label: string }[] = [
  { key: "doctor_showed_empathy", label: "Врач проявил эмпатию" },
  { key: "doctor_interrupted_patient", label: "Врач перебивал пациента" },
  { key: "patient_asked_questions", label: "Пациент задавал вопросы" },
  { key: "doctor_used_medical_jargon", label: "Использование мед. жаргона" },
  { key: "doctor_confirmed_understanding", label: "Подтверждение понимания" },
  { key: "lifestyle_discussed", label: "Обсуждение образа жизни" },
  { key: "allergies_discussed", label: "Обсуждение аллергий" },
  { key: "shared_decision_making", label: "Совместное принятие решений" },
  { key: "patient_compliance_assessment", label: "Оценка комплаентности" },
  { key: "doctor_pacing", label: "Комфортный темп разговора" },
];

class App {
  private audio = new AudioCapture();
  private ws = new WebSocketClient();
  private stage: Stage = "idle";
  private timerInterval: ReturnType<typeof setInterval> | null = null;
  private recordStart = 0;
  private fieldTimeouts = new Map<string, ReturnType<typeof setTimeout>>();

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
    window.addEventListener("beforeunload", () => this.ws.disconnect());
  }

  private setStage(s: Stage): void {
    this.stage = s;

    // Button visibility: show buttons relevant to current AND transitioning stages
    this.btnCalibrate.classList.toggle("hidden", s !== "idle");
    this.btnStopCalibrate.classList.toggle("hidden", s !== "calibrating");
    this.btnRecord.classList.toggle("hidden", s !== "calibrated");
    this.btnStop.classList.toggle("hidden", s !== "recording");
    // Finalize button stays visible during stopped, finalizing, and done
    this.btnFinalize.classList.toggle("hidden", !["stopped", "finalizing", "done"].includes(s));

    // Button states: loading during transitions, enabled when actionable
    this.clearAllBtnLoading();
    this.btnCalibrate.disabled = s !== "idle";
    this.btnStopCalibrate.disabled = s !== "calibrating";
    this.btnRecord.disabled = s !== "calibrated";
    this.btnStop.disabled = s !== "recording";

    // Finalize: loading during finalizing, disabled when done
    if (s === "finalizing") {
      this.setBtnLoading(this.btnFinalize, "Формирование...");
    } else if (s === "done") {
      this.btnFinalize.disabled = true;
      const label = this.btnFinalize.querySelector(".btn-label");
      if (label) label.textContent = "✅ Заключение готово";
    } else {
      this.btnFinalize.disabled = s !== "stopped";
    }

    this.calibrationHint.classList.toggle("visible", s === "calibrating");
    this.processingEl.classList.toggle("visible", s === "processing");

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
      if (this.stage === "processing") this.setStage("recording");
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
    } catch (e) {
      const msg = e instanceof DOMException && e.name === "NotAllowedError"
        ? "Не удалось получить доступ к микрофону. Разрешите доступ в настройках браузера."
        : "Ошибка инициализации аудио. Попробуйте обновить страницу.";
      alert(msg);
    }
  }

  private stopCalibration(): void {
    this.setBtnLoading(this.btnStopCalibrate, "Обработка...");
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
    } catch (e) {
      const msg = e instanceof DOMException && e.name === "NotAllowedError"
        ? "Не удалось получить доступ к микрофону. Разрешите доступ в настройках браузера."
        : "Ошибка инициализации аудио. Попробуйте обновить страницу.";
      alert(msg);
    }
  }

  private stopRecording(): void {
    this.setBtnLoading(this.btnStop, "Остановка...");
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
        <div class="proto-title">Пациент</div>
        <div class="patient-fields">
          <div class="proto-field">
            <label class="proto-label">ФИО</label>
            <textarea class="field-input" id="field-full_name" data-field="full_name" rows="1" readonly></textarea>
          </div>
          <div class="patient-row">
            <div class="proto-field" style="flex:1">
              <label class="proto-label">Возраст</label>
              <textarea class="field-input" id="field-age" data-field="age" rows="1" readonly></textarea>
            </div>
            <div class="proto-field" style="flex:1">
              <label class="proto-label">Пол</label>
              <textarea class="field-input" id="field-gender" data-field="gender" rows="1" readonly></textarea>
            </div>
          </div>
        </div>
      </div>
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
      <div class="proto-section cds-section">
        <div class="proto-title">Система помощи принятия врачебного решения</div>
        <div class="cds-layout">
          <div class="cds-col">
            <div class="cds-subtitle">Критерии качества</div>
            ${QC_LABELS.map(q => `
              <div class="cds-row" id="qc-${q.key}">
                <span class="cds-row-label">${q.label}</span>
                <span class="cds-badge cds-score-na" id="qc-val-${q.key}">—</span>
              </div>
            `).join("")}
          </div>
          <div class="cds-col">
            <div class="cds-score-block" id="cds-score-block">
              <div class="cds-score-value" id="cds-overall-score">0.0</div>
              <div class="cds-score-label">общая оценка</div>
              <div class="cds-completed" id="cds-completed">0 / 11 критериев</div>
            </div>
            <div class="cds-subtitle">Аналитика диалога</div>
            ${DA_LABELS.map(d => `
              <div class="cds-row" id="da-${d.key}">
                <span class="cds-row-label">${d.label}</span>
                <span class="cds-analytics-icon" id="da-val-${d.key}">—</span>
              </div>
            `).join("")}
          </div>
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
    // Patient info
    const pi = proto.patient_info ?? {};
    const nameEl = document.getElementById("field-full_name") as HTMLTextAreaElement | null;
    const ageEl = document.getElementById("field-age") as HTMLTextAreaElement | null;
    const genderEl = document.getElementById("field-gender") as HTMLTextAreaElement | null;
    if (nameEl && pi.full_name) { nameEl.value = pi.full_name; this.flashUpdate("full_name", nameEl); }
    if (ageEl && pi.age != null) { ageEl.value = String(pi.age); this.flashUpdate("age", ageEl); }
    if (genderEl && pi.gender) { genderEl.value = pi.gender; this.flashUpdate("gender", genderEl); }

    // Also update header
    if (pi.full_name || pi.age) {
      this.patientInfoEl.textContent = `${pi.full_name || "—"}, ${pi.age || "?"} лет, ${pi.gender || "?"}`;
      this.patientInfoEl.classList.add("visible");
    }

    // Exam data
    const exam = proto.exam_data ?? {};
    for (const f of EXAM_FIELDS) {
      const el = document.getElementById(`field-${f.key}`) as HTMLTextAreaElement | null;
      if (!el) continue;
      const val = exam[f.key];
      if (val && typeof val === "string" && val.trim()) {
        if (el.value !== val) {
          el.value = val;
          this.flashUpdate(`field-${f.key}`, el);
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
          this.flashUpdate(`vital-${v.key}`, card);
        }
      }
    }

    // Clinical Decision Support
    const cds = proto.clinical_decision_support;
    if (cds) {
      const qc = cds.quality_criteria ?? {};
      for (const q of QC_LABELS) {
        const badge = document.getElementById(`qc-val-${q.key}`);
        if (!badge) continue;
        const val = qc[q.key];
        if (val != null) {
          badge.textContent = String(val);
          badge.className = `cds-badge cds-score-${val}`;
        }
      }

      const eq = cds.examination_quality;
      if (eq) {
        const scoreEl = document.getElementById("cds-overall-score");
        const completedEl = document.getElementById("cds-completed");
        if (scoreEl) scoreEl.textContent = (eq.overall_score ?? 0).toFixed(1);
        if (completedEl) completedEl.textContent = `${eq.criteria_completed ?? 0} / ${eq.criteria_total ?? 11} критериев`;
      }

      const da = cds.dialogue_analytics ?? {};
      for (const d of DA_LABELS) {
        const icon = document.getElementById(`da-val-${d.key}`);
        if (!icon) continue;
        const val = da[d.key];
        if (val != null) {
          icon.textContent = val ? "\u2714" : "\u2718";
          icon.className = `cds-analytics-icon ${val ? "da-yes" : "da-no"}`;
        }
      }
    }
  }

  private fmtTime(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2,"0")}:${s.toString().padStart(2,"0")}`;
  }

  private flashUpdate(key: string, el: Element, cls: string = "updated"): void {
    const prev = this.fieldTimeouts.get(key);
    if (prev) clearTimeout(prev);
    el.classList.add(cls);
    this.fieldTimeouts.set(key, setTimeout(() => { el.classList.remove(cls); this.fieldTimeouts.delete(key); }, 1500));
  }

  private setBtnLoading(btn: HTMLButtonElement, text: string): void {
    btn.classList.add("btn-loading");
    btn.disabled = true;
    const label = btn.querySelector(".btn-label");
    if (label) {
      btn.dataset.origLabel = label.textContent || "";
      label.textContent = text;
    }
  }

  private clearAllBtnLoading(): void {
    for (const btn of [this.btnCalibrate, this.btnStopCalibrate, this.btnRecord, this.btnStop, this.btnFinalize]) {
      btn.classList.remove("btn-loading");
      const label = btn.querySelector(".btn-label");
      if (label && btn.dataset.origLabel) {
        label.textContent = btn.dataset.origLabel;
        delete btn.dataset.origLabel;
      }
    }
  }

  private esc(t: string): string {
    const d = document.createElement("div"); d.textContent = t; return d.innerHTML;
  }
}

new App().init();
