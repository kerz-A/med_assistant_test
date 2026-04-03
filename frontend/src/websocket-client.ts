// ---- Types ----

export interface Utterance {
  speaker: string;
  text: string;
  start: number;
  end: number;
}

export interface PatientInfo {
  full_name?: string | null;
  age?: number | null;
  gender?: string | null;
}

export interface ExamData {
  complaints?: string; complaints_details?: string;
  anamnesis?: string; life_anamnesis?: string;
  allergies?: string; medications?: string;
  diagnosis?: string; treatment_plan?: string;
  patient_recommendations?: string;
}

export interface Vitals {
  height_cm?: number | null; weight_kg?: number | null;
  bmi?: number | null; pulse?: number | null;
  spo2?: number | null; systolic_bp?: string | null;
}

export interface QualityCriteria {
  greeting_and_contact?: number;
  conversation_structure?: number;
  needs_identification?: number;
  current_complaints_identification?: number;
  disease_history?: number;
  general_medical_history?: number;
  medication_history?: number;
  family_history?: number;
  prevention_and_risk_control?: number;
  treatment_planning?: number;
  visit_closure?: number;
}

export interface ExaminationQuality {
  overall_score?: number;
  criteria_completed?: number;
  criteria_total?: number;
}

export interface DialogueAnalytics {
  doctor_showed_empathy?: number;
  doctor_interrupted_patient?: number;
  patient_asked_questions?: number;
  doctor_used_medical_jargon?: number;
  doctor_confirmed_understanding?: number;
  lifestyle_discussed?: number;
  allergies_discussed?: number;
  shared_decision_making?: number;
  patient_compliance_assessment?: number;
  doctor_pacing?: number;
}

export interface ClinicalDecisionSupport {
  quality_criteria?: QualityCriteria;
  examination_quality?: ExaminationQuality;
  dialogue_analytics?: DialogueAnalytics;
}

export interface Protocol {
  patient_info?: PatientInfo;
  exam_data?: ExamData;
  vitals?: Vitals;
  clinical_decision_support?: ClinicalDecisionSupport;
}

export interface CalibrationCompleteMsg {
  type: "calibration_complete";
  patient_info: PatientInfo;
  message: string;
}

export interface TranscriptUpdateMsg {
  type: "transcript_update";
  utterances: Utterance[];
  processing_time_ms: number;
}

export interface ProtocolUpdateMsg {
  type: "protocol_update";
  protocol: Protocol;
  filled_fields: string[];
}

export interface StatusMsg {
  type: "status";
  status: string;
  message: string;
}

type ServerMessage = CalibrationCompleteMsg | TranscriptUpdateMsg | ProtocolUpdateMsg | StatusMsg;

// ---- Client ----

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private shouldReconnect = true;
  private url: string;

  onCalibrationComplete: ((msg: CalibrationCompleteMsg) => void) | null = null;
  onTranscriptUpdate: ((msg: TranscriptUpdateMsg) => void) | null = null;
  onProtocolUpdate: ((msg: ProtocolUpdateMsg) => void) | null = null;
  onStatus: ((msg: StatusMsg) => void) | null = null;
  onConnectionChange: ((connected: boolean, connecting?: boolean) => void) | null = null;

  constructor(path = "/ws/session") {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    this.url = `${proto}//${location.host}${path}`;
  }

  connect(): void {
    this.shouldReconnect = true;
    this.doConnect();
  }

  private doConnect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;
    this.onConnectionChange?.(false, true);
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => { this.reconnectDelay = 1000; this.onConnectionChange?.(true); };
    this.ws.onclose = () => { this.onConnectionChange?.(false); this.scheduleReconnect(); };
    this.ws.onerror = () => this.ws?.close();

    this.ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== "string") return;
      const msg = JSON.parse(event.data) as ServerMessage;
      switch (msg.type) {
        case "calibration_complete": this.onCalibrationComplete?.(msg); break;
        case "transcript_update": this.onTranscriptUpdate?.(msg); break;
        case "protocol_update": this.onProtocolUpdate?.(msg); break;
        case "status": this.onStatus?.(msg); break;
      }
    };
  }

  private scheduleReconnect(): void {
    if (!this.shouldReconnect) return;
    this.reconnectTimer = setTimeout(() => this.doConnect(), this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 16000);
  }

  sendAudio(chunk: ArrayBuffer): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(chunk);
  }

  send(msg: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  get connected(): boolean { return this.ws?.readyState === WebSocket.OPEN; }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }
}
