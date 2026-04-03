from pydantic import BaseModel


class PatientInfo(BaseModel):
    full_name: str | None = None
    age: int | None = None
    gender: str | None = None  # "М" / "Ж"


class ExamData(BaseModel):
    complaints: str | None = None
    complaints_details: str | None = None
    anamnesis: str | None = None
    life_anamnesis: str | None = None
    allergies: str | None = None
    medications: str | None = None
    diagnosis: str | None = None
    treatment_plan: str | None = None
    patient_recommendations: str | None = None


class Vitals(BaseModel):
    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None
    pulse: float | None = None
    spo2: float | None = None
    systolic_bp: str | None = None


class QualityCriteria(BaseModel):
    greeting_and_contact: int = 0
    conversation_structure: int = 0
    needs_identification: int = 0
    current_complaints_identification: int = 0
    disease_history: int = 0
    general_medical_history: int = 0
    medication_history: int = 0
    family_history: int = 0
    prevention_and_risk_control: int = 0
    treatment_planning: int = 0
    visit_closure: int = 0


class ExaminationQuality(BaseModel):
    overall_score: float = 0.0
    criteria_completed: int = 0
    criteria_total: int = 11
    recording_duration_sec: float = 0.0


class DialogueAnalytics(BaseModel):
    doctor_showed_empathy: int = 0
    doctor_interrupted_patient: int = 0
    patient_asked_questions: int = 0
    doctor_used_medical_jargon: int = 0
    doctor_confirmed_understanding: int = 0
    lifestyle_discussed: int = 0
    allergies_discussed: int = 0
    shared_decision_making: int = 0
    patient_compliance_assessment: int = 0
    doctor_pacing: int = 0


class ClinicalDecisionSupport(BaseModel):
    quality_criteria: QualityCriteria = QualityCriteria()
    examination_quality: ExaminationQuality = ExaminationQuality()
    dialogue_analytics: DialogueAnalytics = DialogueAnalytics()


class MedicalProtocol(BaseModel):
    patient_info: PatientInfo = PatientInfo()
    exam_data: ExamData = ExamData()
    vitals: Vitals = Vitals()
    clinical_decision_support: ClinicalDecisionSupport = ClinicalDecisionSupport()
