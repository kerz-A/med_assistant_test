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
    data_completeness: int = 0
    complaints_quality: int = 0
    anamnesis_quality: int = 0
    vitals_collected: int = 0
    life_history_quality: int = 0


class ExaminationQuality(BaseModel):
    overall_score: float = 0.0
    recording_duration_sec: float = 0.0


class ClinicalDecisionSupport(BaseModel):
    quality_criteria: QualityCriteria = QualityCriteria()
    examination_quality: ExaminationQuality = ExaminationQuality()


class MedicalProtocol(BaseModel):
    patient_info: PatientInfo = PatientInfo()
    exam_data: ExamData = ExamData()
    vitals: Vitals = Vitals()
    clinical_decision_support: ClinicalDecisionSupport = ClinicalDecisionSupport()
