import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  BrowserRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
  useNavigate,
  useParams,
} from 'react-router-dom'

import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  Cpu,
  Download,
  FileText,
  HeartPulse,
  Info,
  LayoutDashboard,
  LogOut,
  Menu,
  Plus,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Stethoscope,
  TrendingUp,
  Users,
  X,
} from 'lucide-react'

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import './styles.css'


/* =========================================================
   TYPES & SCHEMAS
   ========================================================= */

export type RiskLevel = 'High' | 'Medium' | 'Low'

export interface Patient {
  id: string
  name: string
  age: number
  gender: 'Male' | 'Female'
  diagnosis: string
  risk: RiskLevel
  score: number
  lastVisit: string
  initials: string
  color: string
  email: string

  clinical: {
    hba1c: string
    glucose: string
    creatinine: string
    egfr: string
    albumin: string
    bp: string
    heartRate: string
    diabetesDuration: string
    bmi: number
    uacr?: string
    urineDipstick?: string
  }

  wearable: {
    steps: number
    sleep: number
    rhr: number
    exercise: number
    spo2: number
    hrv: number
    stress: string
  }
}

export interface FeatureContribution {
  feature: string
  display_name?: string
  contribution: number
}

export interface ClinicalMarkers {
  creatinine_mgdL?: number
  creatinine_status?: string
  egfr_ml_min?: number
  ckd_stage?: string
  uacr_mg_g?: number
  albuminuria_category?: string
  urine_dipstick?: string
  dipstick_note?: string
  marker_layer_note?: string
}

export interface ApiPrediction {
  prediction: string
  probability: number
  risk_score: number
  risk_category: string
  threshold: number
  model_version: string
  model_type?: string
  top_features: FeatureContribution[]
  attribution_method?: string
  attribution_available?: boolean
  input_values?: Record<string, number>
  assessment_timestamp?: string
  clinical_markers?: ClinicalMarkers
  disclaimer: string
}

export type ModelFeatures = {
  Age: number
  Gender: string
  BMI: number
  Systolic_BP: number
  Diastolic_BP: number
  Heart_Rate: number
  SpO2: number
  HbA1c: number
  creatinine?: number
  egfr?: number
  uacr?: number
  urine_dipstick?: string
}


/* =========================================================
   CONFIGURATION & CONSTANTS
   ========================================================= */

const API_URL = 'http://127.0.0.1:8000'
const MODEL_VERSION = 'nephromind_dnn_v1'
const LOCKED_THRESHOLD = 0.56

export const FEATURE_METADATA: Record<
  string,
  { label: string; unit: string; category: string; format: (val: number | string) => string }
> = {
  Age: {
    label: 'Age',
    unit: 'yrs',
    category: 'Demographics',
    format: val => `${val} yrs`,
  },
  Gender: {
    label: 'Gender',
    unit: '',
    category: 'Demographics',
    format: val => `${val}`,
  },
  BMI: {
    label: 'BMI',
    unit: 'kg/m²',
    category: 'Anthropometric',
    format: val => `${Number(val).toFixed(1)} kg/m²`,
  },
  Systolic_BP: {
    label: 'Systolic BP',
    unit: 'mmHg',
    category: 'Renal Sentinel Vitals',
    format: val => `${Math.round(Number(val))} mmHg`,
  },
  Diastolic_BP: {
    label: 'Diastolic BP',
    unit: 'mmHg',
    category: 'Renal Sentinel Vitals',
    format: val => `${Math.round(Number(val))} mmHg`,
  },
  Heart_Rate: {
    label: 'Heart Rate',
    unit: 'bpm',
    category: 'Renal Sentinel Vitals',
    format: val => `${Math.round(Number(val))} bpm`,
  },
  SpO2: {
    label: 'SpO₂',
    unit: '%',
    category: 'Renal Sentinel Vitals',
    format: val => `${Number(val).toFixed(0)}%`,
  },
  HbA1c: {
    label: 'HbA1c',
    unit: '%',
    category: 'Glycemic Biomarker',
    format: val => `${Number(val).toFixed(1)}%`,
  },
}


/* =========================================================
   DEMO PATIENT REGISTRY
   ========================================================= */

const names = [
  'Eleanor Brooks',
  'Marcus Chen',
  'Priya Sharma',
  'Robert Wilson',
  'Aisha Patel',
  'Daniel Kim',
  'Sarah Thompson',
  'James Miller',
  'Kavita Rao',
  'Michael Garcia',
  'Linda Anderson',
  'Rahul Verma',
  'Olivia Brown',
  'David Lee',
  'Meera Iyer',
  'John Carter',
  'Sofia Martinez',
  'Arjun Nair',
  'Emily Davis',
  'William Moore',
  'Nina Kapoor',
  'Henry Clark',
  'Ananya Singh',
  'George Hall',
  'Maya Lewis',
  'Thomas Young',
  'Riya Shah',
  'Edward King',
  'Zara Khan',
  'Paul White',
  'Ishita Bose',
  'Frank Adams',
  'Neha Gupta',
  'Victor Scott',
  'Ava Martin',
  'Noah Turner',
  'Simran Kaur',
  'Luke Evans',
  'Pooja Das',
  'Jack Harris',
  'Leah Walker',
  'Sam Reed',
  'Tara Roy',
  'Owen Green',
  'Nisha Jain',
  'Leo Baker',
  'Anita Menon',
  'Adam Wood',
  'Rhea Mehta',
  'Ben Cooper',
]

const avatarColors = [
  '#dbeafe',
  '#ede9fe',
  '#dcfce7',
  '#fce7f3',
  '#fef3c7',
]

export const patients: Patient[] = names.map((name, i) => {
  const risk: RiskLevel =
    i % 7 < 2
      ? 'High'
      : i % 7 < 4
        ? 'Medium'
        : 'Low'

  const age = 42 + ((i * 7) % 37)

  // Derive realistic clinical marker status for demo patients
  const hasKidneyLab = i % 4 !== 3 // 75% have lab data, 25% pending
  const creatVal = hasKidneyLab ? (0.8 + (i % 22) / 10).toFixed(1) : ''
  const egfrVal = hasKidneyLab ? `${38 + (i % 53)}` : ''
  const uacrVal = hasKidneyLab ? `${15 + (i * 19) % 290}` : undefined
  const dipstickVal = hasKidneyLab ? (i % 5 === 0 ? '2+' : i % 3 === 0 ? '1+' : 'Negative') : undefined

  return {
    id: `NM-${String(2401 + i).padStart(4, '0')}`,
    name,
    age,
    gender: i % 2 ? 'Male' : 'Female',

    diagnosis:
      i % 3 === 0
        ? 'Diabetic Kidney Disease — Stage 3'
        : i % 3 === 1
          ? 'Type 2 Diabetes Mellitus'
          : 'Essential Hypertension with Microalbuminuria',

    risk,

    score:
      risk === 'High'
        ? 72 + (i % 21)
        : risk === 'Medium'
          ? 42 + (i % 22)
          : 12 + (i % 22),

    lastVisit: `Jul ${30 - (i % 20)}, 2026`,

    initials: name
      .split(' ')
      .map(part => part[0])
      .join(''),

    color: avatarColors[i % avatarColors.length],

    email: `${name.toLowerCase().replace(/\s+/g, '.')}@example.com`,

    clinical: {
      hba1c: (6.8 + (i % 35) / 10).toFixed(1) + '%',
      glucose: `${112 + (i % 78)} mg/dL`,
      creatinine: creatVal ? `${creatVal} mg/dL` : 'Pending lab',
      egfr: egfrVal ? `${egfrVal} mL/min` : 'Pending lab',
      albumin: (3.4 + (i % 11) / 10).toFixed(1) + ' g/dL',
      bp: `${112 + (i % 31)}/${72 + (i % 18)} mmHg`,
      heartRate: `${64 + (i % 20)} bpm`,
      diabetesDuration: `${5 + (i % 19)} years`,
      bmi: Number((23.5 + (i % 48) / 10).toFixed(1)),
      uacr: uacrVal ? `${uacrVal} mg/g` : undefined,
      urineDipstick: dipstickVal,
    },

    wearable: {
      steps: 3200 + ((i * 137) % 6900),
      sleep: 5.6 + (i % 25) / 10,
      rhr: 62 + (i % 20),
      exercise: 18 + (i % 48),
      spo2: 95 + (i % 5),
      hrv: 24 + ((i * 3) % 34),
      stress: ['Low', 'Moderate', 'High'][i % 3],
    },
  }
})


/* =========================================================
   COLORS & UTILITIES
   ========================================================= */

const riskColors: Record<RiskLevel, string> = {
  High: '#d95763',
  Medium: '#d99a35',
  Low: '#24966b',
}

function getRiskFromApi(
  data: ApiPrediction | null,
  fallback: RiskLevel = 'Low'
): RiskLevel {
  if (!data) return fallback

  const category = data.risk_category?.toLowerCase() || ''

  if (category.includes('higher') || data.probability >= LOCKED_THRESHOLD + 0.19) {
    return 'High'
  }

  if (category.includes('moderate') || data.probability >= LOCKED_THRESHOLD) {
    return 'Medium'
  }

  return 'Low'
}

function getPatientFeatures(patient: Patient): ModelFeatures {
  const bpParts = patient.clinical.bp.split('/')
  const systolic = Number.parseFloat(bpParts[0])
  const diastolic = Number.parseFloat(bpParts[1])

  const creatClean = patient.clinical.creatinine.replace(/[^0-9.]/g, '')
  const creatinineVal = Number.parseFloat(creatClean)

  const egfrClean = patient.clinical.egfr.replace(/[^0-9.]/g, '')
  const egfrVal = Number.parseFloat(egfrClean)

  const uacrClean = patient.clinical.uacr?.replace(/[^0-9.]/g, '')
  const uacrVal = uacrClean ? Number.parseFloat(uacrClean) : undefined

  return {
    Age: patient.age,
    Gender: patient.gender === 'Male' ? 'Male' : 'Female',
    BMI: patient.clinical.bmi,
    Systolic_BP: Number.isFinite(systolic) ? systolic : 120,
    Diastolic_BP: Number.isFinite(diastolic) ? diastolic : 80,
    Heart_Rate: Number.parseFloat(patient.clinical.heartRate) || 70,
    SpO2: patient.wearable.spo2,
    HbA1c: Number.parseFloat(patient.clinical.hba1c) || 7.0,
    creatinine: Number.isFinite(creatinineVal) ? creatinineVal : undefined,
    egfr: Number.isFinite(egfrVal) ? egfrVal : undefined,
    uacr: uacrVal && Number.isFinite(uacrVal) ? uacrVal : undefined,
    urine_dipstick: patient.clinical.urineDipstick,
  }
}

/**
 * Generates an automated, clinically oriented explanation narrative
 * analyzing the DNN probability vs locked threshold and individual feature attributions.
 */
function generateClinicalNarrative(
  patientName: string,
  probability: number,
  threshold: number,
  riskCategory: string,
  features: FeatureContribution[],
  inputValues?: Record<string, number>
): string {
  const probPct = (probability * 100).toFixed(1)
  const threshPct = (threshold * 100).toFixed(1)
  const isAboveThreshold = probability >= threshold

  const positiveContribs = features
    .filter(f => f.contribution > 0.001)
    .sort((a, b) => b.contribution - a.contribution)

  const negativeContribs = features
    .filter(f => f.contribution < -0.001)
    .sort((a, b) => a.contribution - b.contribution)

  const formatFeat = (feat: FeatureContribution, isPos: boolean) => {
    const meta = FEATURE_METADATA[feat.feature]
    const name = feat.display_name || meta?.label || feat.feature
    const rawVal = inputValues ? inputValues[feat.feature] : undefined
    const valText = rawVal !== undefined && meta ? ` (${meta.format(rawVal)})` : ''
    const sign = isPos ? '+' : ''
    return `${name}${valText} [${sign}${feat.contribution.toFixed(3)}]`
  }

  let text = `For ${patientName}, the PyTorch DNN model evaluates an output risk probability of ${probPct}% (${riskCategory}), which ${
    isAboveThreshold
      ? `exceeds the locked decision threshold of ${threshPct}% (t = ${threshold})`
      : `remains below the locked decision threshold of ${threshPct}% (t = ${threshold})`
  }. `

  if (positiveContribs.length > 0) {
    const topPos = positiveContribs.slice(0, 2).map(f => formatFeat(f, true)).join(' and ')
    text += `The primary physiological risk drivers elevating the assessment are ${topPos}. `
  }

  if (negativeContribs.length > 0) {
    const topNeg = negativeContribs.slice(0, 2).map(f => formatFeat(f, false)).join(' and ')
    text += `In contrast, ${topNeg} provide protective downward contributions on the model output. `
  } else if (positiveContribs.length === 0) {
    text += `All evaluated physiological features demonstrate neutral or baseline risk influence. `
  }

  text += `Attributions represent Gradient × Input sensitivity on standardized Renal Sentinel hardware features w.r.t. model sigmoid probability.`

  return text
}


/* =========================================================
   SHARED UI COMPONENTS
   ========================================================= */

function Logo() {
  return (
    <div className="brand">
      <span className="brandmark">
        <HeartPulse size={20} />
      </span>
      <span>
        NephroMind <b>AI</b>
      </span>
    </div>
  )
}

function RiskBadge({ risk }: { risk: RiskLevel }) {
  return (
    <span className={`risk ${risk.toLowerCase()}`}>
      <i />
      {risk === 'High'
        ? 'Higher'
        : risk === 'Medium'
          ? 'Moderate'
          : 'Lower'}{' '}
      risk
    </span>
  )
}

function Avatar({
  patient,
  large = false,
}: {
  patient: Patient
  large?: boolean
}) {
  return (
    <div
      className={large ? 'avatar avatar-lg' : 'avatar'}
      style={{
        background: patient.color,
      }}
    >
      {patient.initials}
    </div>
  )
}

function Metric({
  label,
  value,
  hint,
}: {
  label: string
  value: string | number
  hint?: string
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint && <small>{hint}</small>}
    </div>
  )
}

function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string
  title: string
  description?: string
  action?: React.ReactNode
}) {
  return (
    <div className="page-title">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="muted">{description}</p>}
      </div>
      {action}
    </div>
  )
}

function Stat({
  label,
  value,
  detail,
  kind,
}: {
  label: string
  value: string
  detail: string
  kind: 'patients' | 'High' | 'Medium' | 'Low'
}) {
  const Icon =
    kind === 'patients'
      ? Users
      : kind === 'High'
        ? AlertTriangle
        : kind === 'Medium'
          ? Clock3
          : CheckCircle2

  return (
    <div className="stat card">
      <div className={`stat-icon ${String(kind).toLowerCase()}`}>
        <Icon size={19} />
      </div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </div>
  )
}


/* =========================================================
   EXPLAINABLE AI (XAI) COMPONENT
   ========================================================= */

interface ExplainableAISectionProps {
  apiData: ApiPrediction
  patientName: string
}

function ExplainableAISection({ apiData, patientName }: ExplainableAISectionProps) {
  const features = apiData.top_features || []
  const inputVals = apiData.input_values || {}
  const prob = apiData.probability
  const threshold = apiData.threshold || LOCKED_THRESHOLD

  // Calculate max absolute contribution to scale zero-center bars
  const maxAbsContrib = useMemo(() => {
    if (!features.length) return 0.2
    const maxVal = Math.max(...features.map(f => Math.abs(f.contribution)))
    return Math.max(0.12, maxVal * 1.1)
  }, [features])

  const narrative = useMemo(() => {
    return generateClinicalNarrative(
      patientName,
      prob,
      threshold,
      apiData.risk_category,
      features,
      inputVals
    )
  }, [patientName, prob, threshold, apiData.risk_category, features, inputVals])

  return (
    <section className="card panel explain-card-extended">
      {/* CARD HEADER */}
      <div className="card-head">
        <div>
          <div className="xai-header-badge">
            <Sparkles size={14} />
            <span>Explainable AI Engine</span>
          </div>
          <h3>Model Feature Attributions & Explainability</h3>
          <p>
            Feature contributions computed via Gradient × Input on standardized features w.r.t. PyTorch DNN output.
          </p>
        </div>
        <div className="xai-meta-tag">
          <span>Method:</span>
          <b>Gradient × Input (Scaled)</b>
        </div>
      </div>

      {/* 1. DNN PROBABILITY VS LOCKED THRESHOLD COMPARISON */}
      <div className="xai-threshold-comparison">
        <div className="threshold-gauge-header">
          <div>
            <span className="gauge-label">DNN Raw Output Probability</span>
            <strong className="gauge-value">{(prob * 100).toFixed(2)}%</strong>
            <small className="gauge-sub">p = {prob.toFixed(4)}</small>
          </div>
          <div className="gauge-center-badge">
            {prob >= threshold ? (
              <span className="badge-exceeds">
                <AlertTriangle size={13} />
                Exceeds Decision Threshold ({threshold})
              </span>
            ) : (
              <span className="badge-below">
                <CheckCircle2 size={13} />
                Below Decision Threshold ({threshold})
              </span>
            )}
          </div>
          <div style={{ textAlign: 'right' }}>
            <span className="gauge-label">Locked Decision Threshold</span>
            <strong className="gauge-value">{(threshold * 100).toFixed(1)}%</strong>
            <small className="gauge-sub">t = {threshold.toFixed(2)} (Validation-locked)</small>
          </div>
        </div>

        {/* Visual Threshold Bar */}
        <div className="threshold-bar-track">
          <div
            className="threshold-bar-fill"
            style={{
              width: `${Math.min(100, Math.max(0, prob * 100))}%`,
              background:
                prob >= threshold
                  ? 'linear-gradient(90deg, #d99a35 0%, #d95763 100%)'
                  : 'linear-gradient(90deg, #24966b 0%, #2878ca 100%)',
            }}
          />
          {/* Threshold Marker */}
          <div
            className="threshold-marker-line"
            style={{ left: `${threshold * 100}%` }}
          >
            <span className="threshold-marker-tooltip">
              Threshold {threshold}
            </span>
          </div>
        </div>

        <div className="threshold-bar-labels">
          <span>0.0 (Low Risk Zone)</span>
          <span className="threshold-mid-label">Threshold: {threshold}</span>
          <span>1.0 (High Risk Zone)</span>
        </div>
      </div>

      {/* 2. ZERO-REFERENCE BIDIRECTIONAL CONTRIBUTION CHART */}
      <div className="xai-chart-section">
        <div className="xai-chart-title-row">
          <h4>Renal Sentinel 8-Feature Vector Attributions</h4>
          <div className="xai-legend">
            <span className="legend-item protective">
              <span className="legend-swatch negative" />
              <span>Protective (Pulls Risk Down)</span>
            </span>
            <span className="legend-item risk-driver">
              <span className="legend-swatch positive" />
              <span>Risk Driver (Pushes Risk Up)</span>
            </span>
          </div>
        </div>

        <div className="xai-zero-chart-container">
          {/* Header row */}
          <div className="zero-chart-head">
            <span className="col-feature">Feature & Input Value</span>
            <span className="col-bar-left">Protective (Negative Influence)</span>
            <span className="col-axis-zero">0.0</span>
            <span className="col-bar-right">Risk Driver (Positive Influence)</span>
            <span className="col-score">Attribution</span>
          </div>

          {/* Feature rows */}
          {features.map(feat => {
            const meta = FEATURE_METADATA[feat.feature]
            const displayName = feat.display_name || meta?.label || feat.feature
            const rawVal = inputVals[feat.feature]
            const formattedVal = rawVal !== undefined && meta ? meta.format(rawVal) : rawVal ?? '—'
            const contrib = feat.contribution
            const isPositive = contrib >= 0
            const absWidth = Math.min(100, (Math.abs(contrib) / maxAbsContrib) * 100)

            return (
              <div className="zero-chart-row" key={feat.feature}>
                {/* Feature Name & Value */}
                <div className="col-feature">
                  <b>{displayName}</b>
                  <span className="feature-raw-pill">{formattedVal}</span>
                </div>

                {/* Left (Negative) Bar */}
                <div className="col-bar-left">
                  {!isPositive && (
                    <div
                      className="bar-negative"
                      style={{ width: `${absWidth}%` }}
                      title={`Negative attribution: ${contrib.toFixed(4)}`}
                    >
                      <span className="bar-val-label">{contrib.toFixed(3)}</span>
                    </div>
                  )}
                </div>

                {/* Center Zero Axis */}
                <div className="col-axis-zero">
                  <div className="zero-line" />
                </div>

                {/* Right (Positive) Bar */}
                <div className="col-bar-right">
                  {isPositive && (
                    <div
                      className="bar-positive"
                      style={{ width: `${absWidth}%` }}
                      title={`Positive attribution: +${contrib.toFixed(4)}`}
                    >
                      <span className="bar-val-label">+{contrib.toFixed(3)}</span>
                    </div>
                  )}
                </div>

                {/* Numeric score & Badge */}
                <div className="col-score">
                  <span
                    className={
                      isPositive
                        ? 'xai-score-badge positive'
                        : 'xai-score-badge negative'
                    }
                  >
                    {isPositive ? (
                      <ArrowUpRight size={12} />
                    ) : (
                      <ArrowDownRight size={12} />
                    )}
                    {isPositive ? `+${contrib.toFixed(3)}` : contrib.toFixed(3)}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 3. DYNAMIC CLINICAL NARRATIVE */}
      <div className="xai-narrative-box">
        <div className="narrative-head">
          <Info size={15} />
          <b>Automated Clinical Attribution Summary</b>
        </div>
        <p>{narrative}</p>
      </div>

      {/* 4. TECHNICAL FOOTNOTE / DISCLAIMER */}
      <div className="xai-footnote">
        <ShieldCheck size={14} />
        <span>
          <b>Attribution Methodology:</b> Computed using PyTorch input gradient multiplied by the normalized feature vector (
          <em>Gradient × Input</em>, ∇x ŷ ⊙ x). Reflects model sensitivity for this patient's feature values.
          Feature contributions explain DNN output formation and do not constitute biological or therapeutic causation.
        </span>
      </div>
    </section>
  )
}


/* =========================================================
   NAVIGATION LAYOUT
   ========================================================= */

const navItems = [
  ['/dashboard', LayoutDashboard, 'Dashboard'],
  ['/patients', Users, 'Patients'],
  ['/predictions', Activity, 'Predictions'],
  ['/reports', FileText, 'Reports'],
  ['/analytics', TrendingUp, 'Analytics'],
  ['/settings', Settings, 'Settings'],
] as const

function Layout({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="app">
      <aside className={open ? 'open' : ''}>
        <Logo />

        <button
          className="close"
          onClick={() => setOpen(false)}
          aria-label="Close navigation"
        >
          <X />
        </button>

        <div className="nav-label">CLINICAL WORKSPACE</div>

        <nav>
          {navItems.map(([to, Icon, label]) => (
            <NavLink
              to={to}
              key={to}
              onClick={() => setOpen(false)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="aside-bottom">
          <div className="system-status">
            <ShieldCheck size={18} />
            <div>
              <b>Clinical system</b>
              <small>DNN Prototype · v1.0.0</small>
            </div>
          </div>

          <NavLink to="/">
            <LogOut size={18} />
            <span>Sign out</span>
          </NavLink>
        </div>
      </aside>

      <main>
        <header>
          <button
            className="mobile-menu"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
          >
            <Menu />
          </button>

          <div className="crumb">
            <span>Clinical workspace</span>
            <ChevronRight size={14} />
            <b>NephroMind AI</b>
          </div>

          <div className="top-actions">
            <button className="icon-btn" aria-label="Notifications">
              <Bell size={18} />
              <em />
            </button>

            <div className="doctor">
              <div className="doctorpic">DR</div>
              <span>
                <b>Dr. R. Sharma</b>
                <small>Nephrology</small>
              </span>
              <ChevronDown size={14} />
            </div>
          </div>
        </header>

        {children}
      </main>
    </div>
  )
}


/* =========================================================
   DASHBOARD VIEW
   ========================================================= */

function Dashboard() {
  const high = patients.filter(p => p.risk === 'High').length
  const medium = patients.filter(p => p.risk === 'Medium').length
  const low = patients.filter(p => p.risk === 'Low').length

  const riskData = [
    { name: 'Higher risk', value: high, color: riskColors.High },
    { name: 'Moderate risk', value: medium, color: riskColors.Medium },
    { name: 'Lower risk', value: low, color: riskColors.Low },
  ]

  const assessmentActivity = [
    { day: 'Mon', assessments: 8 },
    { day: 'Tue', assessments: 12 },
    { day: 'Wed', assessments: 10 },
    { day: 'Thu', assessments: 15 },
    { day: 'Fri', assessments: 9 },
    { day: 'Sat', assessments: 6 },
    { day: 'Sun', assessments: 4 },
  ]

  const priorityPatients = [...patients]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)

  return (
    <Layout>
      <section className="page">
        <PageHeader
          eyebrow="THURSDAY · JUL 31, 2026"
          title="Good morning, Dr. Sharma"
          description="Clinical overview of the prototype renal care population."
          action={
            <button className="primary">
              <Plus size={17} />
              Add patient
            </button>
          }
        />

        <div className="stats">
          <Stat
            label="Total patients"
            value={String(patients.length)}
            detail="Prototype registry"
            kind="patients"
          />
          <Stat
            label="Higher risk"
            value={String(high)}
            detail={`${Math.round((high / patients.length) * 100)}% of registry`}
            kind="High"
          />
          <Stat
            label="Moderate risk"
            value={String(medium)}
            detail={`${Math.round((medium / patients.length) * 100)}% of registry`}
            kind="Medium"
          />
          <Stat
            label="Lower risk"
            value={String(low)}
            detail={`${Math.round((low / patients.length) * 100)}% of registry`}
            kind="Low"
          />
        </div>

        <div className="dashboard-grid">
          {/* ASSESSMENT ACTIVITY */}
          <section className="card panel">
            <div className="card-head">
              <div>
                <h3>Assessment activity</h3>
                <p>Recent prototype assessment volume</p>
              </div>
              <span className="panel-tag">Last 7 days</span>
            </div>

            <ResponsiveContainer width="100%" height={245}>
              <AreaChart data={assessmentActivity}>
                <defs>
                  <linearGradient id="assessmentFill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#2878ca" stopOpacity={0.18} />
                    <stop offset="100%" stopColor="#2878ca" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#e9eef3" />
                <XAxis
                  dataKey="day"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  allowDecimals={false}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="assessments"
                  stroke="#2878ca"
                  strokeWidth={2}
                  fill="url(#assessmentFill)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </section>

          {/* RISK DISTRIBUTION */}
          <section className="card panel">
            <div className="card-head">
              <div>
                <h3>Risk distribution</h3>
                <p>Current prototype registry</p>
              </div>
              <span className="panel-tag">Current</span>
            </div>

            <div className="dashboard-risk">
              <ResponsiveContainer width={190} height={200}>
                <PieChart>
                  <Pie
                    data={riskData}
                    dataKey="value"
                    innerRadius={58}
                    outerRadius={80}
                    paddingAngle={3}
                  >
                    {riskData.map(item => (
                      <Cell key={item.name} fill={item.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>

              <div className="risk-center">
                <strong>{patients.length}</strong>
                <span>patients</span>
              </div>

              <div className="legend">
                {riskData.map(item => (
                  <div key={item.name}>
                    <span>
                      <i style={{ background: item.color }} />
                      {item.name}
                    </span>
                    <b>{item.value}</b>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* PRIORITY PATIENTS */}
          <section className="card panel">
            <div className="card-head">
              <div>
                <h3>Priority patients</h3>
                <p>Highest prototype model scores</p>
              </div>
              <NavLink to="/patients" className="text-link">
                View registry
              </NavLink>
            </div>

            {priorityPatients.map(patient => (
              <NavLink
                key={patient.id}
                to={`/patients/${patient.id}`}
                className="recent"
              >
                <Avatar patient={patient} />
                <span>
                  <b>{patient.name}</b>
                  <small>
                    {patient.id} · Last visit {patient.lastVisit}
                  </small>
                </span>
                <RiskBadge risk={patient.risk} />
                <ChevronRight size={15} />
              </NavLink>
            ))}
          </section>

          {/* CLINICAL WORK QUEUE */}
          <section className="card panel">
            <div className="card-head">
              <div>
                <h3>Clinical work queue</h3>
                <p>Prototype review items</p>
              </div>
              <span className="count">3</span>
            </div>

            <div className="work-item urgent">
              <AlertTriangle />
              <div>
                <b>Review higher-risk assessments</b>
                <small>Review model scores above the locked threshold ({LOCKED_THRESHOLD}).</small>
              </div>
              <ChevronRight />
            </div>

            <div className="work-item">
              <CalendarDays />
              <div>
                <b>Follow-up review</b>
                <small>Check upcoming renal care appointments.</small>
              </div>
              <ChevronRight />
            </div>

            <div className="work-item">
              <Activity />
              <div>
                <b>Connected health data</b>
                <small>Wearable measurements are available for review.</small>
              </div>
              <ChevronRight />
            </div>
          </section>
        </div>

        <div className="prototype-note">
          <ShieldCheck size={17} />
          <div>
            <b>Prototype registry</b>
            <span>
              The 50-patient dashboard population is local demonstration data. It is separate
              from the model's hold-out evaluation dataset.
            </span>
          </div>
        </div>
      </section>
    </Layout>
  )
}


/* =========================================================
   PATIENTS REGISTRY VIEW
   ========================================================= */

function Patients() {
  const [query, setQuery] = useState('')
  const [riskFilter, setRiskFilter] = useState<'All' | RiskLevel>('All')

  const filteredPatients = useMemo(() => {
    return patients.filter(patient => {
      const searchText = `${patient.name} ${patient.id} ${patient.diagnosis}`.toLowerCase()
      const matchesSearch = searchText.includes(query.toLowerCase())
      const matchesRisk = riskFilter === 'All' || patient.risk === riskFilter
      return matchesSearch && matchesRisk
    })
  }, [query, riskFilter])

  return (
    <Layout>
      <section className="page">
        <PageHeader
          eyebrow="PATIENT MANAGEMENT"
          title="Patients"
          description="Manage and review the prototype renal care registry."
          action={
            <button className="primary">
              <Plus size={17} />
              Add patient
            </button>
          }
        />

        <section className="card table-card">
          <div className="table-tools">
            <label className="search">
              <Search size={17} />
              <input
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Search patient, ID or diagnosis"
              />
            </label>

            <select
              className="filter-select"
              value={riskFilter}
              onChange={e => setRiskFilter(e.target.value as 'All' | RiskLevel)}
            >
              <option value="All">All risk levels</option>
              <option value="High">Higher risk</option>
              <option value="Medium">Moderate risk</option>
              <option value="Low">Lower risk</option>
            </select>

            <button className="secondary">
              <SlidersHorizontal size={16} />
              Filters
            </button>
          </div>

          <div className="registry-meta">
            <span>Prototype registry</span>
            <b>{filteredPatients.length} matching patients</b>
          </div>

          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Patient</th>
                  <th>Patient ID</th>
                  <th>Age / Gender</th>
                  <th>Diagnosis</th>
                  <th>Risk status</th>
                  <th>Model score</th>
                  <th>Last visit</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {filteredPatients.slice(0, 15).map(patient => (
                  <tr key={patient.id}>
                    <td>
                      <div className="person">
                        <Avatar patient={patient} />
                        <span>
                          <b>{patient.name}</b>
                          <small>{patient.email}</small>
                        </span>
                      </div>
                    </td>
                    <td>{patient.id}</td>
                    <td>
                      {patient.age} · {patient.gender}
                    </td>
                    <td>{patient.diagnosis}</td>
                    <td>
                      <RiskBadge risk={patient.risk} />
                    </td>
                    <td>
                      <b className="score-text">{patient.score}%</b>
                    </td>
                    <td>{patient.lastVisit}</td>
                    <td>
                      <NavLink
                        to={`/patients/${patient.id}`}
                        className="view"
                      >
                        View
                        <ChevronRight size={14} />
                      </NavLink>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <span>
              Showing 1–{Math.min(15, filteredPatients.length)} of {filteredPatients.length}
            </span>
            <div>
              <button>Previous</button>
              <button className="selected">1</button>
              <button>2</button>
              <button>3</button>
              <button>Next</button>
            </div>
          </div>
        </section>
      </section>
    </Layout>
  )
}


/* =========================================================
   PATIENT PROFILE VIEW
   ========================================================= */

function Profile() {
  const { id } = useParams()
  const patient = patients.find(item => item.id === id) || patients[0]

  const [prediction, setPrediction] = useState<ApiPrediction | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let active = true

    const loadAssessment = async () => {
      setLoading(true)
      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(getPatientFeatures(patient)),
        })

        if (!response.ok) throw new Error()
        const data = await response.json()
        if (active) setPrediction(data)
      } catch {
        if (active) setPrediction(null)
      } finally {
        if (active) setLoading(false)
      }
    }

    loadAssessment()
    return () => {
      active = false
    }
  }, [id])

  const modelRisk = getRiskFromApi(prediction, patient.risk)

  const modelInputs = [
    ['Age', `${patient.age} years`],
    ['Gender', patient.gender],
    ['BMI', `${patient.clinical.bmi} kg/m²`],
    ['Systolic BP', `${patient.clinical.bp.split('/')[0]} mmHg`],
    ['Diastolic BP', `${patient.clinical.bp.split('/')[1]} mmHg`],
    ['Heart rate', patient.clinical.heartRate],
    ['SpO₂', `${patient.wearable.spo2}%`],
    ['HbA1c', patient.clinical.hba1c],
  ]

  return (
    <Layout>
      <section className="page">
        <div className="breadcrumbs">
          <NavLink to="/patients">Patients</NavLink>
          <ChevronRight size={14} />
          <span>{patient.name}</span>
        </div>

        <div className="profile-head card">
          <Avatar patient={patient} large />
          <div className="profile-title">
            <h1>{patient.name}</h1>
            <p>
              {patient.id} · {patient.age} years · {patient.gender}
            </p>
            <span>
              <Stethoscope size={14} />
              Nephrology care workspace
            </span>
          </div>

          <RiskBadge risk={modelRisk} />

          <NavLink to="/predictions" className="secondary">
            Open assessment
            <ChevronRight size={15} />
          </NavLink>
        </div>

        <div className="profile-grid">
          <div className="profile-main">
            {/* CLINICAL MEASUREMENTS */}
            <section className="card panel">
              <div className="card-head">
                <div>
                  <h3>Clinical measurements</h3>
                  <p>Current laboratory measurements for physician review</p>
                </div>
                <span className="panel-tag">Clinical data</span>
              </div>

              <div className="feature-grid">
                <Metric label="Diagnosis" value={patient.diagnosis} />
                <Metric label="HbA1c" value={patient.clinical.hba1c} />
                <Metric label="Creatinine" value={patient.clinical.creatinine} />
                <Metric label="eGFR" value={patient.clinical.egfr} />
                <Metric label="Blood pressure" value={patient.clinical.bp} />
                <Metric label="Heart rate" value={patient.clinical.heartRate} />
                <Metric label="Glucose" value={patient.clinical.glucose} />
                <Metric label="Diabetes duration" value={patient.clinical.diabetesDuration} />
              </div>
            </section>

            {/* MODEL INPUTS */}
            <section className="card panel">
              <div className="card-head">
                <div>
                  <h3>Renal Sentinel Model Inputs</h3>
                  <p>8 physiological & clinical features submitted to PyTorch DNN</p>
                </div>
                <span className="panel-tag">8 features (DNN)</span>
              </div>

              <div className="feature-grid">
                {modelInputs.map(([label, value]) => (
                  <Metric key={label} label={label} value={value} />
                ))}
              </div>
            </section>

            {/* WEARABLE */}
            <section className="card panel">
              <div className="card-head">
                <div>
                  <h3>Connected health</h3>
                  <p>Recent wearable measurements · prototype integration</p>
                </div>
                <span className="connected">● Connected</span>
              </div>

              <div className="wearable-grid">
                <Metric
                  label="Daily steps"
                  value={patient.wearable.steps.toLocaleString()}
                  hint="steps / day"
                />
                <Metric
                  label="Sleep"
                  value={`${patient.wearable.sleep.toFixed(1)} h`}
                  hint="hours / night"
                />
                <Metric
                  label="Activity"
                  value={`${patient.wearable.exercise} min`}
                  hint="minutes / day"
                />
                <Metric label="Resting HR" value={`${patient.wearable.rhr} bpm`} />
                <Metric label="SpO₂" value={`${patient.wearable.spo2}%`} />
                <Metric label="HRV" value={`${patient.wearable.hrv} ms`} />
              </div>
            </section>
          </div>

          <aside className="right-stack">
            {/* ASSESSMENT */}
            <section className="card assessment-mini">
              <div className="eyebrow">LATEST ASSESSMENT</div>
              <div className="assessment-score">
                {loading ? '—' : prediction ? `${prediction.risk_score}%` : `${patient.score}%`}
              </div>
              <RiskBadge risk={modelRisk} />
              <p>Model Risk Score</p>
              <NavLink to="/predictions" className="primary full">
                Review prediction
                <ChevronRight size={16} />
              </NavLink>
            </section>

            {/* MODEL CONTRIBUTORS */}
            <section className="card panel">
              <div className="card-head">
                <div>
                  <h3>Model contributors</h3>
                  <p>Gradient × Input attributions for all 8 features</p>
                </div>
              </div>

              {prediction?.top_features?.length ? (
                prediction.top_features.map(feature => {
                  const isPos = feature.contribution >= 0
                  return (
                    <div className="contributor" key={feature.feature}>
                      <span>{feature.display_name || feature.feature}</span>
                      <b className={isPos ? 'positive-text' : 'negative-text'}>
                        {isPos ? '+' : ''}
                        {feature.contribution.toFixed(3)}
                      </b>
                    </div>
                  )
                })
              ) : (
                <div className="empty-inline">
                  {loading
                    ? 'Loading live contributors…'
                    : 'Start the backend assessment to view model-returned contributors.'}
                </div>
              )}
            </section>
          </aside>
        </div>

        <div className="prototype-note">
          <ShieldCheck size={17} />
          <div>
            <b>Clinical interpretation & Model Architecture</b>
            <span>
              Creatinine, eGFR, UACR, and urine dipstick are evaluated in a separate rule-based clinical marker
              layer. They are intentionally NOT inputs to the 8-feature PyTorch DNN to prevent target leakage.
            </span>
          </div>
        </div>
      </section>
    </Layout>
  )
}


/* =========================================================
   PREDICTION WORKSPACE (LIVE INFERENCE & XAI)
   ========================================================= */

function Prediction() {
  const [selectedId, setSelectedId] = useState(patients[0].id)
  const patient = patients.find(item => item.id === selectedId) || patients[0]

  const [formData, setFormData] = useState<ModelFeatures>(getPatientFeatures(patient))
  const [apiData, setApiData] = useState<ApiPrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runPrediction = async (data: ModelFeatures = formData) => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })

      const payload = await response.json().catch(() => null)

      if (!response.ok) {
        throw new Error(
          payload?.detail ||
            payload?.message ||
            'Inference server error. Please verify the backend service.'
        )
      }

      setApiData(payload)
      localStorage.setItem(
        'nephromind_last_assessment',
        JSON.stringify({
          patient_id: selectedId,
          input: data,
          prediction: payload,
        })
      )
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to connect to the prediction backend.'
      )
    } finally {
      setLoading(false)
    }
  }

  const handlePatientSelect = (id: string) => {
    setSelectedId(id)
    const selected = patients.find(item => item.id === id) || patients[0]
    const updated = getPatientFeatures(selected)
    setFormData(updated)
    setApiData(null)
    runPrediction(updated)
  }

  useEffect(() => {
    runPrediction(getPatientFeatures(patient))
  }, [])

  const risk = getRiskFromApi(apiData, 'Medium')

  const updateField = (key: keyof ModelFeatures, value: string) => {
    setFormData(previous => {
      if (key === 'Gender') {
        return {
          ...previous,
          Gender: value,
        }
      }
      if (key === 'urine_dipstick') {
        return {
          ...previous,
          urine_dipstick: value,
        }
      }
      return {
        ...previous,
        [key]: Number(value),
      }
    })
  }

  const groups = [
    {
      title: 'Demographics',
      fields: [
        ['Age', 'Age'],
        ['Gender', 'Gender'],
        ['BMI', 'BMI'],
      ],
    },
    {
      title: 'Vitals (Renal Sentinel Station)',
      fields: [
        ['Systolic BP', 'Systolic_BP'],
        ['Diastolic BP', 'Diastolic_BP'],
        ['Heart rate', 'Heart_Rate'],
        ['SpO₂', 'SpO2'],
      ],
    },
    {
      title: 'Glycemic Biomarker',
      fields: [['HbA1c', 'HbA1c']],
    },
  ] as const

  return (
    <Layout>
      <section className="page">
        <PageHeader
          eyebrow="CLINICAL DECISION SUPPORT"
          title="AI assessment"
          description={`Live prototype assessment using the NephroMind AI 8-feature PyTorch DNN engine with locked decision threshold ${LOCKED_THRESHOLD}.`}
          action={
            <div className="button-row">
              <select
                className="secondary select"
                value={selectedId}
                onChange={e => handlePatientSelect(e.target.value)}
              >
                {patients.map(item => (
                  <option key={item.id} value={item.id}>
                    {item.name} ({item.id})
                  </option>
                ))}
              </select>

              <button
                className="secondary"
                onClick={() => runPrediction()}
                disabled={loading}
              >
                <Activity size={16} />
                {loading ? 'Analyzing…' : 'Run assessment'}
              </button>
            </div>
          }
        />

        {error && (
          <div className="notice error">
            <AlertTriangle size={17} />
            <div>
              <b>Prediction service notice</b>
              <span>{error}</span>
            </div>
          </div>
        )}

        <div className="prediction-workspace">
          {/* INPUT WORKSPACE */}
          <section className="card assessment-panel">
            <div className="assessment-top">
              <div>
                <span className="section-label">PATIENT ASSESSMENT</span>
                <h2>{patient.name}</h2>
                <p>
                  {patient.id} · {patient.age} years · {patient.gender}
                </p>
              </div>

              {apiData && <RiskBadge risk={risk} />}
            </div>

            <div className="form-groups">
              {groups.map(group => (
                <div className="input-group" key={group.title}>
                  <h4>{group.title}</h4>
                  <div className="input-grid">
                    {group.fields.map(([label, key]) => (
                      <label key={key}>
                        {label}
                        {key === 'Gender' ? (
                          <select
                            value={formData.Gender}
                            onChange={e => updateField('Gender', e.target.value)}
                          >
                            <option value="Female">Female</option>
                            <option value="Male">Male</option>
                          </select>
                        ) : (
                          <input
                            type="number"
                            step={
                              key === 'BMI' || key === 'HbA1c'
                                ? '0.1'
                                : '1'
                            }
                            value={String(formData[key as keyof ModelFeatures] ?? '')}
                            onChange={e => updateField(key as keyof ModelFeatures, e.target.value)}
                          />
                        )}
                      </label>
                    ))}
                  </div>
                </div>
              ))}

              {/* OPTIONAL CLINICAL LAYER INPUTS */}
              <div className="input-group" style={{ opacity: 0.95 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <h4 style={{ margin: 0 }}>Clinical Kidney Markers (Rule-Based Layer only)</h4>
                  <span style={{ fontSize: '9px', color: '#6d8197', fontWeight: 600 }}>Separate from DNN</span>
                </div>
                <div className="input-grid">
                  <label>
                    Creatinine (mg/dL)
                    <input
                      type="number"
                      step="0.1"
                      placeholder="Optional"
                      value={formData.creatinine ?? ''}
                      onChange={e => updateField('creatinine', e.target.value)}
                    />
                  </label>
                  <label>
                    eGFR (mL/min)
                    <input
                      type="number"
                      step="1"
                      placeholder="Optional"
                      value={formData.egfr ?? ''}
                      onChange={e => updateField('egfr', e.target.value)}
                    />
                  </label>
                  <label>
                    UACR (mg/g)
                    <input
                      type="number"
                      step="1"
                      placeholder="Optional"
                      value={formData.uacr ?? ''}
                      onChange={e => updateField('uacr', e.target.value)}
                    />
                  </label>
                </div>
              </div>
            </div>

            <div className="form-footer">
              <span>
                <ShieldCheck size={15} />
                8 approved Renal Sentinel features submitted to DNN.
              </span>

              <button
                className="primary"
                onClick={() => runPrediction()}
                disabled={loading}
              >
                {loading ? 'Analyzing…' : 'Calculate model risk'}
                <ChevronRight size={16} />
              </button>
            </div>
          </section>

          {/* MODEL OUTPUT SUMMARY */}
          <section className="card result-panel">
            <div className="card-head">
              <div>
                <span className="section-label">MODEL OUTPUT</span>
                <h3>Risk assessment</h3>
                <p>NephroMind AI · {MODEL_VERSION}</p>
              </div>
              <span className="panel-tag">Threshold {LOCKED_THRESHOLD}</span>
            </div>

            <div className="score-display">
              <strong>{apiData ? `${apiData.risk_score}%` : '—'}</strong>
              <span>Model Risk Score</span>
            </div>

            {apiData ? (
              <>
                <div className="result-row">
                  <span>Classification</span>
                  <RiskBadge risk={risk} />
                </div>

                <div className="result-row">
                  <span>Probability (p)</span>
                  <b>{apiData.probability.toFixed(4)} ({ (apiData.probability * 100).toFixed(1) }%)</b>
                </div>

                <div className="result-row">
                  <span>Prediction</span>
                  <b>
                    {apiData.prediction === 'Yes'
                      ? 'Kidney disease risk predicted'
                      : 'No kidney disease risk predicted'}
                  </b>
                </div>

                <div className="result-row">
                  <span>Locked threshold</span>
                  <b>{apiData.threshold}</b>
                </div>

                <div className="result-row">
                  <span>Model</span>
                  <b>{apiData.model_type || 'Deep Neural Network (MLP)'}</b>
                </div>

                <div className="result-row">
                  <span>Model version</span>
                  <b>{apiData.model_version}</b>
                </div>

                {/* CLINICAL MARKERS RULE-BASED LAYER */}
                {apiData.clinical_markers && (
                  <div className="clinical-marker-box">
                    <div className="clinical-marker-title">
                      Clinical Marker Assessment (Rule-Based Layer)
                    </div>
                    {apiData.clinical_markers.ckd_stage && (
                      <div className="result-row" style={{ fontSize: '0.8rem', padding: '2px 0' }}>
                        <span>KDIGO Stage:</span>
                        <b>{apiData.clinical_markers.ckd_stage}</b>
                      </div>
                    )}
                    {apiData.clinical_markers.creatinine_status && (
                      <div className="result-row" style={{ fontSize: '0.8rem', padding: '2px 0' }}>
                        <span>Creatinine:</span>
                        <b>
                          {apiData.clinical_markers.creatinine_status}{' '}
                          {apiData.clinical_markers.creatinine_mgdL !== undefined &&
                            `(${apiData.clinical_markers.creatinine_mgdL} mg/dL)`}
                        </b>
                      </div>
                    )}
                    {apiData.clinical_markers.albuminuria_category && (
                      <div className="result-row" style={{ fontSize: '0.8rem', padding: '2px 0' }}>
                        <span>Albuminuria:</span>
                        <b>{apiData.clinical_markers.albuminuria_category}</b>
                      </div>
                    )}
                    {apiData.clinical_markers.urine_dipstick && (
                      <div className="result-row" style={{ fontSize: '0.8rem', padding: '2px 0' }}>
                        <span>Dipstick:</span>
                        <b>{apiData.clinical_markers.urine_dipstick}</b>
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="empty-inline">
                Run an assessment to display the live model output.
              </div>
            )}
          </section>
        </div>

        {/* PRIORITY 1 — EXPLAINABLE AI (XAI) SECTION */}
        {apiData && (
          <ExplainableAISection
            apiData={apiData}
            patientName={patient.name}
          />
        )}

        <div className="prototype-note">
          <ShieldCheck size={17} />
          <div>
            <b>Prototype only</b>
            <span>
              This is a retrospective/cross-sectional decision-support prototype. It is not a diagnosis
              or clinically validated prediction system. Results require qualified clinical review.
            </span>
          </div>
        </div>
      </section>
    </Layout>
  )
}


/* =========================================================
   EVIDENCE-BASED CLINICAL SUMMARY & REPORT COMPONENTS
   ========================================================= */

/* =========================================================
   EVIDENCE-BASED CLINICAL SUMMARY & REPORT COMPONENTS
   ========================================================= */

interface ImportantFinding {
  id: string
  title: string
  evidence: string
  whyItMatters: string
  status: string
  severity: 'high' | 'medium' | 'normal' | 'discrepancy'
}

function ClinicianReportSummary({
  patient,
  prediction,
}: {
  patient: Patient
  prediction: ApiPrediction | null
}) {
  const [showAllXai, setShowAllXai] = useState(false)
  const [showTableDetails, setShowTableDetails] = useState(false)
  const [showFullEvidence, setShowFullEvidence] = useState(false)

  const bpParts = patient.clinical.bp.split('/')
  const systolic = Number.parseFloat(bpParts[0]) || 120
  const diastolic = Number.parseFloat(bpParts[1]) || 80

  const creatStr = patient.clinical.creatinine || ''
  const creatClean = creatStr.replace(/[^0-9.]/g, '')
  const creat = creatClean ? Number.parseFloat(creatClean) : null

  const egfrStr = patient.clinical.egfr || ''
  const egfrClean = egfrStr.replace(/[^0-9.]/g, '')
  const egfr = egfrClean ? Number.parseFloat(egfrClean) : null

  const hba1cClean = patient.clinical.hba1c?.replace(/[^0-9.]/g, '') || ''
  const hba1c = hba1cClean ? Number.parseFloat(hba1cClean) : null

  const uacrClean = patient.clinical.uacr?.replace(/[^0-9.]/g, '') || ''
  const uacr = uacrClean ? Number.parseFloat(uacrClean) : null

  const dipstick =
    prediction?.clinical_markers?.urine_dipstick ||
    patient.clinical.urineDipstick ||
    null

  const isDipstickPositive =
    dipstick !== null &&
    dipstick !== 'Negative' &&
    dipstick !== 'Trace' &&
    dipstick !== ''

  const isUacrNormal = uacr !== null && uacr < 30
  const hasUacrDipstickDiscrepancy = isUacrNormal && isDipstickPositive

  const prob = prediction?.probability ?? null
  const threshold = prediction?.threshold ?? LOCKED_THRESHOLD

  // ---------------------------------------------------------
  // SECTION A: 2-3 SENTENCE CLINICAL INTERPRETATION
  // ---------------------------------------------------------
  const summaryInterpretation = useMemo(() => {
    if (egfr !== null && egfr < 60) {
      return `This encounter demonstrates a reduction in spot glomerular filtration (eGFR ${egfr} mL/min/1.73m², G3b range) alongside ${
        hba1c !== null && hba1c >= 6.5 ? `elevated glycemia (HbA1c ${hba1c.toFixed(1)}%)` : 'metabolic risk'
      }${hasUacrDipstickDiscrepancy ? ' and discordant spot dipstick screening' : ''}. Single spot measurements capture cross-sectional filtration and require documented ≥ 90-day persistence to satisfy KDIGO chronicity criteria and distinguish CKD from acute reversible injury (AKI). Model results reflect learned statistical associations and require qualified physician review.`
    }
    return `The current panel demonstrates preserved spot filtration (eGFR ${egfr ?? '—'} mL/min/1.73m²) alongside ${
      hba1c !== null && hba1c >= 6.5 ? `glycemic exposure (HbA1c ${hba1c.toFixed(1)}%)` : 'stable vital signs'
    }. Preserved filtration does not exclude early glomerular barrier injury or microalbuminuria, warranting periodic longitudinal surveillance. Model risk score represents learned dataset patterns, not a clinical diagnosis.`
  }, [egfr, hba1c, hasUacrDipstickDiscrepancy])

  // ---------------------------------------------------------
  // SECTION B: KEY CLINICAL MEASUREMENTS (4–6 READABLE CARDS)
  // ---------------------------------------------------------
  const keyCards = [
    {
      label: 'Estimated GFR (eGFR)',
      value: egfr !== null ? `${egfr}` : '—',
      unit: 'mL/min/1.73m²',
      status:
        egfr === null
          ? 'Pending Lab'
          : egfr < 30
            ? 'G4 Severely Decreased'
            : egfr < 45
              ? 'G3b Reduced (Spot)'
              : egfr < 60
                ? 'G3a Mild-to-Moderate'
                : 'G1/G2 Preserved',
      severity:
        egfr === null
          ? 'unassessed'
          : egfr < 45
            ? 'high'
            : egfr < 60
              ? 'medium'
              : 'normal',
      ref: '≥ 60 mL/min/1.73m²',
      method: 'CKD-EPI (2021) · Laboratory',
    },
    {
      label: 'Serum Creatinine',
      value: creat !== null ? `${creat}` : '—',
      unit: 'mg/dL',
      status:
        creat === null
          ? 'Pending Lab'
          : creat > 1.3
            ? 'Elevated Marker'
            : creat > 1.1
              ? 'Borderline High'
              : 'Normal Range (Ref: 0.6–1.2)',
      severity:
        creat === null
          ? 'unassessed'
          : creat > 1.3
            ? 'high'
            : creat > 1.1
              ? 'medium'
              : 'normal',
      ref: '0.6 – 1.2 mg/dL',
      method: 'Enzymatic Assay · Laboratory',
    },
    {
      label: 'HbA1c (Glycated Hb)',
      value: hba1c !== null ? `${hba1c.toFixed(1)}` : '—',
      unit: '%',
      status:
        hba1c === null
          ? 'Unassessed'
          : hba1c >= 8.0
            ? 'Sub-optimal Control'
            : hba1c >= 6.5
              ? 'Diabetic Range'
              : hba1c >= 5.7
                ? 'Pre-diabetic'
                : 'Normal Glycemia',
      severity:
        hba1c === null
          ? 'unassessed'
          : hba1c >= 8.0
            ? 'high'
            : hba1c >= 6.5
              ? 'medium'
              : 'normal',
      ref: '< 5.7% Normal, < 7.0% Target',
      method: 'HPLC · Laboratory',
    },
    {
      label: 'Resting Blood Pressure',
      value: `${Math.round(systolic)}/${Math.round(diastolic)}`,
      unit: 'mmHg',
      status:
        systolic >= 140 || diastolic >= 90
          ? 'Stage 2 Range (Office)'
          : systolic >= 130 || diastolic >= 80
            ? 'Stage 1 Range (Office)'
            : 'Normotensive Reading',
      severity:
        systolic >= 140 || diastolic >= 90
          ? 'high'
          : systolic >= 130 || diastolic >= 80
            ? 'medium'
            : 'normal',
      ref: '< 120/80 mmHg',
      method: 'Renal Sentinel Vitals Station',
    },
    {
      label: 'Quantitative Spot UACR',
      value: uacr !== null ? `${uacr}` : '—',
      unit: 'mg/g',
      status:
        uacr === null
          ? 'Pending Quantification'
          : uacr >= 300
            ? 'A3 Severely Increased'
            : uacr >= 30
              ? 'A2 Moderately Increased'
              : 'A1 Normal Excretion',
      severity:
        uacr === null
          ? 'unassessed'
          : uacr >= 300
            ? 'high'
            : uacr >= 30
              ? 'medium'
              : 'normal',
      ref: '< 30 mg/g (A1 Normal)',
      method: 'Immuno-turbidimetric · Spot Urine',
    },
    {
      label: 'Qualitative Urine Dipstick',
      value: dipstick || '—',
      unit: 'protein',
      status:
        !dipstick
          ? 'Not Assessed'
          : isDipstickPositive
            ? `Positive Screen (${dipstick})`
            : 'Negative / Trace',
      severity:
        !dipstick
          ? 'unassessed'
          : isDipstickPositive
            ? 'medium'
            : 'normal',
      ref: 'Negative',
      method: 'Colorimetric Spot Dipstick',
    },
  ]

  // ---------------------------------------------------------
  // SECTION C: IMPORTANT FINDINGS (MAX 3 CONCISE CARDS)
  // ---------------------------------------------------------
  const importantFindings: ImportantFinding[] = []

  if (egfr !== null && egfr < 60) {
    importantFindings.push({
      id: 'filtration',
      title: '1. Reduced Glomerular Filtration (G3b Range)',
      evidence: `Spot eGFR is ${egfr} mL/min/1.73m² with serum creatinine of ${creat !== null ? `${creat} mg/dL` : 'normal'}.`,
      whyItMatters: `Reflects compromised solute clearance capacity; a serum creatinine within standard reference range does not exclude reduced filtration in older adults or individuals with low muscle mass.`,
      status: 'Requires 90-day confirmation',
      severity: 'high',
    })
  } else if (egfr !== null) {
    importantFindings.push({
      id: 'filtration',
      title: '1. Preserved Glomerular Filtration Capacity',
      evidence: `Spot eGFR of ${egfr} mL/min/1.73m² indicates intact clearance capacity at current reading.`,
      whyItMatters: `Preserved filtration does not rule out microvascular or glomerular barrier permeability lesions.`,
      status: 'Preserved at encounter',
      severity: 'normal',
    })
  }

  if (hasUacrDipstickDiscrepancy) {
    importantFindings.push({
      id: 'discrepancy',
      title: '2. Discordant Proteinuria / Albuminuria Screening',
      evidence: `Spot qualitative dipstick demonstrates ${dipstick} protein despite a normal quantitative spot UACR of ${uacr} mg/g (A1 category).`,
      whyItMatters: `Spot dipsticks are prone to false positives from concentrated urine, alkaline pH, radiocontrast, or non-albumin proteinuria.`,
      status: 'Needs repeat spot UACR',
      severity: 'discrepancy',
    })
  } else if (uacr !== null && uacr >= 30) {
    importantFindings.push({
      id: 'albuminuria',
      title: `2. Elevated Albuminuria (${uacr >= 300 ? 'A3 Category' : 'A2 Category'})`,
      evidence: `Spot quantitative UACR is elevated at ${uacr} mg/g.`,
      whyItMatters: `Indicates compromised glomerular podocyte slit diaphragm selectivity and active transglomerular protein leakage.`,
      status: 'Requires repeat confirmation',
      severity: uacr >= 300 ? 'high' : 'medium',
    })
  } else if (uacr === null && isDipstickPositive) {
    importantFindings.push({
      id: 'dipstick',
      title: '2. Positive Dipstick Proteinuria (Unquantified)',
      evidence: `Qualitative dipstick screening shows ${dipstick} protein without quantitative UACR confirmation.`,
      whyItMatters: `Dipstick screening lacks sensitivity and specificity for microalbuminuria quantification.`,
      status: 'Requires quantitative UACR',
      severity: 'medium',
    })
  }

  if (hba1c !== null && hba1c >= 6.5) {
    importantFindings.push({
      id: 'glycemic',
      title: '3. Elevated Glycemic Exposure (Diabetes Range)',
      evidence: `HbA1c is ${hba1c.toFixed(1)}% (Ref: < 5.7% Normal, < 7.0% Target) with BP ${Math.round(systolic)}/${Math.round(diastolic)} mmHg.`,
      whyItMatters: `Chronic hyperglycemia increases microvascular endothelial stress, but does not prove Diabetic Kidney Disease (DKD) without establishing persistent albuminuria or ruling out non-diabetic etiologies.`,
      status: 'Requires clinical review',
      severity: 'medium',
    })
  } else if (systolic >= 130 || diastolic >= 80) {
    importantFindings.push({
      id: 'bp',
      title: '3. Elevated Resting Blood Pressure (Spot Encounter)',
      evidence: `Office blood pressure is ${Math.round(systolic)}/${Math.round(diastolic)} mmHg (Ref: < 120/80 mmHg).`,
      whyItMatters: `Systemic hypertension elevates intraglomerular hydrostatic pressure, but a single reading cannot diagnose chronic hypertensive nephrosclerosis.`,
      status: 'Requires ABPM or home log',
      severity: systolic >= 140 || diastolic >= 90 ? 'medium' : 'normal',
    })
  }

  const displayFindings = importantFindings.slice(0, 3)

  // ---------------------------------------------------------
  // SECTION D: TOP 3 XAI CONTRIBUTORS
  // ---------------------------------------------------------
  const dnnFeatures = prediction?.top_features || []
  const inputVals = prediction?.input_values || {}
  const top3Features = dnnFeatures.slice(0, 3)
  const remainingFeatures = dnnFeatures.slice(3)

  const maxAbsContrib = useMemo(() => {
    if (!dnnFeatures.length) return 0.2
    const maxVal = Math.max(...dnnFeatures.map(f => Math.abs(f.contribution)))
    return Math.max(0.12, maxVal * 1.1)
  }, [dnnFeatures])

  // ---------------------------------------------------------
  // SECTION E: MAX 3 CONFIRMATION STEPS
  // ---------------------------------------------------------
  const confirmationSteps = [
    {
      title: 'Renal Chronicity Verification (≥ 90-Day Repeat Panel)',
      detail:
        'Order repeat serum creatinine and eGFR testing in 3 months to satisfy KDIGO chronicity criteria and distinguish CKD from acute reversible injury (AKI).',
      urgent: egfr !== null && egfr < 60,
    },
    {
      title: 'First-Morning Quantitative Spot UACR Verification',
      detail:
        hasUacrDipstickDiscrepancy || uacr === null
          ? 'Order first-morning spot UACR and clean-catch urinalysis with microscopy to reconcile dipstick discordance and quantify albumin excretion.'
          : 'Confirm persistence with 2 of 3 repeat morning spot UACR measurements over 3–6 months.',
      urgent: hasUacrDipstickDiscrepancy || uacr === null,
    },
    {
      title: 'Microvascular & Blood Pressure Evaluation',
      detail:
        'Order dilated retinal fundoscopy to screen for diabetic retinopathy and 24-hr Ambulatory Blood Pressure Monitoring (ABPM) to assess for sustained hypertension.',
      urgent: (hba1c !== null && hba1c >= 6.5) || systolic >= 130,
    },
  ]

  return (
    <div className="clinician-report-container">
      {/* ---------------------------------------------------------
          SECTION A: PATIENT AND ASSESSMENT SUMMARY
         --------------------------------------------------------- */}
      <section className="report-card-section report-summary-banner">
        <div className="summary-banner-top">
          <div>
            <span className="summary-kicker">CLINICAL DECISION SUPPORT ASSESSMENT</span>
            <h2 className="summary-patient-name">{patient.name}</h2>
            <div className="summary-patient-meta">
              <span><b>ID:</b> {patient.id}</span>
              <span><b>Age:</b> {patient.age} yrs</span>
              <span><b>Gender:</b> {patient.gender}</span>
              <span><b>Encounter:</b> {patient.lastVisit}</span>
            </div>
          </div>

          <div className="summary-probability-box">
            {prediction && prob !== null ? (
              <>
                <span className="prob-label">AI Risk Probability</span>
                <div className="prob-score-row">
                  <strong className="prob-large">{(prob * 100).toFixed(1)}%</strong>
                  <span
                    className={`prob-badge ${
                      prob >= threshold ? 'badge-high' : 'badge-low'
                    }`}
                  >
                    {prediction.risk_category}
                  </span>
                </div>
                <small className="prob-sub">
                  p = {prob.toFixed(4)} · Locked Threshold: {threshold}
                </small>
              </>
            ) : (
              <div className="prob-pending">
                <AlertTriangle size={18} />
                <span>AI Assessment Unavailable</span>
              </div>
            )}
          </div>
        </div>

        <div className="summary-interpretation-text">
          <p>{summaryInterpretation}</p>
        </div>

        <div className="summary-prototype-disclaimer">
          <ShieldCheck size={16} />
          <span>
            <b>Research Prototype Assessment:</b> Output represents learned statistical patterns on 8 physiological inputs. This is a non-diagnostic research tool and does not replace qualified clinical evaluation.
          </span>
        </div>
      </section>

      {/* ---------------------------------------------------------
          SECTION B: KEY CLINICAL FINDINGS (4–6 READABLE CARDS)
         --------------------------------------------------------- */}
      <section className="report-card-section">
        <div className="section-head-readable">
          <div className="section-head-title">
            <Activity size={18} className="text-primary-blue" />
            <h3>Key Clinical Measurements</h3>
          </div>
          <button
            type="button"
            className="text-toggle-btn"
            onClick={() => setShowTableDetails(!showTableDetails)}
          >
            {showTableDetails ? 'Hide details' : 'View reference ranges & details'}
            <ChevronDown size={14} className={showTableDetails ? 'rotate-180' : ''} />
          </button>
        </div>

        <div className="key-cards-grid">
          {keyCards.map(item => (
            <div key={item.label} className={`key-measurement-card severity-${item.severity}`}>
              <div className="key-card-label">{item.label}</div>
              <div className="key-card-val-row">
                <strong className="key-card-val">{item.value}</strong>
                {item.value !== '—' && <span className="key-card-unit">{item.unit}</span>}
              </div>
              <div className={`key-card-badge status-${item.severity}`}>{item.status}</div>
              {showTableDetails && (
                <div className="key-card-details">
                  <small><b>Ref:</b> {item.ref}</small>
                  <small><b>Method:</b> {item.method}</small>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* CLINICAL DISCREPANCY ALERT (UACR vs DIPSTICK) */}
        {hasUacrDipstickDiscrepancy && (
          <div className="discrepancy-callout-readable">
            <div className="discrepancy-callout-head">
              <AlertTriangle size={18} />
              <strong>Clinical Discrepancy: Spot Dipstick ({dipstick}) vs Quantitative UACR ({uacr} mg/g)</strong>
            </div>
            <p>
              Spot qualitative dipstick shows protein despite a normal quantitative spot UACR (A1 category). Qualitative dipsticks have high false-positive rates with concentrated urine, high pH, or non-albumin proteinuria. <em>Do not assume concurrent sample collection. Order first-morning repeat spot UACR and clean-catch urinalysis with microscopy.</em>
            </p>
          </div>
        )}
      </section>

      {/* ---------------------------------------------------------
          SECTION C: IMPORTANT FINDINGS AND EXPLANATION (MAX 3)
         --------------------------------------------------------- */}
      <section className="report-card-section">
        <div className="section-head-readable">
          <div className="section-head-title">
            <FileText size={18} className="text-primary-blue" />
            <h3>Important Findings & Clinical Explanation</h3>
          </div>
          <span className="section-sub-badge">Top {displayFindings.length} Prioritized Findings</span>
        </div>

        <div className="important-findings-stack">
          {displayFindings.map(item => (
            <div key={item.id} className={`finding-readable-card border-${item.severity}`}>
              <div className="finding-readable-head">
                <h4>{item.title}</h4>
                <span className={`finding-readable-badge tag-${item.severity}`}>{item.status}</span>
              </div>
              <div className="finding-readable-body">
                <p><b>Evidence:</b> {item.evidence}</p>
                <p><b>Clinical Significance:</b> {item.whyItMatters}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ---------------------------------------------------------
          SECTION D: AI EXPLAINABILITY (TOP 3 + EXPANDABLE)
         --------------------------------------------------------- */}
      <section className="report-card-section">
        <div className="section-head-readable">
          <div className="section-head-title">
            <Sparkles size={18} className="text-primary-blue" />
            <div>
              <h3>AI Explainability (Model Feature Sensitivity)</h3>
              <p className="section-instruction-text">
                Top patient-specific feature contributions derived via <b>Gradient × Input</b>.
              </p>
            </div>
          </div>
          {remainingFeatures.length > 0 && (
            <button
              type="button"
              className="text-toggle-btn"
              onClick={() => setShowAllXai(!showAllXai)}
            >
              {showAllXai ? 'Show top 3 only' : `View all ${dnnFeatures.length} contributors`}
              <ChevronDown size={14} className={showAllXai ? 'rotate-180' : ''} />
            </button>
          )}
        </div>

        <div className="xai-readable-chart-container">
          <div className="xai-readable-head-row">
            <span>Feature & Input Value</span>
            <span style={{ textAlign: 'right' }}>Protective (−)</span>
            <span style={{ textAlign: 'center' }}>0.0</span>
            <span>Risk Driver (+)</span>
            <span style={{ textAlign: 'right' }}>Score</span>
          </div>

          {(showAllXai ? dnnFeatures : top3Features).map(feat => {
            const meta = FEATURE_METADATA[feat.feature]
            const displayName = feat.display_name || meta?.label || feat.feature
            const rawVal = inputVals[feat.feature]
            const formattedVal =
              rawVal !== undefined && meta
                ? meta.format(rawVal)
                : rawVal !== undefined
                  ? String(rawVal)
                  : '—'
            const contrib = feat.contribution
            const isPositive = contrib >= 0
            const barWidth = Math.min(100, (Math.abs(contrib) / maxAbsContrib) * 100)

            return (
              <div className="xai-readable-row" key={feat.feature}>
                <div className="xai-feat-name-col">
                  <b>{displayName}</b>
                  <small>{formattedVal}</small>
                </div>

                <div className="xai-track-col protective-track">
                  {!isPositive && (
                    <div
                      className="xai-bar negative"
                      style={{ width: `${barWidth}%` }}
                      title={`Contribution: ${contrib.toFixed(3)}`}
                    />
                  )}
                </div>

                <div className="xai-zero-col">
                  <div className="zero-line" />
                </div>

                <div className="xai-track-col risk-track">
                  {isPositive && (
                    <div
                      className="xai-bar positive"
                      style={{ width: `${barWidth}%` }}
                      title={`Contribution: +${contrib.toFixed(3)}`}
                    />
                  )}
                </div>

                <div className="xai-score-col">
                  <strong className={isPositive ? 'text-risk' : 'text-protective'}>
                    {isPositive ? '+' : ''}{contrib.toFixed(3)}
                  </strong>
                </div>
              </div>
            )
          })}
        </div>

        <div className="xai-causation-notice">
          <Info size={15} />
          <span>
            Feature contributions indicate how the input features influence this model's output. They do not establish clinical causation.
          </span>
        </div>
      </section>

      {/* ---------------------------------------------------------
          SECTION E: FURTHER REVIEW & EXPANDABLE DETAILS
         --------------------------------------------------------- */}
      <section className="report-card-section">
        <div className="section-head-readable">
          <div className="section-head-title">
            <Clock3 size={18} className="text-primary-blue" />
            <div>
              <h3>Further Clinical Review & Recommended Verification</h3>
              <p className="section-instruction-text">
                Priority follow-up steps to satisfy KDIGO confirmation requirements.
              </p>
            </div>
          </div>
          <button
            type="button"
            className="text-toggle-btn"
            onClick={() => setShowFullEvidence(!showFullEvidence)}
          >
            {showFullEvidence ? 'Hide clinical details' : 'View full clinical details & audit'}
            <ChevronDown size={14} className={showFullEvidence ? 'rotate-180' : ''} />
          </button>
        </div>

        <div className="confirmation-steps-grid">
          {confirmationSteps.map((step, idx) => (
            <div key={step.title} className="confirmation-step-card">
              <div className="step-number-tag">Step {idx + 1}</div>
              <h4>{step.title}</h4>
              <p>{step.detail}</p>
            </div>
          ))}
        </div>

        {/* EXPANDABLE FULL CLINICAL EVIDENCE & AUDIT DETAILS */}
        {showFullEvidence && (
          <div className="full-evidence-expanded-box">
            <div className="expanded-block">
              <h4>
                <ShieldCheck size={15} className="text-primary-blue" />
                Non-Leakage Feature Architecture Separation
              </h4>
              <p>
                Specific kidney biomarkers (<em>Creatinine, eGFR, UACR, Urine Dipstick</em>) are intentionally <b>excluded</b> from the 8-feature PyTorch DNN input vector to prevent target leakage, and are evaluated in the rule-based clinical evidence layer above.
              </p>
            </div>

            <div className="expanded-block">
              <h4>
                <Cpu size={15} className="text-primary-blue" />
                Model Architecture & Verification Audit
              </h4>
              <div className="audit-table-grid">
                <div><span>Model Architecture:</span> <b>PyTorch MLP (8 → 128 → 64 → 32 → 16 → 1)</b></div>
                <div><span>Decision Threshold:</span> <b>t = {threshold} (Locked Validation Threshold)</b></div>
                <div><span>Hold-out ROC-AUC:</span> <b>0.7618 (PR-AUC 0.9214)</b></div>
                <div><span>Regulatory Classification:</span> <b>Research Prototype · Non-Diagnostic Tool</b></div>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}


/* =========================================================
   REPORT VIEW
   ========================================================= */

function Report() {
  const [selectedId, setSelectedId] = useState(patients[0].id)
  const patient = patients.find(item => item.id === selectedId) || patients[0]

  const [prediction, setPrediction] = useState<ApiPrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const generateAssessment = async (selectedPatient: Patient) => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(getPatientFeatures(selectedPatient)),
      })

      const payload = await response.json().catch(() => null)

      if (!response.ok) {
        throw new Error(payload?.message || 'Unable to generate AI assessment.')
      }

      setPrediction(payload)
    } catch (err) {
      setPrediction(null)
      setError(err instanceof Error ? err.message : 'Inference backend unavailable.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    generateAssessment(patient)
  }, [selectedId])

  const reportRisk = getRiskFromApi(prediction, patient.risk)

  return (
    <Layout>
      <section className="page report-page">
        <PageHeader
          eyebrow="CLINICAL DECISION SUPPORT"
          title="Patient clinical report"
          description="Concise, evidence-based clinical synthesis and transparent AI decision-support documentation."
          action={
            <div className="button-row">
              <select
                className="secondary select"
                value={selectedId}
                onChange={e => setSelectedId(e.target.value)}
              >
                {patients.map(item => (
                  <option key={item.id} value={item.id}>
                    {item.name} ({item.id})
                  </option>
                ))}
              </select>

              <button className="secondary" onClick={() => window.print()}>
                Print
              </button>

              <button className="primary" onClick={() => window.print()}>
                <Download size={16} />
                Export / Print PDF
              </button>
            </div>
          }
        />

        {loading && (
          <div className="notice">
            <Activity size={16} />
            <span>Computing live AI model inference & feature attributions…</span>
          </div>
        )}

        {error && !loading && (
          <div className="notice error">
            <AlertTriangle size={16} />
            <span>{error}</span>
          </div>
        )}

        <article className="report-paper card">
          {/* HOSPITAL ENCOUNTER HEADER */}
          <div className="report-encounter-header">
            <div className="report-encounter-brand">
              <Logo />
              <div className="report-type-meta">
                <span className="report-title-badge">Nephrology Clinical Decision Support</span>
                <span className="report-date-badge">Encounter Date: {patient.lastVisit}</span>
              </div>
            </div>
            <div className="report-encounter-risk">
              <RiskBadge risk={reportRisk} />
              <small className="risk-freshness-text">Status: Active Encounter</small>
            </div>
          </div>

          <div className="report-patient-banner">
            <div className="patient-demographics-stack">
              <h2>{patient.name}</h2>
              <div className="patient-meta-row">
                <span><b>Patient ID:</b> {patient.id}</span>
                <span><b>Age:</b> {patient.age} yrs</span>
                <span><b>Gender:</b> {patient.gender}</span>
                <span><b>Encounter:</b> {patient.lastVisit}</span>
                <span className="telemetry-badge">● Live Telemetry Active</span>
              </div>
            </div>
          </div>

          <hr className="report-divider" />

          {/* REDESIGNED CONCISE CLINICIAN REPORT */}
          <ClinicianReportSummary
            patient={patient}
            prediction={prediction}
          />

          {/* HARDWARE STATUS FOOTER */}
          <div className="hardware-flow-footer">
            <div className="hardware-flow-head">
              <div className="hardware-flow-title">
                <Cpu size={14} className="text-primary-blue" />
                <span>Renal Sentinel Station (Hardware Integration Ready)</span>
              </div>
              <span className="hardware-status-tag">Integration Ready</span>
            </div>
            <div className="hardware-flow-steps">
              <span className="step-label">Data Pipeline:</span>
              <span>Renal Sentinel (MAX30102 + BP Sensor)</span>
              <span>→</span>
              <span>ESP32 Telemetry</span>
              <span>→</span>
              <span>FastAPI Ingestion</span>
              <span>→</span>
              <span>PyTorch DNN Inference</span>
              <span>→</span>
              <span>Evidence-Based Report</span>
            </div>
          </div>

          <footer>
            NephroMind AI · Research Decision-Support Prototype · {MODEL_VERSION} · Decision Threshold {LOCKED_THRESHOLD} · Non-Clinical Diagnostic Tool
          </footer>
        </article>
      </section>
    </Layout>
  )
}


/* =========================================================
   ANALYTICS VIEW
   ========================================================= */

function Analytics() {
  const total = patients.length
  const high = patients.filter(p => p.risk === 'High').length
  const medium = patients.filter(p => p.risk === 'Medium').length
  const low = patients.filter(p => p.risk === 'Low').length

  const average = (selector: (p: Patient) => number) =>
    patients.reduce((sum, patient) => sum + selector(patient), 0) / total

  const riskData = [
    { name: 'Higher risk', value: high, color: riskColors.High },
    { name: 'Moderate risk', value: medium, color: riskColors.Medium },
    { name: 'Lower risk', value: low, color: riskColors.Low },
  ]

  const biomarkers = [
    {
      name: 'HbA1c',
      value: Number(average(p => Number.parseFloat(p.clinical.hba1c)).toFixed(1)),
    },
    {
      name: 'BMI',
      value: Number(average(p => p.clinical.bmi).toFixed(1)),
    },
    {
      name: 'SBP',
      value: Math.round(average(p => Number.parseFloat(p.clinical.bp.split('/')[0]))),
    },
    {
      name: 'DBP',
      value: Math.round(average(p => Number.parseFloat(p.clinical.bp.split('/')[1]))),
    },
    {
      name: 'Heart rate',
      value: Math.round(average(p => Number.parseFloat(p.clinical.heartRate))),
    },
  ]

  const representativeFeatures = [
    { name: 'Systolic BP', value: 0.142 },
    { name: 'HbA1c', value: 0.118 },
    { name: 'Age', value: 0.085 },
    { name: 'BMI', value: 0.062 },
    { name: 'Diastolic BP', value: 0.041 },
    { name: 'Heart rate', value: 0.024 },
    { name: 'Gender', value: 0.008 },
    { name: 'SpO₂', value: -0.045 },
  ]

  const rankedPatients = [...patients]
    .sort((a, b) => b.score - a.score)
    .slice(0, 6)

  return (
    <Layout>
      <section className="page analytics-page">
        <PageHeader
          eyebrow="NEPHROMIND AI"
          title="Clinical analytics"
          description="Population-level prototype insights, model performance and clinical biomarker summaries."
          action={
            <div className="button-row">
              <button className="secondary">
                Last 30 days
                <ChevronDown size={14} />
              </button>
              <button className="secondary" onClick={() => window.print()}>
                <Download size={15} />
                Export
              </button>
            </div>
          }
        />

        <div className="analytics-kpis">
          <Stat
            label="Total patients"
            value={String(total)}
            detail="Prototype registry"
            kind="patients"
          />
          <Stat
            label="Higher risk"
            value={String(high)}
            detail={`${Math.round((high / total) * 100)}% of registry`}
            kind="High"
          />
          <Stat
            label="Moderate risk"
            value={String(medium)}
            detail={`${Math.round((medium / total) * 100)}% of registry`}
            kind="Medium"
          />
          <Stat
            label="Lower risk"
            value={String(low)}
            detail={`${Math.round((low / total) * 100)}% of registry`}
            kind="Low"
          />
        </div>

        <div className="analytics-grid">
          {/* POPULATION RISK */}
          <section className="card analytics-card">
            <div className="card-head">
              <div>
                <h3>Population risk distribution</h3>
                <p>Current prototype registry categorization</p>
              </div>
              <span className="panel-tag">Prototype</span>
            </div>

            <div className="risk-visual">
              <ResponsiveContainer width="55%" height={240}>
                <PieChart>
                  <Pie
                    data={riskData}
                    dataKey="value"
                    innerRadius={70}
                    outerRadius={98}
                    paddingAngle={4}
                  >
                    {riskData.map(item => (
                      <Cell key={item.name} fill={item.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>

              <div className="risk-center">
                <strong>{total}</strong>
                <span>patients</span>
              </div>

              <div className="analytics-legend">
                {riskData.map(item => (
                  <div key={item.name}>
                    <span>
                      <i style={{ background: item.color }} />
                      {item.name}
                    </span>
                    <b>{item.value}</b>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* MODEL PERFORMANCE */}
          <section className="card analytics-card">
            <div className="card-head">
              <div>
                <h3>Model performance</h3>
                <p>Hold-out test performance · PyTorch DNN</p>
              </div>
              <span className="model-version">{MODEL_VERSION}</span>
            </div>

            <div className="performance-list">
              {[
                ['ROC-AUC', '0.7618', 76.18],
                ['PR-AUC', '0.9214', 92.14],
                ['Sensitivity', '63.47%', 63.47],
                ['Specificity', '85.45%', 85.45],
              ].map(([label, value, width]) => (
                <div key={String(label)}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                  <div className="progress">
                    <i style={{ width: `${width}%` }} />
                  </div>
                </div>
              ))}
            </div>

            <div className="threshold-box">
              <span>Locked decision threshold</span>
              <strong>{LOCKED_THRESHOLD}</strong>
            </div>
          </section>

          {/* CLINICAL BIOMARKERS */}
          <section className="card analytics-card">
            <div className="card-head">
              <div>
                <h3>Clinical biomarker overview</h3>
                <p>Average values across the prototype registry</p>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={biomarkers}>
                <CartesianGrid vertical={false} stroke="#e9eef3" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip />
                <Bar dataKey="value" fill="#2878ca" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </section>

          {/* EXPLAINABILITY REPRESENTATIVE */}
          <section className="card analytics-card">
            <div className="card-head">
              <div>
                <h3>Model explainability</h3>
                <p>Representative 8-feature attribution ranking (Gradient × Input)</p>
              </div>
            </div>

            <div className="feature-ranking">
              {representativeFeatures.map(feature => {
                const isPos = feature.value >= 0
                return (
                  <div className="ranking-row" key={feature.name}>
                    <span>{feature.name}</span>
                    <div className="ranking-track">
                      <i
                        className={isPos ? 'positive' : 'negative'}
                        style={{
                          width: `${Math.min(100, Math.abs(feature.value) * 500)}%`,
                        }}
                      />
                    </div>
                    <b className={isPos ? 'positive-text' : 'negative-text'}>
                      {isPos ? '+' : ''}
                      {feature.value.toFixed(3)}
                    </b>
                  </div>
                )
              })}
            </div>

            <div className="explainability-note">
              <AlertTriangle size={15} />
              <span>
                Representative values illustrate model sensitivity via Gradient × Input.
                They do not establish biological causation or treatment effects.
              </span>
            </div>
          </section>

          {/* HIGHEST RISK PATIENTS */}
          <section className="card analytics-card full-width">
            <div className="card-head">
              <div>
                <h3>Highest model risk scores</h3>
                <p>Prototype records for closer review</p>
              </div>
              <NavLink to="/patients" className="text-link">
                View registry
              </NavLink>
            </div>

            <div className="ranked-patients-grid">
              {rankedPatients.map((patient, index) => (
                <NavLink
                  to={`/patients/${patient.id}`}
                  className="ranking-patient"
                  key={patient.id}
                >
                  <span className="rank">{index + 1}</span>
                  <Avatar patient={patient} />
                  <span className="patient-name">
                    <b>{patient.name}</b>
                    <small>
                      {patient.id} · {patient.age} years
                    </small>
                  </span>
                  <RiskBadge risk={patient.risk} />
                  <strong>{patient.score}%</strong>
                  <ChevronRight size={15} />
                </NavLink>
              ))}
            </div>
          </section>
        </div>

        <div className="prototype-note">
          <ShieldCheck size={17} />
          <div>
            <b>Analytics scope</b>
            <span>
              The 50-patient registry shown here is prototype/local data. Model performance metrics
              are from the separate hold-out test evaluation and should not be interpreted as clinical validation.
            </span>
          </div>
        </div>
      </section>
    </Layout>
  )
}


/* =========================================================
   SETTINGS VIEW
   ========================================================= */

function SettingsPage() {
  return (
    <Layout>
      <section className="page">
        <PageHeader
          eyebrow="SYSTEM CONFIGURATION"
          title="Settings"
          description="Manage the clinical workspace and review system information."
        />

        <div className="settings-grid">
          <section className="card settings-section">
            <div className="settings-icon">
              <Stethoscope size={18} />
            </div>
            <div>
              <h3>User profile</h3>
              <p>Dr. R. Sharma · Nephrology</p>
            </div>
            <button className="secondary">Edit</button>
          </section>

          <section className="card settings-section">
            <div className="settings-icon">
              <ShieldCheck size={18} />
            </div>
            <div>
              <h3>Clinical workspace</h3>
              <p>NephroMind AI · Prototype environment</p>
            </div>
            <button className="secondary">Manage</button>
          </section>

          <section className="card settings-section">
            <div className="settings-icon">
              <Bell size={18} />
            </div>
            <div>
              <h3>Notifications</h3>
              <p>Clinical review and system notifications</p>
            </div>
            <button className="secondary">Configure</button>
          </section>

          <section className="card settings-section">
            <div className="settings-icon">
              <Activity size={18} />
            </div>
            <div>
              <h3>Model information</h3>
              <p>
                AI Risk Assessment Engine · {MODEL_VERSION} · threshold {LOCKED_THRESHOLD}
              </p>
            </div>
            <button className="secondary">View</button>
          </section>
        </div>

        <section className="card settings-detail">
          <h3>System information</h3>

          <div className="settings-list">
            <div>
              <span>Frontend</span>
              <b>React + TypeScript + Vite</b>
            </div>

            <div>
              <span>Inference API</span>
              <b>FastAPI · 127.0.0.1:8000</b>
            </div>

            <div>
              <span>Model</span>
              <b>Deep Neural Network (PyTorch MLP) — 8 Features</b>
            </div>

            <div>
              <span>Feature vector</span>
              <b>Age, Gender, BMI, SBP, DBP, Heart Rate, SpO2, HbA1c</b>
            </div>

            <div>
              <span>Decision threshold</span>
              <b>{LOCKED_THRESHOLD}</b>
            </div>

            <div>
              <span>Explainability method</span>
              <b>Gradient × Input (Normalized w.r.t. Sigmoid Output)</b>
            </div>

            <div>
              <span>Clinical status</span>
              <b>Prototype — not clinically validated</b>
            </div>
          </div>
        </section>
      </section>
    </Layout>
  )
}


/* =========================================================
   LOGIN VIEW
   ========================================================= */

function Login() {
  const navigate = useNavigate()

  return (
    <div className="login">
      <section className="login-form">
        <Logo />

        <div>
          <p className="eyebrow">SECURE CLINICAL ACCESS</p>
          <h1>Welcome back</h1>
          <p>Sign in to access the NephroMind clinical workspace.</p>
        </div>

        <form
          onSubmit={event => {
            event.preventDefault()
            navigate('/dashboard')
          }}
        >
          <label>
            Work email
            <input type="email" defaultValue="doctor@citycare.org" />
          </label>

          <label>
            Password
            <input type="password" defaultValue="password" />
          </label>

          <div className="form-row">
            <label className="check">
              <input type="checkbox" defaultChecked />
              Remember me
            </label>

            <a href="#">Forgot password?</a>
          </div>

          <button className="primary full" type="submit">
            Sign in securely
            <ChevronRight size={16} />
          </button>
        </form>

        <small className="legal">
          <ShieldCheck size={14} />
          Protected clinical environment · Prototype access
        </small>
      </section>

      <section className="login-visual">
        <div className="visual-copy">
          <span className="visual-icon">
            <HeartPulse />
          </span>

          <p className="eyebrow">AI-ASSISTED RENAL CARE</p>

          <h2>
            Clinical insight,
            <br />
            without the noise.
          </h2>

          <p>
            NephroMind AI brings clinical and connected-health information into one focused
            decision-support workspace.
          </p>

          <div className="login-feature">
            <CheckCircle2 size={17} />
            <span>
              <b>Connected care workspace</b>
              <small>Clinical + wearable data</small>
            </span>
          </div>

          <div className="login-feature">
            <ShieldCheck size={17} />
            <span>
              <b>Model transparency</b>
              <small>Risk score + Gradient × Input feature attributions</small>
            </span>
          </div>
        </div>
      </section>
    </div>
  )
}


/* =========================================================
   ROUTES & ROOT
   ========================================================= */

function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/patients" element={<Patients />} />
      <Route path="/patients/:id" element={<Profile />} />
      <Route path="/predictions" element={<Prediction />} />
      <Route path="/reports" element={<Report />} />
      <Route path="/analytics" element={<Analytics />} />
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  )
}

createRoot(document.getElementById('root')!).render(
  <BrowserRouter>
    <App />
  </BrowserRouter>
)