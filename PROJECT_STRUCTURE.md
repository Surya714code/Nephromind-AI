# NephroMind AI project structure review

## Relevant files

| Area | Files found | Assessment |
|---|---|---|
| Frontend | `src/main.tsx`, `src/styles.css`, `src/vite-env.d.ts` | React/Vite TypeScript single-page frontend. It contains the dashboard UI, mock patient data, routes, charts, and a frontend-only `PredictionResponse` type. |
| Frontend configuration | `package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `tsconfig.node.json`, `index.html` | Vite project with React, React Router, Recharts, Framer Motion, and Lucide. |
| Dataset | `E:\dataset nephro.csv` | One CSV supplied outside the repository. It is the only dataset located. |
| ML / preprocessing | None found | No training, preprocessing, evaluation, inference, model artifact, or experiment code exists. |
| Backend / API | None found | No FastAPI, Node API, API routes, database, or authentication service exists. |
| Documentation | None found before this review | No README or ML documentation was found. |

## Current frontend structure

The frontend is intentionally self-contained. `src/main.tsx` includes reusable UI elements, routes for login, dashboard, patients, patient profile, predictions, reports, analytics, and settings, as well as mock patient records. `src/styles.css` provides the visual system. No source files were modified during this audit.

## Dataset location

`E:\dataset nephro.csv` is external to the repository. The raw file must remain unchanged. Any future training ingestion should reference it via configuration or copy it into an ignored ML data location with provenance recorded.

## Missing ML components

- Raw-data ingestion and schema validation
- Reproducible cleaning and train/validation/test splitting
- Feature specification and data dictionary
- Baseline and neural-network training code
- Evaluation, calibration, and explainability code
- Serialized model/pipeline artifacts
- Inference interface and API contract
- Python dependency manifest and training instructions

## Recommended training structure after audit approval

```text
ml/
├── data/                 # ignored raw/processed local data; ingestion manifests only in Git
├── preprocessing/        # schema validation and sklearn transformers
├── models/               # baseline and MLP definitions
├── training/             # split, training, tuning orchestration
├── evaluation/           # metrics, plots, and comparison tables
├── explainability/       # SHAP-compatible explanation workflow
├── inference/            # load-and-predict interface
├── artifacts/            # ignored fitted pipelines and model metadata
├── configs/              # immutable training configurations
└── reports/              # generated evaluation and audit reports
```

No directory or code under `ml/` has been created yet because training must wait for approval of the data audit.
