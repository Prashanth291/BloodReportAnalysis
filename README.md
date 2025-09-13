# Explainable AI-Based Analysis and Interpretation of Blood Test Reports

## Overview

This project bridges the gap between raw blood test reports and understandable, actionable health insights for patients and healthcare professionals. Leveraging OCR, Explainable AI (XAI), and data visualization, the platform extracts, analyzes, and explains laboratory parameters to empower users in health management.

---

## Features

- **Multi-format Report Upload:** Supports scanned images (JPG) and PDFs.
- **Intelligent Data Extraction:** Uses Tesseract/EasyOCR and PyMuPDF/PDFPlumber for robust text extraction.
- **Medical Parameter Mapping:** Converts raw text into standardized parameters (e.g., Hemoglobin, WBC).
- **Automated Abnormality Detection:** Flags out-of-range values using reference ranges.
- **Explainable AI (SHAP) Integration:** Provides transparent, human-friendly decision explanations.
- **Doctor-verified Recommendations:** Personalized recovery or lifestyle suggestions.
- **Long-term Monitoring and Visualization:** Secure storage, trend graphs, and data tracking.
- **Clean, User-centric Interface:** Accessible frontend for all users.

---

## Architecture Diagram




---

## Technology Stack

- **Frontend:** React.js
- **Backend:** Django (Python)
- **OCR:** Tesseract, EasyOCR
- **PDF Parsing:** PyMuPDF, PDFPlumber
- **AI & XAI:** scikit-learn, XGBoost, SHAP
- **Database:** PostgreSQL/MongoDB
- **Visualization:** Matplotlib, Plotly, Chart.js

---

## Project Workflow

1. **Report Upload:** Secure frontend upload for JPG/PDF blood test reports.
2. **Text Extraction:** OCR and PDF parsers extract and clean text data.
3. **Parameter Mapping:** Raw text mapped to standardized medical parameters.
4. **Abnormality Detection:** Each value checked against reference ranges.
5. **Explainability:** SHAP-based XAI generates clear abnormality explanations.
6. **Recommendations:** Context-specific, doctor-verified guidance for flagged results.
7. **Storage & Visualization:** Data securely stored and visualized for monitoring and review.

---
