# SignBridge

**A Deep Learning Framework for Continuous Sri Lankan Sign Language (SLSL) Recognition**

| | |
|---|---|
| **Project ID** | KND_CDAP_DS_06 |
| **Module** | IT4010 Research Project, 2026 July |
| **Institution** | SLIIT |
| **Research Group** | CoEAI, Centre of Excellence for AI |
| **Specialisation** | Data Science (DS) |
| **Supervisor** | Ms. Chaumika Ragunathan |
| **Co-Supervisor** | Ms. Dinu Ehelapitiya |
| **SDGs** | SDG 3 (Good Health and Well-being), SDG 10 (Reduced Inequalities) |
| **Plan Version** | v1.0, August 2026 |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Team and Component Ownership](#2-team-and-component-ownership)
3. [Research Problem and Novelty](#3-research-problem-and-novelty)
4. [Objectives](#4-objectives)
5. [Methodology](#5-methodology)
6. [Phase-by-Phase Plan](#6-phase-by-phase-plan)
7. [Timeline Overview](#7-timeline-overview)
8. [Risk Register](#8-risk-register)
9. [Governance and Communication](#9-governance-and-communication)
10. [Supervisor Checkpoints](#10-supervisor-checkpoints)
11. [Assessment Components](#11-assessment-components)
12. [Technology Stack](#12-technology-stack)
13. [Immediate Actions (First Two Weeks)](#13-immediate-actions-first-two-weeks)
14. [Quality Gates](#14-quality-gates)
15. [Publication Plan](#15-publication-plan)
16. [Appendix A: System Architecture](#appendix-a-system-architecture-summary)
17. [Appendix B: CTC Risk and Fallback Architecture](#appendix-b-ctc-risk-and-fallback-architecture)
18. [Notes for Coding Agents](#notes-for-coding-agents)

---

## 1. Executive Summary

SignBridge is a first-of-its-kind deep learning framework that converts continuous Sri Lankan Sign Language (SLSL) sentences into real-time Sinhala text.

- Approximately **357,000** Deaf and Hard of Hearing (DHH) people in Sri Lanka rely on SLSL.
- No automated real-time continuous recognition system exists. All prior work is limited to **isolated single-word classification**.
- The project builds on the **SSL400** dataset (400 word-level signs across 12,708 video clips) by engineering the **first sentence-level SLSL dataset** through a novel word-to-sentence construction pipeline.
- The core model is a **Bidirectional LSTM (BiLSTM) with a CTC decoder**, using **MediaPipe Holistic** full-body landmarks (543 landmarks per frame).
- The target domain is **healthcare**, evaluated in scenarios such as hospital reception and clinical consultation.

---

## 2. Team and Component Ownership

| Reg. No | Name | Component | Ownership |
|---|---|---|---|
| IT23262768 | M.R.F. Reshma | **Component A: Data** | Data Preprocessing, Sentence Construction Pipeline, Dataset Creation |
| IT23289048 | M.S.M. Suhail | **Component B: Model** | MediaPipe Feature Engineering, BiLSTM + CTC Model |
| IT23341104 | M.H.M. Shamhan | **Component C: System** | Real-Time Inference, Robustness Benchmark |

> Every member must be able to defend **all three components** in the viva. Cross-training is mandatory.

---

## 3. Research Problem and Novelty

### 3.1 The Gap

- All existing SLSL systems recognise **one sign at a time** from bounded 3-second clips.
- No peer-reviewed framework addresses **Continuous Sign Language Recognition (CSLR)** for SLSL.
- SSL400 contains only word-level sequences, with no sentence-level data.
- Prior SLSL work relies exclusively on **hand-only landmarks**, discarding body pose and facial expression context.

### 3.2 Novelty Claims

| Novelty | Why It Is New |
|---|---|
| First CSLR system for SLSL | No prior work addresses continuous sentence-level recognition for SLSL |
| First sentence-level SLSL dataset | SSL400 is word-level only; this project constructs the first sentence-level sequences |
| First use of MediaPipe Holistic for SLSL | All prior SLSL work uses hand-only landmarks; this captures 543 pose + hand + face landmarks per frame |
| First healthcare-domain SLSL system | No healthcare-specific SLSL dataset or recognition system has been developed |
| First SLSL robustness benchmark | No SLSL model evaluated under varied lighting, backgrounds, or signer diversity |

All "first ever" claims must be re-verified against the literature in Phase 1 and again before paper submission.

---

## 4. Objectives

### 4.1 Main Objective

To create a real-time continuous SLSL recognition system using the SSL400 dataset and an innovative word-to-sentence mapping pipeline, combined with MediaPipe Holistic body landmark extraction and a BiLSTM + CTC model, targeting healthcare-domain sentence recognition.

### 4.2 Sub-Objectives by Member

| Member | Sub-Objective | Key Deliverables |
|---|---|---|
| **Reshma** (Component A) | Data Preprocessing and Sentence Construction: prepare SSL400; design and build the sentence creation pipeline; produce the first healthcare sentence-level SLSL dataset. | SSL400 audit report; healthcare gloss inventory; sentence construction pipeline (code); sentence-level NumPy landmark dataset; CVAT annotation files |
| **Suhail** (Component B) | BiLSTM Model with MediaPipe Holistic Features: generate full-body spatiotemporal features; develop stacked BiLSTM + CTC; train and evaluate. | MediaPipe Holistic extraction pipeline; feature specification document; trained BiLSTM-CTC model; WER and accuracy evaluation report |
| **Shamhan** (Component C) | Real-Time Integration and Robustness: integrate the trained model into a live inference pipeline; display Sinhala text; benchmark robustness. | Real-time OpenCV + MediaPipe pipeline; Sinhala text display interface; robustness benchmark report; FPS analysis |

---

## 5. Methodology

### 5.1 Pipeline Stages

| Stage | Process | Technology |
|---|---|---|
| 1. Video Input | Live webcam stream capturing the signer performing continuous SLSL gestures | OpenCV |
| 2. Landmark Extraction | MediaPipe Holistic extracts 543 landmarks per frame (pose + hands + face mesh); normalised to NumPy arrays | MediaPipe Holistic |
| 3. Temporal Windowing | Consecutive frames assembled into fixed-length temporal windows for the model | NumPy / Python |
| 4. BiLSTM + CTC | Bidirectional LSTM captures forward and backward temporal dependencies; CTC output layer removes the need for frame-level alignment | TensorFlow / PyTorch |
| 5. Sinhala Output | Recognised glosses mapped to Sinhala text and displayed on screen in real time | Python / Unicode |

### 5.2 Dataset Strategy

- SSL400 word-level clips are **concatenated and smoothed** to form synthetic sentence sequences.
- Healthcare sentence templates are designed with SLSL linguistic guidance from **SLAD**.
- Vocabulary gaps are filled by team members **self-recording** additional signs.
- All data is annotated using **CVAT** and converted to structured NumPy landmark arrays.
- A **real continuous validation set (30 to 50 sentences)** is recorded to test generalisation beyond synthetic data.

### 5.3 Model Strategy

- **Primary:** Stacked BiLSTM + CTC. The BiLSTM reads the landmark sequence in both directions, capturing long-range sign dependencies. CTC automatically learns word boundaries without frame-level annotation.
- **Fallback (if CTC fails):** Sliding-window isolated classifier with duplicate-suppression post-processing. This is a fully valid CSLR approach and a guaranteed working baseline. See [Appendix B](#appendix-b-ctc-risk-and-fallback-architecture).

### 5.4 Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| Word Error Rate (WER) | Primary CSLR metric; measures substitutions, insertions, deletions vs ground truth | < 30% synthetic / < 50% real continuous |
| Sentence Accuracy | Percentage of complete sentences correctly decoded | > 60% on constrained healthcare vocabulary |
| Real-time FPS | Frames processed per second in the live pipeline | >= 15 FPS at 640x480 |
| Robustness Score | WER delta across lighting, background, and signer variation | < 15% WER increase vs baseline |

---

## 6. Phase-by-Phase Plan

Nine phases with **hard gates**: the group does not proceed until exit criteria are met and the supervisor signs off.

### Phase 0: Foundation and Alignment
**Weeks 1-2 | Owner: All Members**

Key tasks:
- Confirm TAF approval; set up GitHub, Drive, Zotero/Mendeley
- Install and verify MediaPipe Holistic on all machines; record exact versions
- Download and audit SSL400 (400 signs, 12,708 videos)
- Draft and submit SLIIT Ethics Review Committee application
- Contact SLAD; begin SLSL familiarisation programme
- Begin first reading block (10 papers each from Doc 2)

**Deliverable:** GitHub repo live; SSL400 downloaded; ethics submitted; MediaPipe running on all machines; SLSL glossary produced

**Exit criteria:** Infrastructure live; ethics submitted; 30+ papers logged; MediaPipe landmark extraction working on each machine

### Phase 1: Literature Review and Problem Refinement
**Weeks 3-5 | Owner: All Members**

Key tasks:
- Complete systematic review (min 60 papers: 20+ per member)
- Produce literature review matrix
- Write comparative analysis of CSLR architectures
- Re-verify all "first ever" novelty claims against the literature
- Refine research questions into testable hypotheses

**Deliverable:** Literature review matrix; draft literature review chapter (2,500-3,500 words); verified novelty claims document

**Exit criteria:** Review matrix complete; draft chapter reviewed by supervisor; all novelty claims verified or revised in writing

### Phase 2: Project Charter and Individual Proposals
**Weeks 5-7 | Owner: All Members**

Key tasks:
- Write Project Charter (group document)
- Write individual proposals (each member independently)
- Include Gantt chart, WBS, risk register in each proposal
- Rehearse proposal presentation with supervisor at least one week before the assessed date

**Deliverable:** Project Charter; three individual proposals; Gantt chart; presentation slides

**Exit criteria:** Charter approved; all proposals reviewed by supervisor and submitted; Turnitin similarity within module threshold

### Phase 3: Data Foundation
**Weeks 6-9 | Owner: Reshma (leads), All**

Key tasks:
- Full SSL400 audit; document class distribution, duration, quality
- Healthcare vocabulary elicitation and gloss inventory v1 (with SLAD guidance)
- Sentence template design (e.g., "I have chest pain")
- Build sentence construction pipeline v1; concatenate and smooth SSL400 sequences
- Self-record supplementary signs for out-of-vocabulary healthcare terms
- Annotate with CVAT; convert to NumPy arrays
- Record real continuous validation set (30-50 sentences)

**Deliverable:** SSL400 audit report; healthcare gloss inventory; sentence construction pipeline (code, versioned); sentence-level landmark dataset; CVAT files; dataset card

**Exit criteria:** Dataset card produced; train/val/test splits documented; real continuous test set available; supervisor checkpoint S4 signed

### Phase 4: Feature Engineering and Baseline Model
**Weeks 9-13 | Owner: Suhail (leads), All**

Key tasks:
- Run all SSL400 frames through MediaPipe Holistic; normalise to NumPy arrays
- Define feature specification (landmark subsets, normalisation method)
- Build stacked BiLSTM + CTC architecture
- Train on synthetic sentence dataset; monitor for CTC blank collapse
- Implement sliding-window fallback classifier in parallel as a safety net
- Tune hyperparameters; evaluate WER and sentence accuracy

**Deliverable:** Feature specification document; trained baseline model; training loss and WER curves; evaluation report

**Exit criteria:** Reproducible training run; honest baseline WER reported; supervisor checkpoint S5 signed

### Phase 5: Progress Presentation I
**Weeks 13-14 | Owner: All Members**

Key tasks:
- Prepare PP1 slides showing a working pipeline with real numbers (not plans)
- Demonstrate end-to-end pipeline live or via recorded fallback video
- Rehearse with supervisor at least once before the assessed presentation

**Deliverable:** PP1 slide deck; working demo (live or recorded); rehearsal completed

**Exit criteria:** PP1 delivered; supervisor checkpoint S6 signed

### Phase 6: Real-Time Integration
**Weeks 14-18 | Owner: Shamhan (leads), All**

Key tasks:
- Integrate BiLSTM-CTC model into the real-time OpenCV + MediaPipe pipeline
- Implement live CTC decoder for gloss-to-text conversion
- Design Sinhala Unicode text display interface; test rendering
- Measure and optimise real-time FPS; target >= 15 FPS at 640x480
- Produce end-to-end live demo (record fallback video)

**Deliverable:** Real-time inference pipeline (code); Sinhala text display UI; live demo (recorded); FPS benchmark

**Exit criteria:** End-to-end live demo at documented FPS; supervisor checkpoint S7 signed

### Phase 7: Robustness Evaluation and Results
**Weeks 18-21 | Owner: Shamhan (leads), All**

Key tasks:
- Benchmark robustness across three axes: lighting, background, signer identity variation
- Run all final experiments; compile complete results tables
- Ensure reproducibility: clean checkout by another member
- Freeze test set for final reporting only (never used for tuning)
- Draft research paper targeting ICTer / MERCon / ICIIS

**Deliverable:** Robustness benchmark report; complete results tables; research paper draft

**Exit criteria:** All experiment tables complete; seeds and CIs reported; test set used only for final reporting; supervisor checkpoint S8 signed

### Phase 8: Final Report, Website and Viva
**Weeks 21-24 | Owner: All Members**

Key tasks:
- Write final report (each member writes their own chapter)
- Build project website (content collected throughout the project)
- Finalise and submit research paper
- Prepare final presentation slides
- Rehearse full system defence twice (once with supervisor); each member defends all three components

**Deliverable:** Final thesis report; project website; research paper submitted; final presentation slides; viva preparation

**Exit criteria:** Report, website, viva rehearsal complete; supervisor checkpoint S9 signed

---

## 7. Timeline Overview

| Phase | Title | Weeks | Primary Owner |
|---|---|---|---|
| 0 | Foundation and Alignment | 1-2 | All Members |
| 1 | Literature Review and Problem Refinement | 3-5 | All Members |
| 2 | Project Charter and Individual Proposals | 5-7 | All Members |
| 3 | Data Foundation | 6-9 | Reshma (leads) |
| 4 | Feature Engineering and Baseline Model | 9-13 | Suhail (leads) |
| 5 | Progress Presentation I | 13-14 | All Members |
| 6 | Real-Time Integration | 14-18 | Shamhan (leads) |
| 7 | Robustness Evaluation and Results | 18-21 | Shamhan (leads) |
| 8 | Final Report, Website and Viva | 21-24 | All Members |

---

## 8. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | CTC fails to converge (blank collapse) | Medium | High | Build sliding-window fallback by Week 10 as guaranteed baseline; time-box CTC; monitor after every 5 epochs | Suhail |
| R2 | SSL400 data quality issues | Medium | High | Full audit in Phase 3 before any modelling; document all exclusions | Reshma |
| R3 | Ethics approval delayed | Medium | High | Submit in Phase 0; pipeline and synthetic work proceeds first | All |
| R4 | Face mesh dominates features; model overfits | High | Medium | Ablate face landmark subsets; reduced keypoint set; strong regularisation | Suhail |
| R5 | Real-time FPS too low | Medium | High | Profile in Phase 4; consider frame skipping and model quantisation | Shamhan |
| R6 | Limited access to Deaf signers | Medium | High | Engage SLAD in Phase 0; team self-records as fallback | All |
| R7 | Novelty claims contradicted late | Medium | Medium | Verify all claims in Phase 1; rerun search before paper submission | All |
| R8 | Synthetic data does not generalise to real continuous signing | High | High | Record real continuous validation set in Phase 3; report synthetic vs real WER separately | All |
| R9 | Compute limits (Colab timeouts) | Medium | Medium | Checkpoint every epoch; apply for SLIIT lab GPU access in Phase 0 | Suhail |
| R10 | Scope creep | High | Medium | Scope frozen after Phase 2; all extensions listed as future work only | Supervisor |

---

## 9. Governance and Communication

### 9.1 Meeting Cadence

| Meeting | Frequency | Duration | Attendees | Output |
|---|---|---|---|---|
| Group stand-up | Twice weekly (Mon, Thu) | 20 min | 3 students | Updated task board |
| Supervisor progress | Weekly | 30-45 min | Students + Supervisor | Signed logbook entry |
| Co-supervisor technical | Fortnightly | 30 min | Students + Co-supervisor | Written technical feedback |
| Milestone gate review | Before each deliverable | 60 min | All | Go / No-go decision |

### 9.2 Meeting Discipline

Students must bring to every supervisor meeting:
- A **one-page written update** (what was planned, done, blocked)
- **Evidence**: a notebook output, a metric, a commit log, a plot. Verbal claims are not accepted.
- **Three specific questions** or decision points requiring supervisor input

### 9.3 Escalation Policy

Any task more than **five working days** behind schedule is raised immediately, not at the next scheduled meeting. An absent member's tasks are reassigned within **48 hours** and minuted.

### 9.4 Logbook

Each member maintains a weekly individual logbook entry: date, hours, tasks completed, artefacts produced (file/commit references), blockers, and next week's plan. The supervisor signs off monthly. The logbook is an **assessed component**.

---

## 10. Supervisor Checkpoints

| # | Checkpoint | Phase | Sign-Off Requires |
|---|---|---|---|
| S1 | Scope and infrastructure lock | 0 | Repository live; ethics submitted; SSL400 downloaded; MediaPipe running on all machines; glossary produced |
| S2 | Literature review adequacy | 1 | Review matrix with 60+ papers; novelty claims verified; draft literature review chapter reviewed |
| S3 | Proposal readiness | 2 | Three individual proposals reviewed and revised; Charter approved; similarity check passed |
| S4 | Dataset freeze | 3 | Dataset card; train/val/test splits documented; real continuous test set available |
| S5 | Baseline credibility | 4 | Reproducible training run; honest baseline WER reported; training curves logged |
| S6 | PP1 readiness | 5 | Rehearsal completed; demo fallback recorded; slide deck reviewed |
| S7 | Integration proof | 6 | Live end-to-end demo at documented FPS; Sinhala text rendering verified |
| S8 | Results freeze | 7 | All experiment tables complete; seeds and CIs reported; test set used only for final reporting |
| S9 | Submission readiness | 8 | Final report, website, and viva rehearsal complete; similarity check passed |

---

## 11. Assessment Components

| Component | Key Preparation Strategy |
|---|---|
| Topic Assessment Form | Complete; approved by supervisors |
| Project Charter | Written in Phase 2; consistent with this plan so panel questions map cleanly |
| Individual Proposals | Each member writes their own; shared sections must **NOT** be copy-pasted, as similarity is checked |
| Progress Presentation I | Show working code and real numbers, not plans. A modest working baseline beats an ambitious slide deck. |
| Research Paper | Draft from Phase 7; target ICTer / MERCon / ICIIS; confirm venue and page limits in Phase 1 |
| Progress Presentation II | Live demo expected. Rehearse on actual presentation hardware. Record a fallback demo video. |
| Final Report | Written continuously from Phase 3, not at the end. Each phase produces text that lands in a chapter. |
| Project Website | Built in Phase 8 but content (screenshots, diagrams, demo video) collected throughout |
| Logbook | Weekly entries, signed monthly. Cannot be reconstructed retrospectively with credibility. |
| Final Presentation and Viva | Each member must defend the whole system, not only their component. Cross-training is mandatory. |

---

## 12. Technology Stack

| Component | Tool / Technology | Notes |
|---|---|---|
| Landmark Extraction | MediaPipe Holistic | Pin exact version; record in `FEATURE_SPEC.md` |
| Video Processing | OpenCV | Used for both offline extraction and real-time webcam capture |
| Deep Learning Framework | TensorFlow / PyTorch | Confirm preferred framework in Week 2; keep consistent throughout |
| Model Architecture | Bidirectional LSTM + CTC | Stacked BiLSTM; CTC loss for alignment-free sequence prediction |
| Data Storage | NumPy (`.npy` arrays) | Landmarks stored per sequence; dataset versioned on GitHub + Drive |
| Annotation | CVAT | For supplementary recorded data; export labels as JSON/CSV |
| Development Environment | Google Colab / VS Code | Colab for GPU training; VS Code for local development |
| Version Control | GitHub | GitHub organisation per Doc 1 section 7.1; branch-per-feature; PR reviews required |
| Reference Management | Zotero / Mendeley | Shared library with folder structure per Doc 2 |
| Language | Python 3.10 or 3.11 | Pin exact version; MediaPipe wheel availability is version-sensitive |

---

## 13. Immediate Actions (First Two Weeks)

| # | Action | Owner | Due |
|---|---|---|---|
| 1 | Transcribe official IT4010 assessed dates into the Gantt chart and circulate | Shamhan | Week 1 Day 1 |
| 2 | Create GitHub organisation, repository skeleton, and shared task board | Suhail | Week 1 Day 1 |
| 3 | Set up shared Zotero / Mendeley library with the folder structure in Doc 2 | Reshma | Week 1 Day 2 |
| 4 | Download SSL400 from Kaggle; verify file counts and checksums; run first integrity audit | Reshma | Week 1 Day 2 |
| 5 | Install and verify MediaPipe Holistic on all three machines; extract landmarks from one SSL400 video | Suhail | Week 1 Day 3 |
| 6 | Webcam capture loop with live landmark overlay running on Shamhan's machine | Shamhan | Week 1 Day 4 |
| 7 | Draft SLIIT Ethics Review Committee application; submit by end of Week 2 | All | Week 2 |
| 8 | Contact SLAD / Deaf community; send introduction request; request an advisory contact | Reshma | Week 2 |
| 9 | Complete SLSL familiarisation module: learn manual alphabet and 30-40 healthcare signs | All | Week 2 |
| 10 | Begin first reading block: 10 papers each from Doc 2; log in shared reference manager | All | End of Week 2 |

---

## 14. Quality Gates

Before any assessed submission, the group runs this checklist.

**Technical**
- [ ] Results reproduced from a clean checkout by a member who did not produce them
- [ ] All numbers in slides/report traceable to a logged experiment run
- [ ] No result reported from a single seed without a stated caveat
- [ ] Test set used only for final reporting, never for tuning
- [ ] Splits are signer-independent and verified to have no signer leakage

**Documentation**
- [ ] Every figure has a caption, axis labels, and units
- [ ] Every novelty claim is supported by a cited literature search
- [ ] References complete and consistently formatted (IEEE)
- [ ] Similarity check passed; individual contribution clearly delineated

**Presentation**
- [ ] Rehearsed at least twice, once in front of the supervisor
- [ ] Demo has a recorded fallback video in case of live failure
- [ ] Each member can answer questions about the other two components

---

## 15. Publication Plan

| Item | Detail |
|---|---|
| Target output | One conference paper covering the full system, with all three members as co-authors |
| Candidate venues | ICTer, ICAC, MERCon, SLAAI, ICIIS, or an IEEE regional conference. Confirm submission deadlines in Phase 1. |
| Positioning | Lead with the resource contribution (first sentence-level SLSL dataset and construction pipeline) and honest synthetic-vs-real evaluation. More publishable than an accuracy number alone. |
| Artefact release | Release the sentence construction pipeline and dataset card publicly if licensing permits, which increases citation and impact. |
| Author order | Agreed in writing in Phase 2, based on contribution; supervisor as corresponding author per SLIIT convention. |

---

## Appendix A: System Architecture Summary

| Layer | Input | Process | Output |
|---|---|---|---|
| 1. Capture | Raw webcam video | OpenCV reads frames at 30 fps | Raw RGB frames |
| 2. Extraction | Raw RGB frames | MediaPipe Holistic: 33 pose + 21x2 hand + 468 face = 543 landmarks x (X, Y, Z) per frame | Normalised NumPy array per frame |
| 3. Windowing | Per-frame arrays | Fixed-length temporal windows assembled; stride controls overlap | Shape: `(T, 543x3)` per window |
| 4. BiLSTM | Temporal window `(T, features)` | Forward + backward LSTM layers capture long-range dependencies in both directions | Hidden state sequence |
| 5. CTC Decoder | Hidden state sequence | CTC loss during training; greedy or beam-search decode at inference; blank token collapse | Gloss sequence |
| 6. Text Output | Gloss sequence | Unicode Sinhala text rendered and displayed on screen in real time | Sinhala text sentence |

```
Webcam -> OpenCV -> MediaPipe Holistic (543 landmarks) -> Temporal windows (T, 543x3)
      -> Stacked BiLSTM -> CTC decoder -> Gloss sequence -> Sinhala Unicode text (live display)
```

---

## Appendix B: CTC Risk and Fallback Architecture

If CTC training fails to converge (blank collapse), the following fallback is activated.

| Step | Process |
|---|---|
| 1. Train isolated classifier | Train an LSTM or GRU on SSL400 individual word clips using standard supervised classification. Very stable to train. |
| 2. Sliding window inference | At inference time, slide a fixed window (e.g., 30 frames, stride 5) across the continuous video stream. Feed each window through the classifier. |
| 3. Duplicate suppression | Apply a confidence threshold filter (e.g., 0.75) to remove low-confidence predictions. Merge consecutive identical tokens into a single token. |
| 4. Output gloss sequence | The resulting sequence of distinct, high-confidence tokens forms the recognised sentence output. |

**Academic framing:** "an active online CSLR framework pairing an optimised isolated gesture classifier with an overlapping sliding-window temporal decoder, utilising non-maximum suppression for token deduplication." This is fully publishable and accurately describes the architecture.

---

## Notes for Coding Agents

Practical rules derived from the plan above. Follow these when generating or modifying code in this repository.

1. **Component boundaries.** Component A (Reshma) owns data and the sentence construction pipeline. Component B (Suhail) owns MediaPipe feature engineering and the BiLSTM + CTC model. Component C (Shamhan) owns real-time inference and the robustness benchmark. Keep code for each in its own module and agree on interfaces (array shapes, file formats) before changing them.
2. **Feature shape.** Each frame is 543 landmarks x (X, Y, Z) = 1629 values (33 pose + 21 left hand + 21 right hand + 468 face). A window has shape `(T, 543*3)`. Any change to landmark subsets or normalisation must be recorded in `FEATURE_SPEC.md`.
3. **Pin versions.** Use Python 3.10 or 3.11. Pin the exact MediaPipe Holistic version, since wheel availability is version-sensitive. Use one deep learning framework (TensorFlow or PyTorch) consistently throughout.
4. **Storage format.** Landmarks are stored per sequence as `.npy` arrays. Annotation labels from CVAT are exported as JSON/CSV.
5. **Test set discipline.** The test set is used only for final reporting, never for tuning. Splits must be **signer-independent** with no signer leakage.
6. **Reproducibility.** Set and log random seeds, report results across multiple seeds with confidence intervals, and make sure every reported number traces to a logged experiment run.
7. **CTC monitoring.** Watch for blank collapse; check after every 5 epochs and checkpoint every epoch (Colab timeouts). Keep the sliding-window fallback classifier (Appendix B) working in parallel.
8. **Report synthetic and real results separately.** Synthetic sentences come from concatenated SSL400 clips; the real continuous validation set (30-50 sentences) measures true generalisation.
9. **Face landmark ablation.** The face mesh (468 points) can dominate features and cause overfitting. Support ablation over landmark subsets and a reduced keypoint set.
10. **Real-time target.** At least 15 FPS at 640x480. Profile early; frame skipping and model quantisation are the planned optimisations.
11. **Sinhala output.** Render Unicode Sinhala text correctly in the display UI and verify rendering (checkpoint S7).
12. **Evaluation targets.** WER < 30% (synthetic) and < 50% (real continuous); sentence accuracy > 60% on the constrained healthcare vocabulary; robustness degradation < 15% WER increase across lighting, background, and signer variation.
13. **Scope is frozen after Phase 2.** Extra ideas go in a "future work" list, not into the implementation.
14. **Branching.** Branch per feature; pull request reviews are required.

---

*SignBridge, KND_CDAP_DS_06, SLIIT IT4010, 2026. Plan version v1.0, August 2026.*
