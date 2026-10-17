"""Data preparation script for Federated RAG.

Generates synthetic non-IID domain-specific document sets across 4 federated clients:
- Client 1: Healthcare / Medical
- Client 2: Education / Academics
- Client 3: Industrial / Engineering
- Client 4: General Knowledge / History

Provides utility functions to save and load client data splits to/from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DEFAULT_SPLITS_PATH = Path(__file__).resolve().parent / "client_splits.json"
KAGGLE_SPLITS_PATH = Path(__file__).resolve().parent / "kaggle_client_splits.json"


def get_synthetic_client_data() -> Dict[int, List[Dict[str, str]]]:
    """Generate synthetic non-IID domain-specific document sets for 4 federated clients.

    Returns:
        Dict[int, List[Dict[str, str]]]: Mapping of client ID to list of document dictionaries.
            Each document conforms to:
            {"doc_id": str, "text": str, "domain": str}
    """
    client_data: Dict[int, List[Dict[str, str]]] = {
        1: [
            {
                "doc_id": "MED-001",
                "text": "Cardiovascular risk reduction strategies emphasize managing blood pressure, statin therapy for LDL reduction, and lifestyle modification.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "MED-002",
                "text": "Type 2 diabetes mellitus clinical guidelines recommend metformin as first-line pharmacotherapy alongside continuous glucose monitoring and HbA1c tracking.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "MED-003",
                "text": "mRNA vaccine platforms utilize lipid nanoparticles to encapsulate spike-protein mRNA transcripts, stimulating both humoral and cellular adaptive immunity.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "MED-004",
                "text": "Targeted oncology therapies including monoclonal antibodies and immune checkpoint inhibitors (PD-1/PD-L1) have revolutionized metastatic melanoma management.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "MED-005",
                "text": "Electronic Health Record (EHR) privacy compliance adheres to HIPAA guidelines for de-identifying protected health information (PHI).",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "MED-006",
                "text": "Antimicrobial stewardship initiatives mitigate hospital-acquired multidrug-resistant bacterial infections such as MRSA and C. difficile.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "MED-007",
                "text": "Acute ischemic stroke treatment requires rapid triage and intravenous administration of tissue plasminogen activator (tPA) within the 4.5-hour therapeutic window.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "MED-008",
                "text": "Sepsis resuscitation protocols highlight early broad-spectrum intravenous antibiotics, lactate clearance tracking, and crystalloid fluid administration.",
                "domain": "Healthcare / Medical",
            },
        ],
        2: [
            {
                "doc_id": "EDU-001",
                "text": "Formative assessment methodologies allow educators to provide diagnostic feedback during instruction, improving student retention and learning outcomes.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "EDU-002",
                "text": "Cognitive Load Theory posits that instructional design should minimize extraneous cognitive load to optimize working memory capacity for schema acquisition.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "EDU-003",
                "text": "Higher education accreditation standards mandate rigorous peer review processes, faculty qualification verifications, and curriculum alignment audits.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "EDU-004",
                "text": "Flipped classroom pedagogies leverage asynchronous online video lectures, dedicating in-person seminar hours to collaborative problem-based inquiry.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "EDU-005",
                "text": "Academic peer review practices ensure scientific reproducibility and methodological rigor through double-blind manuscript evaluation.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "EDU-006",
                "text": "University syllabus guidelines require explicit learning objectives, transparent grading rubrics, and reasonable accommodation statements.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "EDU-007",
                "text": "Active learning frameworks, including peer instruction and interactive concept mapping, significantly boost STEM student performance over passive lecturing.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "EDU-008",
                "text": "Learning Management Systems (LMS) collect behavioral telemetry and quiz analytics to identify students requiring academic advisory interventions.",
                "domain": "Education / Academics",
            },
        ],
        3: [
            {
                "doc_id": "IND-001",
                "text": "Predictive maintenance on multi-axis CNC milling machines leverages vibration spectrum analysis and thermal imaging to prevent spindle bearing failure.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "IND-002",
                "text": "Programmable Logic Controllers (PLCs) automate sequential factory operations using ladder logic and real-time industrial fieldbus protocols like Modbus and PROFINET.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "IND-003",
                "text": "Supervisory Control and Data Acquisition (SCADA) telemetry provides centralized monitoring for high-voltage power substations and hydraulic distribution pipelines.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "IND-004",
                "text": "Structural finite element analysis (FEA) evaluates stress concentrations, fatigue life, and tensile strain under dynamic mechanical loading in steel fabrication.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "IND-005",
                "text": "Autonomous robotic palletizers and automated guided vehicles (AGVs) optimize throughput and warehouse logistics in automotive assembly facilities.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "IND-006",
                "text": "Shell-and-tube industrial heat exchangers utilize thermodynamic counterflow configurations to maximize heat transfer efficiency in petrochemical refineries.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "IND-007",
                "text": "Six Sigma manufacturing quality frameworks apply DMAIC methodologies to minimize dimensional variances and defect rates below 3.4 defects per million parts.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "IND-008",
                "text": "High-temperature refractory furnace linings protect blast furnace structural shells against corrosive molten iron and slag ablation.",
                "domain": "Industrial / Engineering",
            },
        ],
        4: [
            {
                "doc_id": "HIS-001",
                "text": "The Italian Renaissance originated in fourteenth-century Florence, sparked by humanist rediscovery of classical Greek and Roman texts and artistic patronage.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "HIS-002",
                "text": "The First Industrial Revolution in Britain revolutionized textile manufacturing and transportation through James Watt's refined steam engine.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "HIS-003",
                "text": "Ancient Mesopotamian civilizations between the Tigris and Euphrates rivers developed cuneiform script, codified legal systems, and agricultural canal irrigation.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "HIS-004",
                "text": "The fall of the Western Roman Empire in 476 CE resulted from internal economic decline, civil strife, and repeated incursions by Germanic federates.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "HIS-005",
                "text": "The Silk Road network facilitated overland and maritime trade across Afro-Eurasia, enabling the exchange of silk, spices, papermaking, and philosophical traditions.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "HIS-006",
                "text": "The Space Race between the United States and Soviet Union culminated in NASA's Apollo 11 lunar landing in July 1969.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "HIS-007",
                "text": "The Magna Carta, sealed at Runnymede in 1215, established the principle that the English sovereign was subject to the rule of law and feudal charter rights.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "HIS-008",
                "text": "The Age of Enlightenment in eighteenth-century Europe championed rationalism, scientific empiricism, individual liberties, and separation of political powers.",
                "domain": "General Knowledge / History",
            },
        ],
    }
    return client_data


def save_client_splits(
    data: Dict[int, List[Dict[str, str]]],
    filepath: Path | str = DEFAULT_SPLITS_PATH,
) -> Path:
    """Save the client document splits to a JSON file.

    Args:
        data: Mapping from client_id (int) to list of document dictionaries.
        filepath: Target filepath for the JSON output.

    Returns:
        Path: Resolved path to the saved JSON file.
    """
    target_path = Path(filepath).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert integer keys to strings for standardized JSON serializability
    serializable_data = {str(client_id): docs for client_id, docs in data.items()}

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

    return target_path


def load_client_splits(
    filepath: Path | str = DEFAULT_SPLITS_PATH,
) -> Dict[int, List[Dict[str, str]]]:
    """Load client document splits from a JSON file.

    Args:
        filepath: Path to the JSON splits file.

    Returns:
        Dict[int, List[Dict[str, str]]]: Mapping from integer client_id to list of document dicts.
    """
    target_path = Path(filepath).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Client splits file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        raw_data: Dict[str, List[Dict[str, str]]] = json.load(f)

    # Parse keys back to integers for clean typed client indexing
    return {int(client_id): docs for client_id, docs in raw_data.items()}


def get_kaggle_client_data() -> Dict[int, List[Dict[str, str]]]:
    """Generate real-world Kaggle domain-partitioned document sets across 4 clients.

    Covers:
    - Client 1: Healthcare / Medical (Kaggle MIMIC / PubMed / Clinical Notes)
    - Client 2: Education / Academics (Kaggle EdTech / Higher Ed / Academic Analytics)
    - Client 3: Industrial / Engineering (Kaggle NASA Turbofan / SCADA Telemetry / IoT)
    - Client 4: General Knowledge / History (Kaggle Historical Corpus / World History)

    Returns:
        Dict[int, List[Dict[str, str]]]: Mapping of client ID to document dictionaries.
    """
    kaggle_data: Dict[int, List[Dict[str, str]]] = {
        1: [
            {
                "doc_id": "KAG-MED-001",
                "text": "A 58-year-old male presenting with acute ST-elevation myocardial infarction underwent primary percutaneous coronary intervention with drug-eluting stent implantation in the left anterior descending artery.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "KAG-MED-002",
                "text": "Clinical study evaluating sodium-glucose cotransporter-2 (SGLT2) inhibitors in patients with type 2 diabetes mellitus and heart failure with preserved ejection fraction.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "KAG-MED-003",
                "text": "Retrospective cohort analysis of immune checkpoint inhibitor-induced pneumonitis in non-small cell lung cancer treated with pembrolizumab and nivolumab.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "KAG-MED-004",
                "text": "Genomic profiling and next-generation sequencing identify actionable mutations in EGFR, ALK, and ROS1 for targeted tyrosine kinase inhibitor administration.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "KAG-MED-005",
                "text": "Hospital intensive care unit EHR records reveal early hemodynamic stabilization and septic shock fluid responsiveness monitoring using arterial line pulse pressure variation.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "KAG-MED-006",
                "text": "Inpatient antibiotic de-escalation protocols reduce Clostridioides difficile relapse rates following empiric piperacillin-tazobactam therapy for nosocomial pneumonia.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "KAG-MED-007",
                "text": "Continuous subcutaneous insulin infusion pump therapy combined with automated bolus calculators significantly attenuates severe hypoglycemic episodes in pediatric type 1 diabetes.",
                "domain": "Healthcare / Medical",
            },
            {
                "doc_id": "KAG-MED-008",
                "text": "Neurological intensive care management of malignant middle cerebral artery infarction indicates decompressive hemicraniectomy significantly reduces 30-day mortality.",
                "domain": "Healthcare / Medical",
            },
        ],
        2: [
            {
                "doc_id": "KAG-EDU-001",
                "text": "Empirical study on peer assessment accuracy in massive open online courses (MOOCs) reveals calibrated peer grading rubrics correlate strongly with instructor evaluations (r=0.84).",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "KAG-EDU-002",
                "text": "Investigating the impact of generative AI tutoring assistants on undergraduate computer science programming self-efficacy and formative code review engagement.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "KAG-EDU-003",
                "text": "Predicting university student academic probation using institutional canvas LMS clickstream telemetry, assignment submission timestamps, and forum participation frequency.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "KAG-EDU-004",
                "text": "Institutional accreditation review of online degree programs confirms faculty-to-student interaction thresholds and verified remote proctoring enhance degree credibility.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "KAG-EDU-005",
                "text": "Quasi-experimental trial comparing inquiry-based flipped chemistry laboratories with traditional verification experiments indicates higher conceptual transfer on organic synthesis exams.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "KAG-EDU-006",
                "text": "Bibliometric analysis of double-blind versus single-blind peer review acceptance rates across 1,200 scientific publications in computer science and computational biology.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "KAG-EDU-007",
                "text": "Multilevel modeling of early childhood literacy interventions demonstrates daily dialogic reading interventions yield sustained phonemic awareness gains across socioeconomic cohorts.",
                "domain": "Education / Academics",
            },
            {
                "doc_id": "KAG-EDU-008",
                "text": "Evaluating cognitive load and mental fatigue during high-stakes remote graduate admissions testing using pupil diameter dilation and blink frequency tracking.",
                "domain": "Education / Academics",
            },
        ],
        3: [
            {
                "doc_id": "KAG-IND-001",
                "text": "NASA turbofan engine C-MAPSS degradation dataset analysis demonstrates recurrent neural network remaining useful life (RUL) estimation outperforms regression with 12.3 cycles RMSE.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "KAG-IND-002",
                "text": "High-frequency acoustic emission sensor signals detect micro-crack propagation and subsurface fatigue spalling in wind turbine planetary gearbox bearings under variable torque loads.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "KAG-IND-003",
                "text": "SCADA telemetry log anomaly detection for municipal water treatment facility identifies anomalous valve pressure surges and unauthorized Modbus TCP function code injections.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "KAG-IND-004",
                "text": "Automotive robotic spot welding quality classification utilizing dynamic electrode resistance curves and infrared thermal imaging to identify expulsion defects in high-strength steels.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "KAG-IND-005",
                "text": "Closed-loop field-oriented control (FOC) algorithms for permanent magnet synchronous motors (PMSM) mitigate cogging torque ripples and maximize inverter switching efficiency in automated guided vehicles.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "KAG-IND-006",
                "text": "Computational fluid dynamics (CFD) modeling of multiphase fluid flow in offshore crude oil pipelines identifies slug formation boundaries and optimizes corrosion inhibitor injection rates.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "KAG-IND-007",
                "text": "Surface finish roughness optimization in high-speed dry milling of Ti-6Al-4V titanium alloy using coated carbide inserts and Taguchi orthogonal array parameter design.",
                "domain": "Industrial / Engineering",
            },
            {
                "doc_id": "KAG-IND-008",
                "text": "Industrial IoT edge gateway architecture for real-time vibration spectral density calculation on induction motors using MQTT protocol and local anomaly inference.",
                "domain": "Industrial / Engineering",
            },
        ],
        4: [
            {
                "doc_id": "KAG-HIS-001",
                "text": "The Congress of Vienna (1814-1815) reorganized the geopolitical architecture of Europe following the Napoleonic Wars, establishing the Concert of Europe balance of power doctrine.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "KAG-HIS-002",
                "text": "Excavations at the ancient Indus Valley site of Mohenjo-daro reveal sophisticated urban planning, standardized baked brick architecture, and advanced subterranean wastewater drainage networks.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "KAG-HIS-003",
                "text": "The Battle of Midway in June 1942 marked the strategic turning point in the Pacific Theater, as US naval cryptanalysts intercepted Japanese operational transmissions.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "KAG-HIS-004",
                "text": "The development of the printing press by Johannes Gutenberg in Mainz c. 1440 catalyzed the rapid dissemination of scientific and theological tracts throughout Renaissance Europe.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "KAG-HIS-005",
                "text": "The Hellenistic period following Alexander the Great's conquests facilitated Greco-Bactrian syncretism, Mediterranean maritime trade expansion, and the rise of the Library of Alexandria.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "KAG-HIS-006",
                "text": "The Manhattan Project at Los Alamos and Oak Ridge mobilized over 130,000 personnel to achieve nuclear fission under the scientific direction of J. Robert Oppenheimer.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "KAG-HIS-007",
                "text": "The Taiping Rebellion (1850-1864) against the Qing Dynasty resulted in over twenty million casualties and fundamental shifts in regional fiscal and military decentralization in southern China.",
                "domain": "General Knowledge / History",
            },
            {
                "doc_id": "KAG-HIS-008",
                "text": "The Bretton Woods Conference of 1944 established the International Monetary Fund and International Bank for Reconstruction and Development, anchoring global post-war currencies to gold.",
                "domain": "General Knowledge / History",
            },
        ],
    }
    return kaggle_data


def create_or_load_kaggle_dataset(
    filepath: Path | str = KAGGLE_SPLITS_PATH,
) -> Dict[int, List[Dict[str, str]]]:
    """Create or load the Kaggle client document splits JSON file.

    Args:
        filepath: Target filepath for the Kaggle JSON dataset.

    Returns:
        Dict[int, List[Dict[str, str]]]: Mapping from integer client_id to list of document dicts.
    """
    target_path = Path(filepath).resolve()
    if not target_path.exists():
        kaggle_data = get_kaggle_client_data()
        save_client_splits(kaggle_data, target_path)

    return load_client_splits(target_path)


if __name__ == "__main__":
    # Ensure both synthetic and Kaggle datasets are generated and verified
    splits = get_synthetic_client_data()
    saved_path = save_client_splits(splits)
    print(f"Successfully generated and saved synthetic splits to: {saved_path}")

    kaggle_splits = create_or_load_kaggle_dataset()
    print(f"Successfully generated and loaded Kaggle splits: {len(kaggle_splits)} clients loaded.")
    for cid, docs in kaggle_splits.items():
        print(f" - Client {cid} ({docs[0]['domain']}): {len(docs)} documents (Sample: {docs[0]['doc_id']})")
