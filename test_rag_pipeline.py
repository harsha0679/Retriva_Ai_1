import sys
import logging
from app.services.ingestion import KnowledgeIngestionService
from app.services.chat import ChatService
from app.services.document import DocumentService
from app.domain.models import ChatRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAG_Test_Suite")

def run_rag_evaluations():
    logger.info("==================================================================")
    logger.info("  STARTING HYBRID RAG (VECTOR + BM25 RRF) EVALUATION SUITE       ")
    logger.info("==================================================================")

    ingestion_service = KnowledgeIngestionService()
    chat_service = ChatService()
    document_service = DocumentService()

    # Clean up previous test database
    document_service.delete_all()

    # -------------------------------------------------------------------
    # DOCUMENT A: HR & Employee Handbook (Factual Questions)
    # -------------------------------------------------------------------
    doc_a_content = """
    KNOWLEDGEHUB HR POLICY (DOCUMENT A - REVISION 2026)
    
    1. Working Hours & Work Location
    Standard core working hours are 9:00 AM to 5:00 PM Eastern Standard Time, Monday through Friday.
    Employees are permitted to work remotely up to two days per week with prior manager approval.

    2. Health Benefits & Medical Allowance
    All full-time employees are eligible for full health insurance starting on their first day of employment.
    The annual health and wellness allowance is $1,500 per employee, reimbursable for gym memberships,
    mental health applications, or ergonomic home office equipment.

    3. Paid Leave Policy
    Employees receive 20 days of paid time off (PTO) annually, accrued monthly at a rate of 1.66 days per month.
    Unused PTO up to 5 days can be carried over into the next calendar year.
    """.encode('utf-8')

    # -------------------------------------------------------------------
    # DOCUMENT B: Technical Quantum Architecture (Different Topic)
    # -------------------------------------------------------------------
    doc_b_content = """
    PROJECT HELIOS QUANTUM ARCHITECTURE SPECIFICATION (DOCUMENT B)

    Abstract:
    Project Helios is a 128-qubit superconducting quantum processor designed for cryptographic resilience tests.

    Key Hardware Component Specifications:
    - Qubit Topology: Heavy-hexagonal lattice configuration to minimize crosstalk.
    - Dilution Refrigerator Operating Temperature: 15 millikelvin (0.015 K).
    - Single-Qubit Gate Fidelity: 99.92% achieved via drag-pulse calibration.
    - Two-Qubit CZ Gate Fidelity: 99.45% utilizing cross-resonance interaction.
    - Principal Investigator: Dr. Marcus Sterling (Department of Applied Quantum Physics).
    """.encode('utf-8')

    # -------------------------------------------------------------------
    # DOCUMENT C: Financial Strategy & Risk Management (Multi-Chunk Synthesis)
    # -------------------------------------------------------------------
    doc_c_content = """
    GLOBAL ENTERPRISE STRATEGY & RISK MANAGEMENT 2026 (DOCUMENT C)

    SECTION 1: REVENUE TARGETS
    The company projects a net operational revenue target of $120 Million for Fiscal Year 2026.
    Growth will primarily be driven by enterprise SaaS subscriptions, which are forecasted to expand by 35% year-over-year.

    SECTION 2: INFRASTRUCTURE CAPITAL EXPEDITION
    To support this revenue growth, capital expenditures are capped at $25 Million.
    Of this amount, $10 Million is allocated to data center GPU clusters, $8 Million to network redundancy,
    and $7 Million to cybersecurity compliance automation.

    SECTION 3: MITIGATION OF MARKET RISKS
    Market risk is mitigated by maintaining a liquidity reserve ratio of at least 18% of operating expenses.
    If currency exchange volatility exceeds 5%, cash reserves are automatically rebalanced into treasury yield instruments.
    """.encode('utf-8')

    # -------------------------------------------------------------------
    # DOCUMENT D: Software Engineer Resume (Typo & Layout Resilience Test)
    # -------------------------------------------------------------------
    doc_d_content = """
    AMULA MANOJ SAGAR - SOFTWARE DEVELOPMENT ENGINEER
    Email: manoj.sagar@example.com | Location: Remote / USA

    PROFESSIONAL SUMMARY:
    Software Engineer with 4+ years of experience building high-scale full-stack applications,
    distributed AI microservices, and Retrieval-Augmented Generation (RAG) architectures.

    WORK EXPERIENCE:
    1. Senior Software Development Engineer at TechCorp (2023 - Present)
       - Architected hybrid vector search pipelines reducing query latency by 45%.
       - Managed Kubernetes clusters running PyTorch and FastAPI microservices.

    2. Software Engineer at DataCloud Systems (2021 - 2023)
       - Developed distributed REST APIs handling 10M+ daily active requests using Python and React.
    """.encode('utf-8')

    logger.info("\n>>> Step 1: Ingesting Documents A, B, C, and D...")
    res_a = ingestion_service.process_file(doc_a_content, "HR_Policy_Handbook.txt", workspace_id="default")
    res_b = ingestion_service.process_file(doc_b_content, "Quantum_Helios_Spec.txt", workspace_id="default")
    res_c = ingestion_service.process_file(doc_c_content, "Enterprise_Strategy_2026.txt", workspace_id="default")
    res_d = ingestion_service.process_file(doc_d_content, "AmulaManojSagar_SDE_Resume.txt", workspace_id="default")

    doc_a_id = res_a["document_id"]
    doc_b_id = res_b["document_id"]
    doc_c_id = res_c["document_id"]
    doc_d_id = res_d["document_id"]

    logger.info(f"Ingested Doc A ({res_a['total_chunks']} chunks) ID: {doc_a_id}")
    logger.info(f"Ingested Doc B ({res_b['total_chunks']} chunks) ID: {doc_b_id}")
    logger.info(f"Ingested Doc C ({res_c['total_chunks']} chunks) ID: {doc_c_id}")
    logger.info(f"Ingested Doc D ({res_d['total_chunks']} chunks) ID: {doc_d_id}")

    # -------------------------------------------------------------------
    # EVALUATION 1: Factual Question on Document A
    # -------------------------------------------------------------------
    logger.info("\n>>> EVALUATION 1: Document A Factual Query")
    q1 = "What is the annual health and wellness allowance for employees?"
    req1 = ChatRequest(question=q1, session_id="eval-session-1")
    res1 = chat_service.chat(req1)

    logger.info(f"Question: {q1}")
    logger.info(f"Retrieved Sources ({res1.total_sources}): {[s.filename for s in res1.sources]}")
    logger.info(f"AI Answer:\n{res1.answer}\n")

    assert res1.total_sources > 0, "Evaluation 1 Failed: No sources retrieved!"
    assert "$1,500" in res1.answer or "1,500" in res1.answer or "1500" in res1.answer, "Evaluation 1 Failed: Answer inaccurate!"
    logger.info("PASS: Evaluation 1 - Factual query retrieved correctly and answered accurately.")

    # -------------------------------------------------------------------
    # EVALUATION 2: Topic Query on Document B (Scoped Search)
    # -------------------------------------------------------------------
    logger.info("\n>>> EVALUATION 2: Document B Scoped Query")
    q2 = "What temperature does the dilution refrigerator operate at and who leads the project?"
    req2 = ChatRequest(question=q2, document_id=doc_b_id, session_id="eval-session-2")
    res2 = chat_service.chat(req2)

    logger.info(f"Question: {q2}")
    logger.info(f"Retrieved Sources ({res2.total_sources}): {[s.filename for s in res2.sources]}")
    logger.info(f"AI Answer:\n{res2.answer}\n")

    assert res2.total_sources > 0, "Evaluation 2 Failed: No sources retrieved!"
    assert any(s.document_id == doc_b_id for s in res2.sources), "Evaluation 2 Failed: Source missing from Doc B!"
    assert "15 millikelvin" in res2.answer.lower() or "0.015" in res2.answer or "sterling" in res2.answer.lower(), "Evaluation 2 Failed: Answer inaccurate!"
    logger.info("PASS: Evaluation 2 - Scoped search retrieved correctly from Document B.")

    # -------------------------------------------------------------------
    # EVALUATION 3: Multi-Chunk Synthesis on Document C
    # -------------------------------------------------------------------
    logger.info("\n>>> EVALUATION 3: Document C Multi-Chunk Synthesis Query")
    q3 = "What is the revenue target for FY 2026, and how much money is allocated for data center GPU clusters?"
    req3 = ChatRequest(question=q3, document_id=doc_c_id, session_id="eval-session-3")
    res3 = chat_service.chat(req3)

    logger.info(f"Question: {q3}")
    logger.info(f"Retrieved Sources ({res3.total_sources}): {[s.filename for s in res3.sources]}")
    logger.info(f"AI Answer:\n{res3.answer}\n")

    assert res3.total_sources > 0, "Evaluation 3 Failed: No sources retrieved!"
    assert "120" in res3.answer or "120 million" in res3.answer.lower(), "Evaluation 3 Failed: Missing revenue target!"
    assert "10" in res3.answer or "10 million" in res3.answer.lower(), "Evaluation 3 Failed: Missing GPU cluster budget!"
    logger.info("PASS: Evaluation 3 - Multi-chunk information synthesized correctly.")

    # -------------------------------------------------------------------
    # EVALUATION 4: Typo & Informal Query Resilience Test on Document D
    # -------------------------------------------------------------------
    logger.info("\n>>> EVALUATION 4: Typo & Informal Query Resilience Test (Resume Document D)")
    q4 = "what are the expireance"
    req4 = ChatRequest(question=q4, document_id=doc_d_id, session_id="eval-session-4")
    res4 = chat_service.chat(req4)

    logger.info(f"Question (with typo): '{q4}'")
    logger.info(f"Retrieved Sources ({res4.total_sources}): {[s.filename for s in res4.sources]}")
    logger.info(f"AI Answer:\n{res4.answer}\n")

    assert res4.total_sources > 0, "Evaluation 4 Failed: Hybrid search failed to retrieve resume chunk for typo query!"
    assert any(w in res4.answer.lower() for w in ["techcorp", "datacloud", "software engineer", "4+ years", "experience"]), "Evaluation 4 Failed: AI failed to describe experience!"
    logger.info("PASS: Evaluation 4 - Hybrid BM25+Vector search successfully matched typo query to work experience!")

    logger.info("==================================================================")
    logger.info("    ALL HYBRID RAG EVALUATIONS PASSED SUCCESSFULLY! 100% OK       ")
    logger.info("==================================================================")

if __name__ == "__main__":
    run_rag_evaluations()
