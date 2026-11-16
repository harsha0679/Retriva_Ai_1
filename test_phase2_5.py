import sys
import logging
from app.services.ingestion import KnowledgeIngestionService
from app.services.chat import ChatService
from app.services.document import DocumentService
from app.domain.models import ChatRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_tests():
    logger.info("=== STARTING PHASE 2.5 VERIFICATION TESTS ===")
    
    ingestion_service = KnowledgeIngestionService()
    chat_service = ChatService()
    document_service = DocumentService()

    # ----------------------------------------------------
    # TEST 1: Multiple Document Ingestion
    # ----------------------------------------------------
    logger.info("\n--- TEST 1: Ingesting Multiple Documents ---")
    doc_a_content = b"Document A: Company Alpha's policy states that working hours are 9 AM to 5 PM Monday to Friday."
    doc_b_content = b"Document B: Project Beta is a confidential quantum computing project led by Dr. Elizabeth Vance."
    doc_c_content = b"Document C: Server Gamma deployment instructions require running kubectl apply -f deployment.yaml."

    res_a = ingestion_service.process_file(doc_a_content, "doc_alpha.txt", workspace_id="engineering", version=1)
    res_b = ingestion_service.process_file(doc_b_content, "doc_beta.txt", workspace_id="engineering", version=1)
    res_c = ingestion_service.process_file(doc_c_content, "doc_gamma.txt", workspace_id="hr", version=1)

    doc_a_id = res_a["document_id"]
    doc_b_id = res_b["document_id"]
    doc_c_id = res_c["document_id"]

    logger.info(f"Ingested Doc A: {doc_a_id}")
    logger.info(f"Ingested Doc B: {doc_b_id}")
    logger.info(f"Ingested Doc C: {doc_c_id}")

    docs_list = document_service.list_documents()
    found_ids = {d.document_id for d in docs_list}
    assert doc_a_id in found_ids, "Doc A missing in list"
    assert doc_b_id in found_ids, "Doc B missing in list"
    assert doc_c_id in found_ids, "Doc C missing in list"
    logger.info("PASS: Test 1 - All 3 documents stored and listed.")

    # ----------------------------------------------------
    # TEST 2: Document-Specific Search
    # ----------------------------------------------------
    logger.info("\n--- TEST 2: Document-Specific Search ---")
    req_b = ChatRequest(
        question="Who leads Project Beta?",
        document_id=doc_b_id,
        session_id="test-session-b"
    )
    res_b_chat = chat_service.chat(req_b)
    logger.info(f"Answer for Doc B question: {res_b_chat.answer}")
    logger.info(f"Sources count: {res_b_chat.total_sources}")
    
    assert res_b_chat.total_sources > 0, "No sources retrieved for Doc B"
    for src in res_b_chat.sources:
        assert src.document_id == doc_b_id, f"Cross-document leak detected! Found source doc_id={src.document_id}, expected={doc_b_id}"
    logger.info("PASS: Test 2 - 100% of sources strictly belong to Document B.")

    # ----------------------------------------------------
    # TEST 3: Global Search
    # ----------------------------------------------------
    logger.info("\n--- TEST 3: Global Search ---")
    req_global = ChatRequest(
        question="What are the working hours for Company Alpha?",
        session_id="test-session-global"
    )
    res_global_chat = chat_service.chat(req_global)
    logger.info(f"Global Answer: {res_global_chat.answer}")
    assert res_global_chat.total_sources > 0, "Global search failed to retrieve chunks"
    logger.info("PASS: Test 3 - Global search retrieved across documents.")

    # ----------------------------------------------------
    # TEST 4: Low Relevance / Similarity Threshold Filter
    # ----------------------------------------------------
    logger.info("\n--- TEST 4: Similarity Threshold Filtering ---")
    req_unrelated = ChatRequest(
        question="What is the recipe for baking chocolate sourdough bread with strawberries?",
        session_id="test-session-unrelated"
    )
    res_unrelated = chat_service.chat(req_unrelated)
    logger.info(f"Unrelated Question Answer: {res_unrelated.answer}")
    logger.info(f"Sources count: {res_unrelated.total_sources}")
    assert res_unrelated.total_sources == 0, "Low relevance chunks were not filtered out!"
    assert "couldn't find enough relevant information" in res_unrelated.answer.lower(), "Safe fallback message not returned"
    logger.info("PASS: Test 4 - Low similarity chunks filtered out, safe fallback returned.")

    # ----------------------------------------------------
    # TEST 5: Single Document Deletion
    # ----------------------------------------------------
    logger.info("\n--- TEST 5: Single Document Deletion ---")
    del_res = document_service.delete_document(doc_b_id)
    logger.info(f"Deleted Doc B: {del_res.deleted_chunks} chunks removed.")

    updated_docs = document_service.list_documents()
    remaining_ids = {d.document_id for d in updated_docs}
    assert doc_a_id in remaining_ids, "Doc A was accidentally deleted!"
    assert doc_c_id in remaining_ids, "Doc C was accidentally deleted!"
    assert doc_b_id not in remaining_ids, "Doc B was not deleted!"
    logger.info("PASS: Test 5 - Document B deleted without affecting Document A or Document C.")

    logger.info("\n=== ALL PHASE 2.5 TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
