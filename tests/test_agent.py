import json
import tempfile
import unittest
from pathlib import Path

from src.agent import KnowledgeQAAgent
from src.advanced_agent import AdvancedTools, EnterpriseKnowledgeAgent
from src.document_service import DocumentUploadError, DocumentUploadService
from src.ingestion import ParsedPage, StructureAwareChunker
from src.retrieval import HybridRetriever
from src.tools import KnowledgeBase


class AgentAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.agent = KnowledgeQAAgent.from_default_data()

    def test_normal_answer_has_supported_citation(self):
        result = self.agent.run("密码忘了怎么办")
        self.assertEqual(result.status, "completed")
        self.assertIn("绑定手机", result.answer)
        self.assertEqual(result.citations[0]["doc_id"], "D04")
        self.assertTrue(any(event.next_action == "read" for event in result.trace))

    def test_conflict_does_not_choose_one_value(self):
        result = self.agent.run("报销期限多久")
        self.assertEqual(result.status, "needs_confirmation")
        self.assertIn("冲突", result.answer)
        self.assertIn("D01", {item["doc_id"] for item in result.citations})
        self.assertIn("D02", {item["doc_id"] for item in result.citations})
        self.assertIn("D01", result.answer)
        self.assertIn("30天", result.answer)
        self.assertIn("D02", result.answer)
        self.assertIn("15天", result.answer)
        self.assertTrue(all(item["content"] for item in result.citations))
        self.assertNotIn("是15天", result.answer)

    def test_out_of_scope_reports_gap(self):
        result = self.agent.run("公司附近地铁几点停运")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertTrue(result.gaps)
        self.assertEqual(result.citations, [])

    def test_paraphrase_retrieves_same_policy(self):
        result = self.agent.run("费用报账要在多长时间内提交")
        self.assertEqual(result.status, "needs_confirmation")
        self.assertIn("适用范围", result.answer)

    def test_context_follow_up_changes_retrieval(self):
        history = [
            {"role": "user", "content": "普通员工国内出差餐补标准是多少？"},
            {"role": "assistant", "content": "普通员工是100元/天。"},
        ]
        result = self.agent.run("那研发员工呢？", history=history)
        self.assertEqual(result.status, "completed")
        self.assertIn("150元/天", result.answer)
        self.assertEqual(result.citations[0]["doc_id"], "D05")

    def test_tool_failure_is_visible_and_recovered(self):
        agent = KnowledgeQAAgent.from_default_data(fail_search_once=True)
        result = agent.run("密码忘了怎么办")
        self.assertEqual(result.status, "completed")
        self.assertTrue(any("暂时不可用" in event.result_summary for event in result.trace))

    def test_read_failure_is_visible_and_recovered(self):
        agent = KnowledgeQAAgent.from_default_data(fail_read_once=True)
        result = agent.run("密码忘了怎么办")
        self.assertEqual(result.status, "completed")
        self.assertTrue(any("读取服务暂时不可用" in event.result_summary for event in result.trace))

    def test_step_limit_returns_incomplete(self):
        result = KnowledgeQAAgent.from_default_data(max_steps=1).run("密码忘了怎么办")
        self.assertEqual(result.status, "incomplete")
        self.assertIn("上限", result.gaps[0])

    def test_progress_reports_actual_agent_stages(self):
        events = []
        result = self.agent.run("密码忘了怎么办", progress_callback=events.append)
        self.assertEqual(result.status, "completed")
        stages = [event["stage"] for event in events]
        self.assertIn("analyze", stages)
        self.assertIn("search", stages)
        self.assertIn("read", stages)
        self.assertIn("generate", stages)
        self.assertLess(stages.index("search"), stages.index("read"))
        self.assertTrue(any("D04" in event["message"] for event in events if event["stage"] == "read"))


class ToolTests(unittest.TestCase):
    def test_document_data_is_synthetic_and_loadable(self):
        path = Path(__file__).parents[1] / "data" / "documents.json"
        docs = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(docs), 5)
        self.assertTrue(all(item["doc_id"].startswith("D") for item in docs))
        self.assertEqual(len(KnowledgeBase(path).documents), 5)

    def test_example_outputs_keep_verifiable_citation_content(self):
        root = Path(__file__).parents[1] / "examples"
        for name in ("01-normal.json", "02-conflict.json", "04-paraphrase.json", "05-context.json", "06-tool-failure.json", "07-advanced-product.json"):
            payload = json.loads((root / name).read_text(encoding="utf-8"))
            citations = payload["actual_output"]["citations"]
            self.assertTrue(citations, name)
            self.assertTrue(all(item.get("content") for item in citations), name)
        upload_payload = json.loads((root / "08-upload-and-ask.json").read_text(encoding="utf-8"))
        upload_citations = upload_payload["ask"]["actual_output"]["citations"]
        self.assertTrue(all(item.get("content") for item in upload_citations))

    def test_structure_aware_chunking_keeps_parent_page_and_image(self):
        chunks = StructureAwareChunker(max_chars=80).chunk(
            [ParsedPage(3, "2. 开机\n\n长按电源键3秒。\n\n见下图。", "2. 开机", [{"bbox": [1, 2, 3, 4]}], ["images/p3.png"])],
            "M-TEST", "测试手册", "2026-01-01", "2026-01-02", "测试型号",
        )
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].page_start, 3)
        self.assertEqual(chunks[0].parent_chunk_id, "M-TEST-P003")

    def test_hybrid_retrieval_returns_routes_and_metadata(self):
        root = Path(__file__).parents[1]
        result = HybridRetriever(root / "data" / "chunks.json").retrieve("Pro X1 怎么开机", product_id="P100-X1")
        self.assertTrue(result.candidates)
        self.assertIn("RRF(k=60)", result.strategy["fusion"])
        self.assertEqual(result.strategy["routes"], ["dense", "sparse", "hyde"])
        self.assertEqual(
            [event["action"] for event in result.pipeline_trace],
            ["dense_retrieve", "sparse_retrieve", "hyde_generate", "hyde_retrieve", "rrf_fusion", "rerank", "conflict_check"],
        )
        self.assertTrue(result.candidates[0]["retrieval_routes"])
        self.assertIn("page_start", result.candidates[0])

    def test_hyde_uses_current_decision_model_without_mcp_route(self):
        class CurrentModel:
            source = "current_test_model"

            def __init__(self):
                self.queries = []

            def expand_query(self, query):
                self.queries.append(query)
                return "Pro X1 开机时需要核对电源键位置和按压秒数"

        root = Path(__file__).parents[1]
        current_model = CurrentModel()
        progress = []
        tools = AdvancedTools(
            HybridRetriever(root / "data" / "chunks.json"),
            query_expander=current_model,
            progress_callback=progress.append,
        )
        tools.set_scope(product_id="P100-X1")
        observation = tools.search("Pro X1 怎么开机")
        strategy = observation.data["retrieval"]["strategy"]
        self.assertEqual(current_model.queries, ["Pro X1 怎么开机"])
        self.assertEqual(strategy["query_expansion"], "current_test_model")
        self.assertEqual(strategy["routes"], ["dense", "sparse", "hyde"])
        self.assertIn("路线=dense+sparse+hyde", observation.summary)
        actions = [event["action"] for event in observation.data["pipeline_trace"]]
        self.assertEqual(
            actions,
            ["dense_retrieve", "sparse_retrieve", "hyde_generate", "hyde_retrieve", "rrf_fusion", "rerank", "conflict_check"],
        )
        stages = [event["stage"] for event in progress]
        self.assertLess(stages.index("dense_retrieve"), stages.index("hyde_generate"))
        self.assertLess(stages.index("hyde_generate"), stages.index("hyde_retrieve"))
        self.assertFalse(any("mcp" in str(value).lower() for value in (strategy, observation.summary, actions)))

        class FailingModel(CurrentModel):
            source = "failing_current_model"

            def expand_query(self, query):
                raise TimeoutError("model timeout")

        fallback_tools = AdvancedTools(HybridRetriever(root / "data" / "chunks.json"), query_expander=FailingModel())
        fallback_tools.set_scope(product_id="P100-X1")
        fallback = fallback_tools.search("Pro X1 怎么开机")
        self.assertEqual(fallback.data["retrieval"]["strategy"]["query_expansion"], "failing_current_model:offline_fallback")
        hyde_event = next(event for event in fallback.data["pipeline_trace"] if event["action"] == "hyde_generate")
        self.assertEqual(hyde_event["status"], "degraded")
        self.assertIn("失败", hyde_event["result_summary"])

    def test_advanced_trace_keeps_dense_and_hyde_as_separate_actions(self):
        result = EnterpriseKnowledgeAgent().run("Pro X1 怎么开机")
        actions = [event.action for event in result.trace]
        self.assertIn("dense_retrieve", actions)
        self.assertIn("hyde_generate", actions)
        self.assertIn("hyde_retrieve", actions)
        self.assertLess(actions.index("dense_retrieve"), actions.index("hyde_generate"))
        self.assertLess(actions.index("hyde_generate"), actions.index("hyde_retrieve"))
        self.assertTrue(all(event.duration_ms is not None for event in result.trace if event.kind == "retrieval_stage"))

    def test_advanced_pipeline_resolves_product_before_retrieval(self):
        agent = EnterpriseKnowledgeAgent()
        resolved = agent.run("Pro X1 怎么开机")
        self.assertEqual(resolved.status, "completed")
        self.assertIn("3秒", resolved.answer)
        self.assertEqual(resolved.citations[0]["page_start"], 3)
        self.assertTrue(resolved.citations[0]["chunk_id"].startswith("M-P100-X1"))

        ambiguous = agent.run("Pro 怎么开机")
        self.assertEqual(ambiguous.status, "needs_confirmation")
        self.assertFalse(any(event.tool for event in ambiguous.trace))
        self.assertIn("Pro X1", ambiguous.answer)

        unresolved = agent.run("型号 Z9 怎么开机")
        self.assertEqual(unresolved.status, "needs_confirmation")
        self.assertEqual(unresolved.citations, [])
        self.assertFalse(any(event.tool for event in unresolved.trace))

    def test_upload_text_then_ask_only_selected_document(self):
        with tempfile.TemporaryDirectory() as runtime:
            agent = EnterpriseKnowledgeAgent()
            service = DocumentUploadService(agent.retriever, runtime_root=runtime)
            first = service.ingest_bytes(
                "保修政策.md",
                "# 产品保修政策\n\n本设备保修期为24个月，自激活之日起计算。".encode("utf-8"),
                "text/markdown",
            )
            second = service.ingest_bytes(
                "旧版政策.txt",
                "旧版设备保修期为6个月。".encode("utf-8"),
                "text/plain",
            )
            self.assertEqual(first.status, "ready")
            self.assertGreater(first.chunk_count, 0)

            result = agent.run("保修期是多久？", document_ids=[first.document_id])
            self.assertEqual(result.status, "completed")
            self.assertIn("24个月", result.answer)
            self.assertNotIn("6个月", result.answer)
            self.assertEqual(result.citations[0]["source_doc_id"], first.document_id)

            other = agent.run("保修期是多久？", document_ids=[second.document_id])
            self.assertIn("6个月", other.answer)

            restored_agent = EnterpriseKnowledgeAgent()
            restored_service = DocumentUploadService(restored_agent.retriever, runtime_root=runtime)
            restored_service.require_ready([first.document_id])
            restored = restored_agent.run("保修期是多久？", document_ids=[first.document_id])
            self.assertIn("24个月", restored.answer)

    def test_upload_rejects_unsupported_and_oversized_files(self):
        with tempfile.TemporaryDirectory() as runtime:
            agent = EnterpriseKnowledgeAgent()
            service = DocumentUploadService(agent.retriever, runtime_root=runtime, max_upload_bytes=8)
            with self.assertRaises(DocumentUploadError) as unsupported:
                service.ingest_bytes("payload.exe", b"123")
            self.assertEqual(unsupported.exception.status_code, 415)
            with self.assertRaises(DocumentUploadError) as oversized:
                service.ingest_bytes("notes.txt", b"123456789")
            self.assertEqual(oversized.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
