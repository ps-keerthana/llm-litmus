"""
Shared test fixtures for litmus_v2 tests.
Contains minimal synthetic data for unit testing — never used in results/.
"""

from litmus_v2.data.schema import Chunk, CandidateQA, QuestionType


def make_chunk(
    chunk_id: str = "chunk_001",
    doc_name: str = "section_80c_deductions.txt",
    doc_title: str = "Section 80C Deductions",
    text: str = (
        "ELSS mutual funds have a mandatory lock-in period of 3 years. "
        "The maximum aggregate deduction under Section 80C is ₹1,50,000 per financial year. "
        "Tuition fee deductions apply only for two children of the individual."
    ),
    chunk_index: int = 0,
    start_char: int = 0,
    end_char: int = 200,
    token_count: int = 40,
    content_hash: str = "deadbeef" * 8,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_name=doc_name,
        doc_title=doc_title,
        text=text,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
        token_count=token_count,
        content_hash=content_hash,
    )


def make_candidate_qa(
    id: str = "LITMUS_QA_001",
    question: str = "What is the lock-in period for ELSS?",
    reference_answer: str = "3 years",
    question_type: QuestionType = QuestionType.SINGLE_FACT,
    primary_chunk_id: str = "chunk_001",
    supporting_chunk_ids: list = None,
    doc_group: str = "section_80c_deductions.txt",
    generator_model: str = "mistral:7b",
    generator_family: str = "mistral",
    metadata: dict = None,
) -> CandidateQA:
    return CandidateQA(
        id=id,
        question=question,
        reference_answer=reference_answer,
        question_type=question_type,
        primary_chunk_id=primary_chunk_id,
        supporting_chunk_ids=supporting_chunk_ids or ["chunk_001"],
        doc_group=doc_group,
        generator_model=generator_model,
        generator_family=generator_family,
        metadata=metadata or {},
    )


SAMPLE_CHUNKS = [
    make_chunk(
        chunk_id="chunk_001",
        doc_name="section_80c_deductions.txt",
        doc_title="Section 80C Deductions",
        text=(
            "ELSS mutual funds have a mandatory lock-in period of 3 years. "
            "The maximum aggregate deduction under Section 80C is ₹1,50,000. "
            "Tuition fees for two children qualify."
        ),
    ),
    make_chunk(
        chunk_id="chunk_002",
        doc_name="section_80d_health.txt",
        doc_title="Section 80D Health Insurance",
        text=(
            "Preventive health check-up expenses up to ₹5,000 can be claimed. "
            "Senior citizen parents get a limit of ₹50,000. "
            "Cash payments are allowed only for preventive health check-ups."
        ),
        content_hash="cafebabe" * 8,
    ),
    make_chunk(
        chunk_id="chunk_003",
        doc_name="hra_exemption.txt",
        doc_title="HRA Exemption",
        text=(
            "In non-metro cities, 40% of salary is considered for HRA exemption. "
            "The employee must actually incur rental expenditure."
        ),
        content_hash="beefdead" * 8,
    ),
    make_chunk(
        chunk_id="chunk_004",
        doc_name="nps_section_80ccd.txt",
        doc_title="NPS and Section 80CCD",
        text=(
            "Section 80CCD(1B) provides an exclusive additional deduction of ₹50,000 "
            "for NPS Tier-1 contributions, over and above the ₹1,50,000 of Section 80C."
        ),
        content_hash="aabbccdd" * 8,
    ),
]
