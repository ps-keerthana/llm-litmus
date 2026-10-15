"""
Data schemas for litmus_v2.
Pydantic models for chunks, Q/A candidates, human labels, and manifests.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    SINGLE_FACT = "single_fact"
    NUMERIC = "numeric"
    YES_NO = "yes_no"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


class GradeLabel(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    ABSTAIN = "abstain"
    ERROR = "error"


class GoldenDatasetItem(BaseModel):
    query_id: str
    question: str
    ground_truth_answer: str
    question_type: Union[QuestionType, str] = QuestionType.SINGLE_FACT
    context: Optional[str] = None
    reference_correct: bool = True
    supported_by_chunk: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_name: str
    doc_title: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    content_hash: str


class CandidateQA(BaseModel):
    id: str
    question: str
    reference_answer: str
    question_type: QuestionType
    primary_chunk_id: Optional[str] = None  # None for unanswerable
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    doc_group: str  # Document name used for leak-free group splitting
    generator_model: str
    generator_family: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HumanLabel(BaseModel):
    id: str
    question: str
    reference_answer: str
    question_type: str
    primary_chunk_id: Optional[str] = None
    source_chunk_text: Optional[str] = None
    reference_correct: Optional[bool] = None      # True / False (unfilled initially)
    supported_by_chunk: Optional[bool] = None     # True / False (unfilled initially)
    edited_reference: Optional[str] = None        # Human edit if needed
    notes: Optional[str] = None                   # Human audit comments
    labeled_by: Optional[str] = None
    labeled_at: Optional[str] = None


class DatasetManifest(BaseModel):
    dataset_name: str
    version: str
    dataset_hash: str
    total_items: int
    type_distribution: Dict[str, int]
    dev_count: int
    test_count: int
    dev_hash: str
    test_hash: str
    generator_model: str
    generator_family: str
    answering_family: str
    judge_family: str
    created_at: str
