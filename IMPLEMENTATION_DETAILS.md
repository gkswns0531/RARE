# RARE Implementation Details (for Reproducibility)

이 문서는 RARE 코드베이스를 기준으로, 논문 **Implementation Details** 섹션에 바로 포함할 수 있도록 **재현성에 직접 영향을 주는 구현 디테일**만을 정리한 문서임. (코드 기준: `RARE/` 디렉토리)

---

## 1. Environment and Dependencies

### 1.1 Python dependencies

패키지 요구사항은 `RARE/requirements.txt`에 정의됨.

- **Core**: `openai`, `numpy`, `tqdm`
- **PDF parsing**: `pdfplumber`
- **Tokenization**: `tiktoken`
- **Chunking**: `langchain-text-splitters` (fallback로 `langchain.text_splitter`)
- **Ranking / Stats**: `scipy`
- **Retrieval evaluation**: `rank-bm25`, `sentence-transformers`, `FlagEmbedding`

### 1.2 Required environment variables

- **`OPENAI_API_KEY`**: Step3/4/5/6/7에서 OpenAI API 호출에 필요
  - `run_complete_pipeline.py`에서 API가 필요한 step이 선택되면 환경변수 존재 여부를 선검사함.

### 1.3 Default model and pricing table

- Default LLM model: `DEFAULT_MODEL = "gpt5_nano"` (`RARE/rare_const.py`)
- Step7 generation default: `DEFAULT_STEP7_GENERATION_MODEL = "gpt5"` (`RARE/rare_const.py`)
- Embedding model: `EMBEDDING_MODEL = "text-embedding-3-large"` (`RARE/rare_const.py`)

OpenAI 모델명은 내부 shorthand를 실제 OpenAI 모델명으로 매핑함 (`RARE/rare_core/rare_llm_client_service.py`, `LLMClient._map_model_name()`).

---

## 2. Reproducibility-Critical Global Settings

### 2.1 Key thresholds and batch sizes

다음 값들이 pipeline 결과(특히 redundancy/질문 생성)에 직접 영향.

- **`DEFAULT_SIMILARITY_THRESHOLD = 0.5`**: Step5 유사도 후보 추출 임계값 기본값 (`RARE/rare_const.py`)
- **Chunking**: `DEFAULT_CHUNK_SIZE = 512`, `DEFAULT_CHUNK_OVERLAP = 0` (`RARE/rare_const.py`)
- **Step5 embedding batch**: `Step5Settings.batch_size = 2048` (`RARE/rare_entities.py`)
- **Step5 similarity batch**: `Step5Settings.similarity_batch_size = 32` (`RARE/rare_entities.py`)
- **Auto-threshold batch**: `Step5Settings.auto_threshold_batch_size = 1024` (`RARE/rare_entities.py`)
- **Step6 max_workers**: `DEFAULT_REDUNDANCY_MAX_WORKERS = 1024` (`RARE/rare_const.py`)
- **Step6 top_k_per_chunk**: `DEFAULT_REDUNDANCY_TOP_K_PER_CHUNK = 3` (`RARE/rare_const.py`)
- **Step6 max_similar_items**: `DEFAULT_REDUNDANCY_MAX_SIMILAR_ITEMS = 512` (`RARE/rare_const.py`)
- **Step7**:
  - `num_information = 2`
  - `num_questions = 10`
  - `num_samples = 10`
  - `input_pool_size = 100`
  - `max_workers = 128`
  (`RARE/rare_const.py`, `RARE/rare_entities.py`)

### 2.2 Determinism notes (randomness + concurrency)

재현성 관점에서 중요한 비결정 요인들.

- **Python `random` 사용**:
  - Step5에서 `--limit` 옵션 사용 시, chunk subset을 `random.sample()`로 샘플링 (`RARE/rare_steps/run_step5_embedding.py`)
  - Step7에서 pool 구성 시 `random.sample()` 및 `random.shuffle()` 사용 (`RARE/rare_core/rare_orchestration_service.py`, `_step7_sample_chunk_diverse_pool()`)
  - 코드 내에서 seed를 고정하지 않으므로, 동일 입력에서도 결과가 달라질 수 있음.
- **ThreadPoolExecutor 기반 병렬 실행**:
  - Step3(atomic extraction), Step4(검증/선별), Step6(LLM redundancy verification), Step7(sample generation) 모두 멀티스레딩을 사용함.
  - 태스크 완료 순서가 실행마다 달라질 수 있고, LLM 호출의 비결정성과 결합되면 최종 산출물이 달라질 수 있음.
- **LLM / Embedding API 자체 비결정성**:
  - 응답 sampling 설정을 강제하지 않으며(temperature 등 명시 없음), API 결과는 동일 프롬프트라도 변할 수 있음.

논문에는 위 항목을 명시하고, 실험 시에는 (i) `--limit` 미사용, (ii) 동일 seed/환경에서 실행, (iii) 중간 산출물 재사용(동일 `outputs/` 고정) 등의 조건을 함께 기록하는 것이 권장됨.

---

## 3. Data Model and Identifier Conventions

### 3.1 Chunk schema (Step2 output)

Chunk는 다음 메타데이터를 포함하는 dict로 저장됨 (`RARE/rare_core/rare_document_processor.py`, `chunk_text()`).

- `file_name`: 원본 PDF 파일명
- `page_no`: 페이지 번호 (1-index)
- `content`: chunk 텍스트
- `sub_text_index`: page 내 chunk 인덱스 (string)
- `token_count`: `tiktoken` 기준 토큰 수

### 3.2 Chunk ID format

Chunk ID는 아래 규칙으로 생성됨 (`RARE/rare_core/rare_orchestration_service.py`, `_generate_doc_id()`).

```
{base_file_name}_page{page_no:03d}_chunk{sub_text_index:03d}
```

- `base_file_name`: `file_name`에서 `.pdf` 제거
- `page_no`: 3자리 zero padding
- `sub_text_index`: `sub_text_index`가 `"1-1"` 같은 string이면 첫 토큰을 int로 파싱 후 3자리 zero padding

Step7의 chunk mapping 생성도 동일한 규칙을 사용함 (`RARE/rare_core/rare_orchestration_service.py`, `_convert_chunks_to_dict()` 및 `RARE/rare_steps/run_step7_multihop_questions.py`, `_load_chunk_data()`).

### 3.3 Atomic information ID format

Atomic info ID는 chunk 단위로 생성되며 아래 규칙을 따름 (`RARE/rare_core/rare_orchestration_service.py`, `_extract_atomic_info_single()`).

```
{chunk_id}_atomic_{i:03d}
```

---

## 4. Pipeline Entry Points and Output Artifacts

### 4.1 End-to-end runner

- CLI entry: `RARE/run_complete_pipeline.py`
  - 내부적으로 `rare_core.rare_orchestration_service.run_rare_pipeline()` 호출
  - step 선택: `--steps parsing chunking atomic_info_extraction atomic_info_selection embedding_similarity redundancy_detection data_generation`

### 4.2 Output directory conventions (important)

두 가지 실행 방식이 섞여 있으므로, 논문에는 어떤 방식을 사용했는지 명확히 적는 것이 중요함.

- **Orchestrator(flat outputs)**: `run_complete_pipeline.py` → `RareOrchestrator.output_dir`에 다음 파일명으로 저장
  - `step1_parsed_texts.json`
  - `step2_chunks.json`
  - `step3_atomic_info_map.json`
  - `step4_selected_atomic_info.json`
  - `step4_embeddings.pkl`
  - `step4_embedding_data.json`
  - `precomputed_similarities.pkl`
  - `step6_redundancy_mapping.json`
  - `multihop_questions_dataset_num{num_information}.json` 등
- **Step scripts(nested outputs)**: `RARE/rare_steps/run_step*.py`는 `outputs/step{N}/...` 형태를 기본으로 사용 (`RARE/rare_entities.py`)

실험 재현을 위해서는 **단일 모드만 사용**하는 것을 권장하며, 본 문서는 orchestrator 기준 로직을 우선으로 설명함.

---

## 5. Step-by-Step Implementation Details

## 5.1 Step 1: PDF Parsing

### 5.1.1 Library and extraction method

- PDF parsing: `pdfplumber` 사용 (`RARE/rare_core/rare_document_processor.py`)
- Page loop: `page.extract_text()` 결과가 non-empty인 페이지 텍스트만 저장
- Output: dict 형태 `{page_num(int): page_text(str)}`

### 5.1.2 Orchestrator output

`RareOrchestrator.step1_document_parsing()`은 PDF 파일(단일/폴더)을 처리한 뒤,
`{pdf_filename: {page_no: text}}` 구조로 저장함 (`step1_parsed_texts.json`).

## 5.2 Step 2: Text Chunking

### 5.2.1 Token-aware splitter configuration

Chunking은 `RecursiveCharacterTextSplitter` 기반이며, 길이 함수는 **토큰 수**로 정의됨.

- Tokenizer: `tiktoken.get_encoding("cl100k_base")`
- Length function: `len(tokenizer.encode(text))`
- Separators: 문단/리스트/문장부호/공백 순의 계층적 separator를 사용 (`_get_korean_aware_separators()`)
- `keep_separator=True`, `strip_whitespace=True`

즉, “문자 수 기반”이 아니라 **cl100k_base 토큰 수 기반**으로 `chunk_size`를 맞추는 구현임.

### 5.2.2 Orchestrator output

`RareOrchestrator.step2_document_chunking()`은 모든 페이지 텍스트에 대해 chunk list를 생성 후 `step2_chunks.json`으로 저장함.

## 5.3 Step 3: Atomic Information Extraction

### 5.3.1 Prompt and expected JSON schema

Atomic extraction prompt template: `RARE/rare_core/rare_prompts_service.py`, `EXTRACT_ATOMIC_INFO_PROMPT`.

입력:
- `doc_title`: 기본적으로 chunk의 `file_name`
- `doc_content`: chunk 텍스트
- `language`: 기본 `"English"`

출력(LLM에게 요구하는 포맷):

```json
{"atomic_information": [
  {"reasoning": "...", "content": "..."},
  ...
]}
```

### 5.3.2 Parsing and filtering rules

`_extract_atomic_info_single()`에서 다음 규칙으로 atomic unit을 생성함.

- `clean_and_parse_json()`로 JSON 추출 (`RARE/rare_core/rare_json_parser_service.py`)
- `parsed_response`가 dict이고 `"atomic_information"` 키가 list일 때만 처리
- 각 item에서 `"content"`가 있고, `len(content.strip()) >= 10`이면 valid로 채택
- atomic_info_id는 `{chunk_id}_atomic_{i:03d}`

### 5.3.3 Concurrency

Step3는 `ThreadPoolExecutor(max_workers=...)`로 chunk 단위 병렬 추출을 수행함 (`RareOrchestrator.step3_atomic_info_extraction()`).

## 5.4 Step 4: Best Information Selection (Quality Scoring + Ranking)

### 5.4.1 Optional atomic completeness filtering (pre-filter)

Step4는 선택적으로 atomic info completeness 필터를 수행함.

- Prompt: `FILTER_ATOMIC_INFORMATION_COMPLETENESS_PROMPT` (`RARE/rare_core/rare_prompts_service.py`)
- 입력: chunk 별 atomic info list를 batch로 전달
- 출력: list 형태, 각 atomic_id에 대해 `has_information_completeness_error: bool`

이 단계는 `enable_logical_filtering=True`일 때 실행됨 (`RareOrchestrator.step4_best_info_selection()`).

### 5.4.2 Five separate scoring calls per chunk

`_validate_separate_chunk_atomic_info()`는 chunk 내 atomic info list에 대해 **5개의 독립 호출**로 점수를 부여함.

- Validity: `VERIFY_ATOMIC_INFO_VALIDITY_SEPARATE_PROMPT`
- Completeness: `VERIFY_ATOMIC_INFO_COMPLETENESS_SEPARATE_PROMPT`
- Specificity: `VERIFY_ATOMIC_INFO_SPECIFICITY_SEPARATE_PROMPT`
- Clarity: `VERIFY_ATOMIC_INFO_CLARITY_SEPARATE_PROMPT`
- Questionability: `VERIFY_ATOMIC_INFO_QUESTIONABILITY_SEPARATE_PROMPT`

각 응답은 `LLMClient.call_api_with_score_validation()`으로 “아이템 수 일치 + score field 존재”를 검증하며 실패 시 재시도함.

최종 `overall_confidence`는 5개 score의 산술평균으로 계산됨.

### 5.4.3 Chunk-based ranking method

Step4는 chunk별 ranking을 생성하며, 기본 구현은 “rank-average + RRF 형태”임.

- `generate_chunk_based_rankings_by_rank_average(..., rrf_k=0)` (`RARE/rare_core/rare_ranking_utils.py`)
  - 각 dimension별 score로 정렬 → rank(position) 산출
  - RRF score: `1 / (position + rrf_k)`
  - 5개 dimension의 rank-score 평균으로 최종 ranking

### 5.4.4 Output structure

`step4_selected_atomic_info.json`에는 dual mode가 포함됨.

- `comparison_modes.threshold_filtered.atomic_info_by_chunk`
- `comparison_modes.all_items.atomic_info_by_chunk`
- `comparison_modes.*.chunk_rankings`
- `validation_ranking_by_chunk` (간단한 chunk별 content ranking)

## 5.5 Step 5: Embedding Generation + Similarity Precomputation

### 5.5.1 Embedding implementation

Embedding client: `RARE/rare_core/rare_search_client_service.py`, `SearchClient`.

- OpenAI embeddings API 사용 (`client.embeddings.create`)
- model: `text-embedding-3-large`
- bulk embedding: 리스트 입력을 `batch_size` 단위로 잘라 호출
- 비용은 `response.usage.total_tokens`에 기반해 token-price로 누적 추적

### 5.5.2 Similarity computation

Step5는 embeddings를 `torch` 텐서로 변환 후 **L2 normalize**한 뒤 dot product로 cosine similarity를 계산함.

- normalize: `torch.nn.functional.normalize(embeddings, p=2, dim=1)`
- similarity: `torch.mm(batch_embeddings, embeddings_tensor.T)`

유사도 후보는 아래 조건을 만족하는 모든 pair를 포함함.

- `score >= similarity_threshold`
- self-match 제외
- **same-chunk pair 제거** (`similar_item["chunk_id"] != atomic_info.chunk_id`)

### 5.5.3 Auto threshold

`similarity_threshold="auto"`인 경우, 전체 embedding pair의 평균 similarity를 threshold로 사용함.

- 구현: `_calculate_auto_threshold()`
- 계산 방식: 배치별로 전체 similarity matrix를 구성하며, diagonal/self 및 중복 pair는 제외 후 평균

### 5.5.4 Persisted artifacts

- `precomputed_similarities.pkl`: `atomic_info_id -> {target_content, target_chunk_id, valid_items[]}`
- `step4_embeddings.pkl`: `{embeddings, search_documents, all_atomic_info}`
- `step4_embedding_data.json`: embeddings shape, threshold, total atomic count

## 5.6 Step 6: Redundancy Detection (Embedding Filter + LLM Verification)

### 5.6.1 Candidate generation via Step5 similarities

Step6는 Step5에서 미리 만든 `precomputed_similarities.pkl`만을 사용함.

- target set: `similarity_data.keys()`에 존재하는 atomic_info_id
- 각 target에 대해 `valid_items` (threshold 이상, same-chunk 제외)만을 비교 대상으로 사용

### 5.6.2 Optional top-k per chunk limitation

`top_k_per_chunk`가 주어지면, chunk별로 상위 k개의 atomic info만 redundancy target으로 유지하려고 시도함.

- 우선 Step4 산출물을 탐색해서 ranking 기반 top-k를 사용
- Step4 파일이 없거나 실패하면 `overall_confidence`(또는 fallback으로 `validity_score`) 기반 정렬로 top-k 선택

### 5.6.3 LLM-based semantic redundancy verification (batch, 1-call)

각 target atomic info는 비교 후보들을 한 번에 묶어 LLM 호출 1회로 검증함.

- Prompt: `DETECT_SEMANTIC_REDUNDANCY_PROMPT`
- 비교 후보 formatting:
  - `"1: {content}"`, `"2: {content}"`, ... 형태의 line list
- 응답 parsing:
  - list를 기대하며, i번째 결과의 `is_redundant==true`이면 i번째 comparison item의 `atomic_info_id`를 redundancy로 기록

출력:
- `RedundancyMapping.redundant_items`: redundant id list, 없으면 `["unique"]`
- `RedundancyMapping.similarity_scores`: 후보 id -> similarity score

## 5.7 Step 7: Multi-hop Question Generation (LLM Selection Pipeline)

Step7에는 `legacy_mode`가 있으나, 본 문서에서는 기본 경로(LLM selection pipeline)를 기술함.

### 5.7.1 Diverse pool construction (redundancy-aware)

`_step7_prepare_diverse_pool()`은 redundancy group의 대표를 선정하여 pool의 다양성을 확보함.

- 각 item에 대해, 자기 chunk + redundant_items의 chunk들을 합쳐 `redundancy_count` 계산
- redundancy group key: `(redundant_items + atomic_info_id)`를 정렬한 tuple
- group별 대표: `redundancy_count` 최대인 item 1개
- `diverse_items = unique_items + representatives`, 이후 `redundancy_count` 내림차순 정렬

### 5.7.2 Pool sampling (chunk-level diversification)

`_step7_sample_chunk_diverse_pool()`은 chunk별 대표를 먼저 뽑고 부족분을 random으로 보충함.

- chunk별 대표: chunk 내 `redundancy_count`가 가장 큰 item
- 대표 수가 pool_size 이상이면 `random.sample(representatives, pool_size)`
- 부족하면 remaining에서 `random.sample`로 보충 후 `random.shuffle`

즉, Step7 결과는 seed 미고정 시 실행마다 달라질 수 있음.

### 5.7.3 LLM selection + question generation

Prompt: `GENERATE_MULTIHOP_QUESTION_WITH_LLM_SELECTION_PROMPT`.

- 입력: pool item을 `"{idx}. {title}: {content}"` 형태로 제공
- LLM이 **(i) num_information개 선택**, **(ii) num_questions개 질문 생성**을 동시에 수행
- 후보 채택 규칙:
  - `selected_items`로 지정한 item 수가 정확히 `num_information`일 때만 candidate로 유지

### 5.7.4 Logical consistency filtering (4 separate calls, all must pass)

`filter_questions_logical_consistency()`는 candidate 질문들에 대해 4개 필터를 각각 LLM 호출로 수행함.

- Contextual independence: `FILTER_CONTEXTUAL_INDEPENDENCE_PROMPT`
- Answer exclusion: `FILTER_ANSWER_EXCLUSION_PROMPT`
- Information equivalence: `FILTER_INFORMATION_EQUIVALENCE_PROMPT`
- Question ambiguity: `FILTER_QUESTION_AMBIGUITY_PROMPT`

각 질문은 **4개 모두 pass**해야 다음 단계로 진행됨.

### 5.7.5 Answerability check

`check_question_answerability()`는 질문별로 필요한 chunk만 모아 answerability를 판단함.

- Prompt: `FILTER_ANSWERABILITY_PROMPT`
- chunk formatting: “Title i / Chunk i” 형태로 모든 unique chunk를 나열 + “Question-Chunk Mapping” 섹션 추가
- 반환: 각 질문에 대해 `answerability_check` 및 `generated_answer` 포함

### 5.7.6 Quality validation (4 separate calls) + RRF selection

`validate_multihop_questions_batch_separate()`는 answerable 후보들에 대해 4개 dimension을 분리 호출로 채점함.

- Connectivity / Fluency / Essentiality / Validity
- 각 dimension의 점수를 결합한 뒤, 후보들에 대해 dimension rank를 만들고
  - `rrf_score = Σ 1 / rank_dim`
  로 best question을 선택함.

### 5.7.7 Gold chunk groups (evaluation target) construction

선택된 atomic info마다, 해당 atomic info chunk + redundant item들의 chunk를 묶어 group으로 저장함.

- 구현: `_step7_calculate_gold_chunks()`
- output: `gold_chunks: List[List[str]]` (각 group은 chunk_id list)

### 5.7.8 Persisted artifacts

LLM selection pipeline이 성공적으로 샘플을 생성하면 다음 파일을 저장함 (`_step7_persist_results()`).

- `multihop_questions_dataset_num{num_information}.json`
  - `metadata`: sample 수, timestamp, format
  - `samples`: `_Step7QuestionSample`의 dict 리스트
  - `chunk_mapping`: 샘플에 등장한 chunk_id → {source_title, content, page_number}
- `extended_validated_samples.json` (extended_samples가 존재할 때)
- `questions_failed_num{num_information}.json` (logical/answerability 실패 로그)

---

## 6. LLM Call Handling and JSON Parsing

### 6.1 Retry policy

`LLMClient.call_api()`는 실패 시 최대 3회 재시도하며, backoff는 0.5s, 1s, 2s (`(2**attempt)*0.5`).

### 6.2 JSON parsing and validation

`clean_and_parse_json()`는 다음 순서로 JSON을 복구함 (`RARE/rare_core/rare_json_parser_service.py`).

- ```json``` code block 추출 시도
- direct `json.loads()` 시도 (list/dict 모두 허용)
- trailing comma 제거, True/False/None → JSON 표준 변환
- 실패 시 regex 기반 JSON object 추출 시도

Step4/Step7의 일부 경로는 `call_api_with_score_validation()`을 사용하여 “필수 score field 존재”를 강제함.

---

## 7. Retrieval Evaluation (Group-based Metrics)

### 7.1 Dataset format

Evaluation dataset은 `RARE/dataset/*.json`에 저장됨.

- Top-level keys:
  - `metadata`
  - `queries`: 각 query는 `question`, `gold_chunk_ids`, `gold_chunk_groups` 포함
  - `corpus`: `chunk_id -> {"content": ... , ...}` 형태

### 7.2 Metrics definition

`RARE/evaluation/metrics.py`는 group-based metric을 제공함.

- `coverage@k`: top-k 결과가 **gold group**들을 얼마나 커버하는지 (group 단위)
- `PerfRecall@k` (JSON field: `perfect_match@k`): 모든 group이 커버되면 1.0, 아니면 0.0
- `ndcg@k`: group 단위 DCG/IDCG
- `mrr`: group별 reciprocal rank 평균

### 7.3 Retrieval models and caching

`RARE/evaluation/models.py`에서 모델별 검색을 수행하며, embedding 모델은 cache를 사용함.

- BM25: `rank_bm25.BM25Okapi` (tokenization은 `text.split()`)
- OpenAI embeddings 기반 dense retrieval: corpus/query embedding을 cache에 저장 (`evaluation/cache/`)
- HuggingFace/FlagEmbedding 기반 모델 지원 (자세한 모델 리스트는 `evaluation/README.md`)

---

## 8. Checklist for Writing the Paper (Implementation Detail)

논문에 반드시 명시할 것을 권장하는 체크리스트.

- **Chunking**: `cl100k_base` 토크나이저 기반 토큰 길이로 `RecursiveCharacterTextSplitter` 수행, `chunk_size=512`, `overlap=0`.
- **Identifiers**: `chunk_id` 및 `atomic_info_id` 생성 규칙(정확한 zero padding 포함).
- **Atomic extraction filter**: atomic content 길이 최소 10자(`len >= 10`) 필터.
- **Quality selection**: Step4에서 5개 score를 separate call로 산출하고 평균을 overall_confidence로 사용, chunk별 rank-average(RRF)로 정렬.
- **Embedding**: `text-embedding-3-large` 사용, bulk batch size, L2 normalize 후 dot-product cosine similarity.
- **Similarity threshold**: fixed(0.5) 또는 auto(mean similarity of all pairs) 사용 여부.
- **Redundancy**: Step5의 similarity 후보를 LLM으로 batch verification(1-call)하여 redundancy list를 구성.
- **Step7**: pool size(100), num_information(2), num_questions(10), logical filter 4종 + answerability + 4-dim quality validation + RRF selection.
- **Non-determinism**: random sampling, concurrency, API stochasticity를 명시하고 실행 조건을 함께 기록.


