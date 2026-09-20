# Failure Analysis and Iterative System Improvements

This document details three significant failure cases encountered during development, the root causes identified, and the engineering adjustments implemented to resolve them.

---

## Case 1: Proportionate Deduction Omission on Room Rent Overshoot

### 1. Description
* **Target Case**: `PUB-001` (Acute appendicitis, Sum Insured: INR 500,000, 4 days admission, Room rent claimed: INR 30,000).
* **Initial Behavior**: The system capped room rent at INR 5,000/day (1% of Sum Insured = INR 20,000 total allowed; deduction of INR 10,000). However, the associated doctor fees (INR 30,000) and diagnostics (INR 90,000) were approved in full without deduction.
* **Expected Outcome**: Under health insurance norms and the policy wording, when an insured opts for a room category above their eligible sub-limit, **proportionate deduction** must be applied to all associated medical, surgical, and diagnostic expenses.

### 2. Root Cause
* The initial Decision Agent treated room rent strictly as an isolated line-item cap. It calculated `allowed_room = min(claimed_room, daily_cap * days)` and subtracted only the direct room rent difference from the payable amount, without linking room rent eligibility to associated charges.

### 3. Engineering Fix
* Implemented a proportionate deduction calculation in `DecisionAgent` (`src/agents/adjudicator.py`):
  $$\text{Ratio} = \frac{\text{Eligible Room Rent}}{\text{Actual Incurred Room Rent}} = \frac{20,000}{30,000} \approx 66.67\%$$
  $$\text{Associated Deductions} = (1 - \text{Ratio}) \times (\text{Doctor Fees} + \text{Diagnostics}) = 33.33\% \times 120,000 = \text{INR } 40,000$$
* Result: Total payable correctly adjusted, preventing over-reimbursement.

---

## Case 2: Lexical Tokenizer Stripping Hyphenated Disease and Clause Terms

### 2. Description
* **Target Case**: `PUB-003` (Pre-existing thyroid condition, `pre_existing: true`) and `PUB-005` (Day-care cataract).
* **Initial Behavior**: Retrieval recall for BM25 queries containing terms like `"pre-existing"` and `"day-care"` produced poor similarity scores, occasionally retrieving general preamble text rather than the specific 48-month waiting period clause.
* **Expected Outcome**: Top-1 retrieval should pinpoint Clause 1 ("Pre-existing diseases") on Page 8 and Clause i ("Day Care Treatment") on Page 2.

### 2. Root Cause
* The standard BM25 regex tokenizer pattern `\b\w+\b` split hyphenated compound words into distinct disjoint tokens (`"pre"` and `"existing"`, `"day"` and `"care"`).
* In the policy wording, the term `"pre-existing"` often appears with non-breaking hyphens or specific formatting, leading to vocabulary fragmentation.

### 3. Engineering Fix
* Updated `_tokenize` in `HybridRetriever` (`src/rag/retriever.py`) to use:
  ```python
  re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*\b", text.lower())
  ```
* This preserves hyphenated phrases (`"pre-existing"`, `"day-care"`, `"first-year"`) as unified semantic tokens.
* Added clause and section title matching bonus (+2.5 to +3.0) in the reranker stage.
* Result: Top-1 retrieval score for PED queries increased significantly, directly surfacing `POL-P08-C023`.

---

## Case 3: False-Positive Validation Rejections on Paraphrased Findings

### 1. Description
* **Target Case**: `PUB-004` (Domiciliary treatment) and `PUB-006` (Unverified hospital definition).
* **Initial Behavior**: The Validation Agent flagged legitimate findings as `"FAIL"` with unsupported claims, even though the reasoning was legally sound according to the policy text.
* **Expected Outcome**: The Validation Agent should verify that the semantic essence and policy conditions (e.g. 20% cap, bed unavailability) are present without requiring verbatim copy-pasting of full paragraphs.

### 2. Root Cause
* The initial validation logic evaluated strict n-gram phrase matching between the agent's summary statement and the chunk text.
* Because the agents synthesized structured conclusions (e.g. *"Domiciliary treatment sub-limit of 20% Sum Insured applied"*), the phrasing differed slightly from the formal legalese in the policy (*"The Company's liability for Domiciliary Hospitalisation shall be limited to 20% of the Sum Insured"*), resulting in low literal n-gram overlap.

### 3. Engineering Fix
* Refactored `ValidationAgent` (`src/agents/validator.py`) to use token-level conceptual intersection over filtered content tokens (removing stopwords):
  $$\text{Overlap} = \frac{|\text{Tokens}(\text{Claim}) \cap \text{Tokens}(\text{Evidence})|}{|\text{Tokens}(\text{Claim})|}$$
* Established calibrated thresholds:
  - 25% overlap for policy citation claims.
  - 20% overlap for material key findings.
  - Excluded pure mathematical deduction summaries from lexical audits.
* Result: Eliminated false-positive validation rejections while reliably catching hallucinated assertions.
