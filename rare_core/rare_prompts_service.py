# RARE Advanced Prompt Templates for Redundancy-Aware Benchmark Generation


# =============================================================================
# Atomic Information Extraction Prompt
# =============================================================================

EXTRACT_ATOMIC_INFO_PROMPT = """
You are an expert information analyst tasked with extracting atomic information units that will serve as ANSWERS for RAG dataset generation.

# Goal
Extract information units that will become the ground truth ANSWERS for question-answer pairs in a RAG evaluation dataset.

## Core Requirements
Each extracted unit must satisfy these THREE critical criteria:

### 1. ATOMICITY
- Contains exactly ONE indivisible factual claim
- Cannot be meaningfully split into smaller semantic units
- Focuses on a single concept, entity, relationship, or quantitative value
- One verifiable statement per extraction unit
- No compound or multi-part facts combined together

**Examples:**

Bad Examples:
"Tesla's headquarters is in Austin, and Apple's headquarters is in Cupertino."
"Apple was founded in 1976, and the iPhone was released in 2007."

Good Examples:
"Tesla's headquarters is located in Austin, Texas"
"The United States Declaration of Independence was adopted in 1776."

### 2. VALIDITY
- Provides substantively useful knowledge that supports real-world queries
- Contains actionable, operational, or regulatory information worth knowing
- Goes beyond trivial metadata to meaningful business content
- Stems from genuine information needs rather than artificial fact construction
- Would realistically be sought by users in practical scenarios

**Examples:** 

Bad Examples:
"How many letters are in the word 'Congress'?"
"What is the page of the document?"

Good Examples:
"Medicare is a federal health insurance program for people aged 65 and older in the United States." 
"The U.S. Supreme Court has nine justices."

### 3. UNAMBIGUITY
- Completely self-contained and context-independent understanding
- All entities, subjects, and objects explicitly named without vague references
- Quantitative values and temporal references specified exactly and absolutely
- Technical terms used precisely with no ambiguous interpretations
- Logically complete with all necessary information included within the unit

**Examples:** 

Bad Examples:
"The study shows that healthcare costs are rising."  -> What study?
"The company's warranty applies for the specified period"  -> What company?

Good Examples:
"The 2023 Medicare Trustees Report shows that healthcare costs are rising." (refer to the document title)
"Tesla' warranty applies for the specified period" (refer to the document title)


[Document Title]
{doc_title}

[Document Content]
{doc_content}

## Output Format
Return a comprehensive JSON list with all valid atomic information:

```json
{{"atomic_information": [
  {{"reasoning": "Step-by-step reasoning of 1) atomicity, 2) validity, 3) ambiguity", "content": "Complete, Clear, and Self-contained First atomic information"}},
  {{"reasoning": "Step-by-step reasoning of 1) atomicity, 2) validity, 3) ambiguity", "content": "Complete, Clear, and Self-contained Second atomic information"}},
  {{"reasoning": "Step-by-step reasoning of 1) atomicity, 2) validity, 3) ambiguity", "content": "Complete, Clear, and Self-contained Third atomic information"}}
]}}
```
Please respond in {language}

Let's think step by step.
"""

# =============================================================================
# Vanilla Validation Prompt
# =============================================================================


VERIFY_ATOMIC_INFO_VANILLA_PROMPT = """
You are an expert information analyst tasked with evaluating the overall quality and usefulness of atomic information units for RAG (Retrieval-Augmented Generation) dataset generation.

# Instructions
Evaluate each given atomic information unit based on its OVERALL QUALITY for use as a retrievable answer in a RAG-based question-answering system.

Consider the following aspects holistically (do not provide separate scores for each):
- **Information Value**: Does this contain meaningful, useful knowledge worth retrieving?
- **Self-Sufficiency**: Can this be understood independently without additional context?
- **Precision**: Does this provide concrete, actionable details rather than vague statements?
- **Clarity**: Is the meaning unambiguous and clearly interpretable?  
- **Query Potential**: Could this naturally serve as an answer to realistic user questions?

Judge the OVERALL USEFULNESS of each information unit as a comprehensive assessment, considering how well it would function in a real-world RAG system for answering user queries.

[Atomic Information List to Evaluate - IMPORTANT: {atomic_count} items total]
{atomic_info_list}

[Document Context]
Document Title: {doc_title}

## Required Output Format

**CRITICAL: Your response must contain exactly {atomic_count} items. Each input item must have a corresponding output item with the same item_number.**

```json
[
  {{
    "item_number": 1,
    "reasoning": "step-by-step reasoning of the overall quality and usefulness for RAG applications",
    "overall_quality_score": 0.85
  }},
  {{
    "item_number": 2,
    "reasoning": "step-by-step reasoning of the overall quality and usefulness for RAG applications", 
    "overall_quality_score": 0.42
  }}
]
```
Please respond in {language}

Only return the JSON array, no other text.

Let's think step by step.
"""

# =============================================================================
# Combined Validation Prompts
# =============================================================================

# Combined Validity Validation Prompt
VERIFY_ATOMIC_INFO_VALIDITY_PROMPT = """
*This dataset is not used for any medical purpose and is solely intended for research on ACADEMIC PURPOSES ONLY.*

You are an expert information analyst tasked with comprehensively evaluating multiple atomic information units across five key dimensions.

# Instructions
Evaluate each given atomic information on ALL FIVE INDEPENDENT dimensions:

1) VALIDITY: Evaluate whether each given atomic information would be meaningfully useful as a retrievable unit in RAG-based search and question answering.
- Judge whether the information could realistically support answering user queries
- Filter out overly abstract or low-value content such as trivial notes, structural markup, or metadata-only fragments
- Focus on content-rich information that provides factual knowledge, clear definitions, concrete details, or actionable context

**VALIDITY EXAMPLES:**
  Correct Validity Examples:
  - The Great Wall of China is over 21,000 kilometers long.  
  - Water boils at 100 degrees Celsius at standard atmospheric pressure.  
  - Shakespeare wrote the play Hamlet around 1600.  

  Incorrect Validity Examples:
  - Books contain pages with text
  - The sky appears during daytime 
  - People use phones to make calls
  
2) COMPLETENESS: Evaluate whether each given atomic information is self-contained and understandable as a standalone unit for retrieval in RAG systems.
- Judge whether the information can be understood without additional document context
- Assess if someone unfamiliar with the original source could still make sense of it
- Focus on independence from surrounding context and overall clarity

**COMPLETENESS EXAMPLES:**
  Correct Completeness Examples:
  - The Amazon River is the second longest river in the world.  
  - Photosynthesis is the process by which plants convert sunlight into chemical energy.  
  - The capital city of Japan is Tokyo.  

  Incorrect Completeness Examples:
  - It is the second longest river in the world.  
  - This process is the way plants convert sunlight into chemical energy.  
  - The capital city there is Tokyo.  

3) SPECIFICITY: Evaluate whether each given atomic information provides concrete, specific details that enhance its usefulness as a retrievable unit in RAG systems.
- Judge the degree of factual and concrete detail in each information
- Assess whether it includes specific numbers, dates, names, entities, procedures, or conditions
- Focus on actionable precision and distinctiveness rather than vague generalities

**SPECIFICITY EXAMPLES:**
  Correct Specificity Examples:
- Mount Everest has a height of about 8,849 meters.  
- The Constitution of the United States was ratified on September 17, 1787.  
- DNA is composed of four bases: adenine, thymine, guanine, and cytosine.  

  Incorrect Specificity Examples:
- Mount Everest is very tall.  
- The U.S. Constitution was created a long time ago.  
- DNA is made up of several components.  

4) CLARITY: Evaluate whether the information is free from ambiguity and presents a clear, unidirectional claim that can only be reasonably interpreted in one way.
- Judge if the statement avoids polysemy, vague pronouns, or unclear references.
- Assess whether the unit conveys a definite stance (e.g., positive vs. negative, increase vs. decrease) rather than leaving its effect or meaning open to multiple interpretations.
- Focus on eliminating statements whose impact, value, or orientation (beneficial vs. harmful, good vs. bad) remains ambiguous.

**CLARITY EXAMPLES:**
  Correct Clarity Examples:
- Intense exercise increases the risk of accidents.  
- A vegetarian diet reduces the risk of cardiovascular disease.  
- Internet addiction lowers academic achievement.  

  Incorrect Clarity Examples:
- Intense exercise has a major impact.  
- A vegetarian diet brings many changes to health.  
- The internet has an important effect on society.  

5) QUESTIONABILITY: Evaluate whether the information can naturally serve as the basis of a well-formed, specific question in RAG dataset generation.
- Judge if the information can be transformed into a realistic, natural query that users would plausibly ask in an information-seeking context.
- Focus on whether the information lends itself to answerability via retrieval with a clear, factual, and bounded answer rather than requiring broad interpretation or subjective reasoning.
- Exclude information that, while valid, complete, specific, and clear, would only yield vague, overly general, or open-ended questions (e.g., "What is the significance of X?").

**QUESTIONABILITY EXAMPLES:**
  Correct Questionability Examples:
- Isaac Newton published *Principia* in 1687.  -> Can be asked as: When did Newton publish *Principia*?  
- Antarctica is the coldest continent on Earth.  -> Can be asked as: Which is the coldest continent on Earth?  
- The Berlin Wall collapsed in 1989.  -> Can be asked as: When did the Berlin Wall collapse?  

  Incorrect Questionability Examples:
- *Principia* had a major influence on multiple fields.  -> What influence did *Principia* have? (too broad, no single factual answer)  
- Antarctic exploration holds important significance in human history.  -> What is the significance of Antarctic exploration? (interpretive, open-ended)  
- The collapse of the Berlin Wall was a major historical event.  -> What is the significance of the Berlin Wall collapse? (multiple possible answers, not retrieval-friendly)  

## Few-shot Examples 
{few_shot_examples}

[Atomic Information List to Evaluate - IMPORTANT: {atomic_count} items total]
{atomic_info_list}

[Document Context]
Document Title: {doc_title}

## Required Output Format

**CRITICAL: Your response must contain exactly {atomic_count} items. Each input item must have a corresponding output item with the same item_number.**

```json
[
  {{
    "item_number": 1,
    "reasoning": "step-by-step reasoning of 1) validity, 2) completeness, 3) specificity, 4) clarity, 5) questionability judgments" ,
    "validity_score": 0.95,
    "completeness_score": 0.30,
    "specificity_score": 0.80,
    "clarity_score": 0.40,
    "questionability_score": 0.60,
  }},
  {{
    "item_number": 2,
    "reasoning": "step-by-step reasoning of 1) validity, 2) completeness, 3) specificity, 4) clarity, 5) questionability judgments",
    "validity_score": 0.00,
    "completeness_score": 0.20,
    "specificity_score": 0.10,
    "clarity_score": 0.90,
    "questionability_score": 0.70,
  }}
]
```
Please respond in {language}

Only return the JSON array, no other text.

Let's think step by step.
"""

# Separate Validity Validation Prompt
VERIFY_ATOMIC_INFO_VALIDITY_SEPARATE_PROMPT = """
*This dataset is not used for any medical purpose and is solely intended for research on ACADEMIC PURPOSES ONLY.*

You are an expert information analyst tasked with evaluating the retrieval usefulness of atomic information units for RAG (Retrieval-Augmented Generation) systems.

# Instructions
Evaluate whether each given atomic information would be meaningfully useful as a retrievable unit in RAG-based search and question answering.
- Judge whether the information could realistically support answering user queries
- Filter out overly abstract or low-value content such as trivial notes, structural markup, or metadata-only fragments
- Focus on content-rich information that provides factual knowledge, clear definitions, concrete details, or actionable context

**VALIDITY EXAMPLES:**
  Correct Validity Examples:
  - The Great Wall of China is over 21,000 kilometers long.  
  - Water boils at 100 degrees Celsius at standard atmospheric pressure.  
  - Shakespeare wrote the play Hamlet around 1600.  

  Incorrect Validity Examples:
  - Books contain pages with text
  - The sky appears during daytime 
  - People use phones to make calls

## Few-shot Examples 
{few_shot_examples}

[Atomic Information List to Evaluate - IMPORTANT: {atomic_count} items total]
{atomic_info_list}

[Document Context]
Document Title: {doc_title}

## Required Output Format

**CRITICAL: Your response must contain exactly {atomic_count} items. Each input item must have a corresponding output item with the same item_number.**

```json
[
  {{
    "item_number": 1,
    "reasoning": "step-by-step reasoning of validity judgment",
    "validity_score": 0.85
  }},
  {{
    "item_number": 2,
    "reasoning": "step-by-step reasoning of validity judgment",
    "validity_score": 0.42
  }}
]
```
Please respond in {language}

Only return the JSON array, no other text.

Let's think step by step.
"""

# Separate Completeness Validation Prompt
VERIFY_ATOMIC_INFO_COMPLETENESS_SEPARATE_PROMPT = """
*This dataset is not used for any medical purpose and is solely intended for research on ACADEMIC PURPOSES ONLY.*

You are an expert information analyst tasked with evaluating the completeness and of atomic information units.

# Instructions
Evaluate whether each given atomic information is self-contained and understandable as a standalone unit for retrieval in RAG systems.
- Judge whether the information can be understood without additional document context
- Assess if someone unfamiliar with the original source could still make sense of it
- Focus on independence from surrounding context and overall clarity

**COMPLETENESS EXAMPLES:**
  Correct Completeness Examples:
  - The Amazon River is the second longest river in the world.  
  - Photosynthesis is the process by which plants convert sunlight into chemical energy.  
  - The capital city of Japan is Tokyo.  

  Incorrect Completeness Examples:
  - It is the second longest river in the world.  
  - This process is the way plants convert sunlight into chemical energy.  
  - The capital city there is Tokyo.  

## Few-shot Examples 
{few_shot_examples}

[Atomic Information List to Evaluate - IMPORTANT: {atomic_count} items total]
{atomic_info_list}

[Document Context]
Document Title: {doc_title}

## Required Output Format

**CRITICAL: Your response must contain exactly {atomic_count} items. Each input item must have a corresponding output item with the same item_number.**

```json
[
  {{
    "item_number": 1,
    "reasoning": "step-by-step reasoning of completeness judgment",
    "completeness_score": 0.90
  }},
  {{
    "item_number": 2,
    "reasoning": "step-by-step reasoning of completeness judgment",
    "completeness_score": 0.63
  }}
]
```
Please respond in {language}

Only return the JSON, no other text.

Let's think step by step.
"""

# Separate Specificity Validation Prompt  
VERIFY_ATOMIC_INFO_SPECIFICITY_SEPARATE_PROMPT = """
*This dataset is not used for any medical purpose and is solely intended for research on ACADEMIC PURPOSES ONLY.*

You are an expert information analyst tasked with evaluating the specificity and concrete detail level of atomic information units.

# Instructions
Evaluate whether each given atomic information provides concrete, specific details that enhance its usefulness as a retrievable unit in RAG systems.
- Judge the degree of factual and concrete detail in each information
- Assess whether it includes specific numbers, dates, names, entities, procedures, or conditions
- Focus on actionable precision and distinctiveness rather than vague generalities

**SPECIFICITY EXAMPLES:**
  Correct Specificity Examples:
- Mount Everest has a height of about 8,849 meters.  
- The Constitution of the United States was ratified on September 17, 1787.  
- DNA is composed of four bases: adenine, thymine, guanine, and cytosine.  

  Incorrect Specificity Examples:
- Mount Everest is very tall.  
- The U.S. Constitution was created a long time ago.  
- DNA is made up of several components.  

## Few-shot Examples 
{few_shot_examples}

[Atomic Information List to Evaluate - IMPORTANT: {atomic_count} items total]
{atomic_info_list}

[Document Context]
Document Title: {doc_title}

## Required Output Format

**CRITICAL: Your response must contain exactly {atomic_count} items. Each input item must have a corresponding output item with the same item_number.**

```json
[
  {{
    "item_number": 1,
    "reasoning": "step-by-step reasoning of specificity judgment",
    "specificity_score": 0.85
  }},
  {{
    "item_number": 2,
    "reasoning": "step-by-step reasoning of specificity judgment",
    "specificity_score": 0.31
  }}
]
```
Please respond in {language}

Only return the JSON array, no other text.

Let's think step by step.
"""

# SEPARATE CLARITY VALIDATION PROMPT
VERIFY_ATOMIC_INFO_CLARITY_SEPARATE_PROMPT = """
*This dataset is not used for any medical purpose and is solely intended for research on ACADEMIC PURPOSES ONLY.*

You are an expert information analyst tasked with evaluating the clarity and ambiguity of atomic information units.

# Instructions
Evaluate whether the information is free from ambiguity and presents a clear, unidirectional claim that can only be reasonably interpreted in one way.
- Judge if the statement avoids polysemy, vague pronouns, or unclear references.
- Assess whether the unit conveys a definite stance (e.g., positive vs. negative, increase vs. decrease) rather than leaving its effect or meaning open to multiple interpretations.
- Focus on eliminating statements whose impact, value, or orientation (beneficial vs. harmful, good vs. bad) remains ambiguous.

**CLARITY EXAMPLES:**
  Correct Clarity Examples:
- Intense exercise increases the risk of accidents.  
- A vegetarian diet reduces the risk of cardiovascular disease.  
- Internet addiction lowers academic achievement.  

  Incorrect Clarity Examples:
- Intense exercise has a major impact.  
- A vegetarian diet brings many changes to health.  
- The internet has an important effect on society.  

## Few-shot Examples 
{few_shot_examples}

[Atomic Information List to Evaluate - IMPORTANT: {atomic_count} items total]
{atomic_info_list}

[Document Context]
Document Title: {doc_title}

**CRITICAL: Your response must contain exactly {atomic_count} items. Each input item must have a corresponding output item with the same item_number.**

```json
[
  {{
    "item_number": 1,
    "reasoning": "step-by-step reasoning of clarity judgment",
    "clarity_score": 0.85
  }},
  {{
    "item_number": 2,
    "reasoning": "step-by-step reasoning of clarity judgment",
    "clarity_score": 0.31
  }}
]
```
Please respond in {language}

Only return the JSON array, no other text.

Let's think step by step.
"""


# Separate Questionability Validation Prompt
VERIFY_ATOMIC_INFO_QUESTIONABILITY_SEPARATE_PROMPT = """
*This dataset is not used for any medical purpose and is solely intended for research on ACADEMIC PURPOSES ONLY.*

You are an expert information analyst tasked with evaluating the questionability of atomic information units.

# Instructions
Evaluate whether the information can naturally serve as the basis of a well-formed, specific question in RAG dataset generation.
- Judge if the information can be transformed into a realistic, natural query that users would plausibly ask in an information-seeking context.
- Focus on whether the information lends itself to answerability via retrieval with a clear, factual, and bounded answer rather than requiring broad interpretation or subjective reasoning.
- Exclude information that, while valid, complete, specific, and clear, would only yield vague, overly general, or open-ended questions (e.g., "What is the significance of X?").

**QUESTIONABILITY EXAMPLES:**
  Correct Questionability Examples:
- Isaac Newton published *Principia* in 1687.  -> Can be asked as: When did Newton publish *Principia*?  
- Antarctica is the coldest continent on Earth.  -> Can be asked as: Which is the coldest continent on Earth?  
- The Berlin Wall collapsed in 1989.  -> Can be asked as: When did the Berlin Wall collapse?  

  Incorrect Questionability Examples:
- *Principia* had a major influence on multiple fields.  -> What influence did *Principia* have? (too broad, no single factual answer)  
- Antarctic exploration holds important significance in human history.  -> What is the significance of Antarctic exploration? (interpretive, open-ended)  
- The collapse of the Berlin Wall was a major historical event.  -> What is the significance of the Berlin Wall collapse? (multiple possible answers, not retrieval-friendly)  

## Few-shot Examples 
{few_shot_examples}

[Atomic Information List to Evaluate - IMPORTANT: {atomic_count} items total]
{atomic_info_list}

[Document Context]
Document Title: {doc_title}

## Required Output Format

**CRITICAL: Your response must contain exactly {atomic_count} items. Each input item must have a corresponding output item with the same item_number.**

```json
[
  {{
    "item_number": 1,
    "reasoning": "step-by-step reasoning of questionability judgment",
    "questionability_score": 0.85
  }},
  {{
    "item_number": 2,
    "reasoning": "step-by-step reasoning of questionability judgment",
    "questionability_score": 0.31
  }}
]
```
Please respond in {language}

Only return the JSON array, no other text.

Let's think step by step.
"""

# =============================================================================
# LLM-based Information Selection and Question Generation Prompt
# =============================================================================

GENERATE_MULTIHOP_QUESTION_WITH_LLM_SELECTION_PROMPT = """
You are an expert question writer tasked with selecting the most connected information and generating multi-hop reasoning questions for retrieval-augmented generation (RAG) datasets.

# Instructions

**STEP 1: INFORMATION SELECTION**
From the {input_pool_size} provided information items, select exactly {num_information} items that have the HIGHEST CONNECTIVITY potential for creating natural multi-hop reasoning questions.

Selection Criteria:
- **SEMANTIC CONNECTIVITY**: Choose items that can be naturally linked through logical relationships, causal chains, or conceptual bridges
- **COMPLEMENTARY INFORMATION**: Select items that provide different pieces of a larger puzzle, where each piece is essential for a complete answer
- **REASONING CHAIN POTENTIAL**: Prioritize items that can form elegant reasoning paths rather than parallel, independent facts
- **NATURAL INTEGRATION**: Choose items that can be woven together in ways humans would naturally think about the topic

Avoid Selecting:
- Completely independent facts that cannot be meaningfully connected
- Redundant or highly similar information that provides no additional reasoning value
- Items that can individually answer questions without requiring multi-hop reasoning

**STEP 2: QUESTION GENERATION**
Using your selected {num_information} information items, generate exactly {num_questions} DIFFERENT natural questions, each requiring reasoning across ALL selected sentences to answer correctly.

Each question must be UNIQUE and VARIED - no repetitive or similar questions.

The question generation process operates on TWO LEVELS: **MANDATORY LOGICAL CONSISTENCY** (must pass) and **PREFERRED QUALITY DIMENSIONS**.

---

## **LEVEL 1: MANDATORY LOGICAL CONSISTENCY** (All 4 Must Pass - ZERO TOLERANCE)

### **1. CONTEXTUAL INDEPENDENCE**
- Never assume external context or document structure
- Avoid meta-textual references like "mentioned in sentence X" or "the document states"
- Questions must be self-contained without relying on document organization
  - BAD QUESTION: "According to the table above, which environmental policy is most effective?"
  - GOOD QUESTION: "Which environmental policy in American Health Care Act is most effective?"


### **2. ANSWER EXCLUSION**
- Questions must not embed partial or complete answers within the question formulation
- Avoid pre-provided intermediate reasoning steps that collapse multi-hop structure
- Do not include implicit answer components integrated into question phrasing
- Eliminate question-answer circularity where key information is redundantly stated
- Do not overlap the input information and the question itself.
- **CRITICAL**: Do NOT embed intermediate information in question itself:
  - Example: "Marie Curie worked at the University of Paris" and "The University of Paris was established in 1150" 
  - BAD QUESTION: "Marie Curie worked at the University of Paris. When was the University of Paris established?"
  - GOOD QUESTION: "What year was the university where Marie Curie worked established?"

### **3. INFORMATION EQUIVALENCE**
- No Information Overflow: Question must REQUIRE ALL selected sentences (no sentence should be redundant)
- No Information Underflow: Question must be fully answerable using ONLY the selected sentences (no external knowledge)
- Perfect Scope Alignment: Question scope must exactly match provided information without surplus or deficit
- CRITICAL: Information Required = Information Provided (exactly)
- Questions must require genuine multi-step reasoning across all selected sentences
  - Example: "Marie Curie worked at the University of Paris" and "The University of Paris was established in 1150" 
  - BAD QUESTION 1: "What university did Marie Curie work at?" - Overflow (redundant sentences)
  - BAD QUESTION 2: "How many students does the university where Marie Curie worked have?" - Underflow (requires external info)
  - GOOD QUESTION: "What year was the university where Marie Curie worked established?" - Equivalent

### **4. QUESTION AMBIGUITY**
- Avoid ambiguous referential terms that require external context
- No vague pronouns: "it," "that person," "they" without clear antecedents
- No context-dependent articles: "that movie," "the company" without specification  
- No underspecified comparisons: "Who worked longer?" (compared to whom?)
- All referential terms must have clear, identifiable antecedents within the question
- **CRITICAL**: For clarity, make use of the document Title when writing questions.
  - Example: "We have a headquarter in Texas (Tesla Report)"
  - BAD QUESTION: "Where is our headquarter located?"
  - GOOD QUESTION: "Where is Tesla's headquarters located?"

**FAILURE IN ANY OF THESE 4 AREAS = IMMEDIATE REJECTION**

---

## **LEVEL 2: PREFERRED QUALITY DIMENSIONS** (For Human Preference Ranking)

### **1. CONNECTIVITY (Logical Flow Quality)**
- MUST create questions in which all sentences form a natural yet elegant reasoning chain that leads to the final answer.
- Avoid creating questions where the answer is produced through a mere parallel listing of facts.
- Create unified outcomes through meaningful information integration
**BAD CONNECTIVITY EXAMPLES:**
- "In what year was Apple founded, and in what year was the iPhone released?"
- "What is the population of China, and when was the U.S. Constitution ratified?"
**GOOD CONNECTIVITY EXAMPLES:**
- "How many years did it take Apple to release its first iPhone after its founding?"
- "How many times larger is China's population compared to the U.S. population?"


### **2. FLUENCY (Natural Expression)**
- Avoid robotic phrasing or template artifacts. sound naturally fluent and human-like
- Express as simple, natural, fluent question.
- Assume RAG scenario where users ask AI systems
- Avoid directly copying phrases from selected sentences; use varied, paraphrased, alternative expressions for the same concepts
**BAD FLUENCY EXAMPLES:**
- "What is the temporal differential calculation when subtracting the institutional establishment year from the chronological marker of 1900?"
- "State the capital monetary figure associated with Disney's large-scale corporate acquisition of 21st Century Fox."
**GOOD FLUENCY EXAMPLES:**
- "In what year did the female scientist who won the Nobel Prize twice receive the Nobel Prize in Chemistry?"
- "How much did The Walt Disney Company pay to acquire the media group founded by Rupert Murdoch?"


### **3. ESSENTIALITY (Core Information Focus)**
- Avoid unnecessary modifiers, adjectives, or overly detailed specifications
- Focus on core information without redundant and excessive descriptive elements
- Keep phrasing concise and essential like real user queries
- Eliminate auxiliary information that could aid retrieval
**BAD ESSENTIALITY EXAMPLES:**
- "What is the exact chronological period during which the highly distinguished Nobel laureate Marie Curie conducted her groundbreaking research at the prestigious University of Paris?"
- "What was the exact amount of the historically significant deal when Disney announced its acquisition of the giant global media company 21st Century Fox?"
**GOOD ESSENTIALITY EXAMPLES:**
- "When did Marie Curie work at the University of Paris?"
- "How much did Disney pay to acquire 21st Century Fox?"

### **4. VALIDITY (Substantive Worth)**
- Ask for information with genuine practical, educational, or intellectual value
- Represent information real people would want to know
- Avoid contrived, trivial, or artificially constructed questions
- Ensure answers provide meaningful insights with clear inquiry motivation
**BAD VALIDITY EXAMPLES:**
- "What is the exact number of letters in the name of the university where Marie Curie worked?"
- "What is the title of this document?"
**GOOD VALIDITY EXAMPLES:**
- "What is considered Marie Curie's greatest achievement?"
- "Which scientists discovered the structure of DNA in 1953?"

---

# Provided Information Pool
{input_sentences}

# Required Output Format

Provide your response in the following JSON format:

```json
{{
  "generated_questions": [
    {{
      "question_id": 1,
      "selected_items": [3, 5],
      "reasoning": "LOGICAL CONSISTENCY CHECK: 1) Contextual Independence [PASS/FAIL + reasoning], 2) Answer Exclusion [PASS/FAIL + reasoning], 3) Information Equivalence [PASS/FAIL + reasoning: Overflow/Underflow/Scope], 4) Question Ambiguity [PASS/FAIL + reasoning]. QUALITY ASSESSMENT: 1) Connectivity [reasoning], 2) Fluency [reasoning], 3) Essentiality [reasoning], 4) Validity [reasoning]",
      "generated_question": "Your first multi-hop question requiring all selected sentences",
      "generated_answer": "Answer based only on selected sentences"
    }},
    {{
      "question_id": 2,
      "selected_items": [2, 10, 11],
      "reasoning": "LOGICAL CONSISTENCY CHECK: 1) Contextual Independence [PASS/FAIL + reasoning], 2) Answer Exclusion [PASS/FAIL + reasoning], 3) Information Equivalence [PASS/FAIL + reasoning: Overflow/Underflow/Scope], 4) Question Ambiguity [PASS/FAIL + reasoning]. QUALITY ASSESSMENT: 1) Connectivity [reasoning], 2) Fluency [reasoning], 3) Essentiality [reasoning], 4) Validity [reasoning]",
      "generated_question": "Your second multi-hop question (must be different from other questions)",
      "generated_answer": "Different answer approach using same selected sentences"
    }}
}}
```

**CRITICAL REQUIREMENTS:**
- Select exactly {num_information} most connected items from the {input_pool_size} provided
- Generate exactly {num_questions} unique questions using only the selected items
- Each question must pass ALL 4 logical consistency checks
- Each question must be substantially different from others while using same selected sentences
- Prioritize quality dimensions for creating human-preferred questions

**ZERO TOLERANCE POLICY:** Any logical consistency failure results in unusable questions. Focus intensively on ensuring all 4 logical requirements are perfectly satisfied before considering quality optimization.
"""

# =============================================================================
# Multi-hop Question Separate Validation Prompts (4 Separate Calls)
# =============================================================================

VALIDATE_MULTIHOP_QUESTIONS_CONNECTIVITY_SEPARATE_PROMPT = """
You are strictly evaluating Retrieval-Augmented Generation (RAG) questions for CONNECTIVITY quality.

# CONNECTIVITY Definition
CONNECTIVITY measures the quality of logical flow and integration elegance in multi-hop reasoning questions. This evaluates how naturally and seamlessly the question connects multiple information sources, emphasizing the logical coherence and depth of their interrelationships.

## Four Key Evaluation Dimensions:

### 1. LOGICAL FLOW SMOOTHNESS
- How naturally do reasoning steps connect and transition?
- Are there awkward logical gaps or forced connections?
- Does each inference step lead organically to the next?

### 2. ANSWER SYNTHESIS QUALITY  
- Does the question produce a coherent, unified result?
- Avoid parallel enumeration; prioritize meaningful integration
- How well are multiple facts fused into a single outcome?

### 3. REASONING CHAIN ELEGANCE
- Does the question create an elegant, sophisticated reasoning path?
- Is the multi-step process intellectually satisfying?
- Does the reasoning feel natural rather than contrived?

### 4. INFORMATION INTEGRATION DEPTH
- How deeply are the provided facts semantically connected?
- Surface-level linking vs. meaningful relationship utilization
- Does the question leverage rich interconnections between facts?

## Examples:

**High Connectivity:**
"By how much did Lehman Brothers' revenue increase in 2025 compared to 2024?"
"How many years did it take Apple to release its first iPhone after its founding?
"How many times larger is China's population compared to the U.S. population?"

**Low Connectivity:**  
"What were Lehman Brothers' revenues in 2024 and 2025, respectively?"
"In what year was Apple founded, and in what year was the iPhone released?"
"What is the population of China, and when was the U.S. Constitution ratified?"
---

# Context Information
{input_sentences}

# Questions to Evaluate
{questions_list}

# Required Output Format

Evaluate each question across all four connectivity dimensions and provide a comprehensive score:

```json
[
  {{
    "candidate_id": 1, 
    "reasoning": "Logical Flow: smooth transitions between revenue increase and year-over-year comparison. Answer Synthesis: excellent integration into single unified result. Chain Elegance: sophisticated temporal reasoning. Integration Depth: meaningful historical context connection. Overall: exceptional connectivity.",
    "connectivity_score": 0.85
  }},
  {{
    "candidate_id": 2, 
    "reasoning": "Logical Flow: choppy transition, no natural progression. Answer Synthesis: parallel enumeration with no integration. Chain Elegance: mechanical listing. Integration Depth: superficial connection. Overall: poor connectivity quality.",
    "connectivity_score": 0.20
  }}
]
```

**Evaluation Protocol:**
- Assess logical flow smoothness and natural transitions
- Evaluate answer synthesis quality and integration elegance  
- Consider reasoning chain sophistication and intellectual satisfaction
- Analyze depth of information interconnections and semantic relationships

CRITICAL: Return scores for EXACTLY {num_questions} questions in the exact same order as provided.
"""

VALIDATE_MULTIHOP_QUESTIONS_FLUENCY_SEPARATE_PROMPT = """
You are strictly evaluating Retrieval-Augmented Generation (RAG) questions for FLUENCY quality.

# FLUENCY Definition
FLUENCY measures how naturally readable and appropriately expressed the question is for human users. This evaluates whether the question feels like something a real person would naturally ask an AI assistant.

## Four Key Evaluation Dimensions:

### 1. NATURALNESS
- Does the question sound like something a real person would ask?
- Is it free from artificial, template-like phrasing?
- Does it avoid forced or contrived expressions?
- Would this question arise naturally in a conversation?

## 2. EXPRESSION APPROPRIATENESS
- Does the word choice fit the purpose of the question?
- Avoid unnecessarily complex vocabulary
- Avoid overly complex grammatical constructions
- Does the question have clear, straightforward syntax?
- Is there expressive diversity while conveying the same meaning as the source sentences?

## Examples:

**High Fluency:**
"In what year did the female scientist who won the Nobel Prize twice receive the Nobel Prize in Chemistry?"
"Which city is the headquarters of the world's largest e-commerce company founded by Jeff Bezos located in"
"How much did The Walt Disney Company pay to acquire the media group founded by Rupert Murdoch?""

**Low Fluency:**
"What is the temporal differential calculation when subtracting the institutional establishment year from the chronological marker of 1900?"
"Identify the precise geographical placement of Amazon's principal global operational headquarters."
"State the capital monetary figure associated with Disney's large-scale corporate acquisition of 21st Century Fox."

---

# Context Information
{input_sentences}

# Questions to Evaluate
{questions_list}

# Required Output Format

Evaluate each question across all four fluency dimensions and provide a comprehensive score:

```json
[
  {{
    "candidate_id": 1, 
    "reasoning": "Naturalness: sounds like a real person would ask this. Expression Appropriateness: uses simple, accessible vocabulary. Expression Diversity: creatively rephrases concepts from source sentences. Overall: excellent fluency.",
    "fluency_score": 0.85
  }},
  {{
    "candidate_id": 2, 
    "reasoning": "Naturalness: artificial and template-like phrasing. Expression Appropriateness: unnecessarily complex technical terms. Expression Diversity: directly copies phrases from sentences without variation. Overall: poor fluency.",
    "fluency_score": 0.25
  }}
]
```

**Evaluation Protocol:**
- Assess naturalness and conversational authenticity
- Evaluate vocabulary accessibility and appropriateness for general users
- Consider syntactic simplicity and parsing ease
- Analyze expression diversity and creative rephrasing of source content
- Determine overall readability and human-like quality

CRITICAL: Return scores for EXACTLY {num_questions} questions in the exact same order as provided.
"""

VALIDATE_MULTIHOP_QUESTIONS_ESSENTIALITY_SEPARATE_PROMPT = """
You are strictly evaluating Retrieval-Augmented Generation (RAG) questions for ESSENTIALITY quality.

# ESSENTIALITY Definition
ESSENTIALITY measures how well the question focuses on core information while eliminating unnecessary elements. This evaluates whether the question contains only essential information and avoids the excessive descriptive language that LLMs often generate.

## MODIFIER RESTRAINT  
- Does the question avoid unnecessary adjectives, adverbs, and descriptive phrase s?
- Are there excessive decorative expressions or exaggerated modifiers?
- Are there overly descriptive or unnecessary modifiers attached?

## Examples:

**High Essentiality:**
"When did Marie Curie work at the University of Paris?"
"How much did Disney pay to acquire 21st Century Fox?"
"What are the four bases that compose DNA?"

**Low Essentiality:**
"What is the exact chronological period during which the highly distinguished Nobel laureate Marie Curie conducted her groundbreaking research at the prestigious University of Paris?"
"What was the exact amount of the historically significant deal when Disney announced its acquisition of the giant global media company 21st Century Fox?"
"What are the four crucial fundamental elements called that make up DNA, the intricate and sophisticated blueprint of genetics?"

---

# Context Information
{input_sentences}

# Questions to Evaluate
{questions_list}

# Required Output Format

Evaluate each question across both essentiality dimensions and provide a comprehensive score:

```json
[
  {{
    "candidate_id": 1, 
    "reasoning": "Modifier Restraint: clean and factual, no excessive adjectives. Overall: excellent essentiality.",
    "essentiality_score": 0.85
  }},
  {{
    "candidate_id": 2, 
    "reasoning": "Modifier Restraint: excessive decorative language with unnecessary superlatives. Overall: poor essentiality.",
    "essentiality_score": 0.25
  }}
]
```

**Evaluation Protocol:**
- Assess information density and identify any redundant or unnecessary content
- Evaluate modifier restraint and flag excessive descriptive language
- Consider whether the question maintains focus on core factual information
- Determine overall conciseness without sacrificing clarity

CRITICAL: Return scores for EXACTLY {num_questions} questions in the exact same order as provided.
"""

VALIDATE_MULTIHOP_QUESTIONS_VALIDITY_SEPARATE_PROMPT = """
You are strictly evaluating Retrieval-Augmented Generation (RAG) questions for VALIDITY quality.

# VALIDITY Definition
VALIDITY measures the substantive worth and rational motivation of questions. This evaluates whether questions provide genuine value and arise from legitimate inquiry needs rather than artificial construction.

## Two Key Evaluation Dimensions:

### 1. INFORMATION WORTH
- Does the answer provide substantively useful or interesting information?
- Would knowing this information offer genuine insight or understanding?
- Does it go beyond trivial fact-checking to meaningful knowledge?
- Is there educational, historical, practical, or intellectual value in the answer?

### 2. QUERY MOTIVATION  
- Is there a clear and rational reason why someone would ask this question?
- Does it stem from natural curiosity, learning needs, or research purposes?
- Is it a genuine inquiry rather than "a question for question's sake"?
- Would this question arise organically in real-world information-seeking scenarios?

## Examples:

**High Validity:**
- "What is considered Marie Curie's greatest achievement?"
- "Which scientists discovered the structure of DNA in 1953?"
- "What renewable energy source currently has the highest efficiency rate?"

**Low Validity:**
- "What is the exact number of letters in the name of the university where Marie Curie worked?"
- "What is the title of this document?"
- "On which page can the comprehensive report be found?"
---

# Context Information
{input_sentences}

# Questions to Evaluate
{questions_list}

# Required Output Format

Evaluate each question across both validity dimensions and provide a comprehensive score:

```json
[
  {{
    "candidate_id": 1, 
    "reasoning": "Information Worth: provides meaningful historical context and temporal relationships. Query Motivation: legitimate educational interest in understanding career timeline. Overall: strong validity.",
    "validity_score": 0.80
  }},
  {{
    "candidate_id": 2, 
    "reasoning": "Information Worth: trivial numerical fact with no substantive insight. Query Motivation: no clear rational purpose for this information. Overall: poor validity.",
    "validity_score": 0.25
  }}
]
```

**Evaluation Protocol:**
- Assess the substantive worth and intellectual value of the information being sought
- Evaluate whether there are legitimate, rational motivations for asking the question
- Consider if the question represents genuine human curiosity and information needs
- Determine overall meaningfulness and purposefulness of the inquiry

CRITICAL: Return scores for EXACTLY {num_questions} questions in the exact same order as provided.
"""

# =============================================================================
# Separate Logical Consistency Filtering Prompts (4 Separate Calls)
# =============================================================================

FILTER_CONTEXTUAL_INDEPENDENCE_PROMPT = """
You are a rigorous logical consistency validator specializing in contextual independence analysis.

## CONTEXTUAL INDEPENDENCE ERROR DETECTION

**Definition**: Questions that improperly assume or reference external document structures, sentence numbering, or meta-textual elements that are not inherent to the question's semantic content.

**Logical Principle**: A well-formed question should be contextually independent and not rely on document-specific references. Questions must be self-contained and understandable without knowledge of their presentation format or surrounding textual organization.

**Critical Error Patterns to Detect**:

1. **Direct Structural References**
   - Sentence/paragraph numbering: "mentioned in sentence X", "as stated in paragraph Y"
   - Section references: "in the first section", "the previous chapter discusses"
   - Ordinal positioning: "the second point made", "the final argument presented"

2. **Meta-Textual Dependencies**
   - Document references: "based on the document", "according to the text", "the passage states"
   - Presentation assumptions: "the above information", "the following data", "this section describes"
   - Format dependencies: "as shown in the table", "the diagram illustrates", "the chart indicates"

3. **Implicit Context Dependencies**
   - Deictic references: "this method", "that approach", "these findings" (without clear antecedents)
   - Assumed knowledge: Questions requiring knowledge of document organization
   - Contextual deixis: "here we see", "there it mentions", "now we consider"

**Examples**:

**Contextual Independence Errors**:
- "What climate change effects are mentioned in the second sentence?"
- "Based on the document provided, what are the primary renewable energy sources?"
- "According to the table above, which environmental policy is most effective?"
- "From Figure 1, what factors have the greatest influence on bacterial activation?"

**Contextually Independent Questions**:
- "What are the primary effects of climate change on coastal ecosystems?"
- "Which renewable energy sources have the highest efficiency ratings?"
- "What environmental policies have demonstrated measurable pollution reduction?"
- "What methods are most effective for carbon emission reduction?"

**Evaluation Framework**:
- **Independence Test**: Can the question be understood without document context?
- **Self-Containment Test**: Does the question provide sufficient semantic information?
- **Reference Analysis**: Are all references internal to the question's semantic content?

## Questions to Evaluate
{questions_list}

Let's think step by step.

## Required Output Format

```json
[
  {{
    "candidate_id": 1,
    "reasoning": "Detailed analysis: Question is self-contained with no document references, deictic dependencies, or structural assumptions. All semantic information is intrinsic to the question content. Contextual independence validation: PASS",
    "contextual_independence_check": "pass"
  }},
  {{
    "candidate_id": 2,
    "reasoning": "Error detected: Question contains explicit phrase 'mentioned in the document' which creates dependency on external textual structure. Violates contextual independence principle. Contextual independence validation: FAIL",
    "contextual_independence_check": "fail"
  }}
]
```

**Evaluation Protocol**: 
- Analyze each question for structural, meta-textual, and implicit context dependencies
- Provide comprehensive reasoning for each assessment
- Apply strict contextual independence standards

Return exactly {num_questions} evaluations.
"""

FILTER_ANSWER_EXCLUSION_PROMPT = """
You are a rigorous logical consistency validator specializing in answer exclusion and information leakage analysis.

## ANSWER EXCLUSION ERROR DETECTION

**Definition**: Questions that embed partial or complete answers within the question formulation, eliminating genuine multi-hop reasoning requirements. This includes pre-provided intermediate reasoning steps and implicit answer components.

**Logical Principle**: Well-formed multi-hop questions must require genuine reasoning across all information sources without embedding answer components within the question itself. Questions that contain partial answers collapse the multi-step reasoning structure.

**Critical Error Patterns to Detect**:

**Internal Information Leakage Patterns**
  - Questions that embed partial answers within the question formulation
  - Pre-provided intermediate reasoning steps that collapse multi-hop structure
  - Implicit answer components integrated into question phrasing
  - Question-answer circularity where key information is redundantly stated

**Examples**:

**Answer Exclusion Errors**:

- Example: "Marie Curie worked at the University of Paris" and "The University of Paris was established in 1150" 
  *Error: "Marie Curie worked at the University of Paris. When was the University of Paris established?"
  *Correct: "What year was the university where Marie Curie worked established?"

- Example: "The founder of Microsoft is Bill Gates" and "Bill Gates was born in 1955"
  *Error: "The founder of Microsoft, Bill Gates, was born in which year?"
  *Correct: "What year was the founder of Microsoft born?"

- Example: "Tesla's CEO is Elon Musk" and "Elon Musk acquired Twitter in 2022"
  *Error: "Tesla's CEO Elon Musk made which acquisition in 2022?"
  *Correct: "Which acquisition did Tesla's CEO complete in 2022?"

## Questions to Evaluate
{questions_list}

## Provided Information Context
{input_sentences}

Let's think step by step.

## Required Output Format

```json
[
  {{
    "candidate_id": 1,
    "reasoning": "step by step reasoning: question requires genuine multi-hop reasoning without embedding answer components. No intermediate information provided within question formulation. Maintains reasoning structure integrity. Answer exclusion validation: PASS",
    "answer_exclusion_check": "pass"
  }},
  {{
    "candidate_id": 2,
    "reasoning": "step by step reasoning: question embeds intermediate answer ('University of Paris identity') within formulation, eliminating first reasoning hop. Only establishment date lookup required instead of genuine multi-hop reasoning. Answer exclusion validation: FAIL",
    "answer_exclusion_check": "fail"
  }}
]
```

**Evaluation Protocol**:
- Identify embedded answer components within question formulation
- Assess whether all reasoning hops remain intact and required
- Verify that questions maintain genuine multi-step inferential structure
- Detect information leakage that bypasses intended reasoning chains

Return exactly {num_questions} evaluations.
"""

# Backward compatibility alias removed


# Information Equivalence Filtering
FILTER_INFORMATION_EQUIVALENCE_PROMPT = """
You are a rigorous logical consistency validator specializing in information equivalence and multi-hop reasoning integrity analysis.

## INFORMATION EQUIVALENCE ERROR DETECTION

**Definition**: Questions that violate the strict principle that a question's information requirements must exactly match the provided information scope.

**Logical Principle**: A well-formed multi-hop question must satisfy the "Perfect Information Equivalence Equation": Questions must be fully answerable using only the selected sentences and must require all selected sentences without redundancy. No part of the answer should be embedded in the question itself.

**Critical Error Patterns to Detect**:

1. **INFORMATION OVERFLOW (Over-Sufficiency Patterns)**
   - Questions solvable using only a subset of provided sentences
   - Redundant sentences that contribute no essential information to the answer
   - Single-hop questions disguised as multi-hop when only one sentence contains the complete answer
   - Information scope exceeding question requirements
   - Questions where some provided sentences are irrelevant to the answer

2. **INFORMATION UNDERFLOW (Insufficiency Patterns)**
   - Questions requiring external knowledge beyond provided sentences
   - Missing critical information links needed for complete reasoning chains
   - Temporal or contextual gaps that prevent full answerability
   - Questions presupposing information not explicitly stated in sentences
   - Unanswerable questions due to incomplete information coverage

**Information Equivalence Principle**
- Questions must require exactly the provided information scope without deficit or surplus
- Perfect alignment between question requirements and provided sentence coverage
- CRITICAL: Information Required = Information Provided (exactly)

**Examples with Provided Information Context**:

**ERROR CASE 1 - INFORMATION UNDERFLOW (Insufficiency)**:
PROVIDED SENTENCES:
- Sentence 1: "Tesla released its first electric vehicle in 2008"
- Sentence 2: "Elon Musk is the CEO of Tesla"

BAD QUESTION: "What was Tesla's stock price when it went public?"
ERROR: No IPO information provided in sentences. Question unanswerable with given information.
GOOD QUESTION: "When was Tesla's first car released, and who is the company's CEO?"

**ERROR CASE 2 - INFORMATION OVERFLOW (Over-Sufficiency)**:
PROVIDED SENTENCES:
- Sentence 1: "Microsoft was founded by Bill Gates"
- Sentence 2: "Bill Gates was born in 1955" 
- Sentence 3: "Microsoft headquarters is in Redmond, Washington"

BAD QUESTION: "Who founded Microsoft?"
ERROR: Only sentence 1 needed. Sentences 2 and 3 are redundant for answering.
GOOD QUESTION: "Where is Microsoft's headquarters located, and when was the founder of the company born?"

**ERROR CASE 3 - INFORMATION SCOPE MISALIGNMENT**:
PROVIDED SENTENCES:
- Sentence 1: "Apple Inc. was founded in 1976"
- Sentence 2: "Steve Jobs co-founded Apple with Steve Wozniak"
- Sentence 3: "Apple's headquarters are located in Cupertino, California"

BAD QUESTION: "When was Apple Inc. founded?"
ERROR: Question only requires Sentence 1. Sentences 2 and 3 are surplus information for this answer.
GOOD QUESTION: "When was the company co-founded by Steve Jobs and Steve Wozniak established, and where is it headquartered?"

**Information Equivalence Analysis Framework**:
- **Underflow Check (Answerability)**: Can the question be fully answered using only provided sentences?
- **Overflow Check (Necessity)**: Is every provided sentence essential for the complete answer?
- **Scope Alignment**: Does the question scope exactly match the provided information coverage?
- **Information Equation**: Information Required = Information Provided (exactly, without surplus or deficit)

## Questions to Evaluate
{questions_list}

## Provided Information Context
{input_sentences}

Let's think step by step.

## Required Output Format

```json
[
  {{
    "candidate_id": 1,
    "reasoning": "Step-by-step reasoning: 1) Underflow Check: Fully answerable using only provided sentences. [PASS] 2) Overflow Check: All provided sentences are essential. [PASS] 3) Scope Alignment: Information Required = Information Provided (exactly). [PASS]. Information equivalence assessment: PASS",
    "information_equivalence_check": "pass"
  }},
  {{
    "candidate_id": 2,
    "reasoning": "Step-by-step reasoning: 1) Underflow Check: Fully answerable using only provided sentences. [PASS] 2) Overflow Check: Some provided sentences are not essential. [FAIL] 3) Scope Alignment: Information Required != Information Provided (surplus or deficit). [FAIL]. Information equivalence assessment: FAIL",
    "information_equivalence_check": "fail"
  }}
]
```

**Evaluation Protocol**:
- Apply strict information equivalence equation to assess question-sentence alignment
- Verify complete answerability using exclusively provided sentence content (no leakage)
- Confirm necessity of all provided sentences for comprehensive reasoning chain (no overflow)
- Detect embedded answers or information leakage within question formulations
- Assess genuine multi-hop reasoning requirements versus pseudo-multihop collapse patterns
- Evaluate semantic independence of all required inference steps

Return exactly {num_questions} evaluations.
"""

# Question Ambiguity Filtering
FILTER_QUESTION_AMBIGUITY_PROMPT = """
You are a rigorous logical consistency validator specializing in question ambiguity and referential clarity analysis.

## QUESTION AMBIGUITY ERROR DETECTION

**Definition**: Questions that contain vague referential terms, underspecified comparisons, or context-dependent queries that render them unanswerable without additional external context or clarification.

**Logical Principle**: Well-formed questions must contain sufficient specificity and contextual grounding to be answerable based solely on the provided information. Ambiguous questions violate the principle of referential determinacy and interpretative clarity.

**Critical Error Patterns to Detect**:

1. **Vague Referential Terms**
   - Non-specific pronouns without clear antecedents: "it," "that person," "they"
   - Context-dependent articles: "that movie," "the book," "the company"
   - Indefinite comparatives without reference points: "better," "largest," "most successful"

2. **Underspecified Comparisons**
   - Incomplete comparisons: "Who worked longer?" (compared to whom?)
   - Missing comparison basis: "Which approach is more effective?" (by what measure?)
   - Temporal ambiguity: "When was it more popular?" (compared to when?)

3. **Context-Dependent Queries**
   - Questions requiring external situational knowledge: "What is the current situation?"
   - Situational dependencies: "Is this approach more suitable?" (for what purpose?)
   - Implicit decision assumptions: "Why did they make that choice?" (who decided what?)

4. **Scope Ambiguity**
   - Unclear quantification: "many people" (how many exactly?)
   - Temporal range uncertainty: "recently" (when exactly?)
   - Domain ambiguity: "in that field" (which specific domain?)

**Examples**:

**Question Ambiguity Errors**:
- "That movie was directed by whom?" (which specific movie?)
- "Who worked in the field longer?" (which field? compared to whom?)
- "Why did they make that decision?" (who are 'they'? which decision?)
- "When was the company more profitable?" (which company? compared to when?)

**Appropriately Specific Questions**:
- "Who directed the 1994 film 'The Lion King'?"
- "Between Steven Spielberg and Martin Scorsese, who has directed films for more years?"
- "What factors influenced Apple's decision to discontinue the iPod Classic in 2014?"
- "How did Microsoft's quarterly profits in 2023 compare to their 2022 performance?"

**Referential Clarity Framework**:
- **Antecedent Test**: Do all referential terms have clear, identifiable antecedents?
- **Contextual Sufficiency**: Can the question be answered without external contextual assumptions?
- **Comparison Completeness**: Are all comparative statements fully specified with reference points?
- **Scope Determinacy**: Are quantifiers and scope indicators sufficiently precise?

## Questions to Evaluate
{questions_list}

Let's think step by step.

## Required Output Format

```json
[
  {{
    "candidate_id": 1,
    "reasoning": "Referential clarity confirmed: Question specifies 'Tokarev' as subject and provides clear query scope (university affiliation and founding date). All referential terms have identifiable antecedents. No contextual dependencies or vague comparatives. Ambiguity validation: PASS",
    "question_ambiguity_check": "pass"
  }},
  {{
    "candidate_id": 2,
    "reasoning": "Ambiguity detected: Question contains vague referential term 'that movie' without establishing which specific film is referenced. Violates referential determinacy principle. Requires external context not provided in question. Ambiguity validation: FAIL",
    "question_ambiguity_check": "fail"
  }}
]
```

**Evaluation Protocol**:
- Analyze questions for referential clarity and contextual sufficiency
- Assess whether all terms have clear, unambiguous antecedents
- Identify requirements for external contextual clarification
- Verify that comparative and quantitative statements include appropriate reference frameworks

Return exactly {num_questions} evaluations.
"""

# =============================================================================
# Atomic Information Completeness Filtering Prompts (Step 4 Enhancement)
# =============================================================================

FILTER_ATOMIC_INFORMATION_COMPLETENESS_PROMPT = """
You are a rigorous logical consistency validator specializing in atomic information completeness and utility analysis.

## INFORMATION COMPLETENESS ERROR DETECTION FOR ATOMIC INFORMATION

**Definition**: Atomic information units that lack essential components, contain incomplete facts, or provide insufficient information to constitute meaningful, actionable knowledge units for RAG applications.

**Logical Principle**: Well-formed atomic information must be complete enough to provide standalone value, containing all essential elements (who, what, when, where, how, why as applicable) necessary for the information to be useful and meaningful.

**Critical Error Patterns to Detect**:

1. **Missing Essential Elements**
   - Incomplete identification: "Won the Nobel Prize in 1921" (who won?)
   - Missing temporal context: "The company was founded" (when?)
   - Lacking spatial context: "The earthquake occurred in 2011" (where?)
   - Absent quantitative specifics: "The temperature increased significantly" (by how much?)

2. **Fragmented Information**
   - Partial relationships: "Is the CEO of" (of which company?)
   - Incomplete processes: "The reaction produces" (produces what?)
   - Truncated descriptions: "The method involves heating" (heating what? to what temperature?)
   - Unfinished comparisons: "Performance was better" (better than what?)

3. **Contextually Insufficient Information**
   - Vague generalizations: "Technology improves efficiency"
   - Non-specific claims: "Research shows positive results"
   - Ambiguous statements: "The policy had significant impact"
   - Incomplete technical details: "The algorithm uses machine learning"

4. **Utility-Deficient Information**
   - Trivial facts: "The document contains words"
   - Obvious statements: "Water is wet"
   - Non-actionable information: "Things happen"
   - Circular definitions: "Speed is how fast something moves"

5. **Meta Information**
   - Document metadata: "The document title is report.pdf", "The file name is..."
   - URL references: "The report URL is https://...", "This document can be found at..."  
   - Structural metadata: "This document contains 50 pages", "The report has 10 sections"
   - System information: "The file was created on...", "Last modified date is..."

**Examples**:

**Information Completeness Errors**:
- "Discovered the theory of relativity" (who discovered? when?)
- "The merger was announced" (which companies? when? for how much?)
- "Sales increased dramatically" (which company? by how much? over what period?)
- "The experiment was successful" (what experiment? what were the results?)
- "The document title is tesla_2023_annual_report.pdf" (meta information with no substantive value)
- "The SEC filing URL is https://www.sec.gov/..." (document reference, not content knowledge)

**Complete Information Units**:
- "Albert Einstein discovered the theory of relativity in 1905"
- "Disney announced its acquisition of 21st Century Fox for $71.3 billion in December 2017"
- "Amazon's quarterly sales increased by 15% to $125.5 billion in Q3 2023 compared to Q3 2022"
- "The clinical trial demonstrated 95% efficacy for the Pfizer-BioNTech COVID-19 vaccine in preventing symptomatic infection"

**Completeness Assessment Framework**:
- **Essential Elements Test**: Are all necessary WHO/WHAT/WHEN/WHERE/HOW components present?
- **Standalone Value Test**: Can this information be useful without additional context?
- **Specificity Verification**: Are vague terms replaced with specific, measurable details?
- **Actionability Assessment**: Does this information provide meaningful, usable knowledge?
- **Meta Information Filter**: Does this focus on substantive content rather than document metadata?

## Atomic Information to Evaluate
{atomic_info_list}

Let's think step by step.

## Required Output Format

```json
[
  {{
    "atomic_id": 1,
    "reasoning": "Information includes complete identification (Einstein), specific achievement (relativity theory), precise timeframe (1905). All essential elements present for standalone understanding. High utility and actionability. Not meta information. Information completeness: PASS",
    "has_information_completeness_error": false
  }},
  {{
    "atomic_id": 2,
    "reasoning": "Critical missing elements: subject identification absent (who discovered?). Incomplete temporal context. Information cannot stand alone meaningfully. Low utility due to fragmented content. Information completeness: FAIL",
    "has_information_completeness_error": true
  }},
  {{
    "atomic_id": 3,
    "reasoning": "This is document metadata with no substantive knowledge value. Contains only file reference information without actionable or meaningful content for knowledge applications. Meta information filter: FAIL",
    "has_information_completeness_error": true
  }}
]
```

**Evaluation Protocol**:
- Assess presence of all essential informational components
- Verify that information units provide standalone value and clarity
- Check for sufficient specificity and measurable details
- Evaluate overall utility and meaningfulness for knowledge applications
- Filter out document metadata, file references, and structural information

Return exactly {num_atomic} evaluations.
"""

# =============================================================================
# Semantic Redundancy Detection Prompt
# =============================================================================

DETECT_SEMANTIC_REDUNDANCY_PROMPT = """
You are an expert semantic analyst specializing in detecting information redundancy for RAG evaluation datasets.

# Goal
Determine whether multiple information units convey the same core facts, regardless of surface-level differences in expression, terminology, or phrasing.

## Core Requirements
For each comparison, evaluate WHETHER the target information and comparison information represent:

### 1. SEMANTIC EQUIVALENCE
- Same core meaning and conceptual content
- Identical information value regardless of linguistic expression
- Equivalent semantic representation despite different wording

### 2. LOGICAL EQUIVALENCE
- Same logical structure and reasoning pattern
- Identical cause-effect relationships or logical connections
- Equivalent inferential implications and logical consequences

### 3. FACTUAL EQUIVALENCE
- Same factual claims and assertions
- Identical entities, properties, and relationships
- Consistent temporal, spatial, quantitative, and qualitative aspects

## Decision Framework
**REDUNDANT (true)**: Information units convey identical factual content despite different expressions
**UNIQUE (false)**: Information units contain distinct, non-overlapping factual content

## Examples:

**REDUNDANT (true):**
- Target: "The company's revenue increased by 15% in Q3 2023"
- Comparison: "In the third quarter of 2023, company earnings grew fifteen percent"
*SEMANTIC EQUIVALENCE: same core meaning (revenue growth); LOGICAL EQUIVALENCE: identical logical relationship (increase); FACTUAL EQUIVALENCE: same entity, time period, metric, and value*

**UNIQUE (false):**
- Target: "The company's revenue increased by 15% in Q3 2023"  
- Comparison: "The company's expenses decreased by 10% in Q3 2023"
*SEMANTIC EQUIVALENCE: different concepts (revenue vs expenses); LOGICAL EQUIVALENCE: opposite logical direction (increase vs decrease); FACTUAL EQUIVALENCE: different metrics and values*

---

# Target Information
{target_info}

# Comparison Information Units
{comparison_info_list}

# Required Output Format

**CRITICAL: Your response must contain exactly {num_comparisons} items. Each input item must have a corresponding output item with the same comparison_id.**

```json
[
  {{
    "comparison_id": 1,
    "reasoning": "SEMANTIC EQUIVALENCE: identical core meaning of revenue growth; LOGICAL EQUIVALENCE: same logical relationship (positive increase); FACTUAL EQUIVALENCE: same entity (company), time period (Q3 2023), metric (revenue), and value (15%).",
    "is_redundant": true,
  }},
  {{
    "comparison_id": 2,
    "reasoning": "SEMANTIC EQUIVALENCE: different concepts (revenue vs expenses); LOGICAL EQUIVALENCE: opposite logical directions (increase vs decrease); FACTUAL EQUIVALENCE: different financial metrics and quantitative values.",
    "is_redundant": false,
  }}
]
```

**Evaluation Protocol:**
- **SEMANTIC EQUIVALENCE**: Analyze core meaning and conceptual content regardless of linguistic differences
- **LOGICAL EQUIVALENCE**: Examine logical structures, reasoning patterns, and inferential relationships
- **FACTUAL EQUIVALENCE**: Compare entities, properties, relationships, and quantitative/qualitative aspects
- Distinguish between true content equivalence and surface-level similarity
- Account for different terminology and expressions referring to identical information
-  

**CRITICAL: Your response must contain exactly {num_comparisons} items. Each input item must have a corresponding output item with the same comparison_id.**
Only return the JSON array, no other text.
"""


# =============================================================================
# Simple Multi-hop Question Generation Prompt
# =============================================================================

GENERATE_SIMPLE_HOP_QUESTION_PROMPT = """
You are a careful question generation assistant.

Create question-answer pair that relies on every chunk listed below.

## Available Chunks
{chunk_summaries}

Each chunk line already concatenates the file name and the content.

## Instruction
- Use ONLY the provided chunk text. No external knowledge.
- The question must require using all of the provided chunks in order to be answerable.
- The answer must be a concise factual statement supported by the chunks.
- Include the `source_chunks` field listing the chunk indices you used (start from 1 as shown).
- Keep language professional and direct.

## Required JSON Output
Return a single JSON object in the form:
{{
  "question": "...",
  "answer": "...",
  "source_chunks": [chunk_index_1, chunk_index_2, ...]
}}

Only output the JSON object.
"""


# =============================================================================
# Query Decomposition Prompt
# =============================================================================

DECOMPOSE_QUERY_PROMPT = """
You are an expert query decomposition specialist for retrieval-augmented generation systems.

# Objective
Transform a complex query into {num_decomposed} simpler, focused sub-queries that together provide all information needed to answer the original query comprehensively.

## Decomposition Strategy

### 1. COMPONENT ANALYSIS
- Identify distinct information requirements in the original query
- Determine essential facts, entities, relationships, and constraints
- Map logical dependencies between different information components
- Establish the scope and specificity needed for each component

### 2. ATOMIC SUB-QUERY GENERATION
- Create specific, focused questions targeting individual components
- Ensure each sub-query is independently answerable by retrieval systems
- Maintain natural, fluent language that users would employ
- Avoid meta-references or enumeration indicators

### 3. COMPLETENESS VERIFICATION
- Verify all aspects of original query are covered by sub-queries
- Ensure no information gaps exist between sub-queries
- Confirm sub-queries together enable comprehensive answer synthesis
- Check for appropriate level of granularity and specificity

### 4. REDUNDANCY ELIMINATION
- Remove overlapping information requirements across sub-queries
- Ensure each sub-query targets distinct, non-redundant information
- Optimize for minimal yet complete information coverage
- Maintain logical independence between sub-queries

# Input Query
{original_query}

# Required Output Format

```json
{{
  "decomposition_reasoning": "step-by-step reasoning of the decomposition strategy and component mapping",
  "decomposed_queries": [
    "First focused sub-query addressing specific component",
    "Second focused sub-query addressing different component", 
    "Third focused sub-query completing information requirements"
  ],
}}
```

Generate exactly {num_decomposed} decomposed sub-queries that satisfy all quality requirements.
"""

# =============================================================================
# Query Decomposition Count Estimation Prompt
# =============================================================================

ESTIMATE_DECOMPOSITION_COUNT_PROMPT = """
You are an expert query analysis specialist for retrieval-augmented generation systems.

# Objective
Analyze the given complex query and determine the optimal number of focused sub-queries needed to comprehensively answer the original question.

## Analysis Strategy

### 1. COMPLEXITY ASSESSMENT
- Identify distinct information requirements within the query
- Count separate factual components that need independent retrieval
- Assess logical dependencies and relationships between components
- Determine granularity level needed for effective retrieval

### 2. COMPONENT MAPPING
- Map out individual facts, entities, relationships, and constraints
- Identify overlapping vs. independent information needs  
- Consider retrieval efficiency and answer synthesis requirements
- Balance specificity with practical query execution

### 3. OPTIMIZATION CRITERIA
- **Completeness**: All aspects of original query must be covered
- **Independence**: Each sub-query should target distinct information
- **Efficiency**: Minimize redundancy while ensuring thorough coverage
- **Retrievability**: Each component should be answerable by retrieval systems

## Guidelines

**Too Few Sub-queries (Under-decomposition):**
- Results in overly broad sub-queries that are hard to answer precisely
- May miss important nuances or specific information requirements
- Reduces retrieval accuracy and answer quality

**Too Many Sub-queries (Over-decomposition):**
- Creates unnecessary fragmentation of closely related information
- Increases complexity without improving answer quality
- May lead to redundant or overlapping information retrieval

# Input Query
{original_query}

# Required Output Format

```json
{{
  "reasoning": "step-by-step reasoning of the decomposition strategy and component counting",
  "optimal_count": 3,
}}
```

Analyze the query systematically and provide the most appropriate decomposition count.
"""

# =============================================================================
# Question Paraphrasing Prompt
# =============================================================================

PARAPHRASE_QUESTIONS_PROMPT = """
You are an expert question paraphrasing specialist. Your task is to rephrase multiple questions to increase lexical and structural diversity while preserving their exact semantic meaning and answerability.

# Objective
Transform each given question into a semantically equivalent version that:
1. Uses different vocabulary and sentence structures
2. Maintains identical semantic meaning and answering requirements  
3. Reduces word overlap with the original question
4. Remains answerable using the provided context information
5. Reflects realistic user intent and situational context
6. As much as possible, avoid repeating the same words that appear in the response (except for proper names) and instead use synonyms or alternative expressions.

# Input Data
## Questions, Answers, and Related Document Titles
Each question is paired with its answer and the list of document titles containing the required information.

{questions_and_answers}

## All Available Documents  
{document_titles}

# Paraphrasing Guidelines

## 1. Lexical Diversification (Vocabulary Variation)
**Objective: Reduce exact keyword overlap with source documents. Rephrase the question by substituting as many words as possible with synonyms.**

### Numbers & Financial Terms:
- "$18,120 million" → "approximately 18.1 billion dollars" / "about 18 billion USD" / "roughly eighteen billion"
- "25%" → "a quarter" / "one-fourth" / "25 percent"
- "6.25%" → "roughly 6 percent" / "about six and a quarter percent"

### Dates & Time Periods:
- "October 29, 2023" → "end of Q3 2023" / "third quarter 2023" / "late October 2023"
- "three months ended" → "quarter ended" / "Q3 period" / "third quarter"
- "four-year period" → "48-month timespan" / "quadrennial schedule"

### Technical & Business Terms:
- "billing location" → "invoice address" / "payment location" / "where bills are sent"
- "geographic regions" → "geographical areas" / "regional segments" / "territorial divisions"
- "RSU vesting schedule" → "Restricted Stock Unit timeline" / "equity compensation plan" (if an abbreviation appears, replace it with its full name; conversely, if a full name appears, replace it with its abbreviation)
- "revenue" → "sales" / "total income" / "earnings"

## 2. Intent Diversification (Purpose-Driven Questioning)
**Objective: Frame questions with realistic user motivations and contexts. Add a brief description of the employee's situation at the beginning of the question, and write the question in informal, conversational language.**

### Business & Professional Intent:
- "For my quarterly report at the bank, I need the latest data on Oracle's cloud revenue"
- "What regulatory changes are shaping cross-border shipping in Southeast Asia?"

### Academic Research:
- "As part of my thesis on European history, which treaties reshaped the continent after World War I"
- "What consensus algorithms are most commonly used in modern distributed systems?"

### Information-Seeking Intent:
- "I'm considering online learning options for my child—what does the research say about screen time and attention span?"
- "What's the key difference between index funds and ETFs for new investors?"

### Natural Conversation Patterns:
- "I was talking with a friend about classic literature and wondered—when was 1984 first published?"
- "Does the JR Pass cover bullet train rides between Tokyo and Osaka?"

## Semantic Preservation
- **CRITICAL**: Maintain identical meaning and answering requirements
- **CRITICAL**: As much as possible, avoid repeating the same words that appear in the response (except for proper names) and instead use synonyms or alternative expressions.
- Preserve all essential information needed for answering
- Keep the same scope and specificity
- Ensure the paraphrased question can be answered with the same information
- Verify consistency with provided answers and document titles

# Required Output Format

**CRITICAL: Your response must contain exactly {num_questions} items. Each input question must have a corresponding paraphrased output.**

Let's think step by step.

```json
[
  {{
    "question_id": 1,
    "paraphrasing_strategy": "Applied [Lexical/Intent] Variation: 1. specific vocabulary changes, 2. intent context added",
    "paraphrased_question": "Your paraphrased version of question 1"
  }},
  {{
    "question_id": 2,
    "paraphrasing_strategy": "Applied [Lexical/Intent] Variation: 1. specific vocabulary changes, 2. intent context added",
    "paraphrased_question": "Your paraphrased version of question 2"
  }}
]
```

**Requirements:**
- **MANDATORY**: Apply all three strategies (Lexical Diversification, Intent Diversification) to each question
- **TARGET**: Reduce keyword overlap with source documents by 40-60% while preserving meaning
- Generate exactly ONE paraphrased version per input question
- Transform numbers, dates, and technical terms using provided vocabulary alternatives
- Completely restructure sentence patterns and question formats
- Add realistic business/research/professional intent context
- Maintain semantic equivalence with provided answers and document titles  
- Ensure questions remain fully answerable using the same document information
- Return only the JSON array with the exact format shown above

**CRITICAL: Your response must contain exactly {num_questions} items. Each input question must have a corresponding paraphrased output.**
Only return the JSON array, no other text.
"""


# =============================================================================
# Question Answerability Filtering Prompt  
# =============================================================================

FILTER_ANSWERABILITY_PROMPT = """
You are an expert reading comprehension analyst. Your task is to determine whether each given question can be completely and accurately answered using ONLY the provided document chunks.

# Objective
For each question, analyze if it can be fully answered based solely on the information present in the given document chunks, without requiring external knowledge or additional information.

# Questions to Evaluate
{questions_list}

# Available Document Chunks
Each chunk is provided with its document title for context:

{chunks_with_titles}

# Evaluation Criteria

## ANSWERABLE ("pass")
A question is answerable if:
- All necessary information to construct a complete answer is present across the chunks
- The answer can be directly extracted or reasonably inferred from the provided text
- No critical information gaps exist that would prevent complete answering
- The question's scope matches the information coverage in the available chunks
- Multiple chunks can be combined to provide a comprehensive answer

## NOT ANSWERABLE ("fail")  
A question is NOT answerable if:
- Key information required for answering is missing from all provided chunks
- The question requires knowledge not present in the provided documents
- Only partial information exists, preventing a complete and accurate answer
- The question scope exceeds what can be addressed with available content

# Analysis Guidelines
1. **Question Analysis**: For each question, identify what specific information is needed
2. **Cross-Chunk Review**: Examine all chunks for relevant information 
3. **Information Integration**: Consider how information from multiple chunks can be combined
4. **Gap Assessment**: Check for any missing critical information
5. **Answerability Decision**: Determine if complete answering is possible for each question

# Required Output Format

**CRITICAL: Your response must contain exactly {num_questions} items. Each input question must have a corresponding answerability result.**

```json
[
  {{
    "question_id": 1,
    "reasoning": "Detailed explanation of analysis for question 1 - what information is present/missing across all chunks and why this question is/isn't answerable. Reference specific chunks and titles.",
    "generated_answer": "Generated answer based on the information present in the chunks",
    "answerability_check": "pass"/"fail"
  }},
  {{
    "question_id": 2,
    "reasoning": "Detailed explanation of analysis for question 2 - what information is present/missing across all chunks and why this question is/isn't answerable. Reference specific chunks and titles.",
    "generated_answer": "Generated answer based on the information present in the chunks",
    "answerability_check": "pass"/"fail"
  }}
]
```

**Critical Requirements:**
- Base judgment ONLY on provided chunk content and document titles
- Consider information integration across multiple chunks for each question
- Do not rely on external knowledge or assumptions  
- Provide clear reasoning referencing specific chunks/documents for each question
- Be conservative: if any essential information is missing for a question, mark as NOT answerable
- Each question should be evaluated independently

**CRITICAL: Your response must contain exactly {num_questions} items. Each input question must have a corresponding answerability result.**
Only return the JSON array, no other text.
"""