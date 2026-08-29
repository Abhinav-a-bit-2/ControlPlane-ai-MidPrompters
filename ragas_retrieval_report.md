# 📊 Ragas Vector Retrieval Quality Report

## Executive Summary
- **Evaluated Queries:** 5
- **Mean Context Precision (MAP):** `0.733` (Target: `≥0.60`)
- **Mean Context Recall (Reference-Free):** `0.700` (Target: `≥0.60`)
- **Average Signal-to-Noise Ratio:** `40.0%`
- **Retrieval Sufficiency Rate:** `60.0%`

## Metric Definitions
- **Context Precision:** Measures whether retrieved chunks are relevant to the query and penalizes ranking irrelevant chunks above relevant ones (Mean Average Precision).
- **Context Recall (Reference-Free):** Decomposes the user query into key informational requirements and calculates the percentage covered by the retrieved contexts, without requiring human reference answers.

## Evaluation Matrix

| # | Query | Precision | Recall | S/N Ratio | Sufficiency | Verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | In the beginning, what did God create? | `1.00` | `1.00` | `33%` | ✅ PASS | **EXCELLENT** |
| 2 | What was upon the face of the deep before light was ... | `0.83` | `1.00` | `67%` | ✅ PASS | **EXCELLENT** |
| 3 | Who was the husbandman who killed his shepherd broth... | `1.00` | `1.00` | `33%` | ✅ PASS | **EXCELLENT** |
| 4 | What were the exact dimensions of Noah's ark, and ho... | `0.83` | `0.50` | `67%` | ❌ FAIL | **POOR** |
| 5 | What is the capital city of Australia and what are i... | `0.00` | `0.00` | `0%` | ❌ FAIL | **RETRIEVAL_MISS** |

## Detailed Diagnostics per Query

### Query 1: In the beginning, what did God create?
**Verdict:** `EXCELLENT` | **Precision:** `1.00` | **Recall:** `1.00` | **Latency:** `2060.8ms`

**Retrieved Chunks Evaluation:**

| Rank | Chunk ID | Useful? | Justification | Snippet |
|:---:|:---:|:---:|---|---|
| 1 | `chunk-2` | ✅ Yes | Contains the verse "In the beginning God created heaven, and earth." which directly answers the question. | *"Genesis Chapter 1 God createth Heaven and Earth, and all things therein, in six days.  1:1. In the beginning God created..."* |
| 2 | `chunk-10` | ❌ No | Describes creation of whales and birds, which occurs later in Genesis, not at the beginning. | *"1:21. And God created the great whales, and every living and moving creature, which the waters brought forth, according ..."* |
| 3 | `chunk-11` | ❌ No | Describes creation of animals, which also occurs later in Genesis, not at the beginning. | *"1:24. And God said: Let the earth bring forth the living creature in its kind, cattle and creeping things, and beasts of..."* |

**Query Requirement Coverage (Recall Breakdown):**

- **✅ [Covered]:** *Heaven and earth* — Context explicitly states 'In the beginning God created heaven, and earth.'

---
### Query 2: What was upon the face of the deep before light was made, and what did God name the light?
**Verdict:** `EXCELLENT` | **Precision:** `0.83` | **Recall:** `1.00` | **Latency:** `2337.4ms`

**Retrieved Chunks Evaluation:**

| Rank | Chunk ID | Useful? | Justification | Snippet |
|:---:|:---:|:---:|---|---|
| 1 | `chunk-3` | ✅ Yes | Contains the naming of light as "Day" and addresses the second part of the question. | *"1:5. And he called the light Day, and the darkness Night; and there was evening and morning one day.  1:6. And God said:..."* |
| 2 | `chunk-8` | ❌ No | Discusses celestial bodies but does not mention the face of the deep or naming of light. | *"Two great lights.... God created on the first day, light, which being moved from east to west, by its rising and setting..."* |
| 3 | `chunk-2` | ✅ Yes | Describes darkness upon the face of the deep before light was made, addressing the first part of the question. | *"Genesis Chapter 1 God createth Heaven and Earth, and all things therein, in six days.  1:1. In the beginning God created..."* |

**Query Requirement Coverage (Recall Breakdown):**

- **✅ [Covered]:** *What was upon the face of the deep before light was made* — Context states darkness was upon the face of the deep in Genesis 1:2.
- **✅ [Covered]:** *What did God name the light* — Context states God called the light Day in Genesis 1:5.

---
### Query 3: Who was the husbandman who killed his shepherd brother Abel?
**Verdict:** `EXCELLENT` | **Precision:** `1.00` | **Recall:** `1.00` | **Latency:** `21260.6ms`

**Retrieved Chunks Evaluation:**

| Rank | Chunk ID | Useful? | Justification | Snippet |
|:---:|:---:|:---:|---|---|
| 1 | `chunk-42` | ✅ Yes | Chunk 1 identifies Abel as a shepherd and Cain as a husbandman, providing the necessary context that the husbandman who killed Abel is Cain. | *"3:24. And he cast out Adam: and placed before the paradise of pleasure Cherubims, and a flaming sword, turning every way..."* |
| 2 | `chunk-48` | ❌ No | Chunk 2 discusses guilt and mentions Cain and Abel but does not directly state that the husbandman killed Abel. | *"Every one that findeth me shall kill me.... His guilty conscience made him fear his own brothers and nephews; of whom, b..."* |
| 3 | `chunk-54` | ❌ No | Chunk 3 describes Lamech killing Cain, which is unrelated to the question. | *"I have slain a man, etc.... It is the tradition of the Hebrews, that Lamech in hunting slew Cain, mistaking him for a wi..."* |

**Query Requirement Coverage (Recall Breakdown):**

- **✅ [Covered]:** *Name of the husbandman who killed his shepherd brother Abel* — Context identifies Cain as the husbandman and states he killed Abel.

---
### Query 4: What were the exact dimensions of Noah's ark, and how many years did Abraham live?
**Verdict:** `POOR` | **Precision:** `0.83` | **Recall:** `0.50` | **Latency:** `30877.2ms`

**Retrieved Chunks Evaluation:**

| Rank | Chunk ID | Useful? | Justification | Snippet |
|:---:|:---:|:---:|---|---|
| 1 | `chunk-74` | ✅ Yes | Provides ark dimensions (length 300 cubits, area 450,000 square cubits) | *"Three hundred cubits, etc.... The ark, according to the dimensions here set down, contained four hundred and fifty thous..."* |
| 2 | `chunk-84` | ❌ No | Describes flood duration, not ark dimensions or Abraham's lifespan | *"7:17. And the flood was forty days upon the earth: and the waters increased, and lifted up the ark on high from the eart..."* |
| 3 | `chunk-73` | ✅ Yes | Specifies ark dimensions: length 300 cubits, breadth 50 cubits, height 30 cubits | *"6:14. Make thee an ark of timber planks: thou shalt make little rooms in the ark, and thou shalt pitch it within and wit..."* |

**Query Requirement Coverage (Recall Breakdown):**

- **✅ [Covered]:** *Exact dimensions of Noah's ark (length, breadth, height)* — Context specifies 300 cubits length, 50 cubits breadth, and 30 cubits height.
- **❌ [Missing]:** *Number of years Abraham lived* — Context does not mention Abraham or his lifespan.

---
### Query 5: What is the capital city of Australia and what are its geographic coordinates?
**Verdict:** `RETRIEVAL_MISS` | **Precision:** `0.00` | **Recall:** `0.00` | **Latency:** `19129.5ms`

**Retrieved Chunks Evaluation:**

| Rank | Chunk ID | Useful? | Justification | Snippet |
|:---:|:---:|:---:|---|---|
| 1 | `chunk-23` | ❌ No | Chunk does not contain information about Australia or its capital. | *"2:10. And a river went out of the place of pleasure to water paradise, which from thence is divided into four heads.  2:..."* |
| 2 | `chunk-85` | ❌ No | Chunk does not contain information about Australia or its capital. | *"7:20. The water was fifteen cubits higher than the mountains which it covered.  7:21. And all flesh was destroyed that m..."* |
| 3 | `chunk-4` | ❌ No | Chunk does not contain information about Australia or its capital. | *"1:7. And God made a firmament, and divided the waters that were under the firmament, from those that were above the firm..."* |

**Query Requirement Coverage (Recall Breakdown):**

- **❌ [Missing]:** *Capital city of Australia* — Context does not mention Australia or its capital city.
- **❌ [Missing]:** *Geographic coordinates of the capital city of Australia* — Context does not provide any geographic coordinates.

---