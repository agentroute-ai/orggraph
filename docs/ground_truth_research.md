# Ground-Truth Organizational Structure in Email Corpora

## Research Question

Do the Enron or Avocado email corpora contain ground-truth information about employee roles and organizational structure that can serve as a gold standard for evaluating unsupervised relationship extraction models?

---

## Executive Summary

| Dataset     | Ground-Truth Available? | Quality | Accessibility              |
| ----------- | ----------------------- | ------- | -------------------------- |
| **Enron**   | ✅ YES - Comprehensive  | High    | Free (ACL Anthology)       |
| **Avocado** | ⚠️ LIMITED - Partial    | Medium  | Requires manual extraction |

**Recommendation:** Use **Enron with the Agarwal et al. (2012) gold standard** for validating organizational hierarchy extraction. This provides 13,724 annotated dominance pairs with high reliability.

---

## Enron Corpus: Ground-Truth Resources

### 1. Agarwal et al. (2012) Gold Standard ⭐ BEST OPTION

**Citation:** Agarwal, A., Omuya, A., Harnly, A., & Rambow, O. (2012). _A Comprehensive Gold Standard for the Enron Organizational Hierarchy_. ACL 2012.

**What it provides (verified from the paper, 2026-05-07):**

- **1,518 employees** total: 158 "core" (maildir custodians with inboxes) + 1,360 "non-core" (people who only appear as senders/recipients in core inboxes)
- **2,155 immediate manager-subordinate edges** annotated from Enron organizational charts filed with FERC
- **13,724 transitive-closure dominance pairs** derived from those edges
- Pair breakdown (Table 1 of paper): 440 Core-Core, 6,436 Inter (one core, one non-core), 6,847 Non-Core-Non-Core
- Thread structure reconstruction (from Yeh & Harnly, 2006)
- Available as MongoDB database

**Access:**

- Paper: https://aclanthology.org/P12-2032/
- Companion: http://www.cs.columbia.edu/~rambow/enron/
- The MongoDB database is "available by contacting the authors" (paper §3, p. 163) - not direct download. Email Owen Rambow at Columbia.

**Reported results (corrected, the paper reports accuracy NOT F1):**

| Method | All 13,724 | Core-Core (440) | Inter | Non-Core |
|---|---|---|---|---|
| SNA degree centrality | **83.88%** | **79.31%** | 93.75% | 74.57% |
| NLP upper bound | 59.61% | - | - | - |
| Random | 50% | - | - | - |

**Common mis-attribution:** The figure "F1 = 0.70" does NOT appear in this paper. If you see it cited, the attribution is wrong.

### 2. Enron Employee Status Report

**Source:** ISI/USC Enron Dataset
**URL:** http://www.isi.edu/~adibi/Enron/Enron_Employee_Status.xls

**What it provides:**

- Employee names
- Job titles
- Department affiliations
- Employment status

### 3. FERC TRADER Dataset

**What it provides:**

- Organization chart for 47 members of Enron's North American West Power Trading division
- Source: McCullough Research
- Used in Creamer et al. (2022) for validation

### 4. Executive Compensation Data

**Source:** U.S. Congress Joint Committee on Taxation, Towers Perkin reports
**Use:** Proxy for organizational importance/hierarchy level

### 5. Known Network Properties (for validation)

| Metric                 | Enron Value | Source                |
| ---------------------- | ----------- | --------------------- |
| Clustering coefficient | 0.497       | Diesner et al. (2005) |
| Effective diameter     | 4.8         | Multiple studies      |
| Core employees         | 145-158     | CMU version           |

---

## Avocado Corpus: Ground-Truth Limitations

### What IS available:

1. **Account types** (from metadata):
   - Employee accounts (most of the 279)
   - Shared accounts (e.g., "Leads")
   - System accounts (e.g., "Conference Room Upper Canada")

2. **Contact lists and address books** (unexplored):
   - May contain organizational information
   - Requires manual extraction and analysis

3. **Folder structure metadata** (XML):
   - May indicate departmental organization
   - Requires inference

### What is NOT available:

- ❌ Pre-annotated organizational hierarchy
- ❌ Job titles or roles
- ❌ Department assignments
- ❌ Reporting relationships
- ❌ Gold standard annotations

### Partial Solutions for Avocado:

From Zhang et al. (CSCW 2020) "Configuring Audiences":

- Developed procedures to associate names with email addresses
- Inferred job titles at coarse granularity
- Processing code available: https://github.com/tisjune/avocado-data-processing

**Limitation noted by authors:** "More work would be needed to infer precise ranks or org chart-style information."

---

## Recommended Validation Strategy

### Phase 1: Develop with Enron (Ground-Truth Available)

1. Implement unsupervised hierarchy extraction on Enron corpus
2. Validate against Agarwal et al. (2012) gold standard
3. Establish baseline metrics:
   - Precision/recall for dominance relation prediction
   - Accuracy of hierarchy level assignment
   - F1 score for relationship classification

### Phase 2: Apply to Avocado (No Ground-Truth)

1. Apply validated model to Avocado
2. Use qualitative evaluation (human judges)
3. Compare extracted structure against email metadata patterns
4. Consider creating limited annotations for subset validation

### Alternative Validation Approaches:

| Method               | Description                                   | Feasibility |
| -------------------- | --------------------------------------------- | ----------- |
| Cross-validation     | Train on Enron, test organizational patterns  | High        |
| Network metrics      | Compare clustering, centrality distributions  | High        |
| Human evaluation     | Business students judge extracted hierarchies | Medium      |
| Synthetic validation | Use MATRIX to generate known-structure data   | High        |

---

## Key Papers Using Ground-Truth Data

1. **Agarwal et al. (2012)** - Created the gold standard
2. **Creamer et al. (2022)** - CorpRank algorithm validated against org charts
3. **Gilbert (2012)** - "Phrases that signal workplace hierarchy"
4. **Hardin et al. (2015)** - Centrality measures vs. hierarchy
5. **Prabhakaran & Rambow (2014)** - Power relations from language

---

## Conclusion

**Summary:** The Enron corpus with the Agarwal et al. (2012) gold standard is the clear choice for validating an unsupervised organizational structure extraction model. This provides:

1. **Quantitative evaluation** - 13,724 labeled relationship pairs
2. **Reproducibility** - Well-documented, freely available
3. **Comparability** - Many prior studies use this benchmark
4. **Completeness** - Covers hierarchy, not just connections

For Avocado, you can demonstrate generalizability through qualitative evaluation and network property comparison, but cannot provide the same rigorous quantitative validation due to lack of ground-truth annotations.
