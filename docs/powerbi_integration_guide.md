# 🔬 ResearchRadar v2 — Power BI Integration Guide

This guide explains how to connect, relate, and visualize the 5 CSV datasets exported from ResearchRadar's analysis engine to build a premium, multi-page analytics dashboard in **Power BI Desktop**.

---

## 🛠️ Step 1: Loading the 5 Datasets

1. Open **Power BI Desktop**.
2. Click **Get Data** → **Text/CSV**.
3. Select and load each of the 5 files in `F:\Major Projects\Researchradar\data\exports\`:
   - `sentiment_analysis.csv`
   - `paper_similarity.csv`
   - `section_similarity.csv`
   - `topic_model.csv`
   - `keyword_scores.csv`
4. Verify they appear as 5 separate tables in the **Fields** pane on the right.

---

## 🧠 Step 2: Setting Up the Data Model (Star Schema)

To enable cross-filtering and slicing by Category, Organization, or Date across all tables, we will create a central **`Papers` Dimension Table** using DAX.

### 1. Create the `Papers` Calculated Table
1. In Power BI Desktop, click **Table Tools** or **Modeling** → **New Table**.
2. Paste the following DAX formula to extract unique paper metadata automatically:
   ```dax
   Papers = 
   SUMMARIZE(
       sentiment_analysis, 
       sentiment_analysis[paper_title], 
       sentiment_analysis[author_org], 
       sentiment_analysis[category], 
       sentiment_analysis[date]
   )
   ```
3. Press **Enter**. This creates a clean dimension table.

### 2. Configure Table Relationships
Switch to the **Model View** (the relationship icon on the left sidebar) and draw the following **One-to-Many (1 to *)** relationships:

1. **Papers to Sentiment**:
   - Link `Papers[paper_title]` $\rightarrow$ `sentiment_analysis[paper_title]`
   - Cardinality: **One to Many (1:*)**
   - Cross filter direction: **Single** (or **Both**)
2. **Papers to Keywords**:
   - Link `Papers[paper_title]` $\rightarrow$ `keyword_scores[paper_title]`
   - Cardinality: **One to Many (1:*)**
3. **Papers to Paper Similarity**:
   - Link `Papers[paper_title]` $\rightarrow$ `paper_similarity[paper_1]`
   - Cardinality: **One to Many (1:*)**
4. **Papers to Section Similarity**:
   - Link `Papers[paper_title]` $\rightarrow$ `section_similarity[paper1]`
   - Cardinality: **One to Many (1:*)**

---

## 📊 Step 3: Designing the 5 Dashboard Pages

### 💻 Dashboard 1 — Corpus Overview
* **Goal**: Provide a high-level summary of the entire indexed paper library.
* **Layout**:
  - **KPI Cards**:
    - `Papers Count` $\rightarrow$ Card visual with DAX measure: `Papers Count = DISTINCTCOUNT(Papers[paper_title])`
    - `Chunks Count` $\rightarrow$ Card visual: count of `sentiment_analysis[chunk_index]`
    - `Avg Similarity` $\rightarrow$ Card visual: average of `paper_similarity[similarity_score]` (filter out self-similarity score = 1.0)
  - **Slicers (Filter Chips)**:
    - Slicer 1: `Papers[author_org]` (Vertical List or Tile format)
    - Slicer 2: `Papers[category]` (Vertical List or Tile format)
  - **Chunks per Paper (Bar Chart)**:
    - Visual: Clustered Bar Chart.
    - Y-Axis: `Papers[paper_title]`.
    - X-Axis: Count of `sentiment_analysis[chunk_index]`.
  - **Paper Index (Table)**:
    - Visual: Table.
    - Columns: `Papers[paper_title]`, `Papers[author_org]`, `Papers[category]`, `Papers[date]`, and Count of Chunks.

---

### 🟢 Dashboard 2 — Sentiment Deep Dive
* **Goal**: Analyze the tones and rhetorical styles of the research.
* **Layout**:
  - **Overall Sentiment Distribution (Donut Chart)**:
    - Legend: `sentiment_analysis[label]`
    - Values: Count of `sentiment_analysis[chunk_index]`
  - **Sentiment by Research Category (Stacked Bar Chart)**:
    - Y-Axis: `Papers[category]`
    - X-Axis: Count of chunks (Percentage of Grand Total)
    - Legend: `sentiment_analysis[label]`
  - **Sentiment Trajectory Flow (Line Chart)**:
    - Axis (X): `sentiment_analysis[chunk_index]`
    - Values (Y): Average of `sentiment_analysis[score]`
    - Legend: `Papers[paper_title]`
    - *Insight*: This shows exactly where the paper shifts from optimistic (intro/results) to critical/cautious (limitations).

---

### 🌐 Dashboard 3 — Semantic Similarity Matrices
* **Goal**: Identify conceptual overlaps and related work automatically.
* **Layout**:
  - **Paper-to-Paper Heatmap (Matrix Visual)**:
    - Rows: `paper_similarity[paper_1]`
    - Columns: `paper_similarity[paper_2]`
    - Values: `paper_similarity[similarity_score]`
    - *Format*: Under Cell Elements, turn on **Background Color** conditional formatting. Use a blue gradient (Light Blue for low, Deep Indigo for high) to create a heatmap.
  - **Section-to-Section Heatmap (Matrix Visual)**:
    - Rows: `section_similarity[paper1_section]`
    - Columns: `section_similarity[paper2_section]`
    - Values: `section_similarity[score]`
    - *Slicers*: Add dropdown slicers for `section_similarity[paper1]` and `section_similarity[paper2]` to let users select two papers and inspect section alignments.

---

### 🔑 Dashboard 4 — Topics & Keywords
* **Goal**: Map semantic keywords and unsupervised topic clusters.
* **Layout**:
  - **Unsupervised BERTopic Themes (Treemap or Bar Chart)**:
    - Category: `topic_model[keywords]`
    - Values: Sum of `topic_model[chunk_count]`
  - **Keyword Relevance Heatmap (Matrix Visual)**:
    - Rows: `keyword_scores[paper_title]`
    - Columns: `keyword_scores[keyword]`
    - Values: `keyword_scores[relevance_score]`
    - *Format*: Turn on conditional formatting background gradient to easily scan which keywords belong to which papers.

---

### ⚔️ Dashboard 5 — Debate Prep View
* **Goal**: View extracted claims and claim strengths before debate rounds.
* **Layout**:
  - **Claims Index Table (Table Visual)**:
    - Columns: `sentiment_analysis[paper_title]`, `sentiment_analysis[section]`, `sentiment_analysis[chunk_index]`, `sentiment_analysis[chunk_sentiment_score]`, `sentiment_analysis[text]`
    - *Slicers*: Filter by `Papers[paper_title]` and `sentiment_analysis[label]` to view only "critical" claims or "optimistic" assertions.
