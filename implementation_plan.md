# Redesign ResearchRadar UI and Retrieval Logic

## Goal Description

The user wants a cleaner, more responsive chat UI similar to ChatGPT, with a simple black/grey/white theme. The UI should allow uploading multiple PDFs and asking questions limited to the content of those PDFs. Retrieval should use a similarity score threshold (e.g., 0.75) to decide whether to answer from the PDFs or fall back to general LLM knowledge, clearly labeling fallback answers.

## User Review Required

- **Color palette**: Confirm the exact shades for black, dark grey, light grey, and white. Suggested HSL values are provided, but the user may want tweaks.
- **Fallback label wording**: Proposed label is `⚠️ (Fallback)`. Confirm if this is acceptable or if a different wording/style is preferred.
- **Threshold value**: Default set to `0.75`. User may want to adjust.

> [!IMPORTANT]
> Ensure the redesign does not break existing functionality (PDF upload, indexing, chat history, sentiment badges).

## Open Questions

> [!WARNING]
> - Do you want any additional UI elements (e.g., a dark mode toggle) beyond the static black/grey/white theme?
> - Should the sentiment badges be retained, removed, or restyled?
> - Do you prefer the chat messages to have rounded corners or a more rectangular look?

## Proposed Changes

---
### UI Redesign (app.py)
- Replace the custom CSS gradient background with a solid dark background (`#121212`).
- Simplify sidebar styling: solid dark grey (`#1e1e1e`) with white text.
- Update chat bubbles:
  - User messages: background `#2d2d2d`, text `#e0e0e0`.
  - Assistant messages: background `#ffffff`, text `#000000`.
  - Add subtle box‑shadow for depth.
- Remove unnecessary gradients, paddings, and decorative elements.
- Use Google Font **Inter** (already imported) for consistency.
- Adjust header to a plain centered title with no gradient text.
- Reduce visual clutter: hide sentiment badges for now (optional – can be re‑added later).
- Ensure the layout remains responsive (`st.set_page_config(layout="wide")`).

---
### Retrieval Score Handling (core/query_router.py & core/retriever.py)
- Introduce a helper function `apply_score_threshold(results, threshold=0.75)` that returns either the top chunks with a flag indicating whether the score is acceptable.
- In `run_query_pipeline` (app.py), after retrieving chunks, compute `top_score = results["distances"][0][0]` (assuming the existing retriever returns this structure).
- If `top_score < threshold`, proceed with LLM answer as usual.
- Else, generate a fallback answer using the LLM with a prefix like `"⚠️ (Fallback) "` and set `mode = "fallback"` for UI display.
- Update `assistant-msg` rendering to include `mode_reason` when fallback is used.

---
### Minor Adjustments
- Update any CSS class names referenced in the Python code to match new styles (`user-msg`, `assistant-msg`).
- Ensure the sidebar upload component matches the new dark theme (white icons, grey borders).
- Add a small disclaimer under the chat input: "Answers are based on uploaded PDFs unless indicated otherwise."

## Verification Plan

### Automated Tests
- Run `streamlit run app.py` and manually verify:
  1. UI colors match the specified palette.
  2. PDFs can be uploaded and indexed without errors.
  3. Asking a question that matches PDF content returns an answer without the fallback label.
  4. Asking an unrelated question returns a fallback answer prefixed with the chosen label.
- Check the console for any CSS errors or missing class references.

### Manual Verification
- Ask several domain‑specific questions to ensure the retrieval threshold works.
- Capture screenshots of the UI (sidebar, chat bubbles, fallback labeling) for visual confirmation.
