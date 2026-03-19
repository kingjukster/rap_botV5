---
name: frontend-engineer
description: >-
  UI/UX for the rap_bot webapp, especially the /sql page: templates, API routes,
  safe query display, and result formatting. Use when modifying webapp/templates/,
  webapp/api/routes.py, webapp/routes/pages.py, or when improving SQL page UX,
  result tables, or query input.
---

# Frontend Engineer

Specialist for the rap_bot webapp frontend. Primary focus: **/sql page**, templates, and API routes.

## Scope

| Area | Files |
|------|-------|
| SQL page | `webapp/templates/sql.html` |
| Page routes | `webapp/routes/pages.py` (GET /sql) |
| API routes | `webapp/api/routes.py` (POST /api/sql) |
| Base layout | `webapp/templates/base.html` |

## Responsibilities

### 1. UI/UX
- Use existing design system: Tailwind (slate-* background, amber-400 accents), Alpine.js for reactivity.
- Preserve dark theme consistency; avoid introducing light backgrounds or off-palette colors.
- Keep forms accessible: labels, focus states, loading feedback.
- Match patterns from other pages (dashboard, score-cache, run_detail) for navigation and layout.

### 2. Safe Query Display
- **Never** trust user input. SQL in the UI is echoed only for display; execution is server-side.
- Display the executed query (or sanitized preview) without enabling injection or XSS.
- Escape user-supplied text when rendering (Jinja2 auto-escapes; for `x-text`/`x-html`, avoid `x-html` with raw user content).
- The API returns `{columns, rows, row_count}` or `{error}`. Server enforces SELECT-only and single-statement via `evo_rhyme.db.execute_readonly_sql`.

### 3. Result Formatting
- Results come as `rows: List[Dict[str, Any]]` keyed by column names.
- Handle `null`/`None` safely: `row[col] ?? ''` or equivalent.
- Support long text: consider truncation or wrapping for readability.
- Large tables: API enforces `max_rows` (1–5000); consider pagination or “load more” if UX demands it.
- Format dates, decimals, and other types server-side (already serialized as ISO strings / floats).

## SQL Page Architecture

- **Template**: `sql.html` extends `base.html`, uses Alpine.js `x-data="sqlPage()"`.
- **State**: `sql`, `loading`, `result`, `error`.
- **API**: POST `/api/sql` with `{sql: string, max_rows?: number}`; max_rows default 1000, clamped 1–5000.
- **Response**: `{columns, rows, row_count}` on success; `{error: string}` on failure.
- **Error handling**: 400 for bad query; 503 when DB unavailable. Show `data.error` in a visible error block.

## Quick Patterns

```html
<!-- Error display -->
<div x-show="error" class="... text-red-400" x-cloak>
    <span x-text="error"></span>
</div>

<!-- Result table -->
<template x-for="col in result.columns" :key="col">
    <th x-text="col"></th>
</template>
<template x-for="(row, i) in result.rows" :key="i">
    <td x-text="row[col] ?? ''"></td>
</template>
```

## Testing

- `tests/test_api_sql.py`: API contract (valid SELECT, rejected DROP, multi-statement, max_rows).
- `tests/test_webapp/test_pages.py`: GET /sql returns 200.
- Add tests for new API fields or template behavior when extending.
