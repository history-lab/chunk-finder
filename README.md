# HistoryLab Vector Search API

A Cloudflare Worker providing semantic search across ~5 million declassified historical documents.

## Quick Start

**Base URL:** `https://vector-search-worker.nchimicles.workers.dev`

### Search for Documents

```bash
curl -X POST "https://vector-search-worker.nchimicles.workers.dev/api/search" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": "Cuban Missile Crisis",
    "topK": 5
  }'
```

### Fetch Full Document

```bash
curl -X GET "https://vector-search-worker.nchimicles.workers.dev/api/document/{r2Key}" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Documentation

See **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)** for complete documentation including:

- Authentication
- All endpoints and parameters
- Filtering (date ranges, document IDs)
- Response schemas
- Code examples (curl, Python)
- Query tips

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Semantic search |
| `/api/document/{r2Key}` | GET | Fetch full document |
| `/api/health` | GET | Health check |
| `/api/help` | GET | API docs (JSON) |

## Key Features

- **Semantic Search** - Natural language queries using BGE embeddings
- **Date Filtering** - Filter by year/month or exact date (YYYYMM or YYYYMMDD)
- **Full Document Retrieval** - Get complete document text and metadata
- **Multiple Queries** - Search multiple concepts in one request

## Document Sources

~5 million documents from:
- CIA declassified documents
- State Department Central Foreign Policy Files
- Foreign Relations of the United States (FRUS)
- Presidential Daily Briefings
- NATO, UN, World Bank archives
- And more...

## Development

### Prerequisites

- Node.js 18+
- Wrangler CLI

### Deploy

```bash
wrangler deploy
```

### Set API Key

```bash
wrangler secret put API_SECRET_KEY_2
```

## Technical Stack

- **Runtime:** Cloudflare Workers
- **Embeddings:** `@cf/baai/bge-base-en-v1.5` (768 dimensions)
- **Vector DB:** Cloudflare Vectorize
- **Storage:** Cloudflare R2 + KV
