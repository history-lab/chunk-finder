# HistoryLab Search API Documentation

## Overview

The HistoryLab Search API provides programmatic access to semantic search across nearly 5 million declassified historical documents (18+ million pages). The API uses vector embeddings to find semantically relevant documents based on natural language queries.

**Base URL:** `https://vector-search-worker.nchimicles.workers.dev`

**Collection ID:** `80650a98-fe49-429a-afbd-9dde66e2d02b` (optional - this is the default)

---

## Authentication

All API endpoints require Bearer token authentication.

```
Authorization: Bearer YOUR_API_KEY
```

Include this header in every request.

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Semantic search across documents |
| `/api/document/{r2Key}` | GET | Fetch full document text and metadata |
| `/api/health` | GET | Health check |
| `/api/help` | GET | API documentation (JSON) |
| `/api/about` | GET | Technical architecture details |

---

## Search Endpoint

### `POST /api/search`

Perform semantic search across the historical document collection.

### Request Body

```json
{
  "queries": "string or array of strings",
  "topK": 5,
  "filters": { }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `queries` | string \| string[] | Yes | Search query text. Use natural language for best results. |
| `collection_id` | string | No | Defaults to `80650a98-fe49-429a-afbd-9dde66e2d02b` (HistoryLab). You can omit this. |
| `topK` | number | No | Number of documents to return (1-100, default: 5) |
| `filters` | object | No | Filter criteria (see Filtering section) |

### Response

```json
{
  "status": "success",
  "documents": [
    {
      "document_id": "d6055ebe-6139-41cc-9135-dbd4cb5845d6",
      "best_score": 0.85,
      "chunks": [
        {
          "id": "chunk-uuid",
          "text": "The chunk text content that matched the query...",
          "score": 0.85,
          "metadata": {
            "authored_year_month": 196210,
            "authored_year_month_day": 19621015,
            "doc_id": "CIA-RDP79T00429A001400010019-1",
            "file_id": "d6055ebe-6139-41cc-9135-dbd4cb5845d6",
            "file_info": {
              "id": "d6055ebe-6139-41cc-9135-dbd4cb5845d6",
              "name": "document.txt",
              "size": 1728,
              "type": "text/plain",
              "metadata": {
                "doc_id": "CIA-RDP79T00429A001400010019-1",
                "corpus": "cia",
                "classification": "SECRET",
                "title": "Document Title",
                "date": "1962-10-15T00:00:00+00:00",
                "source": "https://www.cia.gov/..."
              }
            }
          }
        }
      ],
      "file_info": { ... }
    }
  ],
  "total_chunks": 10
}
```

### Response Fields

| Field | Description |
|-------|-------------|
| `status` | `"success"`, `"partial_success"`, or `"error"` |
| `documents` | Array of matching documents, sorted by best score |
| `documents[].document_id` | Unique file identifier |
| `documents[].best_score` | Highest chunk similarity score (0-1) |
| `documents[].chunks` | Array of matching text chunks from this document |
| `documents[].chunks[].text` | The actual text content |
| `documents[].chunks[].score` | Similarity score (0-1, higher = more relevant) |
| `documents[].file_info` | Document metadata |
| `total_chunks` | Total number of chunks returned |

### Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| 0.85+ | Excellent match - highly relevant |
| 0.75-0.85 | Good match - relevant content |
| 0.65-0.75 | Moderate match - related content |
| Below 0.65 | Weak match - tangentially related |

---

## Document Endpoint

### `GET /api/document/{r2Key}`

Fetch the full text and metadata for a specific document.

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `r2Key` | The R2 storage path from search results. Format: `{userId}/{collectionId}/{fileId}/{filename}` |

The `r2Key` can be found in search results at `documents[].file_info.r2Key` or constructed from the path components.

### Response

```json
{
  "status": "success",
  "document": {
    "r2Key": "0000000001/80650a98-fe49-429a-afbd-9dde66e2d02b/d6055ebe-6139-41cc-9135-dbd4cb5845d6/document.txt",
    "text": "Full document text content...",
    "metadata": {
      "doc_id": "CIA-RDP79T00429A001400010019-1",
      "corpus": "cia",
      "classification": "SECRET",
      "title": "Document Title",
      "date": "1962-10-15T00:00:00+00:00",
      "authored": -228182400000,
      "source": "https://www.cia.gov/readingroom/docs/..."
    },
    "file_info": {
      "id": "d6055ebe-6139-41cc-9135-dbd4cb5845d6",
      "name": "document.txt",
      "size": 1728,
      "type": "text/plain"
    }
  }
}
```

---

## Filtering

Filters are applied at the vector database level for efficient searching. The most reliable filters are date-based.

### Available Filters

| Filter | Type | Indexed | Description |
|--------|------|---------|-------------|
| `authored_year_month` | number/object | Yes | Date in YYYYMM format |
| `authored_year_month_day` | number/object | Yes | Date in YYYYMMDD format |
| `doc_id` | string | Yes | Specific document ID |

### Date Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"authored_year_month": 196210}` |
| `$gte` | Greater than or equal | `{"$gte": 196201}` |
| `$lte` | Less than or equal | `{"$lte": 196212}` |
| `$gt` | Greater than | `{"$gt": 196200}` |
| `$lt` | Less than | `{"$lt": 196301}` |
| `$in` | In array | `{"$in": [196210, 196211]}` |
| `$nin` | Not in array | `{"$nin": [196210]}` |

### Date Filter Examples

**Single year (1962):**
```json
{
  "filters": {
    "authored_year_month": {
      "$gte": 196201,
      "$lte": 196212
    }
  }
}
```

**Specific date range (Cuban Missile Crisis, Oct 1962):**
```json
{
  "filters": {
    "authored_year_month_day": {
      "$gte": 19621016,
      "$lte": 19621028
    }
  }
}
```

**Single month:**
```json
{
  "filters": {
    "authored_year_month": 196210
  }
}
```

**Multi-year range (Vietnam War era):**
```json
{
  "filters": {
    "authored_year_month": {
      "$gte": 196501,
      "$lte": 197512
    }
  }
}
```

---

## Complete Examples

### Example 1: Basic Search

```bash
curl -X POST "https://vector-search-worker.nchimicles.workers.dev/api/search" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": "Cuban Missile Crisis",
    "topK": 5
  }'
```

### Example 2: Search with Date Filter

```bash
curl -X POST "https://vector-search-worker.nchimicles.workers.dev/api/search" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": "nuclear weapons Soviet Union",
    "topK": 10,
    "filters": {
      "authored_year_month": {
        "$gte": 196201,
        "$lte": 196212
      }
    }
  }'
```

### Example 3: Precise Date Search (JFK Assassination)

```bash
curl -X POST "https://vector-search-worker.nchimicles.workers.dev/api/search" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": "Kennedy assassination reaction",
    "topK": 5,
    "filters": {
      "authored_year_month_day": {
        "$gte": 19631122,
        "$lte": 19631130
      }
    }
  }'
```

### Example 4: Fetch Full Document

After getting search results, use the `r2Key` to fetch the full document:

```bash
curl -X GET "https://vector-search-worker.nchimicles.workers.dev/api/document/0000000001/80650a98-fe49-429a-afbd-9dde66e2d02b/d6055ebe-6139-41cc-9135-dbd4cb5845d6/CIA-RDP79T00429A001400010019-1_unknown.txt" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Example 5: Multiple Queries

You can search for multiple concepts at once:

```bash
curl -X POST "https://vector-search-worker.nchimicles.workers.dev/api/search" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Soviet military capabilities", "nuclear arsenal estimates"],
    "topK": 10
  }'
```

---

## Typical Workflow

1. **Search** for relevant documents using natural language queries
2. **Review** the returned chunks and their scores
3. **Fetch** full documents for the most relevant results using their `r2Key`

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://vector-search-worker.nchimicles.workers.dev"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Step 1: Search (no collection_id needed - defaults to HistoryLab)
search_response = requests.post(
    f"{BASE_URL}/api/search",
    headers=headers,
    json={
        "queries": "Cold War intelligence",
        "topK": 5,
        "filters": {
            "authored_year_month": {"$gte": 196001, "$lte": 196912}
        }
    }
)
results = search_response.json()

# Step 2: Get full document for top result
if results["documents"]:
    top_doc = results["documents"][0]
    r2_key = top_doc["file_info"]["r2Key"]

    doc_response = requests.get(
        f"{BASE_URL}/api/document/{r2_key}",
        headers=headers
    )
    full_document = doc_response.json()
    print(full_document["document"]["text"])
```

---

## Document Collections

The database contains documents from these sources:

| Corpus | Description | Approximate Count |
|--------|-------------|-------------------|
| `cfpf` | Central Foreign Policy Files | 1.67M documents |
| `cia` | CIA declassified documents | 440K documents |
| `frus` | Foreign Relations of the United States | 159K documents |
| `un` | United Nations archives | 93K documents |
| `worldbank` | World Bank archives | 68K documents |
| `clinton` | Clinton administration documents | 30K documents |
| `nato` | NATO archives | 23K documents |
| `cabinet` | U.S. Cabinet meeting records | 20K documents |
| `cpdoc` | Brazilian historical archives | 6K documents |
| `kissinger` | Henry Kissinger papers | 2K documents |
| `briefing` | Presidential daily briefings | 924 documents |

**Note:** The `corpus` field is available in document metadata but is not currently indexed for filtering at the search level. Use date filters for precise searching.

---

## Rate Limits

| Limit | Value |
|-------|-------|
| Requests per minute | 60 |
| Max queries per request | 10 |
| Max topK | 100 |

---

## Error Responses

All errors return JSON with `status: "error"`:

```json
{
  "status": "error",
  "message": "Description of the error"
}
```

| HTTP Status | Meaning |
|-------------|---------|
| 400 | Bad request (missing parameters, invalid filters) |
| 401 | Unauthorized (missing or invalid API key) |
| 404 | Not found (document or collection doesn't exist) |
| 500 | Server error |

---

## Query Tips

1. **Use natural language** - The search is semantic, so "documents about nuclear tensions with Soviet Union" works better than "nuclear Soviet"

2. **Be specific but not too narrow** - "Cold War diplomatic negotiations" is better than just "Cold War"

3. **Use date filters** for historical precision - Narrow down to specific time periods for more relevant results

4. **Try multiple queries** - For complex topics, try different phrasings or use the array format to search multiple concepts

5. **Check chunk scores** - Higher scores (0.75+) indicate stronger relevance

6. **Fetch full documents** - The search returns chunks; use the document endpoint to get complete text for detailed analysis

---

## Technical Details

- **Embedding Model:** `@cf/baai/bge-base-en-v1.5` (768 dimensions)
- **Vector Database:** Cloudflare Vectorize
- **Similarity Metric:** Cosine similarity
- **Document Storage:** Cloudflare R2
- **Metadata Storage:** Cloudflare KV

The search process:
1. Your query is converted to a 768-dimensional vector
2. Similar vectors are found across multiple indexes
3. Chunks are retrieved and re-ranked by exact similarity
4. Results are grouped by document and enriched with metadata
