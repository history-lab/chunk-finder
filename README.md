# Cloudflare Vector Search Worker

This Cloudflare Worker provides functionality for searching through vector embeddings using Cloudflare Vectorize and Workers AI.

## Features

- Search for similar embeddings across multiple query terms
- Return results sorted by similarity score
- Support for parallel processing of multiple queries
- Deduplication of results

## API Usage

### Search for Similar Embeddings

```http
POST /
Content-Type: application/json

{
  "queries": ["Your search query", "Another search query"],
  "collection_id": "your-collection-id",
  "topK": 5
}
```

### Parameters

- `queries`: A string or array of strings to search for
- `collection_id`: The ID of the Vectorize collection to search in
- `topK` (optional): Number of results to return (default: 5)

### Response Format

```json
{
  "status": "success",
  "matches": [
    {
      "id": "chunk-123",
      "score": 0.95,
      "metadata": {
        "text": "The matching text content",
        "additional_field": "Any additional metadata"
      }
    },
    {
      "id": "chunk-456",
      "score": 0.92,
      "metadata": {
        "text": "Another relevant text match",
        "additional_field": "Any additional metadata"
      }
    }
  ]
}
```

## Configuration

Make sure to bind the following:

1. Cloudflare Workers AI with the binding name "AI"
2. Cloudflare Vectorize with the binding name "VECTORIZE"

Example wrangler.jsonc configuration:

```jsonc
{
  "name": "vector-search-worker",
  "main": "src/index.ts",
  "compatibility_date": "2025-02-11",
  "compatibility_flags": ["nodejs_compat"],
  "ai": {
    "binding": "AI"
  },
  "vectorize": [
    {
      "binding": "VECTORIZE",
      "index_name": "files-1"
    }
  ],
  "observability": {
    "enabled": true
  }
}
```

## Error Handling

The API will return appropriate error responses with descriptive messages when:
- Required parameters are missing
- Embedding generation fails
- Vector database queries fail
- Any other errors occur during processing

## Implementation Details

- Uses the `@cf/baai/bge-base-en-v1.5` embedding model for query embedding generation
- Processes multiple query terms in parallel for efficiency
- Results are combined and sorted by similarity score
- Duplicate results are filtered out by ID
- Returns the top K most similar results across all queries 

# Chunk Finder Service

A Cloudflare Worker for finding and retrieving chunks of text from embedded documents.

## API Reference

### Find Similar Embeddings

```
POST /
```

Finds chunks of text similar to the provided query text.

#### Request Body Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| queries | string or array of strings | The query text to search for similar chunks |
| collection_id | string | The ID of the collection to search in |
| topK | number (optional) | Number of results to return, default is 5 |
| corpus | string (optional) | Filter by corpus (e.g., 'cia', 'frus', 'clinton') |
| doc_id | string (optional) | Filter by specific document ID |
| authored_start | string (optional) | Start date for filtering (YYYY-MM-DD format) |
| authored_end | string (optional) | End date for filtering (YYYY-MM-DD format) |

#### Supported Corpus Values

- `cfpf`: Central Federal Policy Files (1.67M docs)
- `cia`: CIA documents (440K docs)
- `frus`: Foreign Relations of the United States (159K docs)
- `un`: United Nations documents (93K docs)
- `worldbank`: World Bank reports (68K docs)
- `clinton`: Clinton administration documents (30K docs)
- `nato`: NATO documents (23K docs)
- `cabinet`: U.S. Cabinet meeting records (20K docs)
- `cpdoc`: Brazilian historical archives (6K docs)
- `kissinger`: Henry Kissinger's diplomatic work (2K docs)
- `briefing`: Government briefing documents (924 docs)

#### Example Request

```json
{
  "queries": "Cold War diplomacy in Eastern Europe",
  "collection_id": "history-lab-1",
  "topK": 10,
  "corpus": "cia",
  "authored_start": "1965-01-01",
  "authored_end": "1975-12-31"
}
```

#### Example Response

```json
{
  "status": "success",
  "matches": [
    {
      "id": "chunk-123",
      "text": "The Soviet Union's influence in Eastern Europe remained strong throughout the Cold War period...",
      "score": 0.92,
      "metadata": {
        "corpus": "cia",
        "doc_id": "doc-456",
        "authored": 157680000,
        "title": "Analysis of Eastern Bloc Politics"
      }
    },
    // Additional matches...
  ]
}
```

## Development

### Prerequisites

- Node.js 18 or later
- Wrangler CLI (`npm install -g wrangler`)

### Setup

1. Clone the repository
2. Install dependencies: `npm install`
3. Configure your environment variables in `wrangler.jsonc`

### Deployment

```
wrangler publish
``` 