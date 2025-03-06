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