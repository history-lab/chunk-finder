import { WorkerEntrypoint } from "cloudflare:workers";
import JSZip from "jszip";

interface Env {
  AI: any;
  RAMUS_EMBEDDINGS: R2Bucket;
  API_SECRET_KEY?: string;
  VECTORIZE_API_TOKEN: string;
  ACCOUNT_ID: string;
  COLLECTIONS_METADATA: KVNamespace;
  FILES_METADATA: KVNamespace;
}

// Define types for Vectorize responses
interface VectorizeMatch {
  id: string;
  score: number;
  metadata?: Record<string, any>;
  values?: number[];
  namespace?: string;
}

interface VectorizeQueryResponse {
  matches: VectorizeMatch[];
}

interface ChunkResult {
  key: string;
  status: number;
  contentType?: string;
  size?: number;
  data?: string;
  error?: string;
}

interface Chunk {
  id: string;
  text: string;
  index: number;
  embedding?: number[];
  metadata?: Record<string, any>;
  score?: number;
}

// Types for collection indexes from KV
interface MetadataIndexConfig {
  propertyName: string;
  indexType: 'string' | 'number' | 'boolean';
}

interface CollectionIndexes {
  collection_id: string;
  indexes: string[];
  metadata_indexes: {
    [indexName: string]: MetadataIndexConfig[];
  };
}

const MODEL = '@cf/baai/bge-base-en-v1.5';
const MAX_CONCURRENT_REQUESTS = 6;
const USE_NAMESPACE_FOR_COLLECTION = true; // Set to false to use collection_id as a metadata field instead of namespace

// Type for index search result including errors
interface IndexSearchResult {
  indexName: string;
  chunks: Chunk[];
  error?: string;
}

export default class extends WorkerEntrypoint<Env> {
  /**
   * Fetch collection indexes information from KV store
   */
  async getCollectionIndexes(collection_id: string): Promise<CollectionIndexes | null> {
    try {
      console.log(`Fetching collection indexes for collection ${collection_id} from KV`);

      // Fetch the collection metadata from KV using the collection_id as the key
      const collectionData = await this.env.COLLECTIONS_METADATA.get(collection_id);

      if (!collectionData) {
        console.error(`No collection found in KV for collection_id: ${collection_id}`);
        return null;
      }

      // Parse the KV data
      const parsedData = JSON.parse(collectionData);
      console.log(`Found collection in KV:`, parsedData);

      // Map the KV data structure to the CollectionIndexes interface
      return {
        collection_id: collection_id,
        indexes: parsedData.indexes || [],
        metadata_indexes: parsedData.metadataIndexes || {},
      };
    } catch (error) {
      console.error(`Error fetching collection indexes from KV:`, error);
      return null;
    }
  }

  /**
   * Main fetch handler that routes between RPC and HTTP API
   */
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    // Handle HTTP API requests with /api prefix
    if (url.pathname.startsWith('/api/')) {
      return this.handleApiRequest(request);
    }

    // Handle RPC requests (original functionality)
    return this.handleRpcRequest(request);
  }

  /**
   * Middleware to check API authentication
   */
  private async authenticateApiRequest(request: Request): Promise<boolean> {
    const authHeader = request.headers.get('Authorization');

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return false;
    }

    const providedKey = authHeader.substring(7); // Remove 'Bearer ' prefix
    const expectedKey = this.env.API_SECRET_KEY;

    if (!expectedKey) {
      console.error('API_SECRET_KEY not configured');
      return false;
    }

    // Constant-time comparison to prevent timing attacks
    if (providedKey.length !== expectedKey.length) {
      return false;
    }

    let equal = true;
    for (let i = 0; i < providedKey.length; i++) {
      if (providedKey[i] !== expectedKey[i]) {
        equal = false;
      }
    }

    return equal;
  }

  /**
   * Handle HTTP API requests with authentication
   */
  private async handleApiRequest(request: Request): Promise<Response> {
    // Check authentication
    const isAuthenticated = await this.authenticateApiRequest(request);

    if (!isAuthenticated) {
      return new Response(
        JSON.stringify({
          status: 'error',
          message:
            'Unauthorized. Please provide a valid API key in the Authorization header.',
        }),
        {
          status: 401,
          headers: {
            'Content-Type': 'application/json',
            'WWW-Authenticate': 'Bearer realm="API"',
          },
        },
      );
    }

    const url = new URL(request.url);

    // Handle different API endpoints
    if (url.pathname === '/api/search' && request.method === 'POST') {
      return this.handleSearchRequest(request);
    }

    // Handle API health check
    if (url.pathname === '/api/health' && request.method === 'GET') {
      return new Response(
        JSON.stringify({
          status: 'ok',
          timestamp: new Date().toISOString(),
        }),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        },
      );
    }

    // Handle API help endpoint
    if (url.pathname === '/api/help' && request.method === 'GET') {
      return this.getHelpResponse();
    }

    // Handle API about endpoint
    if (url.pathname === '/api/about' && request.method === 'GET') {
      return this.getAboutResponse();
    }

    // Return 404 for unknown API endpoints
    return new Response(
      JSON.stringify({
        status: 'error',
        message: 'Endpoint not found',
      }),
      {
        status: 404,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  }

  /**
   * Get help documentation response
   */
  private getHelpResponse(): Response {
    const helpDoc = {
      name: 'History Lab Vector Search API',
      version: '1.0.0',
      description: 'Semantic search API for historical documents using vector embeddings',
      base_url: 'https://vector-search-worker.nchimicles.workers.dev',
      authentication: {
        type: 'Bearer Token',
        header: 'Authorization: Bearer <API_KEY>',
        description:
          'All API endpoints require authentication using the Bearer token',
        example: 'Authorization: Bearer sk-your-api-key-here',
      },
      important_notes: [
        'ALWAYS use collection_id: 80650a98-fe49-429a-afbd-9dde66e2d02b',
        'Invalid collection_id returns empty results, not an error',
        'Queries can be natural language questions or keywords',
        'Scores above 0.7 indicate high relevance',
        'Results include both individual chunk scores and cluster scores',
      ],
      collection: {
        required_id: '80650a98-fe49-429a-afbd-9dde66e2d02b',
        description:
          'The ONLY valid collection ID. Using any other ID will return empty results.',
        warning:
          'Invalid collection IDs do not return errors - they return empty match arrays',
      },
      endpoints: [
        {
          path: '/api/search',
          method: 'POST',
          description: 'Perform semantic search on historical documents',
          authentication: 'Required',
          request: {
            body: {
              queries: {
                type: 'string | string[]',
                required: true,
                description:
                  'Search query or array of queries. Use natural language for best results.',
                examples: [
                  'Cuban Missile Crisis',
                  'Vietnam War',
                  'Cold War tensions',
                  'nuclear weapons',
                ],
              },
              collection_id: {
                type: 'string',
                required: true,
                description:
                  'MUST be exactly: 80650a98-fe49-429a-afbd-9dde66e2d02b',
                value: '80650a98-fe49-429a-afbd-9dde66e2d02b',
                warning: 'Any other value returns empty results without error',
              },
              topK: {
                type: 'number',
                required: false,
                default: 5,
                description:
                  'Number of most similar results to return (1-100)',
                examples: [5, 10, 20],
              },
              corpus: {
                type: 'string',
                required: false,
                description: 'Filter by document corpus/source',
                valid_values: [
                  'cfpf',
                  'cia',
                  'frus',
                  'un',
                  'worldbank',
                  'clinton',
                  'nato',
                  'cabinet',
                  'cpdoc',
                  'kissinger',
                  'briefing',
                ],
                examples: ['cia', 'frus', 'clinton'],
                note: 'Invalid corpus values return an error',
              },
              doc_id: {
                type: 'string',
                required: false,
                description: 'Filter by specific document ID',
                examples: ['CIA-RDP80B01676R002800240004-4', 'P790016-1808'],
              },
              authored_start: {
                type: 'string',
                required: false,
                description:
                  'Filter documents authored on or after this date (YYYY-MM-DD format)',
                example: '1960-01-01',
                note: 'Converted to Unix timestamp internally',
              },
              authored_end: {
                type: 'string',
                required: false,
                description:
                  'Filter documents authored on or before this date (YYYY-MM-DD format)',
                example: '1989-12-31',
                note: 'Converted to Unix timestamp internally',
              },
            },
          },
          response: {
            success: {
              status: 'success',
              matches: [
                {
                  id: 'string - Unique chunk identifier (e.g., "651296f4-14f0-44b7-a773-74a08f58b1dc_0")',
                  text: 'string - The actual text content of the chunk',
                  score:
                    'number - Similarity score (0-1, higher is more similar. 0.7+ is highly relevant)',
                  metadata: {
                    corpus: 'string - Source corpus (e.g., "cia", "cfpf")',
                    doc_id: 'string - Document identifier',
                    authored: 'number - Unix timestamp in milliseconds',
                    date: 'string - ISO date string (may be null)',
                    classification:
                      'string - Security classification (e.g., "unclassified")',
                    title: 'string - Document title',
                    source: 'string - Source URL (may be null)',
                    file_id: 'string - Source file identifier',
                    file_name: 'string - Original file name',
                    file_key: 'string - R2 storage key',
                    batch_size: 'number - Number of chunks in batch',
                    total_chunks: 'number - Total chunks in document',
                    is_anchor: 'boolean - Whether this is an anchor chunk',
                    user_id: 'string - User who created the embedding',
                    collection_id: 'string - Collection identifier',
                    cluster_id: 'string - Cluster identifier for grouped chunks',
                    cluster_score: 'number - Cluster-level similarity score',
                  },
                },
              ],
            },
            error: {
              status: 'error',
              message: 'string - Error description',
            },
          },
          example_requests: [
            {
              description: 'Simple search',
              curl: `curl -X POST https://vector-search-worker.nchimicles.workers.dev/api/search \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer YOUR_API_KEY" \\\n  -d '{\
    "queries": "Cold War",\
    "collection_id": "80650a98-fe49-429a-afbd-9dde66e2d02b",\
    "topK": 5\
  }'`,
            },
            {
              description: 'Search with filters',
              curl: `curl -X POST https://vector-search-worker.nchimicles.workers.dev/api/search \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer YOUR_API_KEY" \\\n  -d '{\
    "queries": "Cuban Missile Crisis",\
    "collection_id": "80650a98-fe49-429a-afbd-9dde66e2d02b",\
    "topK": 10,\
    "corpus": "cia",\
    "authored_start": "1962-01-01",\
    "authored_end": "1963-12-31"\
  }'`,
            },
            {
              description: 'Multiple queries',
              curl: `curl -X POST https://vector-search-worker.nchimicles.workers.dev/api/search \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer YOUR_API_KEY" \\\n  -d '{\
    "queries": ["Cuban Missile Crisis", "Berlin Wall", "Vietnam War"],\
    "collection_id": "80650a98-fe49-429a-afbd-9dde66e2d02b",\
    "topK": 3\
  }'`,
            },
          ],
        },
        {
          path: '/api/health',
          method: 'GET',
          description: 'Check API health status',
          authentication: 'Required',
          response: {
            status: 'ok',
            timestamp: 'ISO 8601 timestamp',
          },
          example_curl:
            `curl -X GET https://vector-search-worker.nchimicles.workers.dev/api/health \\\n  -H "Authorization: Bearer YOUR_API_KEY"`,
        },
        {
          path: '/api/help',
          method: 'GET',
          description: 'Get API documentation (this endpoint)',
          authentication: 'Required',
        },
        {
          path: '/api/about',
          method: 'GET',
          description: 'Learn about the search technology and architecture',
          authentication: 'Required',
        },
      ],
      filters: {
        corpus_values: {
          cfpf:
            'Center for Presidential and Political Files - Presidential correspondence and memos',
          cia: 'Central Intelligence Agency documents - Intelligence reports and analyses',
          frus:
            'Foreign Relations of the United States - Official diplomatic history',
          un: 'United Nations documents - International relations and resolutions',
          worldbank: 'World Bank archives - Development and economic reports',
          clinton: 'Clinton Presidential Library - 1990s presidential records',
          nato: 'NATO archives - Alliance documents and communications',
          cabinet: 'Cabinet papers - High-level government discussions',
          cpdoc: 'CPDOC Brazilian political archives - Brazilian political history',
          kissinger: 'Henry Kissinger papers - 1970s diplomatic correspondence',
          briefing: 'Presidential daily briefings - Intelligence summaries',
        },
        date_range: {
          format: 'YYYY-MM-DD',
          description:
            'Filter by document authored date. Dates are converted to Unix timestamps.',
          examples: [
            'Cold War era: authored_start="1947-01-01", authored_end="1991-12-31"',
            'Vietnam War period: authored_start="1965-01-01", authored_end="1975-12-31"',
            '1960s: authored_start="1960-01-01", authored_end="1969-12-31"',
          ],
        },
      },
      query_tips: {
        best_practices: [
          'Use natural language queries for best results (e.g., "Cuban Missile Crisis negotiations")',
          'Be specific but not too narrow (e.g., "Cold War tensions" vs "Cold War")',
          'Combine filters for precise results (corpus + date range)',
          'Higher topK values (10-20) provide more context',
          'Scores above 0.7 indicate strong relevance, above 0.8 is very strong',
        ],
        example_queries: {
          events: [
            'Cuban Missile Crisis',
            'Berlin Wall',
            'Vietnam War',
            'Kennedy assassination',
          ],
          topics: [
            'nuclear weapons',
            'Cold War tensions',
            'Soviet Union relations',
            'diplomatic negotiations',
          ],
          people: ['John F. Kennedy', 'Henry Kissinger', 'Fidel Castro', 'Nikita Khrushchev'],
          periods: [
            '1960s foreign policy',
            'detente period',
            'Reagan administration',
          ],
        },
      },
      troubleshooting: {
        empty_results: [
          'Check collection_id is exactly: 80650a98-fe49-429a-afbd-9dde66e2d02b',
          'Verify your API key is correct',
          'Try broader search terms',
          'Remove filters to test if they are too restrictive',
        ],
        low_scores: [
          'Rephrase query using different terms',
          'Try more specific or contextual queries',
          'Check if the topic exists in the time period searched',
        ],
      },
      rate_limits: {
        requests_per_minute: 60,
        max_queries_per_request: 10,
        max_topK: 100,
      },
      notes: [
        'Collection ID must be exactly: 80650a98-fe49-429a-afbd-9dde66e2d02b',
        'Invalid collection IDs return empty results, not errors',
        'The API uses semantic search - natural language queries work best',
        'Results include both chunk-level and cluster-level similarity scores',
        'Metadata includes rich context: dates, sources, classifications, document titles',
        'Filters are applied at the vector database level for efficiency',
      ],
    };

    return new Response(JSON.stringify(helpDoc, null, 2), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /**
   * Get about/technical information response
   */
  private getAboutResponse(): Response {
    const aboutDoc = {
      name: 'History Lab Vector Search API',
      version: '1.0.0',
      architecture: {
        overview:
          'This API provides semantic search capabilities for historical documents using state-of-the-art vector embedding technology.',
        components: {
          embedding_model: {
            name: '@cf/baai/bge-base-en-v1.5',
            type: 'BERT-based encoder',
            dimensions: 768,
            description:
              'Converts text queries into high-dimensional vectors that capture semantic meaning',
          },
          vector_database: {
            name: 'Cloudflare Vectorize',
            description:
              'Stores and indexes document embeddings for fast similarity search',
            features: [
              'Cosine similarity search',
              'Metadata filtering',
              'Scalable to millions of vectors',
            ],
          },
          storage: {
            name: 'Cloudflare R2',
            description:
              'Object storage for document chunks and their embeddings',
            structure: 'Organized in compressed batches for efficient retrieval',
          },
        },
      },
      search_process: {
        steps: [
          {
            step: 1,
            name: 'Query Embedding',
            description:
              'Your search query is converted into a 768-dimensional vector using the BGE model. This vector represents the semantic meaning of your query.',
          },
          {
            step: 2,
            name: 'Vector Similarity Search',
            description:
              'The query vector is compared against millions of document vectors using cosine similarity. This finds documents with similar semantic meaning, even if they use different words.',
          },
          {
            step: 3,
            name: 'Metadata Filtering',
            description:
              'Results are filtered based on your criteria (corpus, date range, document ID) at the database level for efficiency.',
          },
          {
            step: 4,
            name: 'Chunk Retrieval',
            description:
              'The most similar document chunks are retrieved from R2 storage. Documents are split into chunks for granular matching.',
          },
          {
            step: 5,
            name: 'Ranking and Response',
            description:
              'Results are ranked by similarity score and returned with their metadata and relevance scores.',
          },
        ],
      },
      clustering_approach: {
        description: 'Documents are processed in clusters for efficiency',
        benefits: [
          'Reduces storage costs by grouping related content',
          'Improves query performance through batch processing',
          'Enables efficient compression of embedding data',
        ],
        implementation:
          'Each cluster contains multiple document chunks with their embeddings stored in compressed ZIP format',
      },
      similarity_scoring: {
        metric: 'Cosine Similarity',
        range: '0.0 to 1.0',
        interpretation: {
          '0.9-1.0': 'Nearly exact match - Often the same document or direct quotes',
          '0.8-0.9': 'Highly relevant - Same event/topic with strong connection',
          '0.7-0.8': 'Very relevant - Related topic with good context match',
          '0.6-0.7': 'Relevant - Related content worth reviewing',
          '0.5-0.6': 'Somewhat relevant - Peripheral connection',
          'Below 0.5': 'Low relevance - Different topic or weak connection',
        },
        practical_guidance:
          'In practice, results with scores above 0.7 are typically what users want',
      },
      data_format: {
        chunk_size:
          'Documents are split into meaningful chunks (typically 1-3 paragraphs)',
        embedding_storage:
          'Float32Array binary format for efficient storage and computation',
        compression: 'ZIP compression reduces storage size by ~60%',
        metadata:
          'Each chunk includes source, date, and document identification',
      },
      performance: {
        query_latency: 'Typically 200-500ms for vector search',
        concurrent_requests: 6,
        index_size: 'Scales to millions of document embeddings',
        accuracy: 'BGE model provides state-of-the-art semantic understanding',
      },
      use_cases: [
        'Historical research: Find all documents about specific events (e.g., Cuban Missile Crisis)',
        'Cross-archive search: Search across CIA, State Department, UN documents simultaneously',
        'Timeline analysis: Filter by date ranges to understand event progression',
        'Entity research: Find all mentions of specific people, places, or organizations',
        'Thematic research: Discover documents about concepts like "nuclear deterrence" or "Cold War tensions"',
        'Comparative analysis: Find similar events or patterns across different time periods',
        'Primary source discovery: Locate original documents and intelligence reports',
      ],
      advantages: {
        semantic_understanding:
          "Finds relevant content even when exact keywords don't match",
        multilingual: 'Can find related content across language barriers',
        contextual: 'Understands the context and meaning of queries',
        efficient:
          'Vector search is much faster than traditional full-text search for semantic queries',
      },
    };

    return new Response(JSON.stringify(aboutDoc, null, 2), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /**
   * Handle RPC requests (original functionality)
   */
  private async handleRpcRequest(request: Request): Promise<Response> {
    try {
      const body = (await request.json()) as {
        queries: string | string[];
        collection_id: string;
        topK?: number;
        filters?: Record<string, any>;
        corpus?: string;
        doc_id?: string;
        authored_start?: string;
        authored_end?: string;
      };
      return this.processSearchRequest(body);
    } catch (error) {
      console.error('Error parsing RPC request body:', error);
      return new Response(
        JSON.stringify({ status: 'error', message: 'Invalid request body.' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } },
      );
    }
  }

  /**
   * Handle API search requests
   */
  private async handleSearchRequest(request: Request): Promise<Response> {
    try {
      const body = (await request.json()) as {
        queries: string | string[];
        collection_id: string;
        topK?: number;
        corpus?: string;
        doc_id?: string;
        authored_start?: string;
        authored_end?: string;
        filters?: Record<string, any>;
      };

      return this.processSearchRequest(body);
    } catch (error) {
      console.error('Error parsing request body:', error);
      return new Response(
        JSON.stringify({
          status: 'error',
          message: 'Invalid request body. Please provide valid JSON.',
        }),
        { status: 400, headers: { 'Content-Type': 'application/json' } },
      );
    }
  }

  /**
   * Process search request (shared logic for both RPC and API)
   */
  private async processSearchRequest(body: {
    queries: string | string[];
    collection_id: string;
    topK?: number;
    corpus?: string;
    doc_id?: string;
    authored_start?: string;
    authored_end?: string;
    filters?: Record<string, any>;
  }): Promise<Response> {
    console.log('Received search request:', JSON.stringify(body, null, 2));

    const { queries, collection_id, topK = 5, filters } = body;

    if (!queries) {
      console.error('Missing required parameter: queries');
      return new Response(
        JSON.stringify({
          status: 'error',
          message: 'Missing required parameter: queries',
        }),
        { status: 400 },
      );
    }

    // Fetch collection indexes from KV using the provided collection_id
    const collectionIndexes = await this.getCollectionIndexes(collection_id);

    if (!collectionIndexes) {
      console.error(`Collection not found: ${collection_id}`);
      return new Response(
        JSON.stringify({
          status: 'error',
          message: `Collection not found: ${collection_id}`,
        }),
        { status: 404 },
      );
    }

    // Validate that the collection has at least one index
    if (!collectionIndexes.indexes || collectionIndexes.indexes.length === 0) {
      console.error(`No indexes found for collection: ${collection_id}`);
      return new Response(
        JSON.stringify({
          status: 'error',
          message: `No indexes found for collection: ${collection_id}`,
        }),
        { status: 400 },
      );
    }

    console.log(
      `Executing vector search with query: ${
        typeof queries === 'string' ? queries : queries.join(', ')
      }`,
    );

    // Process and validate filters if provided
    let validatedFilters: Record<string, any> = {};

    if (filters && Object.keys(filters).length > 0) {
      console.log('Processing provided filters:', JSON.stringify(filters, null, 2));

      // Process each filter to ensure it has correct operators for its type
      try {
        validatedFilters = this.processFilters(filters, collectionIndexes);
        console.log('Validated filters:', JSON.stringify(validatedFilters, null, 2));
      } catch (error) {
        console.error('Error validating filters:', error);
        return new Response(
          JSON.stringify({
            status: 'error',
            message:
              error instanceof Error ? error.message : 'Invalid filters provided',
          }),
          { status: 400 },
        );
      }
    }

    // Execute the similarity search using the provided collection_id
    const result = await this.findSimilarEmbeddings(
      queries,
      collection_id,
      topK,
      validatedFilters,
      collectionIndexes,
    );

    console.log(
      `Search returned ${result?.documents?.length || 0} documents with a total of ${
        result?.total_chunks || 0
      } chunks`,
    );

    return new Response(JSON.stringify(result), {
      headers: { 'Content-Type': 'application/json' },
    });
  }

  /**
   * Find similar embeddings for multiple query terms across multiple indexes
   */
  async findSimilarEmbeddings(
    queries: string | string[],
    collection_id: string,
    topK: number = 5,
    filters?: Record<string, any>,
    collectionIndexes?: CollectionIndexes,
  ) {
    try {
      // Convert input to array if it's a single string
      const queryArray = Array.isArray(queries) ? queries : [queries];
      console.log(
        `Finding similar embeddings for ${queryArray.length} queries in collection ${collection_id}`,
      );

      // Log metadata filters if present
      if (filters) {
        console.log('Using metadata filters:', JSON.stringify(filters, null, 2));
      }

      // Make sure we have valid collection indexes
      if (!collectionIndexes) {
        const fetchedIndexes = await this.getCollectionIndexes(collection_id);
        if (!fetchedIndexes) {
          throw new Error(`Collection not found: ${collection_id}`);
        }
        collectionIndexes = fetchedIndexes;
      }

      // At this point, collectionIndexes is guaranteed
      const indexesInfo = collectionIndexes as CollectionIndexes;

      // Validate that the collection has at least one index
      if (!indexesInfo.indexes || indexesInfo.indexes.length === 0) {
        throw new Error(`No indexes found for collection: ${collection_id}`);
      }

      console.log(
        `Collection has ${indexesInfo.indexes.length} indexes:`,
        indexesInfo.indexes,
      );

      // Generate embeddings for all queries in parallel
      let queryEmbeddings: any;
      try {
        queryEmbeddings = await this.env.AI.run(MODEL, { text: queryArray });
      } catch (error) {
        console.error('Error generating query embeddings:', error);
        return {
          status: 'error',
          message: 'Failed to generate query embeddings',
        };
      }

      console.log('Generated query embeddings');

      // For each query embedding, search across all indexes in parallel
      const allSearchResults: Chunk[][] = [];
      const failedIndexes: { indexName: string; error: string }[] = [];

      for (let i = 0; i < queryEmbeddings.data.length; i++) {
        const embedding = queryEmbeddings.data[i];
        console.log(`Processing query: "${queryArray[i]}" (truncated)`);

        // Calculate how many results to fetch from each index
        const fetchCount = Math.min(Math.max(topK * 2, 20), 10);
        console.log(
          `Fetching top ${fetchCount} results from each index to ensure document diversity`,
        );

        // Query each index in parallel and collect the results
        // Use Promise.allSettled instead of Promise.all to handle partial failures
        const indexSearchPromises = indexesInfo.indexes.map(
          async (indexName): Promise<IndexSearchResult> => {
            try {
              console.log(`Querying index: ${indexName}`);

              // Prepare query options for this index
              const queryOptions: {
                topK: number;
                namespace?: string;
                returnValues: boolean;
                returnMetadata: 'all' | 'none' | 'indexed';
                filter?: Record<string, any>;
              } = {
                topK: fetchCount,
                returnValues: false,
                returnMetadata: 'all',
              };

              // Apply collection filtering based on configuration
              if (USE_NAMESPACE_FOR_COLLECTION) {
                // Use namespace for collection filtering
                queryOptions.namespace = collection_id;
                console.log(`Using namespace for collection filtering: ${collection_id}`);
              } else {
                // Use metadata field for collection filtering
                queryOptions.filter = { collection_id: collection_id };
                console.log(
                  `Using metadata field for collection filtering: ${collection_id}`,
                );
              }

              // Add filters if present
              if (filters && Object.keys(filters).length > 0) {
                // If we're already using filter for collection_id, merge with user filters
                if (queryOptions.filter) {
                  queryOptions.filter = {
                    ...queryOptions.filter,
                    ...filters,
                  };
                } else {
                  queryOptions.filter = filters;
                }
              }

              // Query the Vectorize API for this index
              const vectorMatches = await this.queryVectorizeAPI(
                indexName,
                embedding,
                queryOptions,
              );

              if (
                !vectorMatches ||
                !Array.isArray(vectorMatches.matches) ||
                vectorMatches.matches.length === 0
              ) {
                console.log(
                  `No matches found in index ${indexName} for query "${queryArray[i]}"`,
                );
                return { indexName, chunks: [] };
              }

              console.log(
                `Found ${vectorMatches.matches.length} matches in index ${indexName}`,
              );

              // Add default user_id to ALL matches that don't have one (only if needed)
              for (const match of vectorMatches.matches) {
                if (match.metadata && !match.metadata.user_id) {
                  console.log('Adding default user_id to match');
                  match.metadata.user_id = '0000000001';
                }
              }

              // Construct R2 keys for the matching vectors
              const r2KeysWithMetadata = vectorMatches.matches
                .map((match) => {
                  if (
                    !match.metadata ||
                    !match.metadata.user_id ||
                    !match.metadata.file_id ||
                    (!match.id && !match.metadata.batch_id)
                  ) {
                    console.error('Missing metadata fields in match:', match);
                    return null;
                  }

                  // Construct R2 key using the metadata fields
                  // Format: userId/collectionId/fileId/batchId.zip
                  const userId = match.metadata.user_id;
                  // Get collection_id from metadata if using metadata field approach, or use the parameter if using namespace
                  const collId = USE_NAMESPACE_FOR_COLLECTION
                    ? collection_id
                    : match.metadata.collection_id || collection_id;
                  const fileId = match.metadata.file_id;
                  const batchId = match.id || match.metadata.batch_id;

                  return {
                    key: `${userId}/${collId}/${fileId}/${batchId}.zip`,
                    metadata: match.metadata,
                    score: match.score,
                    id: match.id,
                  };
                })
                .filter((item) => item !== null) as Array<{
                key: string;
                metadata: Record<string, any>;
                score: number;
                id: string;
              }>;

              if (r2KeysWithMetadata.length === 0) {
                console.error('No valid R2 keys could be constructed');
                return { indexName, chunks: [] };
              }

              const r2Keys = r2KeysWithMetadata.map((item) => item.key);
              console.log(
                `Fetching ${r2Keys.length} chunks from R2 for index ${indexName}`,
              );

              // Fetch chunks from R2 bucket with concurrency limit
              const chunkResults = await this.fetchChunksFromR2(r2Keys);

              if (!chunkResults || chunkResults.length === 0) {
                console.error(`No chunks returned from R2 for index ${indexName}`);
                return { indexName, chunks: [] };
              }

              // Process the chunks - extract them from the ZIP files and calculate similarity
              const processedChunks: Chunk[] = [];

              for (let i = 0; i < chunkResults.length; i++) {
                const result = chunkResults[i];
                const keyWithMetadata = r2KeysWithMetadata.find(
                  (item) => item.key === result.key,
                );

                if (!keyWithMetadata) {
                  console.error(`Could not find metadata for key ${result.key}`);
                  continue;
                }

                if (result.status !== 200 || !result.data) {
                  console.error(
                    `Error fetching chunk ${result.key}: ${
                      result.error || 'Unknown error'
                    }`,
                  );
                  continue;
                }

                try {
                  // Convert base64 to array buffer
                  const binaryData = atob(result.data);
                  const bytes = new Uint8Array(binaryData.length);
                  for (let i = 0; i < binaryData.length; i++) {
                    bytes[i] = binaryData.charCodeAt(i);
                  }

                  // Use JSZip to extract the data
                  const zip = await JSZip.loadAsync(bytes);

                  // Extract embeddings binary data and convert to Float32Array
                  const embeddingsFile = zip.file('embeddings.bin');
                  if (!embeddingsFile) {
                    console.error('Missing embeddings.bin in ZIP');
                    continue;
                  }
                  const embeddingsArrayBuffer = await embeddingsFile.async(
                    'arraybuffer',
                  );
                  const embeddings = new Float32Array(embeddingsArrayBuffer);

                  // Extract chunks
                  const chunksFile = zip.file('chunks.json');
                  if (!chunksFile) {
                    console.error('Missing chunks.json in ZIP');
                    continue;
                  }
                  const chunksText = await chunksFile.async('text');
                  const chunks = JSON.parse(chunksText) as Chunk[];

                  // Try to determine batch size either from metadata or chunks length
                  let batchSize = chunks.length;
                  let fileMetadata: any = null;

                  try {
                    const metadataFile = zip.file('metadata.json');
                    if (metadataFile) {
                      const metadataText = await metadataFile.async('text');
                      fileMetadata = JSON.parse(metadataText);
                      if (fileMetadata.batch_size) {
                        batchSize = fileMetadata.batch_size;
                      }
                    }
                  } catch (e) {
                    console.warn(
                      'Could not read metadata.json, using chunks length as batch size',
                      e,
                    );
                  }

                  // Calculate the number of embeddings and dimensions
                  const embeddingDimensions = embeddings.length / batchSize;

                  // Process each chunk with its corresponding embedding
                  for (let j = 0; j < chunks.length; j++) {
                    const chunk = chunks[j];

                    // Extract the embedding for this chunk from the Float32Array
                    const startIdx = j * embeddingDimensions;
                    const chunkEmbedding = Array.from(
                      embeddings.subarray(startIdx, startIdx + embeddingDimensions),
                    );

                    // Calculate cosine similarity
                    const similarity = this.cosineSimilarity(
                      embedding,
                      chunkEmbedding,
                    );

                    // Add embedding, score, and cluster metadata to the chunk
                    const processedChunk: Chunk = {
                      ...chunk,
                      embedding: chunkEmbedding,
                      score: similarity,
                      metadata: {
                        ...keyWithMetadata.metadata,
                        cluster_id: keyWithMetadata.id,
                        cluster_score: keyWithMetadata.score,
                        index_name: indexName, // Add the index name to the metadata
                      },
                    };

                    processedChunks.push(processedChunk);
                  }
                } catch (error) {
                  console.error(`Error processing chunk ${result.key}:`, error);
                }
              }

              console.log(
                `Processed ${processedChunks.length} chunks from index ${indexName}`,
              );

              // Sort chunks by similarity score and return
              processedChunks.sort((a, b) => (b.score || 0) - (a.score || 0));
              return { indexName, chunks: processedChunks };
            } catch (error) {
              console.error(`Error searching index ${indexName}:`, error);

              // Instead of re-throwing, return a result with an error field
              return {
                indexName,
                chunks: [],
                error: error instanceof Error ? error.message : String(error),
              };
            }
          },
        );

        // Wait for all index searches to complete, even ones that fail
        const indexResults = await Promise.allSettled(indexSearchPromises);

        // Combine all successful results and track failures
        const successfulResults: Chunk[] = [];
        const indexErrors: { indexName: string; error: string }[] = [];

        indexResults.forEach((result) => {
          if (result.status === 'fulfilled') {
            // If this index search succeeded but had an error
            if (result.value.error) {
              indexErrors.push({
                indexName: result.value.indexName,
                error: result.value.error,
              });
            }
            // Add chunks from successful indexes
            successfulResults.push(...result.value.chunks);
          } else {
            // Handle rejected promises - should be rare as we catch errors in the index search function
            console.error(`Unexpected rejection for index search:`, result.reason);
            indexErrors.push({
              indexName: 'unknown',
              error:
                result.reason instanceof Error
                  ? result.reason.message
                  : String(result.reason),
            });
          }
        });

        // Track failed indexes for the final response
        failedIndexes.push(...indexErrors);

        // Log summary of results
        console.log(`Combined ${successfulResults.length} results from all indexes`);
        if (indexErrors.length > 0) {
          console.warn(
            `${indexErrors.length} indexes failed: ${indexErrors
              .map((e) => e.indexName)
              .join(', ')}`,
          );
        }

        // Sort by similarity score
        successfulResults.sort((a, b) => (b.score || 0) - (a.score || 0));

        // Take a larger number of top results to ensure we have enough after deduplication
        const topResults = successfulResults.slice(0, fetchCount);
        console.log(
          `Selected top ${topResults.length} chunks by score before document deduplication`,
        );

        allSearchResults.push(topResults);
      }

      // Combine results from all queries
      const allChunks = allSearchResults.flat();

      // Group chunks by document (file ID)
      const documentGroups = new Map<string, Chunk[]>();

      for (const chunk of allChunks) {
        if (chunk.metadata && chunk.metadata.file_id) {
          const docKey = chunk.metadata.file_id;
          if (!documentGroups.has(docKey)) {
            documentGroups.set(docKey, []);
          }
          documentGroups.get(docKey)?.push(chunk);
        }
      }

      console.log(`Grouped chunks into ${documentGroups.size} unique documents`);

      // Sort documents by their best chunk's score
      const sortedDocuments = Array.from(documentGroups.entries()).map(
        ([docId, chunks]) => {
          // Sort chunks within each document by score
          chunks.sort((a, b) => (b.score || 0) - (a.score || 0));

          return {
            docId,
            bestScore: chunks[0]?.score || 0,
            chunks,
          };
        },
      );

      // Sort documents by their best score
      sortedDocuments.sort((a, b) => b.bestScore - a.bestScore);

      // Take the top K unique documents
      const topDocuments = sortedDocuments.slice(0, topK);
      console.log(`Selected top ${topDocuments.length} unique documents`);

      // Collect all chunks from these top documents
      const uniqueDocChunks: Chunk[] = [];

      for (const doc of topDocuments) {
        uniqueDocChunks.push(...doc.chunks);
      }

      console.log(
        `Returning ${uniqueDocChunks.length} chunks from ${topDocuments.length} unique documents`,
      );

      // Determine if we should return partial success or full success
      const hasFailedIndexes = failedIndexes.length > 0;

      // Prepare results grouped by document
      const documentResults = topDocuments.map((doc) => {
        return {
          document_id: doc.docId,
          best_score: doc.bestScore,
          chunks: doc.chunks.map((chunk) => ({
            id: chunk.id,
            text: chunk.text,
            score: chunk.score,
            metadata: chunk.metadata,
          })),
        };
      });

      // Fetch additional metadata from KV for each document (not each chunk)
      const enhancedDocuments = await this.enrichDocumentsWithFileMetadata(
        documentResults,
        collection_id,
      );

      return {
        status: hasFailedIndexes ? 'partial_success' : 'success',
        documents: enhancedDocuments,
        total_chunks: uniqueDocChunks.length,
        errors: hasFailedIndexes ? failedIndexes : undefined,
      };
    } catch (error) {
      console.error('Error in findSimilarEmbeddings:', error);
      return {
        status: 'error',
        message: 'Failed to find similar embeddings',
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Fetch chunks from R2 bucket with concurrency limit
   */
  async fetchChunksFromR2(keys: string[]): Promise<ChunkResult[]> {
    console.log(
      `Fetching ${keys.length} chunks from R2 bucket with concurrency limit ${MAX_CONCURRENT_REQUESTS}`,
    );

    const results: ChunkResult[] = [];

    // If we have fewer keys than the concurrency limit, fetch them all in parallel
    if (keys.length <= MAX_CONCURRENT_REQUESTS) {
      const fetchPromises = keys.map((key) => this.fetchSingleChunk(key));
      const chunkResults = await Promise.all(fetchPromises);
      return chunkResults;
    }

    // Otherwise, fetch in batches with the concurrency limit
    for (let i = 0; i < keys.length; i += MAX_CONCURRENT_REQUESTS) {
      const batchKeys = keys.slice(i, i + MAX_CONCURRENT_REQUESTS);
      console.log(
        `Fetching batch ${i / MAX_CONCURRENT_REQUESTS + 1} with ${
          batchKeys.length
        } keys`,
      );

      const fetchPromises = batchKeys.map((key) => this.fetchSingleChunk(key));
      const batchResults = await Promise.all(fetchPromises);

      results.push(...batchResults);
    }

    return results;
  }

  /**
   * Fetch a single chunk from R2 bucket
   */
  async fetchSingleChunk(key: string): Promise<ChunkResult> {
    try {
      console.log(`Fetching chunk with key: ${key}`);
      const object = await this.env.RAMUS_EMBEDDINGS.get(key);

      if (!object) {
        return {
          key,
          status: 404,
          error: 'Object not found',
        };
      }

      const arrayBuffer = await object.arrayBuffer();
      const base64Data = this.arrayBufferToBase64(arrayBuffer);

      return {
        key,
        status: 200,
        contentType: object.httpMetadata?.contentType,
        size: arrayBuffer.byteLength,
        data: base64Data,
      };
    } catch (error) {
      console.error(`Error fetching chunk ${key}:`, error);
      return {
        key,
        status: 500,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Convert ArrayBuffer to base64 string
   */
  arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same dimensions');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (normA * normB);
  }

  /**
   * Query the Vectorize API directly
   */
  async queryVectorizeAPI(
    indexName: string,
    vector: number[],
    options: {
      topK?: number;
      filter?: Record<string, any>;
      namespace?: string;
      returnValues?: boolean;
      returnMetadata?: 'all' | 'none' | 'indexed';
    },
  ): Promise<VectorizeQueryResponse> {
    try {
      console.log(
        `Querying Vectorize API index ${indexName} with options:`,
        JSON.stringify(options, null, 2),
      );

      const url = `https://api.cloudflare.com/client/v4/accounts/${this.env.ACCOUNT_ID}/vectorize/v2/indexes/${indexName}/query`;

      const requestBody: {
        vector: number[];
        topK: number;
        returnValues: boolean;
        returnMetadata: 'all' | 'none' | 'indexed';
        namespace?: string;
        filter?: Record<string, any>;
      } = {
        vector,
        topK: options.topK || 5,
        returnValues: options.returnValues || false,
        returnMetadata: options.returnMetadata || 'all',
      };

      // Add namespace if provided
      if (options.namespace) {
        requestBody.namespace = options.namespace;
      }

      // Add filter if provided
      if (options.filter && Object.keys(options.filter).length > 0) {
        requestBody.filter = options.filter;
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.env.VECTORIZE_API_TOKEN}`,
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Vectorize API error (${response.status}):`, errorText);
        throw new Error(`Vectorize API error: ${response.status}`);
      }

      interface VectorizeAPIResponse {
        success: boolean;
        errors?: any[];
        result: {
          matches: VectorizeMatch[];
        };
      }

      const responseData = (await response.json()) as VectorizeAPIResponse;

      if (!responseData.success) {
        console.error('Vectorize API returned error:', responseData.errors);
        throw new Error(
          `Vectorize API returned error: ${JSON.stringify(responseData.errors)}`,
        );
      }

      console.log(
        `Vectorize API returned ${responseData.result.matches?.length || 0} matches`,
      );

      return {
        matches: responseData.result.matches || [],
      };
    } catch (error) {
      console.error(`Error querying Vectorize API:`, error);
      throw error;
    }
  }

  /**
   * Process filters and validate them against the collection indexes
   */
  processFilters(
    filters: Record<string, any>,
    collectionIndexes: CollectionIndexes,
  ): Record<string, any> {
    const validatedFilters: Record<string, any> = {};

    // Flatten metadata indexes from all collection indexes to make lookup easier
    const metadataIndexMap: Record<string, MetadataIndexConfig> = {};

    // Process all indexes in the collection
    for (const indexName of collectionIndexes.indexes) {
      // Get metadata indexes for this index
      const indexMetadataConfigs = collectionIndexes.metadata_indexes[indexName] || [];

      // Add each metadata index to the map
      for (const config of indexMetadataConfigs) {
        metadataIndexMap[config.propertyName] = config;
      }
    }

    console.log('Available metadata indexes:', Object.keys(metadataIndexMap));

    // Process each filter key-value pair
    for (const [key, value] of Object.entries(filters)) {
      // Special case for collection_id when not using namespace
      if (key === 'collection_id' && !USE_NAMESPACE_FOR_COLLECTION) {
        // Always add collection_id filter directly without validation
        validatedFilters[key] = value;
        continue;
      }

      // Skip if there's no metadata index config for this key
      if (!metadataIndexMap[key]) {
        console.warn(
          `Skipping filter for '${key}' as it has no metadata index configuration`,
        );
        continue;
      }

      const indexConfig = metadataIndexMap[key];
      console.log(
        `Processing filter for '${key}' with type '${indexConfig.indexType}'`,
      );

      // Process based on the filter's structure and metadata index type
      if (typeof value === 'object' && value !== null) {
        // This is a complex filter with operators
        const processedFilter: Record<string, any> = {};

        for (const [op, opValue] of Object.entries(value)) {
          // Validate operator based on metadata index type
          if (indexConfig.indexType === 'string') {
            // String type supports all operators
            if (
              ['$eq', '$ne', '$in', '$nin', '$lt', '$lte', '$gt', '$gte'].includes(
                op,
              )
            ) {
              processedFilter[op] = opValue;
            } else {
              console.warn(`Skipping invalid operator '${op}' for string type`);
            }
          } else if (indexConfig.indexType === 'number') {
            // Number type supports all operators
            if (
              ['$eq', '$ne', '$in', '$nin', '$lt', '$lte', '$gt', '$gte'].includes(
                op,
              )
            ) {
              // Convert to number if necessary
              if (typeof opValue === 'string') {
                const numValue = Number(opValue);
                if (!isNaN(numValue)) {
                  processedFilter[op] = numValue;
                } else {
                  console.warn(
                    `Skipping invalid number value '${opValue}' for operator '${op}'`,
                  );
                }
              } else if (Array.isArray(opValue) && ['$in', '$nin'].includes(op)) {
                // Convert array values to numbers if necessary
                const numArray = opValue
                  .map((v) => (typeof v === 'string' ? Number(v) : v))
                  .filter((v) => typeof v === 'number' && !isNaN(v));
                processedFilter[op] = numArray;
              } else {
                processedFilter[op] = opValue as any;
              }
            } else {
              console.warn(`Skipping invalid operator '${op}' for number type`);
            }
          } else if (indexConfig.indexType === 'boolean') {
            // Boolean type supports only $eq, $ne, $in, $nin
            if (['$eq', '$ne', '$in', '$nin'].includes(op)) {
              // Convert to boolean if necessary
              if (typeof opValue === 'string') {
                if (opValue === 'true') {
                  processedFilter[op] = true;
                } else if (opValue === 'false') {
                  processedFilter[op] = false;
                } else {
                  console.warn(
                    `Skipping invalid boolean value '${opValue}' for operator '${op}'`,
                  );
                }
              } else if (Array.isArray(opValue) && ['$in', '$nin'].includes(op)) {
                // Convert array values to booleans if necessary
                const boolArray = opValue
                  .map((v) => {
                    if (typeof v === 'string') {
                      return v === 'true' ? true : v === 'false' ? false : null;
                    }
                    return typeof v === 'boolean' ? v : null;
                  })
                  .filter((v) => v !== null);
                processedFilter[op] = boolArray;
              } else {
                processedFilter[op] = opValue as any;
              }
            } else {
              console.warn(`Skipping invalid operator '${op}' for boolean type`);
            }
          }
        }

        if (Object.keys(processedFilter).length > 0) {
          validatedFilters[key] = processedFilter;
        }
      } else {
        // This is a simple filter with implicit $eq
        if (indexConfig.indexType === 'string') {
          // String type
          if (typeof value === 'string') {
            validatedFilters[key] = value;
          } else {
            validatedFilters[key] = { $eq: String(value) };
          }
        } else if (indexConfig.indexType === 'number') {
          // Number type
          if (typeof value === 'number') {
            validatedFilters[key] = value;
          } else if (typeof value === 'string') {
            const numValue = Number(value);
            if (!isNaN(numValue)) {
              validatedFilters[key] = numValue;
            } else {
              console.warn(
                `Skipping invalid number value '${value}' for key '${key}'`,
              );
            }
          } else {
            console.warn(`Skipping non-numeric value for key '${key}'`);
          }
        } else if (indexConfig.indexType === 'boolean') {
          // Boolean type
          if (typeof value === 'boolean') {
            validatedFilters[key] = value;
          } else if (typeof value === 'string') {
            if (value === 'true') {
              validatedFilters[key] = true;
            } else if (value === 'false') {
              validatedFilters[key] = false;
            } else {
              console.warn(
                `Skipping invalid boolean value '${value}' for key '${key}'`,
              );
            }
          } else {
            console.warn(`Skipping non-boolean value for key '${key}'`);
          }
        }
      }
    }

    console.log('Validated filters:', validatedFilters);
    return validatedFilters;
  }

  /**
   * Enrich document results with additional file metadata from KV
   */
  async enrichDocumentsWithFileMetadata(
    documents: any[],
    collection_id: string,
  ): Promise<any[]> {
    // Create a map to store unique file keys to avoid duplicate KV lookups
    const fileMetadataKeys = new Map<string, any>();

    // Extract document metadata from the first chunk in each document
    for (const doc of documents) {
      if (doc.chunks && doc.chunks.length > 0) {
        const firstChunk = doc.chunks[0];
        if (firstChunk.metadata) {
          const userId = firstChunk.metadata.user_id;
          const collectionId = USE_NAMESPACE_FOR_COLLECTION
            ? collection_id
            : firstChunk.metadata.collection_id || collection_id;
          const fileId = firstChunk.metadata.file_id;

          if (userId && collectionId && fileId) {
            // Format the key as "{userid}:{collectionid}:{fileid}"
            const metadataKey = `${userId}:${collectionId}:${fileId}`;

            // Add to our map of keys to fetch
            if (!fileMetadataKeys.has(metadataKey)) {
              fileMetadataKeys.set(metadataKey, {
                docIndex: documents.indexOf(doc),
                docInfo: {
                  userId,
                  collectionId,
                  fileId,
                },
              });
            }
          }
        }
      }
    }

    console.log(
      `Fetching additional file metadata for ${fileMetadataKeys.size} unique documents`,
    );

    // If there are no keys to look up, return the original documents
    if (fileMetadataKeys.size === 0) {
      console.warn('No valid file metadata keys could be constructed');
      return documents;
    }

    try {
      // Convert our keys to an array
      const kvKeys = Array.from(fileMetadataKeys.keys());

      // Fetch all file metadata in batch to reduce API calls
      const metadataPromises = kvKeys.map(async (key) => {
        const result = await this.env.FILES_METADATA.getWithMetadata(key, 'json');
        return { key, result } as const;
      });

      const results = await Promise.all(metadataPromises);

      // Process results and enhance documents
      for (const { key, result } of results) {
        if (result.value !== null) {
          const docInfo = fileMetadataKeys.get(key);
          if (docInfo) {
            const docIndex = docInfo.docIndex;
            const doc = documents[docIndex];

            // Add file info to the document
            doc.file_info = result.value;

            // If there's KV metadata, add it under a separate key
            if (result.metadata) {
              doc.file_metadata = result.metadata;
            }

            console.log(`Enhanced document ${doc.document_id} with file metadata`);

            // Also add to each chunk for backward compatibility
            for (const chunk of doc.chunks) {
              chunk.metadata.file_info = result.value;
              if (result.metadata) {
                chunk.metadata.file_kv_metadata = result.metadata;
              }
            }
          }
        }
      }

      return documents;
    } catch (error) {
      console.error('Error fetching file metadata from KV:', error);

      // If there's an error, still return the original documents
      return documents;
    }
  }
}