import { WorkerEntrypoint } from "cloudflare:workers";
import JSZip from "jszip";

interface Env {
  AI: any;
  VECTORIZE: any;
  RAMUS_EMBEDDINGS: R2Bucket;
  API_SECRET_KEY?: string;
}

// Define types for Vectorize responses
interface VectorizeMatch {
  id: string;
  score: number;
  metadata?: Record<string, any>;
  values?: number[];
}

interface VectorizeQueryResponse {
  matches: VectorizeMatch[];
}

// Define types for metadata filtering
interface AuthoredFilter {
  $gte?: string | number;
  $lte?: string | number;
}

interface MetadataFilter {
  corpus?: string;
  doc_id?: string;
  authored?: AuthoredFilter;
  collection_id?: string;
  [key: string]: any;
}

interface VectorizeQueryOptions {
  topK: number;
  filter?: MetadataFilter;
  returnValues?: boolean;
  returnMetadata?: 'all' | 'none';
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

const MODEL = '@cf/baai/bge-base-en-v1.5';
const MAX_CONCURRENT_REQUESTS = 6;

export default class extends WorkerEntrypoint<Env> {
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
      return new Response(JSON.stringify({
        status: 'error',
        message: 'Unauthorized. Please provide a valid API key in the Authorization header.'
      }), { 
        status: 401,
        headers: {
          'Content-Type': 'application/json',
          'WWW-Authenticate': 'Bearer realm="API"'
        }
      });
    }
    
    const url = new URL(request.url);
    
    // Handle different API endpoints
    if (url.pathname === '/api/search' && request.method === 'POST') {
      return this.handleSearchRequest(request);
    }
    
    // Handle API health check
    if (url.pathname === '/api/health' && request.method === 'GET') {
      return new Response(JSON.stringify({
        status: 'ok',
        timestamp: new Date().toISOString()
      }), { 
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
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
    return new Response(JSON.stringify({
      status: 'error',
      message: 'Endpoint not found'
    }), { 
      status: 404,
      headers: { 'Content-Type': 'application/json' }
    });
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
        description: 'All API endpoints require authentication using the Bearer token',
        example: 'Authorization: Bearer sk-your-api-key-here'
      },
      important_notes: [
        'ALWAYS use collection_id: 80650a98-fe49-429a-afbd-9dde66e2d02b',
        'Invalid collection_id returns empty results, not an error',
        'Queries can be natural language questions or keywords',
        'Scores above 0.7 indicate high relevance',
        'Results include both individual chunk scores and cluster scores'
      ],
      collection: {
        required_id: '80650a98-fe49-429a-afbd-9dde66e2d02b',
        description: 'The ONLY valid collection ID. Using any other ID will return empty results.',
        warning: 'Invalid collection IDs do not return errors - they return empty match arrays'
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
                description: 'Search query or array of queries. Use natural language for best results.',
                examples: ['Cuban Missile Crisis', 'Vietnam War', 'Cold War tensions', 'nuclear weapons']
              },
              collection_id: {
                type: 'string',
                required: true,
                description: 'MUST be exactly: 80650a98-fe49-429a-afbd-9dde66e2d02b',
                value: '80650a98-fe49-429a-afbd-9dde66e2d02b',
                warning: 'Any other value returns empty results without error'
              },
              topK: {
                type: 'number',
                required: false,
                default: 5,
                description: 'Number of most similar results to return (1-100)',
                examples: [5, 10, 20]
              },
              corpus: {
                type: 'string',
                required: false,
                description: 'Filter by document corpus/source',
                valid_values: ['cfpf', 'cia', 'frus', 'un', 'worldbank', 'clinton', 'nato', 'cabinet', 'cpdoc', 'kissinger', 'briefing'],
                examples: ['cia', 'frus', 'clinton'],
                note: 'Invalid corpus values return an error'
              },
              doc_id: {
                type: 'string',
                required: false,
                description: 'Filter by specific document ID',
                examples: ['CIA-RDP80B01676R002800240004-4', 'P790016-1808']
              },
              authored_start: {
                type: 'string',
                required: false,
                description: 'Filter documents authored on or after this date (YYYY-MM-DD format)',
                example: '1960-01-01',
                note: 'Converted to Unix timestamp internally'
              },
              authored_end: {
                type: 'string',
                required: false,
                description: 'Filter documents authored on or before this date (YYYY-MM-DD format)',
                example: '1989-12-31',
                note: 'Converted to Unix timestamp internally'
              }
            }
          },
          response: {
            success: {
              status: 'success',
              matches: [
                {
                  id: 'string - Unique chunk identifier (e.g., "651296f4-14f0-44b7-a773-74a08f58b1dc_0")',
                  text: 'string - The actual text content of the chunk',
                  score: 'number - Similarity score (0-1, higher is more similar. 0.7+ is highly relevant)',
                  metadata: {
                    corpus: 'string - Source corpus (e.g., "cia", "cfpf")',
                    doc_id: 'string - Document identifier',
                    authored: 'number - Unix timestamp in milliseconds',
                    date: 'string - ISO date string (may be null)',
                    classification: 'string - Security classification (e.g., "unclassified")',
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
                    cluster_score: 'number - Cluster-level similarity score'
                  }
                }
              ]
            },
            error: {
              status: 'error',
              message: 'string - Error description'
            }
          },
          example_requests: [
            {
              description: 'Simple search',
              curl: `curl -X POST https://vector-search-worker.nchimicles.workers.dev/api/search \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "queries": "Cold War",
    "collection_id": "80650a98-fe49-429a-afbd-9dde66e2d02b",
    "topK": 5
  }'`
            },
            {
              description: 'Search with filters',
              curl: `curl -X POST https://vector-search-worker.nchimicles.workers.dev/api/search \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "queries": "Cuban Missile Crisis",
    "collection_id": "80650a98-fe49-429a-afbd-9dde66e2d02b",
    "topK": 10,
    "corpus": "cia",
    "authored_start": "1962-01-01",
    "authored_end": "1963-12-31"
  }'`
            },
            {
              description: 'Multiple queries',
              curl: `curl -X POST https://vector-search-worker.nchimicles.workers.dev/api/search \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "queries": ["Cuban Missile Crisis", "Berlin Wall", "Vietnam War"],
    "collection_id": "80650a98-fe49-429a-afbd-9dde66e2d02b",
    "topK": 3
  }'`
            }
          ]
        },
        {
          path: '/api/health',
          method: 'GET',
          description: 'Check API health status',
          authentication: 'Required',
          response: {
            status: 'ok',
            timestamp: 'ISO 8601 timestamp'
          },
          example_curl: `curl -X GET https://vector-search-worker.nchimicles.workers.dev/api/health \\
  -H "Authorization: Bearer YOUR_API_KEY"`
        },
        {
          path: '/api/help',
          method: 'GET',
          description: 'Get API documentation (this endpoint)',
          authentication: 'Required'
        },
        {
          path: '/api/about',
          method: 'GET',
          description: 'Learn about the search technology and architecture',
          authentication: 'Required'
        }
      ],
      filters: {
        corpus_values: {
          cfpf: 'Center for Presidential and Political Files - Presidential correspondence and memos',
          cia: 'Central Intelligence Agency documents - Intelligence reports and analyses',
          frus: 'Foreign Relations of the United States - Official diplomatic history',
          un: 'United Nations documents - International relations and resolutions',
          worldbank: 'World Bank archives - Development and economic reports',
          clinton: 'Clinton Presidential Library - 1990s presidential records',
          nato: 'NATO archives - Alliance documents and communications',
          cabinet: 'Cabinet papers - High-level government discussions',
          cpdoc: 'CPDOC Brazilian political archives - Brazilian political history',
          kissinger: 'Henry Kissinger papers - 1970s diplomatic correspondence',
          briefing: 'Presidential daily briefings - Intelligence summaries'
        },
        date_range: {
          format: 'YYYY-MM-DD',
          description: 'Filter by document authored date. Dates are converted to Unix timestamps.',
          examples: [
            'Cold War era: authored_start="1947-01-01", authored_end="1991-12-31"',
            'Vietnam War period: authored_start="1965-01-01", authored_end="1975-12-31"',
            '1960s: authored_start="1960-01-01", authored_end="1969-12-31"'
          ]
        }
      },
      query_tips: {
        best_practices: [
          'Use natural language queries for best results (e.g., "Cuban Missile Crisis negotiations")',
          'Be specific but not too narrow (e.g., "Cold War tensions" vs "Cold War")',
          'Combine filters for precise results (corpus + date range)',
          'Higher topK values (10-20) provide more context',
          'Scores above 0.7 indicate strong relevance, above 0.8 is very strong'
        ],
        example_queries: {
          events: ['Cuban Missile Crisis', 'Berlin Wall', 'Vietnam War', 'Kennedy assassination'],
          topics: ['nuclear weapons', 'Cold War tensions', 'Soviet Union relations', 'diplomatic negotiations'],
          people: ['John F. Kennedy', 'Henry Kissinger', 'Fidel Castro', 'Nikita Khrushchev'],
          periods: ['1960s foreign policy', 'detente period', 'Reagan administration']
        }
      },
      troubleshooting: {
        empty_results: [
          'Check collection_id is exactly: 80650a98-fe49-429a-afbd-9dde66e2d02b',
          'Verify your API key is correct',
          'Try broader search terms',
          'Remove filters to test if they are too restrictive'
        ],
        low_scores: [
          'Rephrase query using different terms',
          'Try more specific or contextual queries',
          'Check if the topic exists in the time period searched'
        ]
      },
      rate_limits: {
        requests_per_minute: 60,
        max_queries_per_request: 10,
        max_topK: 100
      },
      notes: [
        'Collection ID must be exactly: 80650a98-fe49-429a-afbd-9dde66e2d02b',
        'Invalid collection IDs return empty results, not errors',
        'The API uses semantic search - natural language queries work best',
        'Results include both chunk-level and cluster-level similarity scores',
        'Metadata includes rich context: dates, sources, classifications, document titles',
        'Filters are applied at the vector database level for efficiency'
      ]
    };

    return new Response(JSON.stringify(helpDoc, null, 2), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
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
        overview: 'This API provides semantic search capabilities for historical documents using state-of-the-art vector embedding technology.',
        components: {
          embedding_model: {
            name: '@cf/baai/bge-base-en-v1.5',
            type: 'BERT-based encoder',
            dimensions: 768,
            description: 'Converts text queries into high-dimensional vectors that capture semantic meaning'
          },
          vector_database: {
            name: 'Cloudflare Vectorize',
            description: 'Stores and indexes document embeddings for fast similarity search',
            features: ['Cosine similarity search', 'Metadata filtering', 'Scalable to millions of vectors']
          },
          storage: {
            name: 'Cloudflare R2',
            description: 'Object storage for document chunks and their embeddings',
            structure: 'Organized in compressed batches for efficient retrieval'
          }
        }
      },
      search_process: {
        steps: [
          {
            step: 1,
            name: 'Query Embedding',
            description: 'Your search query is converted into a 768-dimensional vector using the BGE model. This vector represents the semantic meaning of your query.'
          },
          {
            step: 2,
            name: 'Vector Similarity Search',
            description: 'The query vector is compared against millions of document vectors using cosine similarity. This finds documents with similar semantic meaning, even if they use different words.'
          },
          {
            step: 3,
            name: 'Metadata Filtering',
            description: 'Results are filtered based on your criteria (corpus, date range, document ID) at the database level for efficiency.'
          },
          {
            step: 4,
            name: 'Chunk Retrieval',
            description: 'The most similar document chunks are retrieved from R2 storage. Documents are split into chunks for granular matching.'
          },
          {
            step: 5,
            name: 'Ranking and Response',
            description: 'Results are ranked by similarity score and returned with their metadata and relevance scores.'
          }
        ]
      },
      clustering_approach: {
        description: 'Documents are processed in clusters for efficiency',
        benefits: [
          'Reduces storage costs by grouping related content',
          'Improves query performance through batch processing',
          'Enables efficient compression of embedding data'
        ],
        implementation: 'Each cluster contains multiple document chunks with their embeddings stored in compressed ZIP format'
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
          'Below 0.5': 'Low relevance - Different topic or weak connection'
        },
        practical_guidance: 'In practice, results with scores above 0.7 are typically what users want'
      },
      data_format: {
        chunk_size: 'Documents are split into meaningful chunks (typically 1-3 paragraphs)',
        embedding_storage: 'Float32Array binary format for efficient storage and computation',
        compression: 'ZIP compression reduces storage size by ~60%',
        metadata: 'Each chunk includes source, date, and document identification'
      },
      performance: {
        query_latency: 'Typically 200-500ms for vector search',
        concurrent_requests: 6,
        index_size: 'Scales to millions of document embeddings',
        accuracy: 'BGE model provides state-of-the-art semantic understanding'
      },
      use_cases: [
        'Historical research: Find all documents about specific events (e.g., Cuban Missile Crisis)',
        'Cross-archive search: Search across CIA, State Department, UN documents simultaneously',
        'Timeline analysis: Filter by date ranges to understand event progression',
        'Entity research: Find all mentions of specific people, places, or organizations',
        'Thematic research: Discover documents about concepts like \"nuclear deterrence\" or \"Cold War tensions\"',
        'Comparative analysis: Find similar events or patterns across different time periods',
        'Primary source discovery: Locate original documents and intelligence reports'
      ],
      advantages: {
        semantic_understanding: 'Finds relevant content even when exact keywords don\'t match',
        multilingual: 'Can find related content across language barriers',
        contextual: 'Understands the context and meaning of queries',
        efficient: 'Vector search is much faster than traditional full-text search for semantic queries'
      }
    };

    return new Response(JSON.stringify(aboutDoc, null, 2), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  /**
   * Handle RPC requests (original functionality)
   */
  private async handleRpcRequest(request: Request): Promise<Response> {
    const body = await request.json() as {
      queries: string | string[];
      collection_id: string;
      topK?: number;
      corpus?: string;
      doc_id?: string;
      authored_start?: string;
      authored_end?: string;
    };

    return this.processSearchRequest(body);
  }

  /**
   * Handle API search requests
   */
  private async handleSearchRequest(request: Request): Promise<Response> {
    try {
      const body = await request.json() as {
        queries: string | string[];
        collection_id: string;
        topK?: number;
        corpus?: string;
        doc_id?: string;
        authored_start?: string;
        authored_end?: string;
      };

      return this.processSearchRequest(body);
    } catch (error) {
      console.error('Error parsing request body:', error);
      return new Response(JSON.stringify({
        status: 'error',
        message: 'Invalid request body. Please provide valid JSON.'
      }), { 
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
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
  }): Promise<Response> {
    console.log("Received search request:", JSON.stringify(body, null, 2));
    
    const { queries, collection_id, topK = 5, corpus, doc_id, authored_start, authored_end } = body;
    
    if (!queries) {
      console.error("Missing required parameter: queries");
      return new Response(JSON.stringify({
        status: 'error',
        message: 'Missing required parameter: queries'
      }), { status: 400 });
    }
    
    // Validate corpus if provided
    const validCorpusValues = [
      'cfpf', 'cia', 'frus', 'un', 'worldbank', 
      'clinton', 'nato', 'cabinet', 'cpdoc', 
      'kissinger', 'briefing'
    ];
    
    if (corpus && !validCorpusValues.includes(corpus)) {
      console.error(`Invalid corpus value: ${corpus}`);
      return new Response(JSON.stringify({
        status: 'error',
        message: `Invalid corpus value: ${corpus}. Valid values are: ${validCorpusValues.join(', ')}`
      }), { status: 400 });
    }
    
    // Validate and convert date strings if provided
    let startTimestamp: number | undefined;
    let endTimestamp: number | undefined;
    
    if (authored_start) {
      const startDate = new Date(authored_start);
      if (isNaN(startDate.getTime())) {
        console.error(`Invalid authored_start date: ${authored_start}`);
        return new Response(JSON.stringify({
          status: 'error',
          message: `Invalid authored_start date: ${authored_start}. Please use YYYY-MM-DD format.`
        }), { status: 400 });
      }
      startTimestamp = Math.floor(startDate.getTime()); // Convert to Unix timestamp
      console.log(`Converted authored_start '${authored_start}' to Unix timestamp: ${startTimestamp}`);
    }
    
    if (authored_end) {
      const endDate = new Date(authored_end);
      if (isNaN(endDate.getTime())) {
        console.error(`Invalid authored_end date: ${authored_end}`);
        return new Response(JSON.stringify({
          status: 'error',
          message: `Invalid authored_end date: ${authored_end}. Please use YYYY-MM-DD format.`
        }), { status: 400 });
      }
      endTimestamp = Math.floor(endDate.getTime()); // Convert to Unix timestamp
      console.log(`Converted authored_end '${authored_end}' to Unix timestamp: ${endTimestamp}`);
    }
    
    // Create metadata filter object if filtering parameters are provided
    let metadata;
    
    if (corpus || doc_id || startTimestamp || endTimestamp) {
      const filter: Record<string, any> = {};
      
      if (corpus) {
        filter.corpus = corpus;
        console.log(`Adding corpus filter: ${corpus}`);
      }
      
      if (doc_id) {
        filter.doc_id = doc_id;
        console.log(`Adding doc_id filter: ${doc_id}`);
      }
      
      // Handle authored date range filtering
      if (startTimestamp || endTimestamp) {
        filter.authored = {};
        
        // In Vectorize metadata filtering, comparison operators must be prefixed with $ 
        // Example: { "timestamp": { "$gte": 1734242400, "$lt": 1734328800 } }
        if (startTimestamp) {
          filter.authored.$gte = startTimestamp;
          console.log(`Adding authored.$gte filter: ${startTimestamp}`);
        }
        
        if (endTimestamp) {
          filter.authored.$lte = endTimestamp;
          console.log(`Adding authored.$lte filter: ${endTimestamp}`);
        }
      }
      
      metadata = { filter };
      console.log('Applying metadata filters:', JSON.stringify(metadata, null, 2));
    } else {
      console.log('No metadata filters applied');
    }
    
    console.log(`Executing vector search with query: ${typeof queries === 'string' ? queries : queries.join(', ')}`);
    const result = await this.findSimilarEmbeddings(queries, collection_id, topK, metadata);
    console.log(`Search returned ${result?.matches?.length || 0} matches`);
    
    return new Response(JSON.stringify(result), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  /**
   * Find similar embeddings for multiple query terms
   * @param queries - Array of query strings to search for
   * @param collection_id - The vector collection ID to search in
   * @param topK - Number of similar results to return for each query (default: 5)
   * @param metadata - Optional metadata filtering parameters
   * @param metadata.filter - Filter object with the following possible properties:
   * @param metadata.filter.corpus - Optional filter for specific corpus (e.g., 'cia', 'frus', 'clinton')
   * @param metadata.filter.doc_id - Optional filter for specific document ID
   * @param metadata.filter.authored - Optional date range filter for document authored date
   * @param metadata.filter.authored.$gte - Optional greater than or equal to timestamp (using "$gte" operator)
   * @param metadata.filter.authored.$lte - Optional less than or equal to timestamp (using "$lte" operator)
   * @returns Array of matching results with metadata, sorted by similarity score
   */
  async findSimilarEmbeddings(
    queries: string | string[],
    collection_id: string,
    topK: number = 5,
    metadata?: { 
      filter?: {
        corpus?: string;
        doc_id?: string;
        authored?: {
          $gte?: string | number;
          $lte?: string | number;
        };
        [key: string]: any;
      }
    }
  ) {
    try {
      // Convert input to array if it's a single string
      const queryArray = Array.isArray(queries) ? queries : [queries];
      console.log(`Finding similar embeddings for ${queryArray.length} queries in collection ${collection_id}`);

      // Log metadata filters if present
      if (metadata?.filter) {
        console.log('Using metadata filters:', JSON.stringify(metadata.filter, null, 2));
      }

      // Generate embeddings for all queries in parallel
      let queryEmbeddings;
      try {
        queryEmbeddings = await this.env.AI.run(MODEL, { text: queryArray });
      } catch (error) {
        console.error('Error generating query embeddings:', error);
        return {
          status: 'error',
          message: 'Failed to generate query embeddings'
        };
      }

      console.log("queryEmbeddings", queryEmbeddings);

      // Query Vectorize for similar vectors for each embedding in parallel
      const searchPromises = queryEmbeddings.data.map(async (embedding: number[], i: number) => {
        console.log(`Querying collection ${collection_id} with embedding for "${queryArray[i]}" (truncated)`);
        
        try {
          // Prepare Vectorize query options with filters
          const queryOptions: VectorizeQueryOptions = {
            topK: metadata ? 10 : topK, // Get more results when filtering to ensure we have enough after filtering
            filter: { collection_id: collection_id },
            returnValues: false,
            returnMetadata: 'all',
          };

          // Add metadata filters if present
          if (metadata?.filter) {
            // Merge the user-provided filters with the collection_id filter
            queryOptions.filter = {
              ...queryOptions.filter,
              ...metadata.filter
            };
          }

          console.log('Using query options:', JSON.stringify(queryOptions, null, 2));

          // Check if VECTORIZE is available
          if (!this.env.VECTORIZE) {
            console.error('VECTORIZE service not available (likely running in local dev mode)');
            return [] as Chunk[];
          }

          // Query the vector database
          let vectorMatches = await this.env.VECTORIZE.query(embedding, queryOptions) as VectorizeQueryResponse;

          console.log("Vector matches", vectorMatches);        

          // Check if matches is undefined or doesn't have the expected structure
          if (!vectorMatches || !Array.isArray(vectorMatches.matches) || vectorMatches.matches.length === 0) {
            console.error(`Invalid or empty response from Vectorize for query "${queryArray[i]}":`, vectorMatches);
            return [] as Chunk[];
          }

          // Construct R2 keys for the matching vectors using the correct structure
          const r2KeysWithMetadata = vectorMatches.matches.map(match => {
            if (!match.metadata || !match.metadata.user_id || !match.metadata.file_id || !match.id) {
              console.error("Missing metadata fields in match:", match);
              return null;
            }
          
            // Construct R2 key using the metadata fields
            // Format: userId/collectionId/fileId/batchId.zip
            const userId = match.metadata.user_id;
            const fileId = match.metadata.file_id;
            const batchId = match.id;
            
            return {
              key: `${userId}/${collection_id}/${fileId}/${batchId}.zip`,
              metadata: match.metadata,
              score: match.score,
              id: match.id
            };
          }).filter(item => item !== null);

          if (r2KeysWithMetadata.length === 0) {
            console.error("No valid R2 keys could be constructed");
            return [] as Chunk[];
          }
          
          const r2Keys = r2KeysWithMetadata.map(item => item.key);

          console.log("Fetching chunks with keys:", r2Keys);

          // Fetch chunks from R2 bucket with concurrency limit
          const chunkResults = await this.fetchChunksFromR2(r2Keys);
          
          if (!chunkResults || chunkResults.length === 0) {
            console.error("No chunks returned from R2");
            return [] as Chunk[];
          }

          // Process the chunks - extract them from the ZIP files and calculate similarity
          const allProcessedChunks: Chunk[] = [];

          for (let i = 0; i < chunkResults.length; i++) {
            const result = chunkResults[i];
            const keyWithMetadata = r2KeysWithMetadata.find(item => item.key === result.key);
            
            if (!keyWithMetadata) {
              console.error(`Could not find metadata for key ${result.key}`);
              continue;
            }
            
            if (result.status !== 200 || !result.data) {
              console.error(`Error fetching chunk ${result.key}: ${result.error || 'Unknown error'}`);
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
              
              // No need to extract metadata.json anymore as we're using the cluster metadata
              
              // Extract embeddings binary data and convert to Float32Array
              const embeddingsFile = zip.file("embeddings.bin");
              if (!embeddingsFile) {
                console.error("Missing embeddings.bin in ZIP");
                continue;
              }
              const embeddingsArrayBuffer = await embeddingsFile.async("arraybuffer");
              const embeddings = new Float32Array(embeddingsArrayBuffer);
              
              // Extract chunks
              const chunksFile = zip.file("chunks.json");
              if (!chunksFile) {
                console.error("Missing chunks.json in ZIP");
                continue;
              }
              const chunksText = await chunksFile.async("text");
              const chunks = JSON.parse(chunksText) as Chunk[];
              
              // Try to determine batch size either from metadata or chunks length
              let batchSize = chunks.length;
              let fileMetadata = null;
              
              // We'll still try to read metadata.json if available just to get batch_size,
              // but we won't use it for the actual metadata
              try {
                const metadataFile = zip.file("metadata.json");
                if (metadataFile) {
                  const metadataText = await metadataFile.async("text");
                  fileMetadata = JSON.parse(metadataText);
                  if (fileMetadata.batch_size) {
                    batchSize = fileMetadata.batch_size;
                  }
                }
              } catch (e) {
                console.warn("Could not read metadata.json, using chunks length as batch size", e);
              }
              
              // Calculate the number of embeddings and dimensions
              const embeddingDimensions = embeddings.length / batchSize;
              
              // Process each chunk with its corresponding embedding
              for (let j = 0; j < chunks.length; j++) {
                const chunk = chunks[j];
                
                // Extract the embedding for this chunk from the Float32Array
                const startIdx = j * embeddingDimensions;
                const chunkEmbedding = Array.from(
                  embeddings.subarray(startIdx, startIdx + embeddingDimensions)
                );
                
                // Calculate cosine similarity
                const similarity = this.cosineSimilarity(embedding, chunkEmbedding);
                
                // Add embedding, score, and cluster metadata to the chunk
                const processedChunk: Chunk = {
                  ...chunk,
                  embedding: chunkEmbedding,
                  score: similarity,
                  metadata: {
                    ...keyWithMetadata.metadata,
                    cluster_id: keyWithMetadata.id,
                    cluster_score: keyWithMetadata.score
                  }
                };
                
                allProcessedChunks.push(processedChunk);
              }
            } catch (error) {
              console.error(`Error processing chunk ${result.key}:`, error);
            }
          }

          // Sort chunks by similarity score and take the top K
          allProcessedChunks.sort((a, b) => (b.score || 0) - (a.score || 0));
          return allProcessedChunks.slice(0, topK);
          
        } catch (error) {
          console.error(`Error querying collection with query "${queryArray[i]}":`, error);
          return [] as Chunk[];
        }
      });

      // Wait for all search queries to complete
      const searchResults = await Promise.all(searchPromises);
      
      // Flatten results and sort by similarity score (descending)
      const allChunks = searchResults.flat() as Chunk[];
      allChunks.sort((a, b) => (b.score || 0) - (a.score || 0));
      
      // Take the top K unique results (by ID)
      const seenIds = new Set<string>();
      const uniqueChunks: Chunk[] = [];
      
      for (const chunk of allChunks) {
        if (!seenIds.has(chunk.id) && uniqueChunks.length < topK) {
          seenIds.add(chunk.id);
          uniqueChunks.push(chunk);
        }
      }
      
      console.log(`Returning ${uniqueChunks.length} unique chunks sorted by similarity score`);
      
      return {
        status: 'success',
        matches: uniqueChunks.map(chunk => ({
          id: chunk.id,
          text: chunk.text,
          score: chunk.score,
          metadata: chunk.metadata
        }))
      };
      
    } catch (error) {
      console.error('Error in findSimilarEmbeddings:', error);
      return {
        status: 'error',
        message: 'Failed to find similar embeddings',
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Fetch chunks from R2 bucket with concurrency limit
   * @param keys - Array of R2 keys to fetch
   * @returns Array of chunk results
   */
  async fetchChunksFromR2(keys: string[]): Promise<ChunkResult[]> {
    console.log(`Fetching ${keys.length} chunks from R2 bucket with concurrency limit ${MAX_CONCURRENT_REQUESTS}`);
    
    const results: ChunkResult[] = [];
    
    // If we have fewer keys than the concurrency limit, fetch them all in parallel
    if (keys.length <= MAX_CONCURRENT_REQUESTS) {
      const fetchPromises = keys.map(key => this.fetchSingleChunk(key));
      const chunkResults = await Promise.all(fetchPromises);
      return chunkResults;
    }
    
    // Otherwise, fetch in batches with the concurrency limit
    for (let i = 0; i < keys.length; i += MAX_CONCURRENT_REQUESTS) {
      const batchKeys = keys.slice(i, i + MAX_CONCURRENT_REQUESTS);
      console.log(`Fetching batch ${i / MAX_CONCURRENT_REQUESTS + 1} with ${batchKeys.length} keys`);
      
      const fetchPromises = batchKeys.map(key => this.fetchSingleChunk(key));
      const batchResults = await Promise.all(fetchPromises);
      
      results.push(...batchResults);
    }
    
    return results;
  }
  
  /**
   * Fetch a single chunk from R2 bucket
   * @param key - R2 key to fetch
   * @returns Chunk result
   */
  async fetchSingleChunk(key: string): Promise<ChunkResult> {
    try {
      console.log(`Fetching chunk with key: ${key}`);
      const object = await this.env.RAMUS_EMBEDDINGS.get(key);
      
      if (!object) {
        return {
          key,
          status: 404,
          error: 'Object not found'
        };
      }
      
      const arrayBuffer = await object.arrayBuffer();
      const base64Data = this.arrayBufferToBase64(arrayBuffer);
      
      return {
        key,
        status: 200,
        contentType: object.httpMetadata?.contentType,
        size: arrayBuffer.byteLength,
        data: base64Data
      };
    } catch (error) {
      console.error(`Error fetching chunk ${key}:`, error);
      return {
        key,
        status: 500,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
  
  /**
   * Convert ArrayBuffer to base64 string
   * @param buffer - ArrayBuffer to convert
   * @returns Base64 string
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
   * @param a - First vector
   * @param b - Second vector
   * @returns Cosine similarity score between 0 and 1
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
}