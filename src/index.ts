import { WorkerEntrypoint } from "cloudflare:workers";
import JSZip from "jszip";

interface Env {
  AI: any;
  VECTORIZE: any;
  RAMUS_EMBEDDINGS: R2Bucket;
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
  async fetch(request: Request) {
    const body = await request.json() as {
      queries: string | string[];
      collection_id: string;
      topK?: number;
      corpus?: string;
      doc_id?: string;
      authored_start?: string;
      authored_end?: string;
    };

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
    
    return new Response(JSON.stringify(result));
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