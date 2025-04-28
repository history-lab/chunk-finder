import { WorkerEntrypoint } from "cloudflare:workers";
import JSZip from "jszip";

interface Env {
  AI: any;
  RAMUS_EMBEDDINGS: R2Bucket;
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

// Types for collection indexes from D1
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

// Filter operation types
// type StringFilterOperator = '$eq' | '$ne' | '$in' | '$nin' | '$lt' | '$lte' | '$gt' | '$gte';
// type NumberFilterOperator = '$eq' | '$ne' | '$in' | '$nin' | '$lt' | '$lte' | '$gt' | '$gte';
// type BooleanFilterOperator = '$eq' | '$ne' | '$in' | '$nin';

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
   * @param collection_id - Collection ID to fetch indexes for
   * @returns Collection indexes information or null if not found
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
        metadata_indexes: parsedData.metadataIndexes || {}
      };
    } catch (error) {
      console.error(`Error fetching collection indexes from KV:`, error);
      return null;
    }
  }
  
  async fetch(request: Request) {
    const body = await request.json() as {
      queries: string | string[];
      collection_id: string;
      topK?: number;
      filters?: Record<string, any>;
    };

    console.log("Received search request:", JSON.stringify(body, null, 2));
    
    const { queries, collection_id, topK = 5, filters } = body;
    
    if (!queries) {
      console.error("Missing required parameter: queries");
      return new Response(JSON.stringify({
        status: 'error',
        message: 'Missing required parameter: queries'
      }), { status: 400 });
    }
    
    // Fetch collection indexes from KV using the provided collection_id
    const collectionIndexes = await this.getCollectionIndexes(collection_id);
    
    if (!collectionIndexes) {
      console.error(`Collection not found: ${collection_id}`);
      return new Response(JSON.stringify({
        status: 'error',
        message: `Collection not found: ${collection_id}`
      }), { status: 404 });
    }
    
    // Validate that the collection has at least one index
    if (!collectionIndexes.indexes || collectionIndexes.indexes.length === 0) {
      console.error(`No indexes found for collection: ${collection_id}`);
      return new Response(JSON.stringify({
        status: 'error',
        message: `No indexes found for collection: ${collection_id}`
      }), { status: 400 });
    }
    
    console.log(`Executing vector search with query: ${typeof queries === 'string' ? queries : queries.join(', ')}`);
    
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
        return new Response(JSON.stringify({
          status: 'error',
          message: error instanceof Error ? error.message : 'Invalid filters provided'
        }), { status: 400 });
      }
    }
    
    // Execute the similarity search using the provided collection_id
    const result = await this.findSimilarEmbeddings(queries, collection_id, topK, validatedFilters, collectionIndexes);
    
    console.log(`Search returned ${result?.documents?.length || 0} documents with a total of ${result?.total_chunks || 0} chunks`);
    
    return new Response(JSON.stringify(result));
  }

  /**
   * Find similar embeddings for multiple query terms across multiple indexes
   * @param queries - Array of query strings to search for
   * @param collection_id - The collection ID to search in
   * @param topK - Number of similar results to return for each query (default: 5)
   * @param filters - Optional metadata filtering parameters
   * @param collectionIndexes - Collection indexes information
   * @returns Array of matching results with metadata, sorted by similarity score
   */
  async findSimilarEmbeddings(
    queries: string | string[],
    collection_id: string,
    topK: number = 5,
    filters?: Record<string, any>,
    collectionIndexes?: CollectionIndexes
  ) {
    try {
      // Convert input to array if it's a single string
      const queryArray = Array.isArray(queries) ? queries : [queries];
      console.log(`Finding similar embeddings for ${queryArray.length} queries in collection ${collection_id}`);

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

      // Validate that the collection has at least one index
      if (!collectionIndexes.indexes || collectionIndexes.indexes.length === 0) {
        throw new Error(`No indexes found for collection: ${collection_id}`);
      }

      console.log(`Collection has ${collectionIndexes.indexes.length} indexes:`, collectionIndexes.indexes);

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

      console.log("Generated query embeddings");

      // For each query embedding, search across all indexes in parallel
      const allSearchResults: Chunk[][] = [];
      const failedIndexes: {indexName: string, error: string}[] = [];

      for (let i = 0; i < queryEmbeddings.data.length; i++) {
        const embedding = queryEmbeddings.data[i];
        console.log(`Processing query: "${queryArray[i]}" (truncated)`);

        // Calculate how many results to fetch from each index
        // We need to ensure we get enough results even after filtering
        // Increase the fetch count to ensure we get enough chunks for deduplication
        const fetchCount = Math.min(Math.max(topK * 2, 20), 10);
        console.log(`Fetching top ${fetchCount} results from each index to ensure document diversity`);

        // Query each index in parallel and collect the results
        // Use Promise.allSettled instead of Promise.all to handle partial failures
        const indexSearchPromises = collectionIndexes.indexes.map(async (indexName): Promise<IndexSearchResult> => {
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
              console.log(`Using metadata field for collection filtering: ${collection_id}`);
            }

            // Add filters if present
            if (filters && Object.keys(filters).length > 0) {
              // If we're already using filter for collection_id, merge with user filters
              if (queryOptions.filter) {
                queryOptions.filter = {
                  ...queryOptions.filter,
                  ...filters
                };
              } else {
                queryOptions.filter = filters;
              }
            }

            // Query the Vectorize API for this index
            const vectorMatches = await this.queryVectorizeAPI(indexName, embedding, queryOptions);

            if (!vectorMatches || !Array.isArray(vectorMatches.matches) || vectorMatches.matches.length === 0) {
              console.log(`No matches found in index ${indexName} for query "${queryArray[i]}"`);
              return { indexName, chunks: [] };
            }

            console.log(`Found ${vectorMatches.matches.length} matches in index ${indexName}`);

            // Add default user_id to ALL matches that don't have one (only if needed)
            for (const match of vectorMatches.matches) {
              if (match.metadata && !match.metadata.user_id) {
                console.log('Adding default user_id to match');
                match.metadata.user_id = '0000000001';
              }
            }

            // Construct R2 keys for the matching vectors
            const r2KeysWithMetadata = vectorMatches.matches.map(match => {
              if (!match.metadata || !match.metadata.user_id || !match.metadata.file_id || (!match.id && !match.metadata.batch_id)) {
                console.error("Missing metadata fields in match:", match);
                return null;
              }
            
              // Construct R2 key using the metadata fields
              // Format: userId/collectionId/fileId/batchId.zip
              const userId = match.metadata.user_id;
              // Get collection_id from metadata if using metadata field approach, or use the parameter if using namespace
              const collId = USE_NAMESPACE_FOR_COLLECTION ? collection_id : match.metadata.collection_id || collection_id;
              const fileId = match.metadata.file_id;
              const batchId = match.id || match.metadata.batch_id;
              
              return {
                key: `${userId}/${collId}/${fileId}/${batchId}.zip`,
                metadata: match.metadata,
                score: match.score,
                id: match.id
              };
            }).filter(item => item !== null);

            if (r2KeysWithMetadata.length === 0) {
              console.error("No valid R2 keys could be constructed");
              return { indexName, chunks: [] };
            }
            
            const r2Keys = r2KeysWithMetadata.map(item => item.key);
            console.log(`Fetching ${r2Keys.length} chunks from R2 for index ${indexName}`);

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
                      cluster_score: keyWithMetadata.score,
                      index_name: indexName // Add the index name to the metadata
                    }
                  };
                  
                  processedChunks.push(processedChunk);
                }
              } catch (error) {
                console.error(`Error processing chunk ${result.key}:`, error);
              }
            }

            console.log(`Processed ${processedChunks.length} chunks from index ${indexName}`);
            
            // Sort chunks by similarity score and return
            processedChunks.sort((a, b) => (b.score || 0) - (a.score || 0));
            return { indexName, chunks: processedChunks };
            
          } catch (error) {
            console.error(`Error searching index ${indexName}:`, error);
            
            // Instead of re-throwing, return a result with an error field
            return { 
              indexName, 
              chunks: [], 
              error: error instanceof Error ? error.message : String(error)
            };
          }
        });

        // Wait for all index searches to complete, even ones that fail
        const indexResults = await Promise.allSettled(indexSearchPromises);
        
        // Combine all successful results and track failures
        const successfulResults: Chunk[] = [];
        const indexErrors: {indexName: string, error: string}[] = [];
        
        indexResults.forEach(result => {
          if (result.status === 'fulfilled') {
            // If this index search succeeded but had an error
            if (result.value.error) {
              indexErrors.push({
                indexName: result.value.indexName,
                error: result.value.error
              });
            }
            // Add chunks from successful indexes
            successfulResults.push(...result.value.chunks);
          } else {
            // Handle rejected promises - should be rare as we catch errors in the index search function
            console.error(`Unexpected rejection for index search:`, result.reason);
            indexErrors.push({
              indexName: 'unknown',
              error: result.reason instanceof Error ? result.reason.message : String(result.reason)
            });
          }
        });
        
        // Track failed indexes for the final response
        failedIndexes.push(...indexErrors);
        
        // Log summary of results
        console.log(`Combined ${successfulResults.length} results from all indexes`);
        if (indexErrors.length > 0) {
          console.warn(`${indexErrors.length} indexes failed: ${indexErrors.map(e => e.indexName).join(', ')}`);
        }
        
        // Sort by similarity score
        successfulResults.sort((a, b) => (b.score || 0) - (a.score || 0));
        
        // Take a larger number of top results to ensure we have enough after deduplication
        const topResults = successfulResults.slice(0, fetchCount);
        console.log(`Selected top ${topResults.length} chunks by score before document deduplication`);
        
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
      const sortedDocuments = Array.from(documentGroups.entries()).map(([docId, chunks]) => {
        // Sort chunks within each document by score
        chunks.sort((a, b) => (b.score || 0) - (a.score || 0));
        
        return {
          docId,
          bestScore: chunks[0]?.score || 0,
          chunks
        };
      });
      
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
      
      console.log(`Returning ${uniqueDocChunks.length} chunks from ${topDocuments.length} unique documents`);
      
      // Determine if we should return partial success or full success
      const hasFailedIndexes = failedIndexes.length > 0;
      
      // Prepare results grouped by document
      const documentResults = topDocuments.map(doc => {
        return {
          document_id: doc.docId,
          best_score: doc.bestScore,
          chunks: doc.chunks.map(chunk => ({
            id: chunk.id,
            text: chunk.text,
            score: chunk.score,
            metadata: chunk.metadata
          }))
        };
      });
      
      // Fetch additional metadata from KV for each document (not each chunk)
      const enhancedDocuments = await this.enrichDocumentsWithFileMetadata(documentResults, collection_id);
      
      return {
        status: hasFailedIndexes ? 'partial_success' : 'success',
        documents: enhancedDocuments,
        total_chunks: uniqueDocChunks.length,
        errors: hasFailedIndexes ? failedIndexes : undefined
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

  /**
   * Query the Vectorize API directly
   * @param indexName - Name of the index to query
   * @param vector - Vector embedding to search with
   * @param options - Query options
   * @returns Vectorize query response
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
    }
  ): Promise<VectorizeQueryResponse> {
    try {
      console.log(`Querying Vectorize API index ${indexName} with options:`, JSON.stringify(options, null, 2));
      
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
          'Authorization': `Bearer ${this.env.VECTORIZE_API_TOKEN}`
        },
        body: JSON.stringify(requestBody)
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
      
      const responseData = await response.json() as VectorizeAPIResponse;
      
      if (!responseData.success) {
        console.error('Vectorize API returned error:', responseData.errors);
        throw new Error(`Vectorize API returned error: ${JSON.stringify(responseData.errors)}`);
      }
      
      console.log(`Vectorize API returned ${responseData.result.matches?.length || 0} matches`);
      
      return {
        matches: responseData.result.matches || []
      };
    } catch (error) {
      console.error(`Error querying Vectorize API:`, error);
      throw error;
    }
  }

  /**
   * Process filters and validate them against the collection indexes
   * @param filters - Filters to process
   * @param collectionIndexes - Collection indexes information
   * @returns Validated filters object with proper type conversions and operators
   */
  processFilters(filters: Record<string, any>, collectionIndexes: CollectionIndexes): Record<string, any> {
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
        console.warn(`Skipping filter for '${key}' as it has no metadata index configuration`);
        continue;
      }
      
      const indexConfig = metadataIndexMap[key];
      console.log(`Processing filter for '${key}' with type '${indexConfig.indexType}'`);
      
      // Process based on the filter's structure and metadata index type
      if (typeof value === 'object' && value !== null) {
        // This is a complex filter with operators
        const processedFilter: Record<string, any> = {};
        
        for (const [op, opValue] of Object.entries(value)) {
          // Validate operator based on metadata index type
          if (indexConfig.indexType === 'string') {
            // String type supports all operators
            if (['$eq', '$ne', '$in', '$nin', '$lt', '$lte', '$gt', '$gte'].includes(op)) {
              processedFilter[op] = opValue;
            } else {
              console.warn(`Skipping invalid operator '${op}' for string type`);
            }
          } else if (indexConfig.indexType === 'number') {
            // Number type supports all operators
            if (['$eq', '$ne', '$in', '$nin', '$lt', '$lte', '$gt', '$gte'].includes(op)) {
              // Convert to number if necessary
              if (typeof opValue === 'string') {
                const numValue = Number(opValue);
                if (!isNaN(numValue)) {
                  processedFilter[op] = numValue;
                } else {
                  console.warn(`Skipping invalid number value '${opValue}' for operator '${op}'`);
                }
              } else if (Array.isArray(opValue) && ['$in', '$nin'].includes(op)) {
                // Convert array values to numbers if necessary
                const numArray = opValue.map(v => typeof v === 'string' ? Number(v) : v)
                  .filter(v => typeof v === 'number' && !isNaN(v));
                processedFilter[op] = numArray;
              } else {
                processedFilter[op] = opValue;
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
                  console.warn(`Skipping invalid boolean value '${opValue}' for operator '${op}'`);
                }
              } else if (Array.isArray(opValue) && ['$in', '$nin'].includes(op)) {
                // Convert array values to booleans if necessary
                const boolArray = opValue.map(v => {
                  if (typeof v === 'string') {
                    return v === 'true' ? true : v === 'false' ? false : null;
                  }
                  return typeof v === 'boolean' ? v : null;
                }).filter(v => v !== null);
                processedFilter[op] = boolArray;
              } else {
                processedFilter[op] = opValue;
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
            validatedFilters[key] = { '$eq': String(value) };
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
              console.warn(`Skipping invalid number value '${value}' for key '${key}'`);
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
              console.warn(`Skipping invalid boolean value '${value}' for key '${key}'`);
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
   * Enriches document results with additional file metadata from KV
   * @param documents - Array of document results with their chunks to enrich
   * @param collection_id - The collection ID to use in metadata keys
   * @returns Enhanced array of document results with additional file metadata
   */
  async enrichDocumentsWithFileMetadata(documents: any[], collection_id: string): Promise<any[]> {
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
            : (firstChunk.metadata.collection_id || collection_id);
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
                  fileId
                }
              });
            }
          }
        }
      }
    }
    
    console.log(`Fetching additional file metadata for ${fileMetadataKeys.size} unique documents`);
    
    // If there are no keys to look up, return the original documents
    if (fileMetadataKeys.size === 0) {
      console.warn('No valid file metadata keys could be constructed');
      return documents;
    }
    
    try {
      // Convert our keys to an array
      const kvKeys = Array.from(fileMetadataKeys.keys());
      
      // Fetch all file metadata in batch to reduce API calls
      // Handle each key individually to correctly type the response
      const metadataPromises = kvKeys.map(async (key) => {
        const result = await this.env.FILES_METADATA.getWithMetadata(key, 'json');
        return { key, result };
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