import { WorkerEntrypoint } from "cloudflare:workers";
import JSZip from "jszip";

interface Env {
  AI: any;
  RAMUS_EMBEDDINGS: R2Bucket;
  DB: D1Database;
  VECTORIZE_API_TOKEN: string;
  ACCOUNT_ID: string;
  COLLECTIONS_METADATA: KVNamespace;
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
const USE_NAMESPACE_FOR_COLLECTION = false; // Set to false to use collection_id as a metadata field instead of namespace

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
    
    // Fetch collection indexes from D1
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
    
    const result = await this.findSimilarEmbeddings(queries, collection_id, topK, validatedFilters, collectionIndexes);
    console.log(`Search returned ${result?.matches?.length || 0} matches`);
    
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

      for (let i = 0; i < queryEmbeddings.data.length; i++) {
        const embedding = queryEmbeddings.data[i];
        console.log(`Processing query: "${queryArray[i]}" (truncated)`);

        // Calculate how many results to fetch from each index
        // We need to ensure we get enough results even after filtering
        const fetchCount = Math.min(Math.max(topK * 2, 10), 20);
        console.log(`Fetching top ${fetchCount} results from each index`);

        // Query each index in parallel and collect the results
        const indexSearchPromises = collectionIndexes.indexes.map(async (indexName) => {
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
              return [] as Chunk[];
            }

            console.log(`Found ${vectorMatches.matches.length} matches in index ${indexName}`);

            // Construct R2 keys for the matching vectors
            const r2KeysWithMetadata = vectorMatches.matches.map(match => {
              if (!match.metadata || !match.metadata.user_id || !match.metadata.file_id || !match.id) {
                console.error("Missing metadata fields in match:", match);
                return null;
              }
            
              // Construct R2 key using the metadata fields
              // Format: userId/collectionId/fileId/batchId.zip
              const userId = match.metadata.user_id;
              // Get collection_id from metadata if using metadata field approach, or use the parameter if using namespace
              const collId = USE_NAMESPACE_FOR_COLLECTION ? collection_id : match.metadata.collection_id || collection_id;
              const fileId = match.metadata.file_id;
              const batchId = match.id;
              
              return {
                key: `${userId}/${collId}/${fileId}/${batchId}.zip`,
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
            console.log(`Fetching ${r2Keys.length} chunks from R2 for index ${indexName}`);

            // Fetch chunks from R2 bucket with concurrency limit
            const chunkResults = await this.fetchChunksFromR2(r2Keys);
            
            if (!chunkResults || chunkResults.length === 0) {
              console.error(`No chunks returned from R2 for index ${indexName}`);
              return [] as Chunk[];
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
            return processedChunks;
            
          } catch (error) {
            console.error(`Error searching index ${indexName}:`, error);
            return [] as Chunk[];
          }
        });

        // Wait for all index searches to complete
        const indexResults = await Promise.all(indexSearchPromises);
        
        // Combine results from all indexes
        const combinedResults = indexResults.flat();
        console.log(`Combined ${combinedResults.length} results from all indexes`);
        
        // Sort by similarity score
        combinedResults.sort((a, b) => (b.score || 0) - (a.score || 0));
        
        // Take the top K
        const topResults = combinedResults.slice(0, topK);
        console.log(`Returning top ${topResults.length} results`);
        
        allSearchResults.push(topResults);
      }

      // Combine results from all queries
      const allChunks = allSearchResults.flat();
      
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
}