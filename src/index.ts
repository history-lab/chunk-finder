import { WorkerEntrypoint } from "cloudflare:workers";

interface Env {
  AI: any;
  VECTORIZE: any;
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

const MODEL = '@cf/baai/bge-base-en-v1.5';

export default class extends WorkerEntrypoint<Env> {  
  async fetch(request: Request) {
    const body = await request.json() as {
      queries: string | string[];
      collection_id: string;
      topK?: number;
    };

    console.log("received request", body);
    
    const { queries, collection_id, topK = 5 } = body;
    
    if (!queries) {
      return new Response(JSON.stringify({
        status: 'error',
        message: 'Missing required parameter: queries'
      }), { status: 400 });
    }
    
    const result = await this.findSimilarEmbeddings(queries, collection_id, topK);
    return new Response(JSON.stringify(result));
  }

  /**
   * Find similar embeddings for multiple query terms
   * @param queries - Array of query strings to search for
   * @param collection_id - The vector collection ID to search in
   * @param topK - Number of similar results to return for each query (default: 5)
   * @returns Array of matching results with metadata, sorted by similarity score
   */
  async findSimilarEmbeddings(
    queries: string | string[],
    collection_id: string,
    topK: number = 5
  ) {
    try {
      // Convert input to array if it's a single string
      const queryArray = Array.isArray(queries) ? queries : [queries];
      console.log(`Finding similar embeddings for ${queryArray.length} queries in collection ${collection_id}`);

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
          // Query the vector database
          let matches = await this.env.VECTORIZE.query(embedding, {
            topK: topK,
            filter: { collection_id: collection_id },
            returnValues: false,
            returnMetadata: 'all',
          }) as VectorizeQueryResponse;

          console.log("matches", matches);        

          // Check if matches is undefined or doesn't have the expected structure
          if (!matches || !Array.isArray(matches.matches)) {
            console.error(`Invalid response from Vectorize for query "${queryArray[i]}":`, matches);
            return [] as VectorizeMatch[];
          }

          if (matches.matches.length !== 0) {
            // loop through mathches and get match.metadata and print it
            for (const match of matches.matches) {
              console.log("match metadata", match.metadata);
            }
          }
          
          return matches.matches;
        } catch (error) {
          console.error(`Error querying collection with query "${queryArray[i]}":`, error);
          return [] as VectorizeMatch[];
        }
      });

      // Wait for all search queries to complete
      const searchResults = await Promise.all(searchPromises);
      
      // Flatten results and sort by similarity score (descending)
      const allMatches = searchResults.flat() as VectorizeMatch[];
      allMatches.sort((a, b) => b.score - a.score);
      
      // Take the top K unique results (by ID)
      const seenIds = new Set<string>();
      const uniqueMatches: VectorizeMatch[] = [];
      
      for (const match of allMatches) {
        if (!seenIds.has(match.id) && uniqueMatches.length < topK) {
          seenIds.add(match.id);
          uniqueMatches.push(match);
        }
      }
      
      console.log(`Returning ${uniqueMatches.length} unique matches sorted by similarity score`);
      
      return {
        status: 'success',
        matches: uniqueMatches
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
}