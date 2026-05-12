/**
 * ============================================================================
 *  HOUSEHOLD SIMILARITY RECOMMENDATION ENGINE
 *  ML Concept: Content-Based Filtering using Cosine Similarity
 * ============================================================================
 *
 *  WHAT ML CONCEPT IS BEING SIMULATED?
 *  ------------------------------------
 *  This module simulates "Content-Based Filtering" — a core technique used
 *  in recommendation systems (Netflix, Amazon, Spotify, etc.).
 *
 *  Instead of relying on user behavior (collaborative filtering), content-based
 *  filtering compares the *features* (attributes) of items to find similar ones.
 *  Here, each household is treated as an "item" with features like neighborhood,
 *  water usage, and household size.
 *
 *  HOW THE RECOMMENDATION SCORE WORKS:
 *  ------------------------------------
 *  1. Each household is converted into a numerical "feature vector"
 *  2. Neighborhoods are encoded using One-Hot Encoding (a standard ML technique)
 *  3. Numerical features (usage, household size) are normalized to 0-1 range
 *     using Min-Max Normalization (another standard ML technique)
 *  4. Cosine Similarity is computed between the selected household and all others
 *  5. The top-N most similar households are returned as recommendations
 *
 *  WHY THIS APPROACH IS BEGINNER-FRIENDLY ML:
 *  -------------------------------------------
 *  - No external libraries needed (pure JavaScript math)
 *  - Uses real ML concepts (feature engineering, vectorization, similarity)
 *  - Easy to visualize and explain each step
 *  - Same foundational concepts used in production ML systems
 *  - Great for learning: covers normalization, encoding, and distance metrics
 *
 * ============================================================================
 */

// ============================================================================
// STEP 1: SAMPLE HOUSEHOLD DATA
// In a real ML system, this would come from a database or API.
// Here we embed the processed dataset directly for frontend-only operation.
// ============================================================================

const householdData = [
  { id: "HH6578", neighborhood: "Tech Hub",       month: "Nov", usage: 314.2, size: 3, bill: 18.45, perPerson: 104.7, anomaly: false },
  { id: "HH3747", neighborhood: "Market Square",  month: "Dec", usage: 213.2, size: 4, bill:  9.80, perPerson:  53.3, anomaly: false },
  { id: "HH9838", neighborhood: "Old Town",       month: "Dec", usage: 212.8, size: 3, bill: 11.43, perPerson:  70.9, anomaly: false },
  { id: "HH1775", neighborhood: "Tech Hub",       month: "Sep", usage: 282.0, size: 4, bill: 13.52, perPerson:  70.5, anomaly: false },
  { id: "HH4843", neighborhood: "Green Park",     month: "Apr", usage: 197.0, size: 2, bill:  8.44, perPerson:  98.5, anomaly: false },
  { id: "HH2016", neighborhood: "Market Square",  month: "Feb", usage: 219.1, size: 4, bill: 12.72, perPerson:  54.8, anomaly: false },
  { id: "HH9571", neighborhood: "Market Square",  month: "Feb", usage: 282.7, size: 4, bill: 16.28, perPerson:  70.7, anomaly: false },
  { id: "HH6618", neighborhood: "Green Park",     month: "Jan", usage: 151.7, size: 3, bill:  7.03, perPerson:  50.6, anomaly: false },
  { id: "HH3204", neighborhood: "River Side",     month: "Jul", usage: 308.5, size: 2, bill: 15.92, perPerson: 154.3, anomaly: false },
  { id: "HH8841", neighborhood: "Old Town",       month: "Mar", usage: 165.3, size: 5, bill:  9.12, perPerson:  33.1, anomaly: false },
  { id: "HH7712", neighborhood: "Tech Hub",       month: "Jun", usage: 412.6, size: 3, bill: 21.05, perPerson: 137.5, anomaly: false },
  { id: "HH5023", neighborhood: "Green Park",     month: "Aug", usage: 251.3, size: 4, bill: 13.41, perPerson:  62.8, anomaly: false },
  { id: "HH1190", neighborhood: "River Side",     month: "May", usage: 225.0, size: 3, bill: 12.10, perPerson:  75.0, anomaly: false },
  { id: "HH6345", neighborhood: "Market Square",  month: "Jul", usage: 348.9, size: 5, bill: 18.62, perPerson:  69.8, anomaly: false },
  { id: "HH4400", neighborhood: "Old Town",       month: "Jun", usage: 228.4, size: 2, bill: 10.87, perPerson: 114.2, anomaly: false },
  { id: "HH2299", neighborhood: "Tech Hub",       month: "Jan", usage: 268.3, size: 5, bill: 14.98, perPerson:  53.7, anomaly: false },
  { id: "HH8100", neighborhood: "Green Park",     month: "Jul", usage: 240.8, size: 1, bill: 11.50, perPerson: 240.8, anomaly: false },
  { id: "HH3650", neighborhood: "River Side",     month: "Oct", usage: 210.5, size: 4, bill: 11.75, perPerson:  52.6, anomaly: false },
  { id: "HH9900", neighborhood: "Old Town",       month: "Aug", usage: 195.2, size: 3, bill:  8.96, perPerson:  65.1, anomaly: false },
  { id: "HH7788", neighborhood: "Market Square",  month: "Apr", usage: 260.0, size: 2, bill: 13.88, perPerson: 130.0, anomaly: false },
  { id: "HH1122", neighborhood: "Tech Hub",       month: "Jul", usage: 435.1, size: 2, bill: 23.10, perPerson: 217.6, anomaly: false },
  { id: "HH5544", neighborhood: "Green Park",     month: "Nov", usage: 175.6, size: 5, bill:  8.22, perPerson:  35.1, anomaly: false },
  { id: "HH3311", neighborhood: "River Side",     month: "Jun", usage: 295.8, size: 1, bill: 15.44, perPerson: 295.8, anomaly: false },
  { id: "HH4455", neighborhood: "Old Town",       month: "Apr", usage: 158.7, size: 4, bill:  7.20, perPerson:  39.7, anomaly: false },
  { id: "HH6677", neighborhood: "Market Square",  month: "Sep", usage: 245.3, size: 3, bill: 12.95, perPerson:  81.8, anomaly: false },
  { id: "HH8899", neighborhood: "Tech Hub",       month: "Mar", usage: 290.4, size: 1, bill: 15.67, perPerson: 290.4, anomaly: false },
  { id: "HH2233", neighborhood: "Green Park",     month: "Jun", usage: 262.5, size: 3, bill: 14.20, perPerson:  87.5, anomaly: false },
  { id: "HH7766", neighborhood: "River Side",     month: "Dec", usage: 185.3, size: 5, bill:  9.88, perPerson:  37.1, anomaly: false },
  { id: "HH9988", neighborhood: "Old Town",       month: "Jul", usage: 230.1, size: 2, bill: 11.22, perPerson: 115.1, anomaly: false },
  { id: "HH1144", neighborhood: "Market Square",  month: "Jun", usage: 355.0, size: 4, bill: 19.30, perPerson:  88.8, anomaly: false },
];


// ============================================================================
// STEP 2: FEATURE ENGINEERING — One-Hot Encoding for Neighborhoods
// ============================================================================
// In ML, categorical data (like "Tech Hub") can't be used directly in math.
// One-Hot Encoding converts each category into a binary vector.
//
// Example: If neighborhoods are [Green Park, Market Square, Old Town, River Side, Tech Hub]
//   "Tech Hub"      → [0, 0, 0, 0, 1]
//   "Green Park"    → [1, 0, 0, 0, 0]
//   "Market Square" → [0, 1, 0, 0, 0]
// ============================================================================

const NEIGHBORHOODS = ["Green Park", "Market Square", "Old Town", "River Side", "Tech Hub"];

/**
 * One-Hot Encode a neighborhood name into a binary vector.
 * @param {string} neighborhood - The neighborhood name
 * @returns {number[]} Binary vector of length 5
 */
function oneHotEncode(neighborhood) {
  // Create a vector of zeros, set 1 at the matching index
  return NEIGHBORHOODS.map(n => n === neighborhood ? 1 : 0);
}


// ============================================================================
// STEP 3: MIN-MAX NORMALIZATION
// ============================================================================
// Numerical features can have very different scales:
//   - Water usage: 50 to 500 liters
//   - Household size: 1 to 5 people
//
// Without normalization, the feature with the larger range would dominate
// the similarity calculation. Min-Max Normalization scales all values to [0, 1].
//
// Formula: normalized = (value - min) / (max - min)
// ============================================================================

/**
 * Normalize a value to [0, 1] range using Min-Max Normalization.
 * @param {number} value - The value to normalize
 * @param {number} min   - Minimum value in the dataset
 * @param {number} max   - Maximum value in the dataset
 * @returns {number} Normalized value between 0 and 1
 */
function minMaxNormalize(value, min, max) {
  // Edge case: if all values are the same, return 0
  if (max === min) return 0;
  return (value - min) / (max - min);
}

// Pre-compute min and max for each numerical feature across the dataset
const usageValues = householdData.map(h => h.usage);
const sizeValues  = householdData.map(h => h.size);
const billValues  = householdData.map(h => h.bill);

const FEATURE_RANGES = {
  usage: { min: Math.min(...usageValues), max: Math.max(...usageValues) },
  size:  { min: Math.min(...sizeValues),  max: Math.max(...sizeValues)  },
  bill:  { min: Math.min(...billValues),  max: Math.max(...billValues)  },
};


// ============================================================================
// STEP 4: FEATURE VECTOR CONSTRUCTION
// ============================================================================
// Each household is converted into a single numerical vector by combining:
//   - One-hot encoded neighborhood (5 values)
//   - Normalized water usage        (1 value)
//   - Normalized household size     (1 value)
//   - Normalized monthly bill       (1 value)
//
// Total feature vector length = 8
// ============================================================================

/**
 * Convert a household object into a numerical feature vector.
 * @param {Object} household - A household data object
 * @returns {number[]} Feature vector of length 8
 */
function toFeatureVector(household) {
  // One-hot encode the neighborhood (5 binary values)
  const neighborhoodVector = oneHotEncode(household.neighborhood);

  // Normalize numerical features to [0, 1]
  const normalizedUsage = minMaxNormalize(household.usage, FEATURE_RANGES.usage.min, FEATURE_RANGES.usage.max);
  const normalizedSize  = minMaxNormalize(household.size,  FEATURE_RANGES.size.min,  FEATURE_RANGES.size.max);
  const normalizedBill  = minMaxNormalize(household.bill,  FEATURE_RANGES.bill.min,  FEATURE_RANGES.bill.max);

  // Combine all features into a single vector
  // Structure: [GP, MS, OT, RS, TH, usage, size, bill]
  return [...neighborhoodVector, normalizedUsage, normalizedSize, normalizedBill];
}


// ============================================================================
// STEP 5: COSINE SIMILARITY CALCULATION
// ============================================================================
// Cosine Similarity measures the angle between two vectors in multi-dimensional
// space. It ranges from -1 (opposite) to 1 (identical).
//
// Formula: cos(θ) = (A · B) / (||A|| × ||B||)
//
// Where:
//   A · B   = dot product (sum of element-wise multiplication)
//   ||A||   = magnitude (square root of sum of squares)
//
// Why cosine similarity instead of Euclidean distance?
//   - It focuses on the *direction* (pattern) of features, not magnitude
//   - Works well with sparse vectors (like one-hot encodings)
//   - Standard in recommendation systems
// ============================================================================

/**
 * Compute the dot product of two vectors.
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} The dot product
 */
function dotProduct(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

/**
 * Compute the magnitude (L2 norm) of a vector.
 * @param {number[]} vec - Input vector
 * @returns {number} The magnitude
 */
function magnitude(vec) {
  return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
}

/**
 * Compute Cosine Similarity between two feature vectors.
 * @param {number[]} vecA - Feature vector of household A
 * @param {number[]} vecB - Feature vector of household B
 * @returns {number} Similarity score between 0 and 1
 */
function cosineSimilarity(vecA, vecB) {
  const dot = dotProduct(vecA, vecB);
  const magA = magnitude(vecA);
  const magB = magnitude(vecB);

  // Prevent division by zero
  if (magA === 0 || magB === 0) return 0;

  return dot / (magA * magB);
}


// ============================================================================
// STEP 6: RECOMMENDATION ENGINE
// ============================================================================
// This is the main function that ties everything together:
// 1. Takes a selected household ID
// 2. Converts it to a feature vector
// 3. Compares it against all other households using cosine similarity
// 4. Returns the top-N most similar households, sorted by score
// ============================================================================

/**
 * Get the top-N most similar households to the selected one.
 * @param {string} selectedId - The household ID to find recommendations for
 * @param {number} topN       - Number of recommendations to return (default: 5)
 * @returns {Array} Array of {household, score, featureBreakdown} objects
 */
function getRecommendations(selectedId, topN = 5) {
  // Find the selected household in our dataset
  const selected = householdData.find(h => h.id === selectedId);
  if (!selected) return [];

  // Convert the selected household to a feature vector
  const selectedVector = toFeatureVector(selected);

  // Calculate similarity score against every other household
  const scores = householdData
    .filter(h => h.id !== selectedId)  // Exclude the selected household itself
    .map(household => {
      const vector = toFeatureVector(household);
      const score = cosineSimilarity(selectedVector, vector);

      // Also compute a human-readable breakdown of why this is similar
      const featureBreakdown = {
        sameNeighborhood: household.neighborhood === selected.neighborhood,
        usageDifference: Math.abs(household.usage - selected.usage).toFixed(1),
        sizeDifference: Math.abs(household.size - selected.size),
        billDifference: Math.abs(household.bill - selected.bill).toFixed(2),
      };

      return { household, score, featureBreakdown };
    });

  // Sort by similarity score (highest first) and return top N
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topN);
}


// ============================================================================
// STEP 7: FILTER-BASED RECOMMENDATION (User-Driven)
// ============================================================================
// This function allows users to find similar households by specifying
// their own criteria — simulating how ML models accept input features.
// ============================================================================

/**
 * Find recommendations based on user-specified criteria.
 * @param {string} neighborhood - Preferred neighborhood
 * @param {number} maxUsage     - Maximum water usage (liters)
 * @param {number} householdSize - Household size
 * @param {number} topN          - Number of results (default: 5)
 * @returns {Array} Array of {household, score, featureBreakdown} objects
 */
function getFilteredRecommendations(neighborhood, maxUsage, householdSize, topN = 5) {
  // Create a "virtual" household representing the user's ideal profile
  const virtualHousehold = {
    id: "VIRTUAL",
    neighborhood: neighborhood,
    usage: maxUsage,
    size: householdSize,
    bill: maxUsage * 0.05,  // Estimate bill from usage
  };

  // Convert the virtual household to a feature vector
  const virtualVector = toFeatureVector(virtualHousehold);

  // Compare against all real households
  const scores = householdData.map(household => {
    const vector = toFeatureVector(household);
    const score = cosineSimilarity(virtualVector, vector);

    const featureBreakdown = {
      sameNeighborhood: household.neighborhood === neighborhood,
      usageDifference: Math.abs(household.usage - maxUsage).toFixed(1),
      sizeDifference: Math.abs(household.size - householdSize),
      billDifference: Math.abs(household.bill - virtualHousehold.bill).toFixed(2),
    };

    return { household, score, featureBreakdown };
  });

  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topN);
}


// ============================================================================
// STEP 8: UTILITY — Format Similarity Score for Display
// ============================================================================

/**
 * Convert a raw cosine similarity score to a human-friendly percentage.
 * @param {number} score - Raw cosine similarity (0 to 1)
 * @returns {string} Formatted percentage string
 */
function formatScore(score) {
  return (score * 100).toFixed(1) + "%";
}

/**
 * Get a color class based on the similarity score.
 * @param {number} score - Raw cosine similarity (0 to 1)
 * @returns {string} CSS class name for color coding
 */
function getScoreClass(score) {
  if (score >= 0.95) return "score-excellent";
  if (score >= 0.85) return "score-good";
  if (score >= 0.70) return "score-moderate";
  return "score-low";
}
