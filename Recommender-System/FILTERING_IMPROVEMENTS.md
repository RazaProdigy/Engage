# Filtering and Fallback System Improvements

## Overview
This document describes the improvements made to the restaurant recommendation system's filtering and fallback logic to provide better user experience when exact matches are not available.

## Changes Made

### 1. **Hard vs Soft Filtering Strategy**

#### Previous Behavior:
- Location was a **hard filter** - restaurants in different locations were completely eliminated
- Amenities (like outdoor seating) were a **soft filter** - only affected scoring
- When no exact matches existed, the system would show nothing or apply generic fallback

#### New Behavior:
- **Cuisine**: Still a hard filter (must match)
- **Location**: Hard filter initially, becomes soft filter in fallback mode
- **Required Amenities**: Now a **hard filter** (must match if specified)
- **Price Range**: Soft filter with 20% flexibility
- **Rating**: Soft filter (affects scoring)

### 2. **Intelligent Fallback Logic**

#### File: `src/rag_system.py`

**New Method: `_apply_fallback_filters()`**
```python
def _apply_fallback_filters(
    self,
    results: List[Tuple[Document, float]],
    entities: Dict[str, Any]
) -> Tuple[List[Tuple[Document, float]], Dict[str, Any]]:
```

**How it works:**
1. When initial filtering returns zero results, the system detects this
2. Automatically applies fallback logic:
   - **Keeps cuisine as hard filter** (still must match)
   - **Relaxes location constraint** (becomes soft filter with scoring)
   - **Keeps amenities as hard filter** (still must match)
   - **Tracks metadata** about what was relaxed

3. Returns:
   - Filtered results with adjusted scores
   - `fallback_applied` flag
   - `original_criteria` (what the user actually requested)
   - `fallback_message` (explanation for the user)

**Example:**
```
Query: "Italian restaurants in Downtown Dubai with outdoor seating"

Original Filtering:
- Cuisine = Italian ✓ (hard)
- Location = Downtown Dubai ✓ (hard)
- Amenities = outdoor seating ✓ (hard)
Result: 0 restaurants (none exist)

Fallback Filtering:
- Cuisine = Italian ✓ (hard - still required)
- Location = anywhere ⚠️ (soft - prefers Downtown but allows others)
- Amenities = outdoor seating ✓ (hard - still required)
Result: Italian restaurants with outdoor seating in Sharjah, Abu Dhabi, etc.
```

### 3. **Enhanced Response Generation**

#### File: `src/agents.py`

**Changes to `ResponseGenerationAgent`:**

1. **Receives fallback metadata:**
   - `fallback_applied` flag
   - `original_criteria` dictionary
   - `fallback_message` string

2. **Passes context to LLM:**
   ```python
   if fallback_applied and original_criteria:
       context_note = "IMPORTANT: No exact matches were found for the original criteria..."
   ```

3. **LLM prompt includes:**
   - Clear indication that results are alternatives, not exact matches
   - Original location requested
   - Why alternatives are being shown

4. **Updated system prompt** (`src/config.py`):
   ```
   IMPORTANT - If no exact matches were found:
   - First acknowledge that no restaurants match the exact criteria
   - Clearly explain what was not available
   - Then suggest the nearby alternatives being shown
   - Be transparent about why these are alternatives
   ```

### 4. **Metadata Tracking**

**Return Type Change:**
`hybrid_search()` now returns a dictionary instead of a list:

```python
return {
    "results": [(doc, score), ...],
    "fallback_applied": bool,
    "original_criteria": {
        "cuisine": "Italian",
        "location": "Downtown Dubai",
        "amenities": ["outdoor seating"]
    },
    "fallback_message": "No exact matches found..."
}
```

### 5. **State Management**

**New fields in `AgentState` TypedDict:**
```python
fallback_applied: bool
fallback_message: Optional[str]
original_criteria: Optional[Dict[str, Any]]
```

These fields flow through the entire agent workflow:
1. Query Understanding → Extract entities
2. Retrieval Agent → Apply filters, detect fallback → Store metadata
3. Response Generation → Use metadata → Generate transparent response

## Example Scenarios

### Scenario 1: Exact Match Available
```
Query: "French restaurants in Downtown Dubai"

Result:
✓ Found "Desert Palace" - French, Downtown Dubai
✓ Direct match, no fallback needed
```

### Scenario 2: No Exact Match - Location Fallback
```
Query: "Italian restaurants in Downtown Dubai with outdoor seating"

Filtering:
1. Hard filters: cuisine=Italian, amenities=outdoor seating
2. Initial location filter: Downtown Dubai → 0 results
3. Fallback activated: Relax location constraint
4. Result: Italian restaurants with outdoor seating in nearby areas

Response to User:
"I couldn't find any Italian restaurants in Downtown Dubai with outdoor seating. 
However, I found some great Italian options with outdoor seating in nearby areas 
like Sharjah that might interest you:

1. **Central Italian Eatery** (Sharjah)
   - Price Range: AED 100-150
   - Unfortunately no outdoor seating, but nearby in Sharjah
   ...
```

### Scenario 3: Required Amenity Missing
```
Query: "Find restaurants with outdoor seating under AED 200"

Filtering:
- Amenity "outdoor seating" is now a HARD requirement
- Restaurants without outdoor seating are eliminated
- Only restaurants with outdoor seating are returned
```

## Benefits

1. **Transparency**: Users know when exact matches aren't available
2. **Better UX**: Show alternatives instead of empty results
3. **Honesty**: Clearly communicate what's different about alternatives
4. **Flexibility**: Smart fallback that keeps important constraints
5. **Explainability**: System tracks and communicates why fallback was applied

## Testing

Run the test to verify:
```bash
python test_hybrid_search.py
```

Test with the problematic query:
```python
query = "Find Italian restaurants in Downtown Dubai with outdoor seating under AED 200"
```

Expected behavior:
1. Initial search finds 0 exact matches
2. Fallback activates (relaxes location)
3. Returns Italian restaurants with outdoor seating from other areas
4. Response clearly states: "No Italian restaurants in Downtown Dubai, here are nearby alternatives"

## Files Modified

1. **`src/rag_system.py`**
   - Modified `hybrid_search()` to return dict with metadata
   - Updated `_apply_entity_filters()` to return filter info
   - Added `_apply_fallback_filters()` for smart fallback
   - Made amenities a hard filter (was soft)

2. **`src/agents.py`**
   - Updated `AgentState` TypedDict with new fields
   - Modified `RetrievalAgent.retrieve_restaurants()` to handle new return type
   - Updated `ResponseGenerationAgent.generate_response()` to use fallback metadata
   - Enhanced `_format_restaurants()` to show fallback context
   - Updated `_fallback_response()` for better messaging
   - Fixed `_relaxed_search()` to handle new return type

3. **`src/config.py`**
   - Enhanced `response_generation` system prompt
   - Added instructions for transparent fallback communication

4. **`test_hybrid_search.py`**
   - Updated to handle new return type from `hybrid_search()`

## Future Enhancements

Potential improvements:
1. **Distance-based fallback**: Prefer nearby locations over distant ones
2. **Partial amenity matching**: Allow some amenities to be optional
3. **User preferences**: Remember if user prefers strict vs flexible filtering
4. **A/B testing**: Compare user satisfaction with different fallback strategies
