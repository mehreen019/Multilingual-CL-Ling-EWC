# Experiment Notes & Design Rationale

## Core Research Question

**Does explicit linguistic similarity improve parameter preservation in multilingual continual learning?**

## Design Decisions

### Why Bengali → Hindi?

1. **Linguistic relationship**: Both Indo-Aryan, moderate similarity
   - Enables meaningful similarity-based scaling
   - Not too similar (like Hindi-Urdu) nor too distant (like Hindi-Tamil)

2. **Practical relevance**: Important low-resource Indic languages
   - Realistic for multilingual NLP applications
   - Good representation of common continual learning scenarios

3. **Observable effects**: Moderate similarity (0.6) should show clear differences
   - Similar enough for positive transfer
   - Different enough to test forgetting prevention

### Why These Hyperparameters?

#### EWC Lambda = 5000
- Standard range from EWC literature: 10² to 10⁶
- 5000 is moderate: strong enough to prevent forgetting, not too rigid
- Allows linguistic scaling to have observable effects

#### Fisher Sample Size = 1000
- Balance between accuracy and computation
- Large enough for stable Fisher estimation
- Not too large to slow down experiment

#### Bengali-Hindi Similarity = 0.6
Based on linguistic analysis:
```
Script:     0.2 (Different: Bengali vs Devanagari)
Word order: 1.0 (Both SOV)
Morphology: 0.8 (Similar inflection patterns)
Family:     1.0 (Both Indo-Aryan)
Lexical:    0.6 (Moderate overlap via Sanskrit)
─────────────────────────────────────────────
Average:    0.72 → Conservative: 0.6
```

#### Similarity → Penalty Scaling

We use **inverted similarity** for EWC penalty:
```python
ewc_scale = 1.0 - similarity
```

Rationale:
- **High similarity (0.8+)**: Low penalty (0.2) → Allow parameter updates
  - Transfer is likely beneficial
  - Model should adapt to leverage shared structure

- **Low similarity (0.2-)**: High penalty (0.8+) → Protect parameters strongly
  - Transfer less likely to help
  - Risk of catastrophic forgetting higher

- **Moderate similarity (0.5-0.7)**: Balanced approach
  - Our Bengali-Hindi case: 0.6 similarity → 0.4 penalty scale
  - 60% reduction in EWC constraint

### Alternative Scaling Strategies

Could also test:
1. **Direct scaling**: `scale = similarity` (opposite effect)
2. **Quadratic**: `scale = (1 - similarity)²` (more aggressive)
3. **Threshold**: Binary protection based on similarity cutoff
4. **Learned**: Meta-learn optimal scaling function

Our approach (linear inverted) is simplest and most interpretable.

## Expected Outcomes & Interpretation

### Scenario 1: Linguistic EWC Wins
```
Forgetting: Naive > Random EWC > Standard EWC > Ling EWC
```
**Interpretation**: Linguistic bias helps. Fisher Information alone is insufficient.

**Next steps**:
- Test on more language pairs
- Try more sophisticated similarity metrics
- Explore learned similarity functions

### Scenario 2: No Difference
```
Forgetting: Ling EWC ≈ Standard EWC
```
**Interpretation**: Fisher already captures linguistic structure implicitly.

**Value**: Still publishable! Shows that:
- Simple EWC is surprisingly effective
- Explicit linguistic bias is unnecessary
- Fisher Information is linguistically-aware by nature

### Scenario 3: Linguistic EWC Worse
```
Forgetting: Ling EWC > Standard EWC
```
**Interpretation**: Our scaling strategy is incorrect.

**Analysis**:
- Maybe inverted scaling is wrong (try direct scaling)
- Maybe similarity score is inaccurate
- Maybe linguistic similarity is task-dependent

## Experimental Controls

### Why Random EWC?
- Sanity check: Ensures any improvement is from linguistic structure, not just "any scaling"
- If random scaling helps, our linguistic hypothesis is wrong
- If random scaling hurts, confirms structured scaling matters

### Why Include Naive?
- Shows upper bound on forgetting (worst case)
- Demonstrates that continual learning is necessary
- Baseline for measuring improvement

## Limitations & Future Work

### Current Limitations
1. **Two languages only**: Limited generalization
2. **Demo data**: Synthetic, may not reflect real phenomena
3. **One task**: Sentiment analysis only
4. **Fixed similarity**: Hand-crafted, not learned
5. **Single model**: XLM-RoBERTa only

### Immediate Extensions
1. Add third language (e.g., Marathi) for multi-step continual learning
2. Test distant language pair (e.g., Hindi → Tamil)
3. Try different tasks (NER, POS tagging)
4. Use real IndicNLP data

### Advanced Extensions
1. **Meta-learned similarity**: Learn optimal similarity function from data
2. **Task-aware similarity**: Different similarity metrics for different tasks
3. **Dynamic scaling**: Adjust scaling during training based on forgetting
4. **Adapter-based**: Combine with parameter-efficient fine-tuning
5. **Curriculum learning**: Optimize language ordering based on similarity

## Computational Requirements

### Minimal Setup (Demo Data)
- **GPU**: Optional (CPU works, just slower)
- **RAM**: 8GB sufficient
- **Time**: ~30 minutes on CPU, ~10 minutes on GPU
- **Storage**: <2GB (mostly for model checkpoints)

### Production Setup (Real Data)
- **GPU**: Recommended (V100 or better)
- **RAM**: 16GB+
- **Time**: 2-4 hours depending on data size
- **Storage**: 5-10GB

## Statistical Significance

For publishable results, repeat experiment with:
- Multiple random seeds (5-10)
- Report mean ± std for all metrics
- Statistical tests (t-test, ANOVA) for significance
- Confidence intervals on forgetting reduction

## Publication Strategy

### If Positive Result
- **Venue**: *SEM, Findings of ACL/EMNLP, or TACL
- **Angle**: "Linguistic structure improves continual learning"
- **Contributions**:
  1. Novel linguistically-aware EWC method
  2. Empirical evidence that similarity matters
  3. Analysis of what linguistic features are most important

### If Negative Result
- **Venue**: Insights/Negative Results track at NLP conferences
- **Angle**: "Fisher Information is implicitly linguistic"
- **Contributions**:
  1. Thorough experimental analysis
  2. Evidence that simple EWC is sufficient
  3. Analysis of when linguistic bias does/doesn't help

Both outcomes are valuable and publishable!
