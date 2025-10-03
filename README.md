# PAC Learning Visualizations

## What is PAC Learning?

PAC (Probably Approximately Correct) Learning is a framework in computational learning theory that formalizes what it means for an algorithm to successfully "learn" from data. An algorithm PAC-learns if, given enough training examples, it can produce a hypothesis that is:

- **Probably** correct (with high confidence, controlled by parameter δ)
- **Approximately** accurate (with low error, controlled by parameter ε)

## Visualizations

### Rectangle Learning

Watch how a simple PAC learning algorithm learns the boundaries of an axis-aligned rectangle from random samples:

![Rectangle](https://github.com/rfpozas/PAC-Learners/blob/main/rectangle_animation.gif)

**Key observations:**
- The green shaded area is the true (unknown) rectangle we're trying to learn
- Blue dots are positive samples (inside the rectangle)
- Red X's are negative samples (outside the rectangle)
- The blue dashed rectangle is the algorithm's current hypothesis
- As more samples arrive, the learned rectangle converges to the true one

Now, let's try the same thing for a circle:

![Circle](https://github.com/rfpozas/PAC-Learners/blob/main/circle_animation.gif)

## The Math Behind It

For a concept class to be PAC-learnable, we need:

**Sample Complexity**: The number of samples m required satisfies:

```
m ≥ (1/ε) * (ln(1/δ) + ln(|H|))
```

Where:
- ε (epsilon) = maximum acceptable error
- δ (delta) = maximum acceptable failure probability
- |H| = size of the hypothesis space

For our rectangle example, the sample complexity is polynomial in 1/ε and 1/δ, making it efficiently PAC-learnable.
