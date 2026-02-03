# Vector Spaces and Linear Mappings

This note summarizes key concepts of vector spaces and linear mappings from a research-oriented perspective.

The focus is not on formal proofs, but on intuition and how these concepts appear in machine learning and systems research.

---

## Vector Space: Intuition First

A vector space can be understood as a set of objects (vectors) where addition and scalar multiplication are defined and behave consistently.

From an intuitive perspective:
- Vectors represent quantities with both magnitude and direction
- Operations allow us to combine and scale these quantities

In practice, vectors often represent:
- Feature representations
- Model parameters
- Embeddings in machine learning models

---

## Linear Mapping

A linear mapping (or linear transformation) is a function between vector spaces that preserves linear structure.

Formally, a mapping \( f \) is linear if:
- \( f(x + y) = f(x) + f(y) \)
- \( f(\alpha x) = \alpha f(x) \)

Intuitively, linear mappings do not distort relationships between vectors.

In machine learning, linear mappings appear as:
- Fully connected layers
- Matrix multiplications
- Projections into new feature spaces

---

## Why This Matters in Research

Understanding vector spaces and linear mappings is fundamental because:
- Most models operate in high-dimensional vector spaces
- Optimization methods assume linear or locally linear behavior
- Concepts such as eigenvalues and singular values build upon this foundation

A weak understanding of linear algebra often leads to shallow model intuition.

---

## Implementation Notes

By implementing basic vector operations in code, the abstract definition of a vector space becomes more concrete.

Vector addition and scalar multiplication correspond directly to array operations, while linear mappings are naturally represented by matrix multiplication.

This reinforces the idea that many machine learning models operate as compositions of linear mappings.
---

## Next Steps

- Study eigenvalues and eigenvectors
- Understand matrix decompositions (SVD, PCA)
- Relate linear algebra concepts to neural network architectures
