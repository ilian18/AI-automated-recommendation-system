# Clothing Recommendation System (Cosine Similarity AI)

This project is a fashion item recommendation engine developed in Python. It uses linear algebra and the NumPy library to suggest clothing similar to a target item, based on a mathematical analysis of their attributes.

---

## Project Objective

The goal is to create a fast recommendation system without classical for loops, capable of processing an item catalog instantly using vectorization.

Rather than using complex databases or heavy Machine Learning algorithms (like neural networks), this project relies on Cosine Similarity. It calculates the mathematical "angle" between the representative vectors of each garment to define their degree of resemblance.

---

## Technical Features

Data Modeling: Transformation of clothing attributes (price, style, type, color) into mathematical vectors.

Advanced Normalization:

Use of Min-Max Scaling to prevent high values (like price) from squashing other binary attributes (style, type).

Vector normalization ($\|v\| = 1$) with protection against division by zero (1e-9).

Vectorized Matrix Calculation: Use of matrix multiplication ($M \times M^T$) via NumPy to simultaneously calculate similarity scores for the entire catalog.

High-Performance Extraction: Replacement of the classic sorting algorithm (np.argsort) with the Quickselect partitioning algorithm (np.argpartition) to extract the Top 3 recommendations with a time complexity of $\mathcal{O}(N)$.

---

## Tech Stack

Language: Python 3.x

Main Library: NumPy

Mathematical Concepts: Dot product, Vector norm, Cosine Similarity, Linear Algebra.

---

## How it works (Summary)

The system reads the item dictionary and encodes their characteristics.

The data is injected into a NumPy matrix $M$ and normalized.

The system multiplies the matrix by its transpose (M @ M.T) to generate a square matrix containing all similarity scores ($0$ = no common points, $1$ = perfect clones).

When querying for a target item, the algorithm isolates its score row and uses argpartition to instantly return the 3 most relevant items.

---

## Future Improvements

Implement One-Hot Encoding to properly handle colors (currently represented by integers, which creates weighting biases).

Connect the system to a real database (CSV, SQL) or an API.

Add a simple GUI or a Flask/FastAPI API to query the model.
