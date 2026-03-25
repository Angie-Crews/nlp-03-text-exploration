"""
nlp_corpus_explore_crews_p4.py - Module 3 Script

Purpose

  Perform exploratory analysis of a small, controlled text corpus.
  Demonstrate how structure emerges from token distributions,
  category comparisons, and co-occurrence patterns.

Analytical Questions

- What tokens dominate each category?
- How do categories differ in vocabulary?
- What words appear in similar contexts?
- What structure is visible before using any models?

Notes

- This module focuses on exploratory analysis (EDA), not modeling.
- Results here prepare for later work with pipelines and embeddings.

Run from root project folder with:

  uv run python -m nlp.nlp_corpus_explore_crews_p4
"""

# ============================================================
# Section 1. Setup and Imports
# ============================================================

from collections import defaultdict
import logging
from pathlib import Path

from datafun_toolkit.logger import get_logger, log_header, log_path
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import polars as pl

print("Imports complete.")


# ============================================================
# Configure Logging
# ============================================================

LOG: logging.Logger = get_logger("CI", level="DEBUG")

ROOT_PATH: Path = Path.cwd()
NOTEBOOKS_PATH: Path = ROOT_PATH / "notebooks"
SCRIPTS_PATH: Path = ROOT_PATH / "scripts"
PLOTS_PATH: Path = ROOT_PATH / "docs" / "images"
BIGRAM_PLOT_FILE_PATH: Path = PLOTS_PATH / "nlp_corpus_explore_crews_p4_top_bigrams.png"
PLOT_FILE_PATH: Path = PLOTS_PATH / "nlp_corpus_explore_crews_p4_top_tokens.png"

log_header(LOG, "MODULE 3: CORPUS EXPLORATION")
LOG.info("START script.....")

log_path(LOG, "ROOT_PATH", ROOT_PATH)
log_path(LOG, "NOTEBOOKS_PATH", NOTEBOOKS_PATH)
log_path(LOG, "SCRIPTS_PATH", SCRIPTS_PATH)
log_path(LOG, "PLOTS_PATH", PLOTS_PATH)
log_path(LOG, "BIGRAM_PLOT_FILE_PATH", BIGRAM_PLOT_FILE_PATH)

LOG.info("Logger configured.")

# ============================================================
# Section 2. Define Corpus (Labeled Text Documents)
# ============================================================

# A corpus is a collection of documents.
# Each document includes a category and text.

corpus: list[dict[str, str]] = [
    # Dogs
    {"category": "dog", "text": "A dog barks loudly."},
    {"category": "dog", "text": "The puppy runs in the yard."},
    {"category": "dog", "text": "A canine wears a leash."},
    {"category": "dog", "text": "The kennel holds the dog."},
    {"category": "dog", "text": "The dog ran across the yard."},
    {"category": "dog", "text": "The puppy ran across the yard."},
    # Cats
    {"category": "cat", "text": "A cat sleeps quietly."},
    {"category": "cat", "text": "The kitten plays with yarn."},
    {"category": "cat", "text": "A feline purrs softly."},
    {"category": "cat", "text": "The cat has whiskers."},
    {"category": "cat", "text": "The cat slept near the window."},
    {"category": "cat", "text": "The kitten slept near the window."},
    # Cars
    {"category": "car", "text": "A car drives on the road."},
    {"category": "car", "text": "The sedan parks in the garage."},
    {"category": "car", "text": "A vehicle has four wheels."},
    {"category": "car", "text": "The car moves down the highway."},
    {"category": "car", "text": "The car stopped near the garage."},
    {"category": "car", "text": "The sedan stopped near the garage."},
    # Trucks
    {"category": "truck", "text": "A truck carries cargo."},
    {"category": "truck", "text": "The pickup pulls a trailer."},
    {"category": "truck", "text": "The engine powers the truck."},
    {"category": "truck", "text": "The truck hauls heavy loads."},
]

# Show results
print(f"Corpus contains {len(corpus)} documents.")


# ============================================================
# Section 3. Tokenize and Clean Text
# ============================================================

# Tokenization splits text into word-like units.

STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "with",
    "has",
    "down",
    "near",
}


# Define a function to tokenize text by lowercasing, splitting on whitespace,
# and stripping common punctuation. We also filter out very short tokens (length <= 2).
# This simple tokenizer is sufficient for our small, controlled corpus.
# Use the string strip() method to remove punctuation from the beginning and end of each token.
def tokenize(text: str) -> list[str]:
    tokens = text.lower().split()
    return [t.strip(".,:;!?()[]\"'") for t in tokens if len(t) > 2]


# Define a new empty list to hold the token records we will create.
records_list: list[dict[str, str]] = []

# Loop through each document, tokenize the text,
# and create a record for each token with its category and
# add it to our list of records.
for doc in corpus:
    # Call our function to tokenize the text of the current document.
    tokens = tokenize(doc["text"])
    # Loop through each token produced by the tokenizer and
    # create a record that includes the category of the document and the token itself.
    # Append this record to our list of records.
    for token in tokens:
        records_list.append({"category": doc["category"], "token": token})

# Create a Polars DataFrame from the list of token records for easier analysis.
token_df: pl.DataFrame = pl.DataFrame(records_list)
filtered_token_df: pl.DataFrame = token_df.filter(~pl.col("token").is_in(STOPWORDS))

# Show results
print("Tokenization complete.")
print(token_df.head(10))
print(
    f"Removed {token_df.height - filtered_token_df.height} stopword tokens before analysis."
)
print(filtered_token_df.head(10))


# ============================================================
# Section 4. Compute Global Token Frequencies
# ============================================================

# Frequency distribution = how often each token appears.

# Create a DataFrame that groups the tokens by their text and
# counts how many times each token appears across the entire corpus.
global_freq_df: pl.DataFrame = (
    filtered_token_df.group_by("token").len().sort("len", descending=True)
)

# Show results
print("Top global tokens:")
print(global_freq_df.head(10))


# ============================================================
# Section 5. Compute Token Frequencies by Category
# ============================================================

# Compare token usage across categories.

# Create a new DataFrame that groups the tokens by both their category and text,
# counts how many times each token appears within each category,
# and sorts the results first by category and then by frequency in descending order.
# This shows which tokens are most common within each category.
category_freq_df: pl.DataFrame = (
    filtered_token_df.group_by(["category", "token"])
    .len()
    .sort(["category", "len"], descending=True)
)

# Show results
print("Top tokens by category:")
print(category_freq_df.head(12))


# ============================================================
# Section 6. Identify Top Tokens per Category
# ============================================================

# Show top tokens per category.


# Define a new empty dictionary to store the top tokens for each category.
top_per_category_dict: dict[str, list[str]] = {}

# Loop through each unique category in the token DataFrame,
# filter the category frequency DataFrame to get the top 5 tokens for that category,
# and store the list of top tokens in the dictionary.
# Also, print the top tokens for each category.
for category in filtered_token_df["category"].unique().to_list():
    subset_df = category_freq_df.filter(pl.col("category") == category).head(5)
    top_tokens_list = subset_df["token"].to_list()
    top_per_category_dict[category] = top_tokens_list

    # Show results for this category
    print(f"{category.upper()} top tokens: {top_tokens_list}")


# ============================================================
# Section 7. Analyze Co-occurrence (Context Windows)
# ============================================================

# Co-occurrence examines which tokens appear near each other.

# Define how many tokens on each side of a target token we include as context.
# A window size of 2 means:
#   - up to 2 tokens before the target token
#   - up to 2 tokens after the target token
# The target token itself is not included in its context list.
WINDOW_SIZE: int = 2

# Define a new empty dictionary to store the co-occurrence information.
# The keys will be target tokens,
# and the values will be lists of context tokens that appear near the target token.
co_occurrence_dict: dict[str, list[str]] = defaultdict(list)

# Loop through each document in the corpus, tokenize the text,
# and for each token, determine its context tokens based on the defined window size.
for doc in corpus:
    tokens = [token for token in tokenize(doc["text"]) if token not in STOPWORDS]
    for i, token in enumerate(tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        context = tokens[start:end]
        for ctx in context:
            if ctx != token:
                co_occurrence_dict[token].append(ctx)

# Show results
for target in ["dog", "cat", "car", "truck"]:
    print(f"\nContext for '{target}':")
    print(co_occurrence_dict[target][:10])


# ============================================================
# Section 8. Create Bigrams (Local Word Pairs) and Compute Frequencies
# ============================================================

# Bigrams combine each word with the next word in the text.
# This helps us capture local structure: how words are used together,
# not just which words appear individually.

# Bigrams capture pairs of consecutive tokens.

# Define a new empty list to hold the bigram records we will create.
bigrams_list: list[dict[str, str]] = []

# Loop through each document in the corpus, tokenize the text,
# and create bigrams by pairing each token with the next token in the list.
for doc in corpus:
    tokens = [token for token in tokenize(doc["text"]) if token not in STOPWORDS]
    for i in range(len(tokens) - 1):
        bigrams_list.append(
            {
                "category": doc["category"],
                "bigram": f"{tokens[i]} {tokens[i + 1]}",
            }
        )

# Create a DataFrame from the list of bigram records.
bigram_df: pl.DataFrame = pl.DataFrame(bigrams_list)

# Create a new DataFrame that groups the bigrams by their text
# and counts how many times each bigram appears,
# then sorts the results by frequency in descending order.
bigram_freq_df: pl.DataFrame = (
    bigram_df.group_by("bigram").len().sort("len", descending=True)
)

# Show results
print("Top bigrams:")
print(bigram_freq_df.head(10))

# Save a chart of the most frequent bigrams for quick visual comparison.
dominant_bigram_category_df: pl.DataFrame = (
    bigram_df.group_by(["bigram", "category"])
    .len()
    .sort(["bigram", "len", "category"], descending=[False, True, False])
    .unique(subset=["bigram"], keep="first")
    .rename({"category": "dominant_category"})
    .select(["bigram", "dominant_category"])
)

top_bigram_plot_df = (
    bigram_freq_df.head(8)
    .join(dominant_bigram_category_df, on="bigram", how="left")
    .sort(["len", "bigram"])
)

bigram_category_colors = {
    "dog": "steelblue",
    "cat": "darkorange",
    "car": "seagreen",
    "truck": "firebrick",
}
bar_colors = [
    bigram_category_colors.get(category, "slategray")
    for category in top_bigram_plot_df["dominant_category"].to_list()
]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(
    top_bigram_plot_df["bigram"].to_list(),
    top_bigram_plot_df["len"].to_list(),
    color=bar_colors,
    alpha=0.85,
)
ax.set_title("Crews Phase 4: Top Bigrams Without Stopwords")
ax.set_xlabel("Frequency")
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for index, count in enumerate(top_bigram_plot_df["len"].to_list()):
    ax.text(count + 0.03, index, str(count), va="center")

legend_handles = [
    Patch(facecolor=color, label=category.title())
    for category, color in bigram_category_colors.items()
]
ax.legend(handles=legend_handles, title="Category", loc="lower right")

fig.tight_layout()
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
fig.savefig(BIGRAM_PLOT_FILE_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved bigram visualization to {BIGRAM_PLOT_FILE_PATH}")


# ============================================================
# Section 9. Visualize Token Frequencies
# ============================================================

# Create a multi-panel figure so the chart compares all categories at once.
ordered_categories = ["dog", "cat", "car", "truck"]
category_colors = {
    "dog": "steelblue",
    "cat": "darkorange",
    "car": "seagreen",
    "truck": "firebrick",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for axis, category in zip(axes.flat, ordered_categories, strict=True):
    top_category_df = (
        category_freq_df.filter(pl.col("category") == category).head(5).sort("len")
    )
    tokens = top_category_df["token"].to_list()
    counts = top_category_df["len"].to_list()

    axis.barh(tokens, counts, color=category_colors[category], alpha=0.85)
    axis.set_title(f"{category.title()} Top Tokens")
    axis.set_xlabel("Frequency")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for index, count in enumerate(counts):
        axis.text(count + 0.03, index, str(count), va="center")

fig.suptitle(
    "Crews Phase 4: Top Tokens by Category Without Stopwords",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.96))

# Save the figure so the script can run end-to-end without waiting on a GUI window.
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOT_FILE_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved visualization to {PLOT_FILE_PATH}")


# ============================================================
# Section 10. Interpret Results and Identify Patterns
# ============================================================

print("\nCASE GENERAL OBSERVATIONS:")

print("- Tokens cluster by category (dog, cat, car, truck).")
print("- Words that appear in similar contexts behave similarly.")
print("- Co-occurrence reveals contextual relationships between words.")
print("- Bigrams capture local structure beyond single tokens.")
print("- Patterns emerge before any machine learning is applied.")

print("\nCREWS PHASE 4 MODIFICATION:")
print(
    "- I removed common stopwords before computing token frequencies, co-occurrence, and bigrams."
)
print(
    "- I saved category and bigram comparison charts to docs/images instead of opening blocking chart windows."
)

print("\nCREWS SPECIFIC OBSERVATIONS:")
print(
    "- After stopword removal, category-specific terms like 'yard', 'garage', and 'truck' became more visible in the top rankings."
)
print(
    "- The updated chart makes it easier to compare the strongest tokens in each category side by side."
)
print(
    "- The filtered bigrams emphasize content pairs such as 'dog barks' and 'pickup pulls' instead of function-word combinations."
)
print(
    "- The bigram chart makes repeated word-pair patterns easier to scan than the table alone, and the legend clarifies which category each color represents."
)
print(
    "- Saving the plot to a file made the script easier to rerun because it now finishes without manual interaction."
)


# ============================================================
# END
# ============================================================

LOG.info("========================")
LOG.info("Pipeline executed successfully!")
LOG.info("========================")
LOG.info("END main()")
