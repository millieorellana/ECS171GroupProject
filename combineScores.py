import pandas as pd
from pathlib import Path

resultsDir = Path(__file__).parent / "results"

#load results from both models
nbScores = pd.read_csv(resultsDir / "naiveBayes_scores.csv")
rfScores = pd.read_csv(resultsDir / "randomForest_scores.csv")

#verify both experiments are complete
assert len(nbScores) == 2275, "Naive Bayes results incomplete"
assert len(rfScores) == 2275, "Random Forest results incomplete"

#identify matching experiments
keys = ["Feature 1", "Feature 2", "Seed", "Noise Level"]

#reuse information loss values for Random Forest
rfScores = rfScores.merge(
    nbScores[keys + ["Information Loss"]],
    on=keys,
    how="left",
    validate="one_to_one"
)

assert rfScores["Information Loss"].notna().all()

#combine both models' results
scores = pd.concat(
    [nbScores, rfScores],
    ignore_index=True
)

#save combined results
scores.to_csv(resultsDir / "scores.csv", index=False)

print("Total results:", len(scores))
print("Results saved to results/scores.csv")