# Publish this prepared repository

Recommended repository name: `agentic-financial-rag`.

Description: Agentic retrieval-augmented generation for financial document QA, with reproducible benchmark runners, validation and refinement ablations, tests, and verified results.

Visibility: public. Keep the default repository empty when creating it on GitHub: do not add another README, license or gitignore there.

From `/Users/raghav/Desktop/Workspace/agentic-financial-rag`:

```bash
git status
git add .
git diff --cached --stat
git commit -m "Release financial RAG benchmark implementation and verified results"
```

Copy the exact remote URL from your new GitHub repository, then:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_REPOSITORY_URL`; it is not a literal command argument to keep unchanged. GitHub may prompt you to authenticate using your configured method. If an origin already exists, inspect it before changing it.

Create a GitHub Release tagged `v1.0.0` and attach `artifacts/benchmark-artifacts.zip`. That asset contains the selected datasets/indexes, raw model records and complete trajectories needed for offline replay. These large files are intentionally ignored by Git. `ARTIFACTS.json` gives the asset checksum. Publishing only the Git source without its release asset does not provide the complete offline reproduction package.

After publication, insert the actual repository/release URL into the paper's availability paragraph, `CITATION.cff` and repository About website field. Do not invent a DOI or say a release is already publicly available before uploading it. The author-review manuscript is delivered separately and should be reviewed and revised in accordance with the venue's AI policy before submission.
