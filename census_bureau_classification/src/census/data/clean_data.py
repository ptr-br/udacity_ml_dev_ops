import sys

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

# Put the code for your API here.


PROFILE = False


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=0, dtype=str)


def clean(df: pd.DataFrame, drop_cols=None, keep_sample_frac=None, verbose=True) -> pd.DataFrame:
    """
    Clean census CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DF containing data from the input CSV file.
    drop_cols : list[str] | None
        Columns to drop from the cleaned DataFrame.
    keep_sample_frac : float | None
        If provided, sample this fraction of rows (0 < frac <= 1) for faster iteration.
    verbose : bool
        If True, print a brief info summary and sample rows.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with normalized column names and string values,
        common numeric columns converted to nullable integers, missing values
        normalized, categorical values canonicalized, duplicate rows removed
    """

    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # trim whitespace
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    df.replace({"?": np.nan, "": np.nan}, inplace=True)

    if keep_sample_frac is not None:
        df = df.sample(frac=keep_sample_frac, random_state=42).reset_index(drop=True)

    int_cols = ["age", "education-num", "hours-per-week", "fnlgt", "capital-gain", "capital-loss"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")  # nullable integer

    cat_cols = [
        c
        for c in df.columns
        if c not in int_cols and c not in ("salary",) and df[c].dtype == "object"
    ]
    for c in cat_cols:
        df[c] = df[c].str.lower().str.replace(r"[\s\-]+", "_", regex=True)

    if "salary" in df.columns:
        bins_required = {"<=50k", ">50k"}
        present = set(df["salary"].dropna().astype(str).str.strip().str.lower().unique())

        if present != bins_required:
            missing = bins_required - present
            extras = present - bins_required
            raise ValueError(
                f"salary column values mismatch. Missing: {missing or 'None'}. "
                f"Unexpected: {extras or 'None'}."
            )

    if drop_cols:
        # drop_cols.append("salary")
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = df.drop_duplicates().reset_index(drop=True)

    if verbose:
        print("cleaning summary:")
        print(df.info(memory_usage="deep"))
        print("sample:")
        print(df.head())

    return df


def profile_data(path: pd.DataFrame) -> None:
    profile = ProfileReport(df, title="Census Dataset Report", explorative=True)
    # save to HTML
    profile.to_file("census_profile_report.html")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_data <file_path_to_data>")
        raise SystemExit(2)

    df = clean(load(path=sys.argv[1]), drop_cols=["fnlgt"], keep_sample_frac=None, verbose=True)

    if PROFILE:
        profile_data(df)

    df.to_csv("./cleaned_data.csv")
    print("Wrote dat to ./cleaned_data.csv")
