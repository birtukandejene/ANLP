import numpy as np

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)


class TFIDFKeywordExtractor:

    def __init__(
        self,
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        max_df=0.95
    ):

        self.vectorizer = TfidfVectorizer(

            ngram_range=ngram_range,

            max_features=max_features,

            min_df=min_df,

            max_df=max_df,

            dtype=np.float32,

            token_pattern=r"(?u)\b\w+\b"
        )


    def fit(self, documents):

        print(
            "Building TF-IDF vocabulary..."
        )

        self.vectorizer.fit(
            documents
        )

        print(
            "Vocabulary size:",
            len(
                self.vectorizer
                .get_feature_names_out()
            )
        )


    def extract_keywords(
        self,
        document,
        top_k=5
    ):

        matrix = self.vectorizer.transform(
            [document]
        )

        scores = matrix.toarray()[0]

        feature_names = np.array(

            self.vectorizer
            .get_feature_names_out()
        )

        nonzero_indices = np.where(
            scores > 0
        )[0]

        if len(nonzero_indices) == 0:

            return []

        sorted_indices = nonzero_indices[
            np.argsort(
                scores[nonzero_indices]
            )[::-1]
        ]

        keywords = []

        for index in sorted_indices:

            keyword = feature_names[index]

            keywords.append(keyword)

            if len(keywords) >= top_k:
                break

        return keywords