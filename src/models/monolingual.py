"""Baseline Model

This module contains the class(es) and functions that implement the CWI baseline model.

"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from src.features.feature_transfomers import Selector, Word_Feature_Extractor, Spacy_Feature_Extractor, Sentence_Feature_Extractor


class MonolingualCWI(object):
    """
    A basic CWI model implementing simple features that serves as baseline.

    """

    def __init__(self, language):
        """Defines the basic properties of the model.

        Args:
            language (str): The language of the data.

        """
        self.model = LogisticRegression(random_state=0)
        self.features_pipeline = self.join_pipelines(language)

    def build_pipelines(self, language):
        """
        Builds all feature pipelines
        Returns pipelines in format suitable for sklearn FeatureUnion
        Args:
            language: The language of the data.
        Returns:
            list. list of ('pipeline_name', Pipeline) tuples
        """
        pipe_dict = {}
        pipe_dict['word_features'] = Pipeline([
            ('select', Selector(key=["language","target_word"])),
            ('extract', Word_Feature_Extractor(language)),
            ('vectorize', DictVectorizer())])

        pipe_dict['sent_features'] = Pipeline([
            ('select', Selector(key=["language","sentence"])),
            ('extract', Sentence_Feature_Extractor(language)),
            ('vectorize', DictVectorizer())])

        pipe_dict['bag_of_words'] = Pipeline([
            ('select', Selector(key=["language","target_word"])),
            ('vectorize', CountVectorizer())])

        pipe_dict['spacy_features'] = Pipeline([
            ('select', Selector(key=["language","target_word", "spacy", "sentence"])),
            ('extract', Spacy_Feature_Extractor(language)),
            ('vectorize', DictVectorizer())])

        return list(pipe_dict.items())

    def join_pipelines(self, language):

        pipelines = self.build_pipelines(language)
        feature_union = Pipeline([('join pipelines', FeatureUnion(transformer_list=pipelines))])

        return feature_union

    def train(self, train_set):
        """Trains the model with the given instances.

        Args:
            train_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases and their gold labels.

        """

        X = self.features_pipeline.fit_transform(train_set)
        y = train_set['gold_label']
        self.model.fit(X, y)

    def predict(self, test_set):
        """Predicts the label for the given instances.

        Args:
            test_set (list): A list of dictionaries that contain the information of each instance in the dataset.
                In particular, the target words/phrases.

        Returns:
            numpy array. The predicted label for each target word/phrase.

        """

        X = self.features_pipeline.transform(test_set)

        return self.model.predict(X)
