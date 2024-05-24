# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

try:
    import nltk
except ImportError:
    nltk = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars if nltk else object):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


def build_splitter(args):
    if nltk and args.split_sentences:
        nltk.download("punkt", quiet=True)
    if args.split_sentences:
        if not nltk:
            logger.error("NLTK is not available to split sentences.")
            raise Exception("nltk is not available")
        splitter = nltk.load("tokenizers/punkt/english.pickle")
        if args.keep_newlines:
            # this prevents punkt from eating newlines after sentences
            final_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                train_text=splitter._params,
                lang_vars=CustomLanguageVars())
        else:
            final_splitter = splitter

    else:
        final_splitter = IdentitySplitter()
    return final_splitter
