# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 20:42
@Desc       :
"""

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from src.utils import read_json_lines


class Evaluator:
    def __init__(self, key):
        self.key = key

    def evaluate(self, ref_file, hyp_file, to_lower):
        list_of_references = []
        for line in read_json_lines(ref_file):
            ref = line[self.key]  # ref is a list of words
            if to_lower:
                ref = list(map(str.lower, ref))
            list_of_references.append([ref])

        hypotheses = []
        for line in read_json_lines(hyp_file):
            hyp = line[self.key]  # hyp is a list of words
            if to_lower:
                hyp = list(map(str.lower, hyp))
            hypotheses.append(hyp)

        assert len(list_of_references) == len(hypotheses)

        bleu1 = 100 * corpus_bleu(list_of_references, hypotheses, (1., 0., 0., 0.), SmoothingFunction().method4)
        bleu2 = 100 * corpus_bleu(list_of_references, hypotheses, (0.5, 0.5, 0., 0.), SmoothingFunction().method4)
        bleu3 = 100 * corpus_bleu(list_of_references, hypotheses, (0.33, 0.33, 0.33, 0.), SmoothingFunction().method4)
        bleu4 = 100 * corpus_bleu(list_of_references, hypotheses, (0.25, 0.25, 0.25, 0.25), SmoothingFunction().method4)
        print('{:>.4f}, {:>.4f}, {:>.4f}, {:>.4f}'.format(bleu1, bleu2, bleu3, bleu4))
        res = {
            'Bleu_1': bleu1,
            'Bleu_2': bleu2,
            'Bleu_3': bleu3,
            'Bleu_4': bleu4,
        }
        return res
