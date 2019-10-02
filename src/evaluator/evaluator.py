import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def get_bleu_score(truth_file, pred_file, to_lower):
        f_truth = open(truth_file, 'r', encoding='utf-8')
        f_pred = open(pred_file, 'r', encoding='utf-8')

        list_of_references = []
        hypotheses = []
        for truth in f_truth:
            pred = f_pred.readline()

            truth = json.loads(truth)
            truth = truth['question']
            if to_lower:
                truth = list(map(str.lower, truth))

            pred = json.loads(pred)
            pred = pred['question']
            if to_lower:
                pred = list(map(str.lower, pred))

            list_of_references.append([truth])
            hypotheses.append(pred)

        bleu1 = 100 * corpus_bleu(list_of_references, hypotheses, (1., 0., 0., 0.), SmoothingFunction().method4)
        bleu2 = 100 * corpus_bleu(list_of_references, hypotheses, (0.5, 0.5, 0., 0.), SmoothingFunction().method4)
        bleu3 = 100 * corpus_bleu(list_of_references, hypotheses, (0.33, 0.33, 0.33, 0.), SmoothingFunction().method4)
        bleu4 = 100 * corpus_bleu(list_of_references, hypotheses, (0.25, 0.25, 0.25, 0.25), SmoothingFunction().method4)
        print('{:>.4f}, {:>.4f}, {:>.4f}, {:>.4f}'.format(bleu1, bleu2, bleu3, bleu4))
        return (bleu1 + bleu2 + bleu3 + bleu4) / 4
