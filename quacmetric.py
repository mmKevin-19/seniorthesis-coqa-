import collections
import logging
import json
import math
import re, string

import numpy as np


from collections import defaultdict, Counter
from transformers import BasicTokenizer


logger = logging.getLogger(__name__)

MAX_FLOAT = 1e30
MIN_FLOAT = -1e30


def normalize_answer(s):
    """normalization."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
  # printed
  # print(precision)
  # print (recall)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def single_score(prediction, ground_truth):
  # classification
    if prediction == "CANNOTANSWER" and ground_truth == "CANNOTANSWER":
        return 1.0
    elif prediction == "CANNOTANSWER" and ground_truth == "CANNOTANSWER":
        return 0.0
    else:
        return f1_score(prediction, ground_truth)


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Projection."""

    # parse accordingly.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    #string manipulation
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def read_target_dict(input_file):
    target = json.load(open(input_file))['data']
    target_dict = {}
    for p in target:
        for par in p['paragraphs']:
            p_id = par['id']
            qa_list = par['qas']
            for qa in qa_list:
                q_idx = qa['id']
                val_spans = [anss['text'] for anss in qa['answers']]
                target_dict[q_idx] = val_spans
    
    return target_dict


def leave_one_out(refs):
    if len(refs) == 1:
        return 1.
    splits = []
    for r in refs:
        splits.append(r.split())
    t_f1 = 0.0
    for i in range(len(refs)):
        m_f1 = 0
        for j in range(len(refs)):
            if i == j:
                continue
            f1_ij = f1_score(refs[i], refs[j])
            if f1_ij > m_f1:
                m_f1 = f1_ij
        t_f1 += m_f1
    return t_f1 / len(refs)


def leave_one_out_max(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        scores_for_ground_truths.append(single_score(prediction, ground_truth))

    if len(scores_for_ground_truths) == 1:
        return scores_for_ground_truths[0]
    else:
        # leave out one ref every time
        t_f1 = []
        for i in range(len(scores_for_ground_truths)):
            t_f1.append(max(scores_for_ground_truths[:i] + scores_for_ground_truths[i+1:]))
        return 1.0 * sum(t_f1) / len(t_f1)


def handle_cannot(refs):
    num_cannot = 0
    num_spans = 0
    for ref in refs:
        if ref == 'CANNOTANSWER':
            num_cannot += 1
        else:
            num_spans += 1
    if num_cannot >= num_spans:
        refs = ['CANNOTANSWER']
    else:
        refs = [x for x in refs if x != 'CANNOTANSWER']
    return refs


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def compute_predictions_logits(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    verbose_logging,
    null_score_diff_threshold,
    tokenizer,
    write_predictions=True,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")
    
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    yes_no_predictions = collections.OrderedDict()
    follow_up_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        
        example_no_answer_score = MAX_FLOAT
        
        example_yes_no_id = 0
        example_yes_no_score = MIN_FLOAT
        example_yes_no_probs = None
        example_follow_up_id = 0
        example_follow_up_score = MIN_FLOAT
        example_follow_up_probs = None



        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            ## add no_answer, yes_no, follow_up 
            example_no_answer_score = min(example_no_answer_score, float(result.no_answer_probs[0]))

            yes_no_probs = [float(yes_no_prob) for yes_no_prob in result.yes_no_probs]
            yes_no_id = int(np.argmax(yes_no_probs))
            yes_no_score = yes_no_probs[yes_no_id]
            if example_yes_no_score < yes_no_score:
                example_yes_no_id = yes_no_id
                example_yes_no_score = yes_no_score
                example_yes_no_probs = yes_no_probs
            
            follow_up_probs = [float(follow_up_prob) for follow_up_prob in result.follow_up_probs]
            follow_up_id = int(np.argmax(follow_up_probs))
            follow_up_score = follow_up_probs[follow_up_id]
            if example_follow_up_score < follow_up_score:
                example_follow_up_id = follow_up_id
                example_follow_up_score = follow_up_score
                example_follow_up_probs = follow_up_probs



            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            feature_null_score = result.start_logits[0] + result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = feature_index
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]
                        )
                    )
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=min_null_feature_index,
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit
            )
        )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = 'CANNOTANSWER'
                seen_predictions[final_text] = True
                
            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

        # if we didn't include the empty option in the n-best, include it
        if "CANNOTANSWER" not in seen_predictions:
            nbest.append(_NbestPrediction(text="CANNOTANSWER", start_logit=null_start_logit, end_logit=null_end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="CANNOTANSWER", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1, "No valid predictions"

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text != "CANNOTANSWER":
                    best_non_null_entry = entry
                    
        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"
        
        all_nbest_json[example.qas_id] = nbest_json

        if not best_non_null_entry:
            score_diff = 10
        else:
            # predict "CANNOTANSWER" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
        scores_diff_json[example.qas_id] = score_diff
        ############# comparison
        # if score_diff > null_score_diff_threshold:
        #     all_predictions[example.qas_id] = "CANNOTANSWER"
        # else:
        #     all_predictions[example.qas_id] = best_non_null_entry.text

        # example_best_predict = nbest[0]
        
        # if example_no_answer_score >= 0.3 and example_follow_up_probs[0] < 0.7:
        #     all_predictions[example.qas_id] = "CANNOTANSWER"
        # else:
        #     all_predictions[example.qas_id] = example_best_predict.text

        ############# second compare
        example_top_predicts = []
        for entry in nbest:
            if len(example_top_predicts) >= 5:
                break
            example_top_predicts.append({
                    "predict_text": entry.text,
                    "predict_score": float(np.log(entry.start_logit) + np.log(entry.end_logit)) 
                })        

        predictions = [i["predict_text"] for i in example_top_predicts]
        scores = [i["predict_score"] for i in example_top_predicts]
        if len(predictions)>1:
            cross_f1 = cross_f1_mean(predictions)
            delta = 0.6
            final_score = [delta*math.exp(i) + (1-delta)*j for (i,j) in zip(scores, cross_f1)]
            max_index = final_score.index(max(final_score))
            
            if example_no_answer_score >= 0.3 and example_follow_up_probs[0] < 0.7:
                all_predictions[example.qas_id] = "CANNOTANSWER"
            else:
                all_predictions[example.qas_id] = predictions[max_index]
        else:
            example_best_predict = nbest[0]
            if example_no_answer_score >= 0.3 and example_follow_up_probs[0] < 0.7:
                all_predictions[example.qas_id] = "CANNOTANSWER"
            else:
                all_predictions[example.qas_id] = example_best_predict.text
        


        yes_no_list = ["y", "x", "n"]
        yes_no_predictions[example.qas_id] = yes_no_list[example_yes_no_id]

        follow_up_list = ["y", "m", "n"]
        follow_up_predictions[example.qas_id] = follow_up_list[example_follow_up_id]


        

    if write_predictions:
        if output_prediction_file:
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")

        if output_nbest_file:
            with open(output_nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


    return all_predictions, all_nbest_json, yes_no_predictions, follow_up_predictions


def write_quac(prediction, nbest_pred, in_file, out_file, yes_no_predictions, follow_up_predictions):
    dialog_pred = defaultdict(list)
    for qa_id, span in prediction.items():
        dialog_id = qa_id.split("_q#")[0]

        if len(span) == 0:
            span = 'CANNOTANSWER'
        dialog_pred[dialog_id].append([qa_id, span])

    # for those we don't predict anything
    target = json.load(open(in_file))['data']
    for p in target:
        for par in p['paragraphs']:
            dialog_id = par['id']
            qa_list = par['qas']
            if dialog_id not in dialog_pred:
                for qa in qa_list:
                    qa_id = qa['id']
                    dialog_pred[dialog_id].append([qa_id, 'CANNOTANSWER'])

    # predict
    with open(out_file, 'w') as fout:
        for dialog_id, dialog_span in dialog_pred.items():
            output_dict = {'best_span_str': [], 'qid': [], 'yesno':[], 'followup': []}
            for qa_id, span in dialog_span:
                output_dict['best_span_str'].append(span)
                output_dict['qid'].append(qa_id)
                # output_dict['yesno'].append('y')
                # output_dict['followup'].append('y')
                output_dict['yesno'].append(yes_no_predictions[qa_id])
                output_dict['followup'].append(follow_up_predictions[qa_id])
            fout.write(json.dumps(output_dict) + '\n')


def quac_performance(prediction, target_dict):
    pred, truth = [], []

    for qa_id, span in prediction.items():
        dialog_id = qa_id.split("_q#")[0]
        
        if len(span) == 0:
            span = 'CANNOTANSWER'
        
        pred.append(span)
        truth.append(target_dict[qa_id])
    min_F1 = 0.4
    clean_pred, clean_truth = [], []

    all_f1 = []
    for p, t in zip(pred, truth):
        clean_t = handle_cannot(t)
        # compute human performance
        human_F1 = leave_one_out(clean_t)
        if human_F1 < min_F1: continue

        clean_pred.append(p)
        clean_truth.append(clean_t)
        all_f1.append(leave_one_out_max(p, clean_t))

    cur_f1, best_f1 = sum(all_f1), sum(all_f1)

    return 100.0 * best_f1 / len(clean_pred)



def cross_f1_mean(predictions):
    cross_f1_mean = []
    for i in range(len(predictions)):
        index = list(range(len(predictions)))
        index.pop(i)
        cross_f1_mean.append(sum([f1_score(predictions[i], predictions[j]) for j in index])/len(index))
    return cross_f1_mean
