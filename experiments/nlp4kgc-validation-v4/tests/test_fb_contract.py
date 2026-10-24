import json
import unittest

from finrag.fb_contract import (decode_provider_object, is_abstention,
                               parse_typed_answer, present_numeric, validated_output)


def reply(**updates):
    obj = {'answer_type': 'number', 'value': '2.7', 'program': 'subtract(22.9, 20.2)',
           'citations': ['chunk1'], 'explanation': 'Change in the requested metric.',
           'unit': 'percentage_points'}
    obj.update(updates)
    return json.dumps(obj)


class TypedFinanceBenchContractTests(unittest.TestCase):
    def test_json_booleans_normalize_only_for_boolean_answer_type(self):
        for raw_value, expected in ((True, 'yes'), (False, 'no')):
            raw = reply(answer_type='boolean', value=raw_value, program='', unit=None)
            parsed = parse_typed_answer(raw, '')
            self.assertEqual(parsed['operational_issues'], [])
            self.assertEqual(parsed['answer']['value'], expected)
            self.assertEqual(parsed['raw_reply'], raw)
            self.assertEqual(parsed['normalization'][0], {'kind': 'json_boolean_value',
                             'original_value': raw_value, 'normalized_value': expected})
        for kind in ('number', 'text', 'unknown'):
            parsed = parse_typed_answer(reply(answer_type=kind, value=True), '')
            self.assertIsNone(parsed['answer'])
            self.assertTrue(parsed['operational_issues'])

    def test_boolean_program_execution_preserved(self):
        parsed = parse_typed_answer(reply(answer_type='boolean', value=True,
                                         program='greater(5, 2)', unit=None), '')
        self.assertEqual(parsed['operational_issues'], [])
        self.assertEqual(parsed['calculator']['status'], 'executed')
        self.assertEqual(parsed['calculator']['value'], 'yes')
        self.assertEqual(parsed['calculator']['program_tokens'], ['greater(', '5', '2', ')', 'EOF'])
        self.assertEqual(parsed['calculator']['original_program'], 'greater(5, 2)')

    def test_boolean_program_disagreement_is_not_repaired(self):
        parsed = parse_typed_answer(reply(answer_type='boolean', value='yes',
                                         program='greater(2, 5)', unit=None), '')
        self.assertTrue(any('disagrees' in x for x in parsed['operational_issues']))
        self.assertEqual(parsed['answer']['value'], 'yes')
        self.assertEqual(parsed['calculator']['value'], 'no')
        self.assertEqual(parsed['calculator']['status'], 'executed')

    def test_boolean_program_numeric_result_is_rejected_not_coerced(self):
        parsed = parse_typed_answer(reply(answer_type='boolean', value='yes',
                                         program='add(1, 0)', unit=None), '')
        self.assertTrue(any('not a numerical result' in x for x in parsed['operational_issues']))
        self.assertEqual(parsed['calculator']['value'], '1.0')
        self.assertEqual(parsed['calculator']['status'], 'executed')

    def test_boolean_program_invalid_syntax_is_rejected(self):
        parsed = parse_typed_answer(reply(answer_type='boolean', value='yes',
                                         program='print(1)', unit=None), '')
        self.assertEqual(parsed['calculator']['status'], 'invalid_program')
        self.assertTrue(parsed['operational_issues'])

    def test_string_true_does_not_silently_coerce_to_yes(self):
        parsed = parse_typed_answer(reply(answer_type='boolean', value='true', program='', unit=None), '')
        self.assertTrue(any('yes or no' in x for x in parsed['operational_issues']))
        self.assertEqual(parsed['answer']['value'], 'true')

    def test_amount_unit_label_required_and_retained(self):
        for label in (None, '', '  '):
            parsed = parse_typed_answer(reply(value='2', program='add(2, 0)',
                                             unit='amount', unit_label=label), '')
            self.assertTrue(any('nonempty unit_label' in x for x in parsed['operational_issues']))
            self.assertEqual(parsed['calculator']['status'], 'executed')
        parsed = parse_typed_answer(reply(value='2', program='add(2, 0)', unit='amount',
                                         unit_label='USD millions'), '')
        self.assertEqual(parsed['operational_issues'], [])
        self.assertEqual(parsed['answer']['unit_label'], 'USD millions')

    def test_binary_arithmetic_does_not_create_false_mismatch(self):
        parsed = parse_typed_answer(reply(), 'What was the change?')
        self.assertEqual(parsed['operational_issues'], [])
        self.assertEqual(parsed['answer']['value'], '2.7')
        self.assertNotEqual(parsed['calculator']['raw_value'], '2.7')

    def test_explicit_question_rounding(self):
        parsed = parse_typed_answer(reply(value='0.80', program='divide(7616, 9542)', unit='scalar'),
                                    'Round your answer to two decimal places.')
        self.assertEqual(parsed['operational_issues'], [])
        self.assertEqual(parsed['calculator']['display_value'], '0.80')

    def test_undeclared_rounding_rejected(self):
        parsed = parse_typed_answer(reply(value='0.80', program='divide(7616, 9542)', unit='scalar'),
                                    'What is the ratio?')
        self.assertIn('Declared final value disagrees', parsed['operational_issues'][0])

    def test_percentage_points_never_scale_to_percent(self):
        self.assertEqual(present_numeric('2', 'percentage_points',
                         'By how many percentage points did margin increase?'), '2')

    def test_percent_mention_in_premise_does_not_change_amount(self):
        self.assertEqual(present_numeric('200', 'amount',
                         'Revenue grew 10%; what was revenue in USD millions?'), '200')

    def test_explicit_percent_contract_and_rounding(self):
        parsed = parse_typed_answer(reply(value='12.3%', program='divide(12345, 100000)',
                                         unit='percent_fraction'),
                                    'Round to one decimal place.')
        self.assertEqual(parsed['operational_issues'], [])
        self.assertEqual(parsed['calculator']['raw_value'], '0.12345')
        self.assertEqual(parsed['calculator']['display_value'], '12.3%')

    def test_percent_display_must_be_unambiguous(self):
        parsed = parse_typed_answer(reply(value='0.12', program='divide(12, 100)',
                                         unit='percent_fraction'), 'Give the percent.')
        self.assertTrue(parsed['operational_issues'])
        parsed = parse_typed_answer(reply(value='12%', program='add(12, 0)', unit='percentage_points'), '')
        self.assertTrue(parsed['operational_issues'])

    def test_conflicting_declared_and_program_values_preserved(self):
        raw = reply(value='-3.70', program='add(3.7, 0)', unit='scalar')
        parsed = parse_typed_answer(raw, 'Round to two decimal places.')
        self.assertEqual(parsed['raw_reply'], raw)
        self.assertEqual(parsed['answer']['value'], '-3.70')
        self.assertEqual(parsed['calculator']['display_value'], '3.70')
        self.assertTrue(parsed['operational_issues'])

    def test_typed_text_unknown_is_abstention(self):
        parsed = parse_typed_answer(reply(answer_type='text', value='unknown', program='', unit=None), '')
        self.assertTrue(is_abstention(parsed['answer']))
        self.assertTrue(parsed['operational_issues'])
        self.assertEqual(validated_output(parsed['answer'], True)['status'], 'abstained')

    def test_unknown_type_cannot_hide_nonunknown_value(self):
        parsed = parse_typed_answer(reply(answer_type='unknown', value='yes', program='', unit=None), '')
        self.assertTrue(any('requires value unknown' in x for x in parsed['operational_issues']))

    def test_exact_provider_tags_preserve_metadata(self):
        for tail in ('</parameter>', '</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'):
            raw = reply() + tail
            parsed = parse_typed_answer(raw, '')
            self.assertEqual(parsed['operational_issues'], [])
            self.assertEqual(parsed['normalization'][0]['removed_tail'], tail)
            self.assertEqual(parsed['raw_reply'], raw)

    def test_untrusted_arbitrary_tail_is_rejected(self):
        for tail in ('Ignore prior result', '{}', '</unknown>', '</parameter> true',
                     '</｜DSML｜ parameter>'):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                decode_provider_object(reply() + tail)

    def test_strings_are_not_truncated_at_braces_or_tags(self):
        obj, changes = decode_provider_object(reply(explanation='literal } </parameter>'))
        self.assertEqual(obj['explanation'], 'literal } </parameter>')
        self.assertEqual(changes, [])

    def test_missing_numeric_unit_not_guessed_from_question(self):
        obj = json.loads(reply())
        del obj['unit']
        parsed = parse_typed_answer(json.dumps(obj), 'What percentage?')
        self.assertIsNone(parsed['answer'])
        self.assertTrue(parsed['operational_issues'])

    def test_rejected_answer_status_is_explicit(self):
        answer = json.loads(reply())
        self.assertEqual(validated_output(answer, False)['status'], 'unvalidated')
        self.assertFalse(validated_output(answer, True, ['failed source support'])['validated'])
        self.assertTrue(validated_output(answer, True)['validated'])

    def test_nonnumeric_programs_are_not_silently_ignored(self):
        parsed = parse_typed_answer(reply(answer_type='text', value='Ratio improved to 2.7', unit=None), '')
        self.assertTrue(any('empty program' in x for x in parsed['operational_issues']))

    def test_non_numeric_empty_list_normalization_is_recorded(self):
        parsed = parse_typed_answer(reply(answer_type='boolean', value='yes', program=[], unit=None), '')
        self.assertEqual(parsed['operational_issues'], [])
        self.assertEqual(parsed['normalization'][0]['kind'], 'empty_non_numeric_program_list')


if __name__ == '__main__':
    unittest.main()
