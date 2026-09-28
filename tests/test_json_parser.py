from rare_core.rare_json_parser_service import clean_and_parse_json


def test_field_fallback_keeps_atomic_information_as_list() -> None:
    # Unterminated string: not valid JSON, so only the field-level fallback can recover the list.
    response = '{"atomic_information": [{"content": "Revenue was $60.9 billion."}], "note": "cut off}'

    parsed = clean_and_parse_json(response)

    assert parsed["atomic_information"] == [{"content": "Revenue was $60.9 billion."}]
