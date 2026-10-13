"""Exercise the actual constrained decoder without loading model weights."""
import json

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig


def accepts_json(value, schema):
    parser = JsonSchemaParser(schema, CharacterLevelParserConfig(force_json_field_order=True))
    for character in json.dumps(value):
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()
