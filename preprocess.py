# -*- coding: utf-8 -*-
import itertools
import os
import random
import string
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from glob import glob

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex
import seaborn as sns
from loguru import logger
from transformers.tokenization_bert import BasicTokenizer, _is_control, _is_whitespace

import file_utils
from annotation import (
    BinaryRelationAnnotation,
    TextAnnotations,
    TextBoundAnnotationWithText,
)

logger.add("logs/{time}.log")

sns.set(context="paper", style="whitegrid", font_scale=1.5, font="serif")

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

TOKENIZER = BasicTokenizer(do_lower_case=False)

TRIGGER_TYPES = {"Action"}
ENTITY_TYPES = {
    "Amount",
    "Concentration",
    "Device",
    "Generic-Measure",
    "Location",
    "Measure-Type",
    "Mention",
    "Method",
    "Modifier",
    "Numerical",
    "Reagent",
    "Seal",
    "Size",
    "Speed",
    "Temperature",
    "Time",
    "pH",
}
RELATION_ROLES = {
    "Acts-on",
    "Commands",
    "Coreference-Link",
    "Count",
    "Creates",
    "Measure",
    "Measure-Type-Link",
    "Meronym",
    "Mod-Link",
    "Of-Type",
    "Or",
    "Product",
    "Setting",
    "Site",
    "Using",
}
DEFINED_RULES = {
    "Acts-on": "Action|Action,Device,Location,Mention,Reagent,Seal",  # Checked
    "Creates": "Action|Reagent,Mention",  # Checked
    "Site": "Action|Device,Location,Mention,Reagent,Seal",  # Checked
    "Using": "Action,Method|Action,Device,Location,Mention,Method,Reagent,Seal",  # Checked
    "Setting": "Action,Device,Location,Modifier,Reagent|Action,ENTITIES",  # Checked
    "Count": "Action|Numerical",  # Checked
    "Measure-Type-Link": "Action|Measure-Type",  # Checked
    "Coreference-Link": "Action,ENTITIES|Action,ENTITIES",  # Checked
    "Mod-Link": "Action,ENTITIES|Modifier,Size",  # Checked
    "Measure": "Amount,Device,Location,Measure-Type,Mention,Method,Reagent,Seal|Amount,Concentration,Generic-Measure,Measure-Type,Numerical,Size,Temperature,pH",  # Checked # noqa: E501
    "Meronym": "Amount,Device,Location,Measure-Type,Mention,Reagent,Seal|Amount,Device,Location,Mention,Reagent,Seal",  # Checked # noqa: E501
    "Or": "Action,ENTITIES|Action,ENTITIES",  # Checked
    "Of-Type": "Amount,Generic-Measure,Numerical|Measure-Type",  # Checked
    "Commands": "Action|Action",  # Checked
    "Product": "Action|Location,Mention,Reagent",  # Checked
}


@click.group()
def cli():
    pass


def is_special_character(char):
    cp = ord(char)
    uc = unicodedata.category(char)

    return (
        cp == 0
        or cp == 0xFFFD
        or _is_control(char)
        or _is_whitespace(char)
        or uc.startswith("Z")
    )


def normalise_special_characters(s):
    return "".join(map(lambda c: " " if is_special_character(c) else c, s))


def shrink_offsets(doc, start_offset, end_offset):
    while start_offset < end_offset and is_special_character(doc[start_offset]):
        start_offset += 1

    while start_offset < end_offset and is_special_character(doc[end_offset - 1]):
        end_offset -= 1

    return start_offset, end_offset


def extend_offset(offset, doc, reverse=False):
    if reverse:
        while offset < len(doc) and regex.match(r"\S", doc[offset]):
            offset += 1
    else:
        while offset > 0 and regex.match(r"\S", doc[offset - 1]):
            offset -= 1

    return offset


def process_string(s):
    s = normalise_special_characters(s)
    s = regex.sub(r"\s+", " ", s)
    s = s.strip()

    return TOKENIZER.tokenize(s)


def split_sentences(doc):
    if doc and normalise_special_characters(doc).strip():
        original_start_offset = 0

        for sentence in doc.split("\n"):
            # Skip empty lines
            if normalise_special_characters(sentence).strip():
                original_end_offset = original_start_offset + len(sentence)

                start_offset, end_offset = shrink_offsets(
                    doc,
                    start_offset=original_start_offset,
                    end_offset=original_end_offset,
                )

                if start_offset < end_offset:
                    yield start_offset, end_offset

                original_start_offset = original_end_offset + 1


@lru_cache(maxsize=None)
def generate_relation_rules():
    relation_rules = defaultdict(set)

    for relation_role, rules in DEFINED_RULES.items():
        assert relation_role in RELATION_ROLES

        left_items, right_items = map(str.strip, rules.split("|"))

        left_items = set(map(str.strip, left_items.split(",")))
        right_items = set(map(str.strip, right_items.split(",")))

        for items in (left_items, right_items):
            if "ENTITIES" in items:
                items.remove("ENTITIES")
                items.update(ENTITY_TYPES)

            for item in items:
                assert item in TRIGGER_TYPES | ENTITY_TYPES

        for left_item, right_item in itertools.product(left_items, right_items):
            relation_rules[left_item, right_item].add(relation_role)

    return relation_rules


def fix_doc(doc, entities):
    marked_offsets = set()

    for entity in entities.values():
        span_start = entity["start"]
        span_end = entity["end"]

        if span_start > 0 and not is_special_character(doc[span_start - 1]):
            marked_offsets.add(span_start)

        if span_end < len(doc) and not is_special_character(doc[span_end]):
            marked_offsets.add(span_end)

    fixed_doc = list(doc)

    for offset in sorted(marked_offsets, reverse=True):
        fixed_doc.insert(offset, " ")

    fixed_doc = "".join(fixed_doc)

    original_pos = 0
    fixed_pos = 0

    offset_map = {}
    inverse_offset_map = {}

    while original_pos < len(doc) and fixed_pos < len(fixed_doc):
        original_char = doc[original_pos]
        fixed_char = fixed_doc[fixed_pos]

        if original_char == fixed_char:
            offset_map[original_pos] = fixed_pos
            inverse_offset_map[fixed_pos] = original_pos

            original_pos += 1
            fixed_pos += 1
        else:
            if original_char == " ":
                offset_map[original_pos] = fixed_pos
                original_pos += 1
            elif fixed_char == " ":
                inverse_offset_map[fixed_pos] = original_pos
                fixed_pos += 1

    if offset_map:
        offset_map[max(offset_map) + 1] = max(offset_map.values()) + 1

    if inverse_offset_map:
        inverse_offset_map[max(inverse_offset_map) + 1] = (
            max(inverse_offset_map.values()) + 1
        )

    for entity in entities.values():
        span_start, span_end = shrink_offsets(
            fixed_doc,
            start_offset=offset_map[entity["start"]],
            end_offset=offset_map[entity["end"]],
        )

        assert span_start < span_end and entity["text"].replace(" ", "") == fixed_doc[
            span_start:span_end
        ].replace(" ", "")

        entity["start"] = span_start
        entity["end"] = span_end

    return fixed_doc, entities, offset_map, inverse_offset_map


def align_doc(doc_id, doc, entities, relations):
    offset_map = {}
    sentence_boundaries = []

    for sentence_index, (sentence_start, sentence_end) in enumerate(
        split_sentences(doc)
    ):
        sentence_boundaries.append({"start": sentence_start, "end": sentence_end})

        for offset in range(sentence_start, sentence_end + 1):
            offset_map[offset] = sentence_index

    # Fix broken entities
    for entity in entities.values():
        left_sentence_index = offset_map[entity["start"]]
        right_sentence_index = offset_map[entity["end"]]

        if left_sentence_index != right_sentence_index:
            left_sentence = sentence_boundaries[
                min(left_sentence_index, right_sentence_index)
            ]

            left_sentence["broken"] = max(
                left_sentence.get("broken", -1),
                left_sentence_index,
                right_sentence_index,
            )

    # Fix broken relations
    for relation in relations.values():
        relation_left_arg_id = relation["left_arg_id"]
        relation_right_arg_id = relation["right_arg_id"]

        left_sentence_index = min(
            offset_map[entities[relation_left_arg_id]["start"]],
            offset_map[entities[relation_right_arg_id]["start"]],
        )
        right_sentence_index = max(
            offset_map[entities[relation_left_arg_id]["end"]],
            offset_map[entities[relation_right_arg_id]["end"]],
        )

        if left_sentence_index != right_sentence_index:
            left_sentence = sentence_boundaries[
                min(left_sentence_index, right_sentence_index)
            ]

            left_sentence["broken"] = max(
                left_sentence.get("broken", -1),
                left_sentence_index,
                right_sentence_index,
            )

    sentence_index = 0
    normalised_sentences = []

    while sentence_index < len(sentence_boundaries):
        original_sentence_index = sentence_index

        sentence_start = sentence_boundaries[sentence_index]["start"]
        sentence_end = sentence_boundaries[sentence_index]["end"]

        while (
            sentence_index < len(sentence_boundaries)
            and "broken" in sentence_boundaries[sentence_index]
        ):
            broken_sentence_index = sentence_boundaries[sentence_index]["broken"]

            assert broken_sentence_index >= 0

            sentence_end = sentence_boundaries[broken_sentence_index]["end"]
            sentence_index = broken_sentence_index

        if original_sentence_index != sentence_index:
            assert original_sentence_index < sentence_index

            logger.warning(
                "Sentences from `{}` to `{}` in `{}` will be merged together",
                original_sentence_index,
                sentence_index,
                doc_id,
            )

        normalised_sentences.append(doc[sentence_start:sentence_end])

        sentence_index += 1

    normalised_sentences = list(map(process_string, normalised_sentences))

    original_pos = 0
    normalised_pos = 0

    offset_map = {}
    inverse_offset_map = {}

    original_doc = normalise_special_characters(doc)
    normalised_doc = " ".join(" ".join(sentence) for sentence in normalised_sentences)

    while original_pos < len(original_doc) and normalised_pos < len(normalised_doc):
        original_char = original_doc[original_pos]
        normalised_char = normalised_doc[normalised_pos]

        if original_char == normalised_char:
            offset_map[original_pos] = normalised_pos
            inverse_offset_map[normalised_pos] = original_pos

            original_pos += 1
            normalised_pos += 1
        else:
            if original_char == " ":
                offset_map[original_pos] = normalised_pos
                original_pos += 1
            elif normalised_char == " ":
                inverse_offset_map[normalised_pos] = original_pos
                normalised_pos += 1

    if offset_map:
        offset_map[max(offset_map) + 1] = max(offset_map.values()) + 1

    if inverse_offset_map:
        inverse_offset_map[max(inverse_offset_map) + 1] = (
            max(inverse_offset_map.values()) + 1
        )

    for entity in entities.values():
        span_start, span_end = shrink_offsets(
            normalised_doc,
            start_offset=offset_map[entity["start"]],
            end_offset=offset_map[entity["end"]],
        )

        assert span_start < span_end

        span_start = extend_offset(span_start, normalised_doc, reverse=False)
        span_end = extend_offset(span_end, normalised_doc, reverse=True)

        assert span_start < span_end and normalise_special_characters(
            entity["text"]
        ).replace(" ", "") == normalised_doc[span_start:span_end].replace(" ", "")

        entity["start"] = span_start
        entity["end"] = span_end
        entity["text"] = normalised_doc[span_start:span_end]

    normalised_doc = "\n".join(" ".join(sentence) for sentence in normalised_sentences)

    return normalised_doc, offset_map, inverse_offset_map


def load_annotation_file(doc_file):
    with TextAnnotations(document=doc_file) as annotator:
        doc = annotator.get_document_text()

        entities = list(annotator.get_textbounds())
        relations = list(annotator.get_relations())
        events = list(annotator.get_events())

        return doc, entities, relations, events


def create_annotation_file(doc_file, doc, entities, relations, inverse_offset_map=None):
    with TextAnnotations(text=doc) as annotator:
        if isinstance(entities, dict):
            for entity_id in sorted(entities, key=lambda k: int(k.lstrip("TR"))):
                entity = entities[entity_id]

                TextBoundAnnotationWithText(
                    id=entity_id,
                    spans=[(entity["start"], entity["end"])],
                    type=entity["type"],
                    text=annotator,
                )
        else:
            for entity in sorted(entities, key=lambda e: int(e.id.lstrip("TR"))):
                TextBoundAnnotationWithText(
                    id=entity.id.replace("TR", "T"),
                    spans=entity.spans,
                    type=entity.type,
                    text=annotator,
                )

        if isinstance(relations, dict):
            for relation_id in sorted(relations, key=lambda k: int(k.lstrip("R"))):
                relation = relations[relation_id]

                annotator.add_annotation(
                    BinaryRelationAnnotation(
                        id=relation_id,
                        type=relation["role"],
                        arg1l="Arg1",
                        arg1=relation["left_arg_id"],
                        arg2l="Arg2",
                        arg2=relation["right_arg_id"],
                        tail="",
                    )
                )
        else:
            for relation in sorted(relations, key=lambda r: int(r.id.lstrip("R"))):
                annotator.add_annotation(
                    BinaryRelationAnnotation(
                        id=relation.id,
                        type=relation.type,
                        arg1l=relation.arg1l,
                        arg1=relation.arg1.replace("TR", "T"),
                        arg2l=relation.arg2l,
                        arg2=relation.arg2.replace("TR", "T"),
                        tail="",
                    )
                )

        file_utils.write_text(doc, filename=doc_file + ".txt")
        file_utils.write_text(str(annotator), filename=doc_file + ".ann")

        if inverse_offset_map:
            file_utils.write_json(inverse_offset_map, filename=doc_file + ".imap")


def normalise_entities(doc_id, doc, entities):
    normalised_entities = {}

    unique_entities = set()

    for entity in entities:
        entity_id = entity.id
        entity_type = entity.type

        if len(entity.spans) > 1:
            logger.warning(
                "Discontinuous spans of entity ID `{}` in `{}` "
                "will be merged together",
                entity_id,
                doc_id,
            )

        span_starts, span_ends = zip(*entity.spans)

        span_start, span_end = shrink_offsets(
            doc,
            start_offset=min(span_starts),
            end_offset=max(span_ends),
        )

        unique_id = f"{entity_type}:({span_start},{span_end})"

        assert (
            entity_id not in normalised_entities
            and entity_type in TRIGGER_TYPES | ENTITY_TYPES
            and span_start < span_end
            and unique_id not in unique_entities
        )

        unique_entities.add(unique_id)

        normalised_entities[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "start": span_start,
            "end": span_end,
            "text": doc[span_start:span_end],
        }

    return normalised_entities


def build_relations(doc_id, entities, relations, events):
    normalised_events = {}

    for event in events:
        event_id = event.id
        trigger_id = event.trigger
        event_type = event.type
        event_args = event.args

        assert (
            event_id not in normalised_events
            and event_type in TRIGGER_TYPES
            and event_type == entities[trigger_id]["type"]
        )

        normalised_events[event_id] = {
            "id": event_id,
            "trigger_id": trigger_id,
            "args": event_args,
        }

    max_relation_id = 0
    normalised_relations = {}

    for relation in relations:
        relation_id = relation.id
        relation_role = relation.type
        relation_left_arg_id = relation.arg1
        relation_right_arg_id = relation.arg2

        assert (
            relation_id not in normalised_relations and relation_role in RELATION_ROLES
        )

        max_relation_id = max(max_relation_id, int(relation_id.lstrip("R")))

        normalised_relations[relation_id] = {
            "id": relation_id,
            "role": relation_role,
            "left_arg_id": relation_left_arg_id,
            "right_arg_id": relation_right_arg_id,
        }

    generated_relations = {}

    for event in normalised_events.values():
        for arg_role, arg_id in event["args"]:
            arg_role = arg_role.rstrip(string.digits)

            assert arg_role in RELATION_ROLES and (
                arg_id in entities or arg_id in normalised_events
            )

            if arg_id in normalised_events:
                arg_id = normalised_events[arg_id]["trigger_id"]

            assert arg_id in entities and arg_id != event["trigger_id"]

            max_relation_id += 1

            relation_id = "R" + str(max_relation_id)

            assert relation_id not in generated_relations

            generated_relations[relation_id] = {
                "id": relation_id,
                "role": arg_role,
                "left_arg_id": event["trigger_id"],
                "right_arg_id": arg_id,
            }

    for relation in normalised_relations.values():
        relation_id = relation["id"]
        relation_left_arg_id = relation["left_arg_id"]
        relation_right_arg_id = relation["right_arg_id"]

        if relation_left_arg_id in normalised_events:
            relation_left_arg_id = normalised_events[relation_left_arg_id]["trigger_id"]

        if relation_right_arg_id in normalised_events:
            relation_right_arg_id = normalised_events[relation_right_arg_id][
                "trigger_id"
            ]

        assert (
            relation_id not in generated_relations
            and relation_left_arg_id in entities
            and relation_right_arg_id in entities
        )

        generated_relations[relation_id] = {
            "id": relation_id,
            "role": relation["role"],
            "left_arg_id": relation_left_arg_id,
            "right_arg_id": relation_right_arg_id,
        }

    unique_relations = defaultdict(set)

    for relation_id in list(generated_relations):
        relation = generated_relations[relation_id]

        unique_id = f'({relation["left_arg_id"]},{relation["right_arg_id"]})'

        if unique_id in unique_relations:
            if relation["role"] in unique_relations[unique_id]:
                logger.warning(
                    "Relation ID `{}` in `{}` will be skipped due to duplication",
                    relation_id,
                    doc_id,
                )

                del generated_relations[relation_id]
            else:
                logger.warning("Found a multi-label relation in `{}`", doc_id)

        unique_relations[unique_id].add(relation["role"])

    return generated_relations


@cli.command()
@click.option(
    "-i", "--input_dir", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option("-o", "--output_dir", required=True, type=click.Path(file_okay=False))
@click.option("-e", "--export_relation_rules", type=click.Path(dir_okay=False))
def preprocess_corpus(input_dir, output_dir, export_relation_rules):
    relation_rules = defaultdict(lambda: {"left_args": set(), "right_args": set()})

    for ann_file in glob(os.path.join(input_dir, "**", "*.ann"), recursive=True):
        doc_file, _ = os.path.splitext(ann_file)
        doc_id = os.path.basename(doc_file)

        logger.info("Processing: {}", doc_id)

        doc, entities, relations, events = load_annotation_file(doc_file=doc_file)

        entities = normalise_entities(doc_id=doc_id, doc=doc, entities=entities)

        relations = build_relations(
            doc_id=doc_id, entities=entities, relations=relations, events=events
        )

        if export_relation_rules:
            logger.info("Extracting relation rules...")

            for relation in relations.values():
                relation_rules[relation["role"]]["left_args"].add(
                    entities[relation["left_arg_id"]]["type"]
                )
                relation_rules[relation["role"]]["right_args"].add(
                    entities[relation["right_arg_id"]]["type"]
                )

        doc, entities, _, inverse_offset_map_s1 = fix_doc(doc=doc, entities=entities)

        processed_doc, _, inverse_offset_map_s2 = align_doc(
            doc_id=doc_id, doc=doc, entities=entities, relations=relations
        )

        inverse_offset_map = {"s1": inverse_offset_map_s1, "s2": inverse_offset_map_s2}

        create_annotation_file(
            doc_file=os.path.join(output_dir, os.path.relpath(doc_file, input_dir)),
            doc=processed_doc,
            entities=entities,
            relations=relations,
            inverse_offset_map=inverse_offset_map,
        )

    if export_relation_rules:
        relation_rules = [
            f"{relation_role}: "
            f'{",".join(sorted(relation_rules[relation_role]["left_args"]))}|'
            f'{",".join(sorted(relation_rules[relation_role]["right_args"]))}'
            for relation_role in sorted(relation_rules)
        ]

        file_utils.write_lines(relation_rules, filename=export_relation_rules)


@cli.command()
@click.option(
    "-i", "--input_dir", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option("-o", "--output_dir", required=True, type=click.Path(file_okay=False))
def flatten_events(input_dir, output_dir):
    for ann_file in glob(os.path.join(input_dir, "**", "*.ann"), recursive=True):
        doc_file, _ = os.path.splitext(ann_file)
        doc_id = os.path.basename(doc_file)

        logger.info("Processing: {}", doc_id)

        doc, gold_entities, relations, events = load_annotation_file(doc_file=doc_file)

        entities = normalise_entities(doc_id=doc_id, doc=doc, entities=gold_entities)

        relations = build_relations(
            doc_id=doc_id, entities=entities, relations=relations, events=events
        )

        create_annotation_file(
            doc_file=os.path.join(output_dir, os.path.relpath(doc_file, input_dir)),
            doc=doc,
            entities=gold_entities,
            relations=relations,
        )


@cli.command()
@click.option(
    "-g",
    "--original_corpus_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-p",
    "--predicted_corpus_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-o", "--output_dir", required=True, type=click.Path(file_okay=False))
@click.option(
    "-f",
    "--filter_relations",
    default=False,
    show_default=True,
    type=click.BOOL,
    is_flag=True,
)
def recover_corpus(
    original_corpus_dir, predicted_corpus_dir, output_dir, filter_relations
):
    relation_rules = generate_relation_rules()

    for ann_file in glob(os.path.join(original_corpus_dir, "*.ann")):
        doc_file, _ = os.path.splitext(ann_file)
        doc_id = os.path.basename(doc_file)

        logger.info("Processing: {}", doc_id)

        gold_doc, entities, *_ = load_annotation_file(doc_file=doc_file)

        _, predicted_entities, predicted_relations, _ = load_annotation_file(
            doc_file=os.path.join(predicted_corpus_dir, doc_id)
        )

        gold_entities = {entity.id: entity for entity in entities}

        predicted_entities = {
            entity.id.replace("TR", "T"): entity for entity in predicted_entities
        }

        if len(gold_entities) != len(predicted_entities):
            logger.warning(
                "Missing gold entities in prediction: {}/{}",
                len(predicted_entities),
                len(gold_entities),
            )

        for entity_id, gold_entity in gold_entities.items():
            if entity_id not in predicted_entities:
                logger.warning("Missing gold entity ID: {}", entity_id)

                continue

            predicted_entity = predicted_entities[entity_id]

            assert normalise_special_characters(gold_entity.text).replace(
                " ", ""
            ) == predicted_entity.text.replace(" ", "")

        filtered_predicted_relations = []

        unique_predicted_relations = set()

        for relation in predicted_relations:
            unique_id = (relation.type, relation.arg1, relation.arg2)

            assert unique_id not in unique_predicted_relations

            unique_predicted_relations.add(unique_id)

            relation_pair = (
                gold_entities[relation.arg1].type,
                gold_entities[relation.arg2].type,
            )

            if filter_relations and relation.type not in relation_rules.get(
                relation_pair, {}
            ):
                logger.warning(
                    "Ignored violated rule: {} {}:{}-{}",
                    relation.id,
                    relation.type,
                    *relation_pair,
                )

                continue

            filtered_predicted_relations.append(relation)

        create_annotation_file(
            os.path.join(output_dir, doc_id),
            doc=gold_doc,
            entities=entities,
            relations=filtered_predicted_relations,
        )


def count_overlapping_entities(entities):
    num_overlapping_entities = 0

    if len(entities) > 1:
        entities = sorted(map(lambda e: (e["start"], e["end"]), entities))

        has_overlapping_entities = False

        (_, previous_entity_end), *entities = entities

        for entity_start, entity_end in entities:
            if entity_start <= previous_entity_end:
                num_overlapping_entities += 1
                has_overlapping_entities = True
                previous_entity_end = max(previous_entity_end, entity_end)
            else:
                num_overlapping_entities += has_overlapping_entities
                has_overlapping_entities = False
                previous_entity_end = entity_end

        num_overlapping_entities += has_overlapping_entities

    return num_overlapping_entities


@cli.command()
@click.option(
    "-i", "--input_dir", required=True, type=click.Path(exists=True, file_okay=False)
)
def print_statistics(input_dir):
    docs = {}

    for ann_file in glob(os.path.join(input_dir, "**", "*.ann"), recursive=True):
        doc_file, _ = os.path.splitext(ann_file)
        doc_id = os.path.basename(doc_file)

        logger.info("Processing: {}", doc_id)

        doc, entities, relations, _ = load_annotation_file(doc_file=doc_file)

        entities = normalise_entities(doc_id=doc_id, doc=doc, entities=entities)

        cursor = 0
        token_start_offsets = {}
        token_end_offsets = {}

        docs[doc_id] = {"sentences": [], "entities": entities}

        for sentence_index, sentence in enumerate(doc.split("\n")):
            tokens = sentence.split(" ")

            for token_index, token in enumerate(tokens):
                token_start_offsets[cursor] = (sentence_index, token_index)
                token_end_offsets[cursor + len(token)] = (
                    sentence_index,
                    token_index,
                )
                cursor += len(token) + 1

            docs[doc_id]["sentences"].append(
                {"tokens": tokens, "entities": [], "relations": []}
            )

        assert len(doc) == cursor == 0 or len(doc) == cursor - 1

        for entity in entities.values():
            left_sentence_index, start_offset = token_start_offsets[entity["start"]]
            right_sentence_index, end_offset = token_end_offsets[entity["end"]]

            assert (
                left_sentence_index == right_sentence_index
                and start_offset <= end_offset
            )

            entity["start"] = start_offset
            entity["end"] = end_offset

            entity["sentence_index"] = left_sentence_index

            docs[doc_id]["sentences"][left_sentence_index]["entities"].append(entity)

        for relation in relations:
            left_sentence_index = entities[relation.arg1]["sentence_index"]
            right_sentence_index = entities[relation.arg2]["sentence_index"]

            assert left_sentence_index == right_sentence_index

            relation.sentence_index = left_sentence_index

            docs[doc_id]["sentences"][left_sentence_index]["relations"].append(relation)

    # Statistics
    num_docs = 0
    num_sentences = 0
    num_tokens = 0

    num_entities = 0
    num_overlapping_entities = 0

    num_relations = 0

    entity_width_distribution = Counter()

    entity_type_distribution = Counter()

    sentence_length_distribution = Counter()

    entity_end_offset_distribution = Counter()

    relation_role_distribution = Counter()

    relation_min_span_distribution = Counter()
    relation_max_span_distribution = Counter()

    relation_end_offset_distribution = Counter()

    for doc in docs.values():
        num_docs += 1

        for sentence in doc["sentences"]:
            num_sentences += 1
            num_tokens += len(sentence["tokens"])

            num_entities += len(sentence["entities"])
            num_overlapping_entities += count_overlapping_entities(sentence["entities"])

            num_relations += len(sentence["relations"])

            sentence_length_distribution[len(sentence["tokens"])] += 1

            for entity in sentence["entities"]:
                entity_width_distribution[entity["end"] - entity["start"] + 1] += 1

                entity_type_distribution[entity["type"]] += 1

                entity_end_offset_distribution[entity["end"] // 64 * 64] += 1

            for relation in sentence["relations"]:
                relation_role_distribution[relation.type] += 1

                relation_min_span_distribution[
                    max(
                        doc["entities"][relation.arg1]["start"],
                        doc["entities"][relation.arg2]["start"],
                    )
                    - min(
                        doc["entities"][relation.arg1]["end"],
                        doc["entities"][relation.arg2]["end"],
                    )
                ] += 1

                relation_max_span_distribution[
                    max(
                        doc["entities"][relation.arg1]["end"],
                        doc["entities"][relation.arg2]["end"],
                    )
                    - min(
                        doc["entities"][relation.arg1]["start"],
                        doc["entities"][relation.arg2]["start"],
                    )
                ] += 1

                relation_end_offset_distribution[
                    max(
                        doc["entities"][relation.arg1]["end"],
                        doc["entities"][relation.arg2]["end"],
                    )
                    // 64
                    * 64
                ] += 1

    logger.info("Number of documents: {}", num_docs)
    logger.info("Number of sentences: {}", num_sentences)
    logger.info("Number of tokens: {}", num_tokens)

    logger.info("Number of entities: {}", num_entities)
    logger.info("Number of overlapping entities: {}", num_overlapping_entities)
    logger.info("Number of entity types: {}", len(entity_type_distribution))

    logger.info("Number of relations: {}", num_relations)
    logger.info("Number of relation roles: {}", len(relation_role_distribution))

    logger.info("Min entity span width: {}", min(entity_width_distribution.keys()))
    logger.info("Max entity span width: {}", max(entity_width_distribution.keys()))

    logger.info(
        "Min relation min span width: {}", min(relation_min_span_distribution.keys())
    )
    logger.info(
        "Max relation min span width: {}", max(relation_min_span_distribution.keys())
    )

    logger.info(
        "Min relation max span width: {}", min(relation_max_span_distribution.keys())
    )
    logger.info(
        "Max relation max span width: {}", max(relation_max_span_distribution.keys())
    )

    logger.info("Min sentence length: {}", min(sentence_length_distribution.keys()))
    logger.info("Max sentence length: {}", max(sentence_length_distribution.keys()))

    series = pd.Series(entity_width_distribution).sort_index().cumsum() / num_entities
    series.plot(title="Entity Coverage Rate over Span Length")
    plt.show()

    pd.Series(relation_min_span_distribution).sort_index().plot(
        kind="bar",
        title="Relation Min-gap Distribution",
        color=sns.color_palette(),
    )
    plt.show()

    pd.Series(relation_max_span_distribution).sort_index().plot(
        kind="bar",
        title="Relation Max-gap Distribution",
        color=sns.color_palette(),
    )
    plt.show()

    series = (
        pd.Series(entity_end_offset_distribution).sort_index().cumsum() / num_entities
    )
    series.plot(
        title="Entity Coverage Rate per Sentence Segment (64)",
        xticks=range(
            0, (max(sentence_length_distribution.keys()) // 64 + 1) * 64 + 1, 64
        ),
    )
    plt.show()

    series = (
        pd.Series(relation_end_offset_distribution).sort_index().cumsum()
        / num_relations
    )
    series.plot(
        title="Relation Coverage Rate over Sentence Segment (64)",
        xticks=range(
            0, (max(sentence_length_distribution.keys()) // 64 + 1) * 64 + 1, 64
        ),
    )
    plt.show()

    pd.Series(entity_type_distribution).sort_index().plot(
        kind="bar",
        title="Entity Type Distribution",
        color=sns.color_palette(),
    )
    plt.show()

    pd.Series(relation_role_distribution).sort_index().plot(
        kind="bar",
        title="Relation Role Distribution",
        color=sns.color_palette(),
    )
    plt.show()

    pd.Series(sentence_length_distribution).sort_index().plot(
        kind="bar",
        title="Sentence Length Distribution",
        color=sns.color_palette(),
    )
    plt.show()


@cli.command()
@click.option(
    "-i", "--input_dir", required=True, type=click.Path(exists=True, file_okay=False)
)
def validate_relation_rules(input_dir):
    relation_rules = generate_relation_rules()

    for ann_file in glob(os.path.join(input_dir, "**", "*.ann"), recursive=True):
        doc_file, _ = os.path.splitext(ann_file)
        doc_id = os.path.basename(doc_file)

        logger.info("Processing: {}", doc_id)

        doc, entities, relations, _ = load_annotation_file(doc_file=doc_file)

        entities = normalise_entities(doc_id=doc_id, doc=doc, entities=entities)

        for relation in relations:
            relation_pair = (
                entities[relation.arg1]["type"],
                entities[relation.arg2]["type"],
            )

            if relation.type not in relation_rules.get(relation_pair, {}):
                logger.warning(
                    "Found violated rule: {} {}:{}-{}",
                    relation.id,
                    relation.type,
                    *relation_pair,
                )


@cli.command()
@click.option(
    "-g",
    "--original_corpus_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-p",
    "--prediction_dirs",
    required=True,
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-o", "--output_dir", required=True, type=click.Path(file_okay=False))
def ensemble(original_corpus_dir, prediction_dirs, output_dir):
    logger.warning(
        "The prediction directories should be provided in descending order of scores."
    )

    logger.info("Num predictions:{}", len(prediction_dirs))

    gold_docs = {}

    for ann_file in glob(os.path.join(original_corpus_dir, "*.ann")):
        doc_file, _ = os.path.splitext(ann_file)
        doc_id = os.path.basename(doc_file)

        gold_doc, gold_entities, *_ = load_annotation_file(doc_file=doc_file)

        gold_docs[doc_id] = (gold_doc, gold_entities)

    assert gold_docs, "Original corpus folder has no files"

    num_accepted_votes = len(prediction_dirs) // 2

    predicted_docs = defaultdict(lambda: defaultdict(Counter))

    for prediction_dir in prediction_dirs:
        ann_files = glob(os.path.join(prediction_dir, "*.ann"))

        assert ann_files, f"Found no files in `{prediction_dir}`"

        for ann_file in ann_files:
            doc_file, _ = os.path.splitext(ann_file)
            doc_id = os.path.basename(doc_file)

            _, _, predicted_relations, _ = load_annotation_file(doc_file=doc_file)

            for relation in predicted_relations:
                predicted_docs[doc_id][relation.arg1, relation.arg2][relation.type] += 1

    for doc_id, (gold_doc, gold_entities) in gold_docs.items():
        logger.info("Processing: {}", doc_id)

        relation_id = 0
        predicted_relations = {}

        for (left_arg_id, right_arg_id), counter in predicted_docs[doc_id].items():
            for relation_role, num_votes in counter.most_common(n=1):  # Boost recall
                if num_votes >= num_accepted_votes:  # Boost precision
                    relation_id += 1

                    predicted_relations[f"R{relation_id}"] = {
                        "id": f"R{relation_id}",
                        "role": relation_role,
                        "left_arg_id": left_arg_id,
                        "right_arg_id": right_arg_id,
                    }

        create_annotation_file(
            os.path.join(output_dir, doc_id),
            doc=gold_doc,
            entities=gold_entities,
            relations=predicted_relations,
        )


if __name__ == "__main__":
    cli()
