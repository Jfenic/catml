"""JSON Schema validation and canonical tool schema definitions for the agent catalog."""
from __future__ import annotations

from typing import Any


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Validate a python dictionary against a JSON Schema structure without external dependencies.
    
    Returns a list of error descriptions (empty list if valid).
    """
    errors: list[str] = []
    if not isinstance(arguments, dict):
        return [f"Arguments must be an object/dict, got {type(arguments).__name__}"]

    schema_type = schema.get("type", "object")
    if schema_type != "object":
        errors.append(f"Root schema must be of type 'object', got '{schema_type}'")

    # Check required fields
    for req in schema.get("required", []):
        if req not in arguments:
            errors.append(f"Missing required parameter: '{req}'")

    # Check additionalProperties
    allow_additional = schema.get("additionalProperties", True)
    properties = schema.get("properties", {})
    if not allow_additional:
        for key in arguments:
            if key not in properties:
                errors.append(f"Unexpected additional parameter: '{key}'")

    # Validate each present property
    for key, val in arguments.items():
        if key in properties:
            prop_schema = properties[key]
            prop_type = prop_schema.get("type")

            if prop_type == "string":
                if not isinstance(val, str):
                    errors.append(f"Parameter '{key}' must be a string, got {type(val).__name__}")
                elif "enum" in prop_schema and val not in prop_schema["enum"]:
                    errors.append(
                        f"Parameter '{key}' has value '{val}' not in allowed enum: {prop_schema['enum']}"
                    )
            elif prop_type == "integer":
                if not isinstance(val, int) or isinstance(val, bool):
                    errors.append(f"Parameter '{key}' must be an integer, got {type(val).__name__}")
                else:
                    if "minimum" in prop_schema and val < prop_schema["minimum"]:
                        errors.append(
                            f"Parameter '{key}' value {val} is below minimum {prop_schema['minimum']}"
                        )
                    if "maximum" in prop_schema and val > prop_schema["maximum"]:
                        errors.append(
                            f"Parameter '{key}' value {val} exceeds maximum {prop_schema['maximum']}"
                        )
            elif prop_type == "number":
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    errors.append(f"Parameter '{key}' must be a number, got {type(val).__name__}")
                else:
                    if "minimum" in prop_schema and val < prop_schema["minimum"]:
                        errors.append(
                            f"Parameter '{key}' value {val} is below minimum {prop_schema['minimum']}"
                        )
                    if "maximum" in prop_schema and val > prop_schema["maximum"]:
                        errors.append(
                            f"Parameter '{key}' value {val} exceeds maximum {prop_schema['maximum']}"
                        )
            elif prop_type == "boolean":
                if not isinstance(val, bool):
                    errors.append(f"Parameter '{key}' must be a boolean, got {type(val).__name__}")
            elif prop_type == "array":
                if not isinstance(val, list):
                    errors.append(f"Parameter '{key}' must be an array/list, got {type(val).__name__}")
                elif "items" in prop_schema:
                    item_type = prop_schema["items"].get("type")
                    for i, item in enumerate(val):
                        if item_type == "string" and not isinstance(item, str):
                            errors.append(f"Item {i} in array '{key}' must be a string")
                        elif item_type == "integer" and (not isinstance(item, int) or isinstance(item, bool)):
                            errors.append(f"Item {i} in array '{key}' must be an integer")
                        elif item_type == "number" and (not isinstance(item, (int, float)) or isinstance(item, bool)):
                            errors.append(f"Item {i} in array '{key}' must be a number")
            elif prop_type == "object":
                if not isinstance(val, dict):
                    errors.append(f"Parameter '{key}' must be an object/dict, got {type(val).__name__}")

    return errors


# Canonical schemas for the initial V0.9 catalog
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_dataset_profile": {
        "input": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
            },
            "required": ["dataset_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "n_rows": {"type": "integer"},
                "n_columns": {"type": "integer"},
                "columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["dataset_id"],
        },
    },
    "list_models": {
        "input": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": [
                        "binary_classification",
                        "multiclass_classification",
                        "regression",
                        "clustering",
                    ],
                },
            },
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "models": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["models"],
        },
    },
    "list_plugins": {
        "input": {
            "type": "object",
            "properties": {
                "plugin_type": {
                    "type": "string",
                    "enum": ["model", "metric", "preprocessor", "optimizer", "modality"],
                },
            },
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "plugins": {"type": "array"},
            },
            "required": ["plugins"],
        },
    },
    "list_experiments": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "experiments": {"type": "array"},
            },
            "required": ["experiments"],
        },
    },
    "get_leaderboard": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "leaderboard": {"type": "array"},
            },
            "required": ["leaderboard"],
        },
    },
    "get_feature_evidence": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "feature_id": {"type": "string"},
            },
            "required": ["run_id", "feature_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "evidence": {"type": "object"},
            },
            "required": ["evidence"],
        },
    },
    "get_feature_ranking": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "ranks": {"type": "array"},
            },
            "required": ["ranks"],
        },
    },
    "create_experiment": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "model_name": {"type": "string"},
                "feature_names": {"type": "array", "items": {"type": "string"}},
                "parameters": {"type": "object"},
            },
            "required": ["run_id", "model_name"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string"},
            },
            "required": ["experiment_id"],
        },
    },
    "prioritize_feature": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "feature_name": {"type": "string"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["run_id", "feature_name", "priority"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "feature_name": {"type": "string"},
                "priority": {"type": "string"},
            },
            "required": ["feature_name", "priority"],
        },
    },
    "run_experiment": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "experiment_id": {"type": "string"},
            },
            "required": ["run_id", "experiment_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "trial_id": {"type": "string"},
                "metric_value": {"type": "number"},
            },
            "required": ["trial_id"],
        },
    },
    "optimize_experiment": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "experiment_id": {"type": "string"},
                "n_trials": {"type": "integer", "minimum": 1, "maximum": 100},
                "optimizer": {"type": "string"},
            },
            "required": ["run_id", "experiment_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "best_trial_id": {"type": "string"},
                "best_score": {"type": "number"},
            },
            "required": ["best_trial_id"],
        },
    },
    "get_operation_status": {
        "input": {
            "type": "object",
            "properties": {
                "operation_id": {"type": "string"},
            },
            "required": ["operation_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "operation_id": {"type": "string"},
                "status": {"type": "string"},
                "action": {"type": "string"},
                "run_id": {"type": "string"},
            },
            "required": ["operation_id", "status"],
        },
    },
    "list_operations": {
        "input": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "operations": {"type": "array"},
            },
            "required": ["operations"],
        },
    },
    "cancel_operation": {
        "input": {
            "type": "object",
            "properties": {
                "operation_id": {"type": "string"},
                "reason": {"type": "string"},
                "force": {"type": "boolean"},
            },
            "required": ["operation_id"],
            "additionalProperties": False,
        },
        "output": {
            "type": "object",
            "properties": {
                "operation_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["operation_id", "status"],
        },
    },
    "analysis_create_study": {
        "input": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "name": {"type": "string"},
                "target_column": {"type": "string"},
            },
            "required": ["dataset_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "dataset_id": {"type": "string"},
            },
            "required": ["id"],
        },
    },
    "analysis_run_study": {
        "input": {
            "type": "object",
            "properties": {
                "study_id": {"type": "string"},
            },
            "required": ["study_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "study_id": {"type": "string"},
                "run_id": {"type": "string"},
                "findings_count": {"type": "integer"},
            },
            "required": ["study_id", "run_id"],
        },
    },
    "analysis_propose_experiment": {
        "input": {
            "type": "object",
            "properties": {
                "study_id": {"type": "string"},
                "finding_id": {"type": "string"},
                "hypothesis_description": {"type": "string"},
                "description": {"type": "string"},
                "hypothesis": {"type": "string"},
                "proposed_action": {"type": "string"},
                "transformation_spec": {"type": "object"},
                "experiment_delta": {"type": "object"},
            },
            "required": ["study_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
                "study_id": {"type": "string"},
            },
            "required": ["hypothesis_id"],
        },
    },
    "analysis_verify_hypothesis": {
        "input": {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
                "run_id": {"type": "string"},
                "min_improvement": {"type": "number"},
            },
            "required": ["hypothesis_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "evidence_link_id": {"type": "string"},
                "hypothesis_id": {"type": "string"},
                "status": {"type": "string"},
            },
        },
    },
    "recipe_create_from_evidence": {
        "input": {
            "type": "object",
            "properties": {
                "evidence_link_id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "model_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["evidence_link_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "recipe_id": {"type": "string"},
            },
            "required": ["recipe_id"],
        },
    },
    "recipe_select": {
        "input": {
            "type": "object",
            "properties": {
                "recipe_id": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["recipe_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "recipe_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["recipe_id"],
        },
    },
    "recipe_submit": {
        "input": {
            "type": "object",
            "properties": {
                "recipe_id": {"type": "string"},
                "run_id": {"type": "string"},
                "model_ids": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "required": ["recipe_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "experiment_ids": {"type": "array"},
            },
            "required": ["experiment_ids"],
        },
    },
    "recipe_promote": {
        "input": {
            "type": "object",
            "properties": {
                "recipe_id": {"type": "string"},
                "reason": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["recipe_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "recipe_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["recipe_id"],
        },
    },
    "recipe_archive": {
        "input": {
            "type": "object",
            "properties": {
                "recipe_id": {"type": "string"},
            },
            "required": ["recipe_id"],
        },
        "output": {
            "type": "object",
            "properties": {
                "recipe_id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["recipe_id"],
        },
    },
}
